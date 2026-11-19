"""An offline stand-in for OpenAI, active only when LLM_MOCK=1.

It exists so the pipeline, the app and the experiments can be exercised without a key or any cost. Its answers are
simple keyword rules, not intelligence, so results made with it say nothing about real quality. The app shows a warning
banner whenever it is active.
"""
import json
import re

from src import llm
from src.config import cfg
from src.schemas import (EnrichBatch, EnrichedPlace, Explanation, Explanations, MediatorDecision, MemberReply,
                         MemberWeight, PersonaText, PlaceScore, PlannedDay, PlannerOut, ProfileOut, ScoreBatch,
                         ThemeScores)

KEYWORDS = {
    "art": ["art", "paint", "museum", "gallery", "impressionis", "sculpt", "monet"],
    "history": ["history", "histor", "royal", "medieval", "old ", "revolution", "napoleon"],
    "architecture": ["architect", "church", "churches", "cathedral", "building", "gothic", "palace"],
    "food": ["food", "eat", "restaurant", "lunch", "dinner", "bistro", "wine", "cheese", "gastronom"],
    "nature": ["park", "garden", "nature", "green", "walk", "outdoor"],
    "views": ["view", "panorama", "rooftop", "photo", "tower"],
    "shopping": ["shop", "fashion", "boutique", "department"],
    "nightlife": ["nightlife", "bar", "club", "party", "cabaret", "evening"],
    "family": ["kid", "child", "family", "teen", "zoo", "aquarium"],
    "local_life": ["local", "market", "neighbourhood", "neighborhood", "everyday", "street"],
}
CATEGORY_THEMES = {
    "museum": {"art": 4, "history": 3}, "gallery": {"art": 5}, "worship": {"architecture": 4, "history": 3},
    "park": {"nature": 5, "family": 2}, "cemetery": {"history": 3, "nature": 2}, "neighbourhood": {"local_life": 5, "food": 2},
    "square": {"architecture": 3, "history": 2}, "bridge": {"views": 3, "architecture": 3}, "canal": {"local_life": 3, "nature": 2},
    "street": {"local_life": 4, "shopping": 2}, "shopping": {"shopping": 5}, "market": {"local_life": 5, "food": 4},
    "monument": {"history": 4, "architecture": 3, "views": 2}, "palace": {"history": 4, "architecture": 4},
    "zoo": {"family": 5, "nature": 3}, "cruise": {"views": 4, "family": 2}, "viewpoint": {"views": 5},
    "attraction": {"views": 2, "history": 2}, "restaurant": {"food": 4}, "cafe": {"food": 2, "local_life": 3},
}
ID = r"(?:wd:Q\d+|osm:(?:node|way|relation)/\d+|[ar]\d_\d|h\d)"    # the last two are the test city's ids


def _themes(text: str, default: int = 2) -> dict[str, int]:
    low = f" {text.lower()} "
    return {t: min(5, default + 2 * sum(low.count(k) for k in ks)) for t, ks in KEYWORDS.items()}


def enrich(system: str, user: str) -> EnrichBatch:
    places = []
    for line in user.splitlines():
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) < 5 or not re.fullmatch(ID, parts[0]):
            continue
        pid, name, cat, text = parts[0], parts[1], parts[2], parts[3]
        scores = {t: 0 for t in cfg().themes}
        scores.update(CATEGORY_THEMES.get(cat, {}))
        low = f"{name} {text}".lower()
        for t, ks in KEYWORDS.items():
            if any(k in low for k in ks):
                scores[t] = min(5, scores[t] + 1)
        tier = None
        if cat in ("restaurant", "cafe"):
            tier = 1 + sum(ord(c) for c in name) % 2 if cat == "cafe" else 1 + sum(ord(c) for c in name) % 3
        places.append(EnrichedPlace(id=pid, themes=ThemeScores(**scores), price_tier=tier,
                                    indoor=cat in ("museum", "gallery", "worship", "shopping")))
    return EnrichBatch(places=places)


def profile(system: str, user: str) -> ProfileOut:
    text = user.split("Wishes:", 1)[-1]
    low = text.lower()
    must = re.findall(r"must see ([A-Z][^.,;]+)", text)
    avoid = re.findall(r"avoid ([A-Z][^.,;]+)", text)
    budget = re.search(r"(\d{3,5})\s*(?:eur|euros|€)", low)
    pace = "relaxed" if "relax" in low or "slow" in low else "packed" if "packed" in low or "see everything" in low else "normal"
    return ProfileOut(theme_importance=ThemeScores(**_themes(text)), budget_eur=float(budget.group(1)) if budget else None,
                      vegetarian="vegetarian" in low or "vegan" in low,
                      mobility="wheelchair" in low or "difficulty walking" in low or "little walking" in low,
                      must_see_names=must, avoid_names=avoid, pace=pace, language="English")


def score(system: str, user: str) -> ScoreBatch:
    wishes = user.split("Wishes:", 1)[-1].split("Profile:", 1)[0]
    want = _themes(wishes, default=0)
    out = []
    for line in user.splitlines():
        parts = [p.strip() for p in line.split(" | ")]
        if not re.fullmatch(ID, parts[0]):
            continue
        strong = dict(re.findall(r"([a-z_]+) (\d)", parts[4])) if len(parts) > 4 else {}
        s = 1 + sum(want.get(t, 0) * int(v) for t, v in strong.items()) / 8
        out.append(PlaceScore(id=parts[0], score=int(min(5, round(s)))))
    return ScoreBatch(scores=out)


def reply(system: str, user: str) -> MemberReply:
    own = float(re.search(r"Yours: (\d+\.\d+)", user).group(1))
    mean = float(re.search(r"Group mean: (\d+\.\d+)", user).group(1))
    if own >= mean or own >= 0.9:
        return MemberReply(accept=True, add=[], remove=[], give_up="", reason="This works for me.")
    candidates = re.search(r"Places you may add.*?:\n(.*?)\nEarlier rounds", user, re.S).group(1)
    ids = re.findall(rf"^({ID}) \|", candidates, re.M)
    return MemberReply(accept=False, add=ids[:1], remove=[], give_up="a slower morning",
                       reason="I would like one more place I care about.")


def mediate(system: str, user: str) -> MediatorDecision:
    members = re.findall(r"^(m\d+) \| [^|]+ \| ([\d., ]+) \| ", user, re.M)
    sats = {m: float(s.split(",")[-1]) for m, s in members}
    low = re.search(r"between (\d+\.\d+) and (\d+\.\d+)", system)
    lo, hi = (float(low.group(1)), float(low.group(2))) if low else (0.1, 0.9)
    worst = min(sats, key=sats.get) if sats else None
    weights = [MemberWeight(member_id=m, weight=hi if m == worst else lo) for m in sats]
    asked = re.findall(rf"add \[({ID})", user)   # the first place each member asked for
    return MediatorDecision(include=asked[:2], exclude=[], weights=weights, note="Favouring the least satisfied member.")


def explain(system: str, user: str) -> Explanations:
    items = []
    for line in user.split("Members", 1)[-1].splitlines():
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) >= 3 and parts[0].startswith("m"):
            items.append(Explanation(member_id=parts[0], text=f"TEST TEXT from the offline stand-in, not a real explanation, for {parts[1]}."))
    return Explanations(items=items)


def planner(system: str, user: str) -> PlannerOut:
    """A naive planner on purpose: top of the list, no attention to distance or hours, so violations appear."""
    days = len(re.search(r"Dates: (.*)", user).group(1).split(","))
    rows = [p.split(" | ") for p in user.split("Candidates", 1)[-1].splitlines()[1:]]
    pick = lambda kind: [r[0] for r in rows if len(r) > 2 and r[2] == kind]   # noqa: E731
    att, res, hot = pick("attraction"), pick("restaurant"), pick("hotel")
    plan_days = [PlannedDay(place_ids=att[d * 4:d * 4 + 4], lunch_after=2, lunch_id=res[d * 2], dinner_id=res[d * 2 + 1])
                 for d in range(days)]
    return PlannerOut(hotel_id=hot[0], days=plan_days, reasoning="Offline stand-in.")


def persona(system: str, user: str) -> PersonaText:
    fields = dict(re.findall(r"^([A-Za-z ]+): (.*)$", user, re.M))
    must = fields.get("Must see", "none")
    parts = [f"My passion is {fields.get('Specific passion')}.", f"I really love {fields.get('Loves')}", f"and I also like {fields.get('Likes')}.",
             f"I am not into {fields.get('Not interested in')}."]
    if must != "none":
        parts.append(f"I must see {must}.")
    if fields.get("Needs", "none") != "none":
        parts.append(f"Note: {fields['Needs']}.")
    return PersonaText(wishes=" ".join(parts))


def install() -> None:
    llm.MOCKS.update({"enrich": enrich, "profile": profile, "score": score, "reply": reply, "mediator": mediate,
                      "explain": explain, "llm_planner": planner, "persona": persona})


__all__ = ["install", "json"]
