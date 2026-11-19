"""Simulated travelers. Each has a hidden true profile; the system only sees the wishes an LLM writes from it,
plus the form fields a real person would fill in (budget, vegetarian, mobility, pace)."""
import json
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np

from src.config import ROOT, cfg
from src.llm import MODEL_STRONG, LLMError, call_many, embed
from src.prompts import load
from src.schemas import MemberInput, PersonaText, TripRequest

KINDS = ("friends", "family", "colleagues", "minority")
THEME_WORDS = {
    "art": "art and museums", "history": "history and historic sites", "architecture": "architecture and beautiful buildings",
    "food": "food and good restaurants", "nature": "parks and nature", "views": "views and panoramas",
    "shopping": "shopping", "nightlife": "nightlife", "family": "activities that work for children and teenagers",
    "local_life": "neighbourhoods, markets and everyday local life",
}
NAMES = ["Sara", "Omar", "Lina", "Karim", "Maya", "Hadi", "Nour", "Rami", "Yara", "Ziad"]
BUDGETS = (450, 900, 1600)
SENSITIVITY = {450: 1.0, 900: 0.3, 1600: 0.0}
# Real people want more than a mix of broad themes. Each traveler has one specific passion that no theme can express, so
# the simulation can reward a system that reads the wishes closely, as a real person would. Without it the hidden taste is
# built from the same themes as the profile, and nothing an agent understands from the text could ever help.
INTERESTS = ("Impressionist painting", "medieval churches and Gothic architecture", "the French Revolution and Napoleon",
             "modern and contemporary art", "science and natural history", "street markets and food shopping", "fashion and design",
             "gardens and quiet green walks", "photography and big city views", "opera, music and theatre",
             "ancient Rome and archaeology", "writers and literary Paris", "Asian and world cultures",
             "street art and bohemian neighbourhoods")


@dataclass
class HiddenMember:
    member_id: str
    name: str
    w: list[float]              # 10 theme weights, sum 1
    budget_eur: float
    price_sensitivity: float    # how much a high price lowers their true utility
    vegetarian: bool
    mobility: bool
    must_see: list[str]         # 0 or 1 place id
    pace: str
    noise_seed: int
    interest: str = ""          # one specific passion, written into the wishes but not into the themes

    def passion(self, places) -> np.ndarray:
        """Extra true utility for the attractions closest to the passion, in meaning. About the top 15% get some, up to 2 points."""
        out = np.zeros(len(places.items))
        if not self.interest:
            return out
        q = embed([self.interest])[0]
        att = places.idx("attraction")
        cos = places.unit[att] @ (q / np.linalg.norm(q))
        out[att] = cfg().experiment.interest_weight * np.clip((cos - cos.mean()) / cos.std() - 1, 0, 2)
        return out

    def true_utility(self, places) -> np.ndarray:
        u = places.themes @ np.asarray(self.w) + self.passion(places)
        u = u + np.random.default_rng(self.noise_seed).normal(0, 0.3, len(u))      # drawn once: it is part of the person
        for kind in ("attraction", "restaurant", "hotel"):                         # price scaled 0..1 within each kind
            i = places.idx(kind)
            u[i] -= self.price_sensitivity * places.price[i] / max(places.price[i].max(), 1)
        if self.vegetarian:
            u[[i for i, p in enumerate(places.items) if p.kind == "restaurant" and p.veg_friendly is not True]] = 0
        return np.clip(u, 0, 5)


@dataclass
class Group:
    group_id: str
    kind: str
    members: list[HiddenMember]
    wishes: dict[str, str] = field(default_factory=dict)
    start_date: str = ""
    days: int = 3

    @property
    def outsider(self) -> str | None:
        return self.members[-1].member_id if self.kind == "minority" else None

    def trip(self) -> TripRequest:
        return TripRequest(trip_id=self.group_id, start_date=date.fromisoformat(self.start_date), days=self.days,
                           members=[MemberInput(member_id=m.member_id, name=m.name, wishes=self.wishes[m.member_id],
                                                budget_eur=m.budget_eur, vegetarian=m.vegetarian, mobility=m.mobility,
                                                pace=m.pace) for m in self.members])


def draw_weights(kind: str, rng) -> list[np.ndarray]:
    themes = cfg().themes
    flat = np.full(len(themes), 0.5)
    if kind == "friends":                               # similar tastes, 3 or 4 people
        base = rng.dirichlet(flat)
        return [rng.dirichlet(30 * base + 0.1) for _ in range(rng.integers(3, 5))]
    if kind == "family":                                # two parents and a teenager
        base = rng.dirichlet(flat)
        teen = np.full(len(themes), 0.3)
        teen[[themes.index(t) for t in ("nightlife", "shopping", "views", "family")]] = 3
        return [rng.dirichlet(30 * base + 0.1) for _ in range(2)] + [rng.dirichlet(teen)]
    if kind == "colleagues":                            # mixed tastes, 4 or 5 people
        return [rng.dirichlet(flat) for _ in range(rng.integers(4, 6))]
    if kind == "minority":                              # three alike and one outsider
        base = rng.dirichlet(flat)
        while True:
            out = rng.dirichlet(flat)
            if out @ base / (np.linalg.norm(out) * np.linalg.norm(base)) < 0.3:
                return [rng.dirichlet(30 * base + 0.1) for _ in range(3)] + [out]
    raise ValueError(kind)


def make_groups(places, seed: int | None = None, per_type: int | None = None) -> list[Group]:
    C = cfg().experiment
    rng = np.random.default_rng(seed if seed is not None else C.seed)
    per_type = per_type or C.groups_per_type
    attractions = places.idx("attraction")
    popular = attractions[np.argsort([-places.items[i].sitelinks for i in attractions])[:100]]
    groups = []
    for kind in KINDS:
        for n in range(per_type):
            weights = draw_weights(kind, rng)
            shared = int(rng.choice(BUDGETS, p=[0.3, 0.5, 0.2]))
            members = []
            for k, w in enumerate(weights):
                budget = shared if kind == "family" else int(rng.choice(BUDGETS, p=[0.3, 0.5, 0.2]))
                m = HiddenMember(member_id=f"m{k + 1}", name=NAMES[k], w=[float(x) for x in w], budget_eur=float(budget),
                                 price_sensitivity=SENSITIVITY[budget], vegetarian=bool(rng.random() < 0.2),
                                 mobility=bool(rng.random() < 0.1), must_see=[],
                                 pace=str(rng.choice(["normal", "relaxed", "packed"], p=[0.6, 0.2, 0.2])),
                                 noise_seed=int(rng.integers(1 << 31)), interest=str(rng.choice(INTERESTS)))
                if rng.random() < 0.3:                  # their best true attraction among the popular ones
                    u = m.true_utility(places)
                    m.must_see = [places.ids[popular[np.argmax(u[popular])]]]
                members.append(m)
            groups.append(Group(group_id=f"{kind}-{n:02d}", kind=kind, members=members,
                                start_date=C.start_date, days=C.trip_days))
    return groups


def describe(m: HiddenMember, places) -> dict[str, str]:
    themes = cfg().themes
    order = np.argsort(-np.asarray(m.w))
    needs = [x for x, on in (("I am vegetarian", m.vegetarian), ("I have difficulty walking far", m.mobility)) if on]
    return {"top_theme": THEME_WORDS[themes[order[0]]],
            "next_two": ", ".join(THEME_WORDS[themes[i]] for i in order[1:3]),
            "bottom_two": ", ".join(THEME_WORDS[themes[i]] for i in order[-2:]),
            "budget_words": "tight" if m.budget_eur < 600 else "medium" if m.budget_eur < 1200 else "comfortable",
            "pace": m.pace, "needs": "; ".join(needs) or "none", "interest": m.interest or "none",
            "must_see": places.by_id[m.must_see[0]].name if m.must_see else "none"}


def write_wishes(groups: list[Group], places) -> int:
    """P7 for every traveler without wishes. Returns how many failed."""
    prompt = load("p7_wishes")
    todo = [(g, m) for g in groups for m in g.members if m.member_id not in g.wishes]
    jobs = [{"schema": PersonaText, "system": prompt.system, "user": prompt.fill_user(**describe(m, places)),
             "purpose": "persona", "model": MODEL_STRONG, "version": f"{prompt.version}|{g.group_id}|{m.member_id}"} for g, m in todo]
    failed = 0
    for (g, m), res in zip(todo, call_many(jobs)):
        if isinstance(res, LLMError):
            failed += 1
        else:
            g.wishes[m.member_id] = res.wishes
    return failed


def save_groups(groups: list[Group], path=ROOT / "results" / "sim_groups.json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(g) for g in groups], ensure_ascii=False, indent=1))


def load_groups(path=ROOT / "results" / "sim_groups.json") -> list[Group]:
    out = []
    for g in json.loads(path.read_text()):
        members = [HiddenMember(**m) for m in g.pop("members")]
        out.append(Group(members=members, **g))
    return out
