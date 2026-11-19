"""Checks the place file before anything is built on it. Offline, unless --names finds a key.

  python scripts/check_places.py                 data checks; the exit code is 1 if any fails
  python scripts/check_places.py --sample 30     also writes results/hand_check.csv, places to verify by hand
  python scripts/check_places.py --names         how written names match; with real embeddings, this sets name_min_cos
"""
import argparse
import csv
import os
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import cfg, minutes  # noqa: E402
from src.data.hours import day_intervals, is_valid  # noqa: E402
from src.data.places import fold  # noqa: E402
from src.llm import embed  # noqa: E402
from src.pipeline import load_context  # noqa: E402

MUST_EXIST = ["louvre", "eiffel", "orsay", "notre dame", "sainte chapelle", "arc de triomphe", "sacre coeur", "pantheon", "orangerie", "catacomb"]
CLOSED = ["pompidou", "palais de la decouverte", "montparnasse tower", "tour montparnasse"]      # closed for all of 2026 and 2027, see README
# Clear cases only: a real enrichment run must get these right.
THEME_CASES = [("louvre", "art", 4), ("eiffel", "views", 3), ("sainte chapelle", "architecture", 3), ("jardin du luxembourg", "nature", 3),
               ("galeries lafayette", "shopping", 4), ("catacomb", "history", 3)]
# What a traveler might write, and the words of the place it should reach. None: it must not match anything.
DESCRIPTIONS = {"Monet's water lilies", "Napoleon's tomb", "natural history gallery", "boat trip on the river"}       # not names: no match is fine, a wrong place is not
NAMES = [("Eiffel Tower", ["eiffel"]), ("Tour Eiffel", ["eiffel"]), ("the eiffel tower", ["eiffel"]), ("Eifel Tower", ["eiffel"]), ("the Louvre", ["louvre"]), ("Le Louvre", ["louvre"]),
         ("Musee d'Orsay", ["orsay"]), ("Orsay", ["orsay"]), ("Orsay Museum", ["orsay"]), ("Notre Dame", ["notre dame cathedral", "notre dame de paris"]),
         ("Notre-Dame de Paris", ["notre dame cathedral"]), ("Sacre Coeur", ["sacre coeur"]), ("Sacré-Cœur", ["sacre coeur"]), ("Sainte Chapelle", ["sainte chapelle"]),
         ("Arc de Triomphe", ["arc de triomphe"]), ("Pere Lachaise", ["lachaise"]), ("the Catacombs", ["catacomb"]), ("les catacombes", ["catacomb"]),
         ("Luxembourg Gardens", ["luxembourg gardens"]), ("Jardin du Luxembourg", ["luxembourg gardens"]), ("Tuileries", ["tuileries"]), ("Champs-Elysees", ["champs"]),
         ("Montmartre", ["montmartre"]), ("the Marais", ["marais"]), ("Palais Garnier", ["garnier"]), ("Opera Garnier", ["garnier"]), ("Grand Palais", ["grand palais"]),
         ("Petit Palais", ["petit palais"]), ("Rodin museum", ["rodin"]), ("Picasso museum", ["picasso"]), ("Fondation Louis Vuitton", ["vuitton"]), ("Les Invalides", ["invalides"]),
         ("Galeries Lafayette", ["lafayette"]), ("the aquarium", ["aquarium de paris"]), ("a Seine cruise", ["bateaux"]),
         ("Monet's water lilies", ["orangerie", "marmottan"]), ("Napoleon's tomb", ["invalides"]), ("natural history gallery", ["evolution"]), ("boat trip on the river", ["bateaux"]),
         ("Moulin Rouge", None), ("Disneyland Paris", None), ("Big Ben", None), ("the Colosseum", None), ("Palace of Versailles", None), ("McDonald's", None),
         ("Mont Saint-Michel", None), ("my cousin's flat", None), ("Lyon", None), ("a museum", None), ("the garden", None), ("Paris", None)]


class Report:
    def __init__(self):
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.failed += not ok
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def closed_days(note: str) -> list[int]:
    """'Closed Mondays and Tuesdays.' gives [0, 1]."""
    day_words = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    m = re.search(rf"closed ((?:(?:{'|'.join(day_words)})s?(?:, | and | or )?)+)", note.lower())
    return [day_words.index(w) for w in re.findall("|".join(day_words), m.group(1))] if m else []


def find(places, word: str):
    return [p for p in places.items if word in fold(p.name) or word in fold(p.name_local or "") or word in fold(p.name_en or "")]


def run_checks(ctx) -> int:
    P, C, r = ctx.places, cfg(), Report()
    att, res, hot = [[p for p in P.items if p.kind == k] for k in ("attraction", "restaurant", "hotel")]
    real = ctx.build_info.get("enrichment") == "openai"
    print(f"Built {ctx.build_info.get('built')}: {len(att)} attractions, {len(res)} restaurants, {len(hot)} hotels")
    r.check("themes and embeddings come from the real OpenAI build", real,
            "" if real else "they come from the offline stand-in: run scripts/build_places.py --stage enrich")
    r.check("enough places", len(att) >= 100 and len(res) >= 200 and len(hot) >= 50)
    r.check("ids are unique", len(set(P.ids)) == len(P.ids))
    r.check("no empty names", all(p.name.strip() for p in P.items))
    south, west, north, east = C.bbox
    r.check("all coordinates are in the Paris box", all(south <= p.lat <= north and west <= p.lon <= east for p in P.items))
    e = P.embeddings
    r.check("one finite, non-zero embedding per place", e.shape == (len(P.items), 1536) and bool(np.isfinite(e).all()) and bool((np.linalg.norm(e, axis=1) > 0).all()), str(e.shape))

    n = C.places.n_zones
    r.check("zone numbers are valid", all(0 <= p.zone < n for p in P.items))
    nb = ctx.neighbours
    r.check("zone neighbours go both ways", all(k in nb[j] for k in nb for j in nb[k]))
    reached, todo = {0}, [0]
    while todo:
        for j in nb[todo.pop()]:
            if j not in reached:
                reached.add(j)
                todo.append(j)
    r.check("every zone can be reached from every other", len(reached) == n)
    for name, items in (("attractions", att), ("restaurants", res), ("hotels", hot)):
        r.check(f"every zone has {name}", {p.zone for p in items} == set(range(n)))

    themes_ok = all(set(p.themes) == set(C.themes) and all(isinstance(v, int) and 0 <= v <= 5 for v in p.themes.values()) for p in P.items)
    r.check("10 themes per place, each 0 to 5", themes_ok)
    no_theme = [p.name for p in att if max(p.themes.values()) == 0]
    r.check("every attraction has a theme", not no_theme, ", ".join(no_theme[:5]))
    strong = np.mean([sum(v >= 3 for v in p.themes.values()) for p in att])
    r.check("themes separate places (mean strong themes per attraction between 1 and 4)", 1 <= strong <= 4, f"{strong:.1f}")
    for word, theme, floor in THEME_CASES:
        hits = find(P, word)
        if real and hits:
            r.check(f"{hits[0].name} scores at least {floor} on {theme}", hits[0].themes[theme] >= floor, f"it has {hits[0].themes[theme]}")

    r.check("durations are plausible (15 to 300 min)", all(15 <= p.duration_min <= 300 for p in att))
    r.check("no negative prices", all(p.price_eur >= 0 for p in P.items))
    r.check("free places cost nothing", all(p.price_eur == 0 for p in att if p.price_source == "free"))
    r.check("restaurant price follows its tier", all(p.price_eur == C.meal_eur_by_tier[p.price_tier] for p in res))
    r.check("hotel price follows its stars", all(p.price_eur == C.hotel_eur_by_stars[p.stars] for p in hot))
    r.check("all opening hours can be read", all(is_valid(p.opening_hours) for p in P.items))
    r.check("no opening hours are guessed", all(p.hours_source in ("osm", "curated", "open_air") for p in P.items))
    week = [date(2027, 3, 8) + timedelta(days=i) for i in range(7)]                  # an ordinary week starting on a Monday
    clash = [p.name for p in att for d in closed_days(p.note) if day_intervals(p.opening_hours, week[d])]
    r.check("the hours agree with what the note says about closed days", not clash, ", ".join(clash))
    print("     hours sources, attractions:", {s: sum(p.hours_source == s for p in att) for s in sorted({p.hours_source for p in att})})
    print("     price sources, attractions:", {s: sum(p.price_source == s for p in att) for s in sorted({p.price_source for p in att})})

    lunch_s, lunch_e = minutes(C.day.lunch_must_open_by), minutes(C.day.lunch_must_open_by) + C.day.lunch_min
    dinner_s, dinner_e = minutes(C.day.dinner_at), minutes(C.day.dinner_at) + C.day.dinner_min

    def serves(p, day, s, e):
        return any(a <= s and e <= b for a, b in day_intervals(p.opening_hours, day))

    for season, first in (("December", date(2026, 12, 7)), ("June", date(2027, 6, 7))):
        open_days, lunch, dinner, veg, zone_veg = [], [], [], [], []
        for offset in range(7):
            day = first + timedelta(days=offset)
            open_days.append(sum(bool(day_intervals(p.opening_hours, day)) for p in att))
            lunch.append(sum(serves(p, day, lunch_s, lunch_e) for p in res))
            dinner.append(sum(serves(p, day, dinner_s, dinner_e) for p in res))
            veg.append(min(sum(serves(p, day, s, e) for p in res if p.veg_friendly) for s, e in ((lunch_s, lunch_e), (dinner_s, dinner_e))))
            zone_veg.append(min(sum(serves(p, day, s, e) for p in res if p.veg_friendly and p.zone == z) for z in range(n) for s, e in ((lunch_s, lunch_e), (dinner_s, dinner_e))))
        detail = f"open attractions {min(open_days)} to {max(open_days)}, lunch {min(lunch)} to {max(lunch)}, dinner {min(dinner)} to {max(dinner)}, vegetarian meals at least {min(veg)}, at least {min(zone_veg)} in every zone"
        r.check(f"every weekday in {season} has places to go", min(open_days) >= 50 and min(lunch) >= 30 and min(dinner) >= 30 and min(veg) >= 5 and min(zone_veg) >= 1, detail)

    print(f"     vegetarian, from OpenStreetMap: {sum(p.veg_friendly is True for p in res)} yes, {sum(p.veg_friendly is False for p in res)} no, {sum(p.veg_friendly is None for p in res)} unknown")
    print(f"     wheelchair, attractions: {sum(p.wheelchair == 'yes' for p in att)} yes, {sum(p.wheelchair == 'limited' for p in att)} limited, "
          f"{sum(p.wheelchair == 'no' for p in att)} no, {sum(p.wheelchair == 'unknown' for p in att)} unknown")
    print(f"     photos: {sum(bool(p.image_url) for p in att)} of {len(att)} attractions")
    r.check("all links are https or http", all((p.website or "http").startswith("http") and (p.image_url or "http").startswith("http") for p in P.items))
    for word in MUST_EXIST:
        r.check(f"the list includes {word}", bool(find(P, word)))
    for word in CLOSED:
        r.check(f"the list leaves out {word} (closed)", not find(P, word))
    print(f"\n{r.failed} checks failed" if r.failed else "\nAll checks passed")
    return r.failed


def write_sample(places, size: int) -> None:
    rng = random.Random(cfg().experiment.seed)
    kinds = {"attraction": size // 2, "restaurant": size // 3, "hotel": size - size // 2 - size // 3}
    chosen = [p for kind, count in kinds.items() for p in rng.sample([x for x in places.items if x.kind == kind], count)]
    path = ROOT / "results" / "hand_check.csv"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "name", "address", "website", "stored_hours", "hours_source", "stored_price", "price_source", "visit_minutes", "vegetarian",
                    "still_open_yes_no", "hours_right_yes_no", "price_within_25_percent_yes_no", "what_you_checked", "comment"])
        for p in chosen:
            w.writerow([p.id, p.kind, p.name, p.address or "", p.website or "", p.opening_hours, p.hours_source, p.price_eur, p.price_source, p.duration_min if p.kind == "attraction" else "",
                        p.veg_friendly if p.kind == "restaurant" else "", "", "", "", "", ""])
    print(f"Wrote {path.relative_to(ROOT)}: check each place on its own site or on Google Maps, then fill the last columns.")


def reaches(place, expected: list[str]) -> bool:
    words = fold(f"{place.name} {place.name_local or ''} {place.name_en or ''}")
    return any(e in words for e in expected)


def check_names(ctx) -> None:
    P, C = ctx.places, cfg().places
    real = ctx.build_info.get("enrichment") == "openai" and os.getenv("OPENAI_API_KEY") and os.getenv("LLM_MOCK") != "1"
    if not real:
        print("Word step only: the embedding step needs the key and the real build, so name_min_cos stays unchecked.\n")
    att = P.idx("attraction")
    wrong, good_cos, bad_cos = 0, [], []
    for written, expected in NAMES:
        row = P.name_hit(written)
        how, place = "words", None if row is None else P.items[row]
        if row is None and real:
            q = embed([written])[0]
            cos = P.unit[att] @ (q / np.linalg.norm(q))
            nearest, best = P.items[int(att[cos.argmax()])], float(cos.max())
            how = f"cos {best:.2f}"
            (good_cos if expected and reaches(nearest, expected) else bad_cos).append(best)
            place = nearest if best >= C.name_min_cos else None
            if place is None:
                how += f", nearest {nearest.name}"
        elif row is None:
            how = "not checked" if written in DESCRIPTIONS else "no match"
        if written in DESCRIPTIONS:
            ok = place is None or reaches(place, expected)
        else:
            ok = place is None if expected is None else place is not None and reaches(place, expected)
        wrong += not ok
        print(f"[{'ok' if ok else 'WRONG'}] {written!r:28} -> {place.name if place else '(nothing)':38} {how}")
    print(f"\n{wrong} wrong of {len(NAMES)}")
    if good_cos or bad_cos:
        low, high = min(good_cos, default=None), max(bad_cos, default=None)
        print(f"Embedding similarity: lowest right match {low}, highest wrong match {high}, current name_min_cos {C.name_min_cos}")
        if low is not None and high is not None and low > high:
            print(f"Any value between them separates the two. Suggested name_min_cos: {(low + high) / 2:.2f} (config.yaml, places).")
        else:
            print("The two overlap, so pick the value that makes the fewest mistakes above, and prefer no match over a wrong match.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--names", action="store_true")
    args = parser.parse_args()
    ctx = load_context()
    failed = run_checks(ctx)
    if args.sample:
        write_sample(ctx.places, args.sample)
    if args.names:
        print()
        check_names(ctx)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
