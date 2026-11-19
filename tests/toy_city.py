"""A made-up city for tests: 3 zones, 12 attractions, 9 restaurants, 3 hotels, 3 members who each prefer a zone."""
from datetime import date

import numpy as np

from src.config import cfg
from src.data.places import Places
from src.pipeline import Context
from src.planner.optimizer import Problem
from src.prefs.model import place_features
from src.schemas import MemberInput, Place, Profile, TripRequest

THEMES = cfg().themes
NEIGHBOURS = {0: [1], 1: [0, 2], 2: [1]}          # zones 0 and 2 are not neighbours
MEMBERS = ["m0", "m1", "m2"]
CATEGORIES = ["museum", "park", "monument", "palace"]
FRIDAY = date(2026, 12, 4)


def place(pid, kind, zone, **kw) -> Place:
    base = {"id": pid, "name": pid, "kind": kind, "category": kind, "lat": 48.85 + zone * 0.01, "lon": 2.33 + zone * 0.02, "zone": zone,
            "themes": {t: 0 for t in THEMES}, "duration_min": 90, "price_eur": 10, "opening_hours": "Mo-Su 09:00-18:00"}
    base.update(kw)
    return Place(**base)


def make_items() -> list[Place]:
    items = []
    for z in range(3):
        for a in range(4):
            hours = "Sa-Th 09:00-18:00" if (z, a) == (0, 0) else "Mo-Su 09:00-18:00"     # a0_0 is closed on Fridays
            items.append(place(f"a{z}_{a}", "attraction", z, category=CATEGORIES[a], duration_min=60 + 20 * a, price_eur=5 * a,
                               opening_hours=hours, themes={**{t: 0 for t in THEMES}, THEMES[(z * 3 + a) % 10]: 5, THEMES[(z + 1) % 10]: 3}))
        for r in range(3):
            hours = "Mo-Su 11:30-15:00" if r == 0 else "Mo-Su 11:30-23:00"                # r*_0 only serves lunch
            items.append(place(f"r{z}_{r}", "restaurant", z, category="restaurant", duration_min=75, price_eur=20 + 5 * r,
                               opening_hours=hours, veg_friendly=(r != 2), themes={**{t: 0 for t in THEMES}, "food": 3 + r}))
        items.append(place(f"h{z}", "hotel", z, category="hotel", duration_min=0, price_eur=[120, 170, 260][z],
                           opening_hours="24/7", stars=z + 1, website="example.com"))
    return items


ITEMS = make_items()
PLACES = Places(ITEMS, np.random.default_rng(0).normal(size=(len(ITEMS), 1536)))


def make_util() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    util = {}
    for k, mid in enumerate(MEMBERS):
        u = rng.uniform(0.5, 2.0, len(ITEMS))
        u[[i for i, p in enumerate(ITEMS) if p.zone == k]] += 2.5
        util[mid] = np.clip(u, 0, 5)
    return util


UTIL = make_util()
TRIP = TripRequest(trip_id="toy", start_date=FRIDAY, days=2,      # a Friday and a Saturday
                   members=[MemberInput(member_id=m, name=m, wishes=f"I love art and history, I am {m}.", budget_eur=800, pace="normal") for m in MEMBERS])


def profiles(budgets=(600, 800, 1000), vegetarian=False, mobility=False) -> list[Profile]:
    return [Profile(member_id=m, name=m, wishes=f"wishes of {m}", theme_weights={t: 0.1 for t in THEMES},
                    budget_eur=b, vegetarian=vegetarian and m == "m1", mobility=mobility and m == "m2")
            for m, b in zip(MEMBERS, budgets)]


def problem(**kw) -> Problem:
    profs = kw.pop("profiles", None) or profiles()
    trip = kw.pop("trip", TRIP)
    return Problem(trip=trip, profiles=profs, util=UTIL, places=PLACES, neighbours=NEIGHBOURS, time_limit_s=10, **kw)


def toy_context() -> Context:
    return Context(PLACES, place_features(PLACES, PLACES.embeddings, 8), NEIGHBOURS, {"enrichment": "mock"})


def selection_satisfaction(sel, s_star) -> dict[str, float]:
    ids = [i for day in sel.days for i in day.attractions + [day.lunch_id, day.dinner_id]]
    return {m: sum(UTIL[m][PLACES.index[i]] for i in ids) / s_star[m] for m in MEMBERS}
