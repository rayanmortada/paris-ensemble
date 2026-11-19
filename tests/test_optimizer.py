import math
from datetime import date

import pytest

from src.config import cfg
from src.pipeline import load_context
from src.schemas import TripRequest
from src.planner.optimizer import Problem, best_alone, candidates, dinner_ok, lunch_ok, run, solve_method, visitable
from tests.toy_city import FRIDAY, ITEMS, NEIGHBOURS, PLACES as P, place, problem, profiles, selection_satisfaction

WEIGHTS = {"m0": 0.2, "m1": 0.2, "m2": 0.6}


@pytest.fixture(scope="module")
def s_star():
    return best_alone(problem())


def cost(sel) -> int:
    hotel = P.by_id[sel.hotel_id]
    ids = [i for day in sel.days for i in day.attractions + [day.lunch_id, day.dinner_id]]
    return (math.ceil(hotel.price_eur * 2 * 2 / 3) + sum(math.ceil(P.by_id[i].price_eur) for i in ids)
            + sum(3 * (len(day.attractions) + 3) for day in sel.days) + 10 * 2)


@pytest.mark.parametrize("method", ["M1", "M2", "M3", "M4"])
def test_every_constraint_holds(method, s_star):
    pb = problem()
    sel = solve_method(pb, method, s_star, weights=WEIGHTS if method == "M4" else None)
    zones_used = set()
    for d, day in enumerate(sel.days):
        assert 1 <= len(day.zones) <= 2
        if len(day.zones) == 2:
            assert day.zones[1] in NEIGHBOURS[day.zones[0]]
        ids = [*day.attractions, day.lunch_id, day.dinner_id]
        assert all(P.by_id[i].zone in day.zones for i in ids)
        assert 2 <= len(day.attractions) <= 4
        assert sum(P.by_id[i].duration_min for i in day.attractions) <= 330     # normal pace
        assert sum(P.by_id[i].duration_min >= 120 for i in day.attractions) <= 1
        assert not (d == 0 and "a0_0" in day.attractions)                        # closed on Fridays
        assert P.by_id[day.dinner_id].opening_hours != "Mo-Su 11:30-15:00"       # lunch only
        zones_used |= set(day.zones)
    assert len(zones_used) >= 3                                                   # days + 1
    ids = [i for day in sel.days for i in [*day.attractions, day.lunch_id, day.dinner_id]]
    assert len(ids) == len(set(ids))
    hotel = P.by_id[sel.hotel_id]
    assert hotel.zone in zones_used or set(NEIGHBOURS[hotel.zone]) & zones_used
    assert cost(sel) <= min(p.budget_eur for p in pb.profiles)


def test_least_misery_lifts_the_lowest_member(s_star):
    m1 = selection_satisfaction(solve_method(problem(), "M1", s_star), s_star)
    m2 = selection_satisfaction(solve_method(problem(), "M2", s_star), s_star)
    assert min(m2.values()) >= min(m1.values())


def test_fairness_threshold_is_met(s_star):
    sel = solve_method(problem(), "M3", s_star)
    assert min(selection_satisfaction(sel, s_star).values()) >= sel.log["tau"] - 1e-3


def test_weights_help_the_favoured_member(s_star):
    m1 = selection_satisfaction(solve_method(problem(), "M1", s_star), s_star)
    m4 = selection_satisfaction(solve_method(problem(), "M4", s_star, weights=WEIGHTS), s_star)
    assert m4["m2"] >= m1["m2"]


def test_include_and_exclude(s_star):
    sel = solve_method(problem(include=["a2_3"], exclude={"a1_0"}), "M1", s_star)
    ids = [i for d in sel.days for i in d.attractions]
    assert "a2_3" in ids and "a1_0" not in ids


def test_exclude_wins_over_include(s_star):
    sel = solve_method(problem(include=["a2_3"], exclude={"a2_3"}), "M1", s_star)
    assert "a2_3" not in [i for d in sel.days for i in d.attractions]


def test_a_vegetarian_keeps_only_confirmed_restaurants_in_the_candidate_list():
    _, restaurants, _ = candidates(problem(profiles=profiles(vegetarian=True)))
    assert restaurants and all(problem().places.items[i].veg_friendly is True for i in restaurants)
    _, everyone, _ = candidates(problem())
    assert len(everyone) > len(restaurants)


def test_vegetarian_member_rules_out_restaurants(s_star):
    sel = solve_method(problem(profiles=profiles(vegetarian=True)), "M1", s_star)
    meals = [i for d in sel.days for i in (d.lunch_id, d.dinner_id)]
    assert all(P.by_id[i].veg_friendly for i in meals)


def test_better_hotel_when_the_budget_allows(s_star):
    rich = solve_method(problem(profiles=profiles(budgets=(2000, 2000, 2000))), "M1", s_star)
    assert P.by_id[rich.hotel_id].stars == 3


def test_impossible_budget_gives_the_cheapest_workable_plan_and_keeps_requests(s_star):
    pb = problem(profiles=profiles(budgets=(150, 800, 1000)), include=["a0_1"])
    sel = solve_method(pb, "M1", s_star)
    assert sel.relaxed == ["budget"]                                  # nothing else had to give way
    assert "a0_1" in [i for d in sel.days for i in d.attractions]     # the request survived
    cheapest = solve_method(problem(profiles=profiles(budgets=(100000, 100000, 100000))), "M1", s_star)
    assert cost(sel) <= sel.log["budget_cap"] <= cost(cheapest)       # far below what the group would otherwise spend


def test_a_request_that_cannot_fit_is_dropped_and_reported(s_star, monkeypatch):
    # both are long visits in non-neighbouring zones, and a third long one would break the limit of one a day
    monkeypatch.setattr(cfg(), "long_visit_minutes", 120)
    long_visits = [i.id for i in ITEMS if i.kind == "attraction" and i.duration_min >= 120]
    sel = solve_method(problem(include=long_visits), "M1", s_star)
    assert any(r.startswith("dropped request") for r in sel.relaxed)
    assert sum(i in long_visits for d in sel.days for i in d.attractions) == 2     # one a day for two days


def test_opening_hours_checks():
    museum = place("x", "attraction", 0, duration_min=120, opening_hours="Tu-Su 09:00-18:00")
    assert not visitable(museum, FRIDAY.replace(day=7))          # a Monday
    assert visitable(museum, FRIDAY)
    evening = place("x", "attraction", 0, duration_min=90, opening_hours="Mo-Su 20:00-23:00")
    assert not visitable(evening, FRIDAY)                        # only open after the planning day ends
    split = place("r", "restaurant", 0, opening_hours="Mo-Su 12:00-14:30,19:00-22:30", category="restaurant")
    assert lunch_ok(split, FRIDAY) and dinner_ok(split, FRIDAY)
    cafe = place("c", "restaurant", 0, opening_hours="Mo-Su 08:00-23:00", category="cafe")
    assert lunch_ok(cafe, FRIDAY) and not dinner_ok(cafe, FRIDAY)


def real_problem(names: list[str]) -> tuple[Problem, list[str]]:
    ctx = load_context()
    must = [p.id for p in ctx.places.items if p.name in names]
    assert len(must) == len(names)
    people = profiles(budgets=(2000, 2000, 2000))
    util = {p.member_id: ctx.places.themes.sum(axis=1) for p in people}
    trip = TripRequest(trip_id="real", start_date=date(2026, 12, 4), days=3, members=problem().trip.members)
    return Problem(trip=trip, profiles=people, util=util, places=ctx.places, neighbours=ctx.neighbours, include=must, exclude=set()), must


def test_four_two_hour_must_sees_fit_in_three_days_on_real_places():
    """Regression: neighbourhood walks counted as long visits, so these could never fit together and one was dropped."""
    pb, must = real_problem(["Musée d'Orsay", "Notre-Dame Cathedral", "Sainte-Chapelle", "Le Marais", "Montmartre", "Eiffel Tower"])
    sel = run(pb, "sum")
    assert sel is not None and set(must) <= {a for d in sel.days for a in d.attractions}


def test_a_request_is_not_blocked_by_our_cap_on_churches_a_day():
    """Regression: Notre-Dame and Sainte-Chapelle, 400 m apart, had to be on different days, and that pushed out the Orangerie."""
    pb, must = real_problem(["Musée d'Orsay", "Musée de l'Orangerie", "Notre-Dame Cathedral", "Sainte-Chapelle", "Le Marais", "Montmartre", "Eiffel Tower"])
    sel = solve_method(pb, "M1", best_alone(pb))
    assert not sel.relaxed and set(must) <= {a for d in sel.days for a in d.attractions}


def test_the_cap_on_churches_still_holds_for_what_nobody_asked_for():
    pb, _ = real_problem(["Louvre Museum"])
    pb.util = {m: pb.places.themes[:, cfg().themes.index("architecture")] for m in pb.util}          # churches score highest
    sel = run(pb, "sum")
    worship = [sum(pb.places.by_id[a].category == "worship" for a in d.attractions) for d in sel.days]
    assert max(worship) <= cfg().category_max_per_day.worship
