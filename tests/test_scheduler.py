import copy

import numpy as np
import pytest

from src.config import cfg, minutes
from src.planner.optimizer import best_alone, solve_method
from src.planner.scheduler import next_five, plan_day, schedule, simulate_fixed
from src.planner.scoring import daily_satisfaction, not_doable, plan_rows, satisfaction
from src.planner.validator import validate
from tests.toy_city import FRIDAY, PLACES as P, place, problem, profiles

HOTEL, LUNCH, DINNER = (P.by_id[i] for i in ("h0", "r0_1", "r0_2"))
WEIGHTS = {"m0": 0.2, "m1": 0.2, "m2": 0.6}


@pytest.fixture(scope="module")
def good_plan():
    pb = problem()
    return schedule(solve_method(pb, "M1", best_alone(pb)), pb)


@pytest.mark.parametrize("method", ["M1", "M2", "M3", "M4"])
def test_validator_finds_nothing_in_optimizer_plans(method):
    pb = problem()
    sel = solve_method(pb, method, best_alone(pb), weights=WEIGHTS if method == "M4" else None)
    assert validate(schedule(sel, pb), pb) == []


def test_times_are_on_five_minute_marks(good_plan):
    assert next_five(547) == 550 and next_five(550) == 550
    assert all(s.start_min % 5 == 0 for d in good_plan.days for s in d.stops if s.role != "hotel")


def test_place_closing_early_goes_first():
    early = place("early", "attraction", 0, duration_min=60, opening_hours="Mo-Su 09:00-11:00", lat=48.851, lon=2.331)
    plain = place("plain", "attraction", 0, duration_min=60, lat=48.853, lon=2.333)
    stops, dropped = plan_day(FRIDAY, HOTEL, [plain, early], LUNCH, DINNER, False, np.ones(2), {"early": 0, "plain": 1})
    assert stops[0].place_id == "early" and not dropped


def test_place_that_can_never_fit_is_dropped_with_a_reason():
    never = place("never", "attraction", 0, duration_min=120, opening_hours="Mo-Su 09:00-10:00", lat=48.852, lon=2.332)
    plain = place("plain", "attraction", 0, duration_min=60, lat=48.853, lon=2.333)
    stops, dropped = plan_day(FRIDAY, HOTEL, [plain, never], LUNCH, DINNER, False, np.ones(2), {"never": 0, "plain": 1})
    assert [(d.place_id, d.reason) for d in dropped] == [("never", "closed at arrival")]
    lunch = next(s for s in stops if s.role == "lunch")
    assert minutes("11:45") <= lunch.start_min <= minutes("14:00")
    assert [s.role for s in stops][-2:] == ["dinner", "hotel"]


def test_lunch_respects_a_restaurant_that_closes_for_the_afternoon():
    tight = place("tight", "restaurant", 0, category="restaurant", opening_hours="Mo-Su 12:00-14:00,19:00-22:00")
    a = place("a", "attraction", 0, duration_min=200, lat=48.851, lon=2.331)
    stops, _ = plan_day(FRIDAY, HOTEL, [a], tight, DINNER, False, np.ones(1), {"a": 0})
    lunch = next(s for s in stops if s.role == "lunch")
    assert minutes("12:00") <= lunch.start_min and lunch.end_min <= minutes("14:00")


def test_overfull_day_drops_the_least_liked_visit():
    visits = [place(f"long{i}", "attraction", 0, duration_min=200, lat=48.85 + i * 0.001) for i in range(4)]
    index = {p.id: i for i, p in enumerate(visits)}
    stops, dropped = plan_day(FRIDAY, HOTEL, visits, LUNCH, DINNER, False, np.array([4.0, 1.0, 3.0, 2.0]), index)
    assert [d.place_id for d in dropped] == ["long1", "long3"]
    assert all(d.reason == "day too long" for d in dropped)
    assert max(s.end_min for s in stops if s.role == "visit") <= minutes(cfg().day.end)


def test_fixed_order_is_never_repaired_so_the_validator_sees_the_problem():
    pb = problem()
    orders = [{"place_ids": ["a0_0", "a2_3", "a0_1", "a2_2"], "lunch_after": 2, "lunch_id": "r0_1", "dinner_id": "r2_1"},
              {"place_ids": ["a1_1", "a1_2"], "lunch_after": 1, "lunch_id": "r1_1", "dinner_id": "r1_2"}]
    plan = simulate_fixed("h0", orders, pb)
    assert [s.place_id for s in plan.days[0].stops if s.role == "visit"] == orders[0]["place_ids"]
    assert "V3" in {v.code for v in validate(plan, pb)}          # a0_0 is closed on Fridays and stays in the plan


def codes(plan, pb=None):
    return {v.code for v in validate(plan, pb or problem())}


def test_v1_unknown_place(good_plan):
    bad = copy.deepcopy(good_plan)
    bad.days[0].stops[0].place_id = "nope"
    assert "V1" in codes(bad)


def test_v1_unknown_hotel(good_plan):
    bad = copy.deepcopy(good_plan)
    bad.hotel_id = "r0_1"                     # a real place, but not a hotel
    assert "V1" in codes(bad)


def test_v2_place_used_twice(good_plan):
    bad = copy.deepcopy(good_plan)
    bad.days[1].stops[0].place_id = good_plan.days[0].stops[0].place_id
    assert "V2" in codes(bad)


def test_v3_outside_opening_hours(good_plan):
    bad = copy.deepcopy(good_plan)
    next(s for s in bad.days[0].stops if s.role == "visit").end_min = minutes("23:20")
    assert "V3" in codes(bad)


def test_v4_no_time_to_travel(good_plan):
    bad = copy.deepcopy(good_plan)
    bad.days[0].stops[1].start_min = bad.days[0].stops[0].start_min
    assert "V4" in codes(bad)


def test_v5_day_runs_late(good_plan):
    bad = copy.deepcopy(good_plan)
    next(s for s in bad.days[0].stops if s.role == "visit").end_min = minutes("19:30")
    assert "V5" in codes(bad)


def test_v6_missing_lunch(good_plan):
    bad = copy.deepcopy(good_plan)
    bad.days[0].stops = [s for s in bad.days[0].stops if s.role != "lunch"]
    assert "V6" in codes(bad)


def test_v7_over_budget(good_plan):
    bad = copy.deepcopy(good_plan)
    bad.cost_per_person = 5000
    assert "V7" in codes(bad)


def test_v8_avoided_place(good_plan):
    pb = problem()
    pb.profiles[0].avoid = [next(s.place_id for s in good_plan.days[0].stops if s.role == "visit")]
    assert "V8" in codes(good_plan, pb)


def test_v8_a_vegetarian_group_eats_only_where_it_is_confirmed(good_plan, monkeypatch):
    bad = copy.deepcopy(good_plan)
    stop = next(s for s in bad.days[0].stops if s.role == "lunch")
    stop.place_id = "r0_2"                                                         # marked not vegetarian friendly
    assert "V8" in codes(bad, problem(profiles=profiles(vegetarian=True)))
    monkeypatch.setattr(problem().places.by_id["r0_2"], "veg_friendly", None)      # unknown is not good enough
    assert "V8" in codes(bad, problem(profiles=profiles(vegetarian=True)))


def test_v9_wrong_number_of_days(good_plan):
    bad = copy.deepcopy(good_plan)
    bad.days = bad.days[:1]
    assert "V9" in codes(bad)


def test_scheduled_cost_never_exceeds_the_budget():
    pb = problem()
    for method in ("M1", "M2", "M3"):
        plan = schedule(solve_method(pb, method, best_alone(pb)), pb)
        assert plan.cost_per_person <= min(p.budget_eur for p in pb.profiles)
        assert plan.cost_per_person == pytest.approx(sum(plan.cost_breakdown.values()), abs=0.02)


def overfull_plan():
    """Seven visits on the first day, as an LLM might plan them, timed but not repaired."""
    pb = problem()
    places = [p for p in pb.places.items if p.kind == "attraction"]
    meals = [p for p in pb.places.items if p.kind == "restaurant" and p.veg_friendly]
    first = {"place_ids": [p.id for p in places[:7]], "lunch_after": 2, "lunch_id": meals[0].id, "dinner_id": meals[1].id}
    others = [{"place_ids": [places[7 + k].id], "lunch_after": 0, "lunch_id": meals[2 + 2 * k].id, "dinner_id": meals[3 + 2 * k].id} for k in range(pb.trip.days - 1)]
    hotel = next(p.id for p in pb.places.items if p.kind == "hotel")
    return simulate_fixed(hotel, [first, *others], pb), pb


def test_v10_too_many_visits_for_the_pace():
    plan, pb = overfull_plan()
    assert "V10" in codes(plan, pb)


def test_visits_past_the_pace_earn_no_satisfaction():
    plan, pb = overfull_plan()
    skipped = not_doable(plan, pb)
    assert len([d for d, _ in skipped if d == 0]) >= 3                      # the pace allows 4 of the 7
    util = {m: np.ones(len(pb.places.items)) for m in pb.util}
    star = {m: 1000.0 for m in util}
    everything = satisfaction(plan, util, star, pb.places)
    plan.log["not_counted"] = skipped
    assert all(satisfaction(plan, util, star, pb.places)[m] < everything[m] for m in util)
    assert all(sum(daily_satisfaction(plan, util, star, pb.places)[m]) * star[m] / len(plan.days) == len(plan_rows(plan, pb.places)) for m in util)


def test_a_plan_inside_the_rules_has_nothing_left_out(good_plan):
    assert not_doable(good_plan, problem()) == []
