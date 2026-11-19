"""The whole path on the made-up city, with the offline stand-in for the LLM."""
import pytest

from src import llm
from src.pipeline import apply_feedback, clone_prepared, prepare, run_method, set_apart_agent_requests
from src.schemas import Feedback, PlannedDay, Plan, PlannerOut
from tests.toy_city import PLACES as P, TRIP, toy_context


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")


@pytest.fixture(scope="module")
def context():
    return toy_context()


@pytest.fixture
def prep(context, monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")
    return prepare(TRIP, context)


@pytest.mark.parametrize("method", ["M1", "M2", "M3", "M4"])
def test_methods_give_plans_with_no_rule_problems(prep, method):
    plan = run_method(prep, method)
    assert plan.method == method
    assert plan.log["violations"] == []
    assert set(plan.satisfaction) == {"m0", "m1", "m2"}
    assert all(0 <= v <= 1 for v in plan.satisfaction.values())


def test_single_planner_keeps_its_mistakes_on_record(prep, monkeypatch):
    def bad_plan(system, user):
        return PlannerOut(hotel_id="invented_hotel", reasoning="", days=[
            PlannedDay(place_ids=["a0_0", "invented_place", "a2_3"], lunch_after=1, lunch_id="r0_1", dinner_id="r0_0"),
            PlannedDay(place_ids=["a1_1"], lunch_after=0, lunch_id="r1_1", dinner_id="invented_dinner")])

    monkeypatch.setitem(llm.MOCKS, "llm_planner", bad_plan)
    plan = run_method(prep, "M5", explain=False)
    found = {v["code"] for v in plan.log["violations"]}
    assert "V1" in found          # invented places and hotel
    assert "V3" in found          # a0_0 is closed on Fridays; r0_0 does not serve dinner
    assert "V6" in found          # the second day has no dinner
    assert "invented_place" not in {s.place_id for d in plan.days for s in d.stops}


def test_feedback_removes_adds_and_teaches(prep):
    first = run_method(prep, "M1", explain=False)
    in_plan = [s.place_id for d in first.days for s in d.stops if s.role == "visit"]
    missing = next(p.id for p in P.items if p.kind == "attraction" and p.id not in in_plan and p.id != "a0_0")
    before = prep.pb.util["m0"][P.index[missing]]
    apply_feedback(prep, [Feedback(member_id="m0", place_id=in_plan[0], action="remove"),
                          Feedback(member_id="m0", place_id=missing, action="add")])
    second = run_method(prep, "M1", explain=False)
    now = {s.place_id for d in second.days for s in d.stops}
    assert in_plan[0] not in now and missing in now
    assert prep.pb.util["m0"][P.index[missing]] > before
    assert any("removed" in n for n in prep.profiles[0].notes)
    assert second.log["violations"] == []


def test_hard_constraints_only_leave_the_model_alone(prep):
    first = run_method(prep, "M1", explain=False)
    visit = next(s.place_id for d in first.days for s in d.stops if s.role == "visit")
    util = prep.pb.util["m0"].copy()
    apply_feedback(prep, [Feedback(member_id="m0", place_id=visit, action="remove")], update_model=False)
    assert (prep.pb.util["m0"] == util).all() and visit in prep.pb.exclude


def test_a_clone_is_independent(prep):
    other = clone_prepared(prep)
    first = run_method(prep, "M1", explain=False)
    visit = next(s.place_id for d in first.days for s in d.stops if s.role == "visit")
    apply_feedback(other, [Feedback(member_id="m0", place_id=visit, action="remove")])
    assert visit not in prep.pb.exclude and not prep.profiles[0].notes
    assert len(other.model.feedback["m0"][0]) == len(prep.model.feedback["m0"][0]) + 1


def test_explanations_are_blind_to_the_method(prep, monkeypatch):
    seen = {}

    def spy(system, user):
        seen["text"] = system + user
        from src.schemas import Explanations
        return Explanations(items=[])

    monkeypatch.setitem(llm.MOCKS, "explain", spy)
    run_method(prep, "M4", max_rounds=1)
    low = seen["text"].lower()
    assert "negotiat" not in low and "mediator" not in low and "round" not in low


def test_only_what_members_asked_for_is_reported_on_the_plan():
    plan = Plan(hotel_id="h", days=[], cost_per_person=0, log={"relaxed": ["coverage", "dropped request a", "dropped request b"]})
    set_apart_agent_requests(plan, own_requests=["a"])
    assert plan.log["relaxed"] == ["coverage", "dropped request a"]
    assert plan.log["agent_requests_dropped"] == ["b"]
