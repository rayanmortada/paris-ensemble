import re

import numpy as np
import pytest

from src import llm
from src.agents.mediator import enforce_rules, gave_in
from src.agents.traveler import TravelerAgent, clean_reply, plan_lines
from src.config import cfg
from src.consensus.negotiation import negotiate
from src.pipeline import prepare
from src.planner.optimizer import best_alone, solve_method
from src.planner.scheduler import schedule
from src.schemas import MediatorDecision, MemberInput, MemberReply, MemberWeight, PlaceScore, ProfileOut, ScoreBatch, ThemeScores
from tests.toy_city import PLACES as P, TRIP, problem, profiles, toy_context


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")


def profile_out(**kw):
    base = {"theme_importance": ThemeScores(art=5, history=0, architecture=0, food=0, nature=0, views=0, shopping=0, nightlife=0, family=0, local_life=0),
            "budget_eur": 700.0, "vegetarian": False, "mobility": False, "must_see_names": [], "avoid_names": [], "pace": "relaxed", "language": "English"}
    return ProfileOut(**{**base, **kw})


def agent(member=None):
    return TravelerAgent(member or MemberInput(member_id="m1", name="Sara", wishes="I love art."), P)


def test_profile_weights_sum_to_one_and_form_fields_win(monkeypatch):
    monkeypatch.setitem(llm.MOCKS, "profile", lambda s, u: profile_out())
    a = agent(MemberInput(member_id="m1", name="Sara", wishes="I love art.", budget_eur=1200, vegetarian=True, pace="packed"))
    p = a.build_profile()
    assert sum(p.theme_weights.values()) == pytest.approx(1) and p.theme_weights["art"] == 1
    assert p.budget_eur == 1200 and p.vegetarian and p.pace == "packed"


def test_profile_uses_the_text_when_the_form_is_empty(monkeypatch):
    monkeypatch.setitem(llm.MOCKS, "profile", lambda s, u: profile_out())
    p = agent().build_profile()
    assert p.budget_eur == 700 and p.pace == "relaxed" and not p.vegetarian


def test_all_zero_importance_becomes_uniform(monkeypatch):
    zero = ThemeScores(art=0, history=0, architecture=0, food=0, nature=0, views=0, shopping=0, nightlife=0, family=0, local_life=0)
    monkeypatch.setitem(llm.MOCKS, "profile", lambda s, u: profile_out(theme_importance=zero))
    assert agent().build_profile().theme_weights["art"] == pytest.approx(0.1)


def test_unknown_must_see_names_are_reported(monkeypatch):
    monkeypatch.setitem(llm.MOCKS, "profile", lambda s, u: profile_out(must_see_names=["a1_2", "Disneyland"]))
    p = agent().build_profile()
    assert p.must_see == ["a1_2"] and p.unmatched == ["Disneyland"]


def test_sample_holds_must_see_and_is_repeatable(monkeypatch):
    monkeypatch.setitem(llm.MOCKS, "profile", lambda s, u: profile_out(must_see_names=["a1_2"]))
    a = agent()
    a.build_profile()
    first = a.sample_places()
    assert "a1_2" in first and first == a.sample_places()
    assert all(P.by_id[i].kind != "hotel" for i in first)


def test_a_few_failed_scoring_batches_fall_back_to_neutral_and_are_counted(monkeypatch):
    monkeypatch.setitem(llm.MOCKS, "profile", lambda s, u: profile_out())
    seen = []

    def one_batch_fails(s, u):
        seen.append(u)
        if len(seen) == 1:
            raise llm.LLMError("down")
        return ScoreBatch(scores=[PlaceScore(id=pid, score=4) for pid in re.findall(r"^(\S+) \|", u, re.M)])

    monkeypatch.setitem(llm.MOCKS, "score", one_batch_fails)
    a = agent()
    a.build_profile()
    scores = a.score_sample()
    assert len(seen) > 2 and a.fallback_scores > 0
    assert sum(v == 2.0 for v in scores.values()) == a.fallback_scores and sum(v == 4.0 for v in scores.values()) == len(scores) - a.fallback_scores


def test_most_scoring_batches_failing_is_an_error_not_a_neutral_plan(monkeypatch):
    monkeypatch.setitem(llm.MOCKS, "profile", lambda s, u: profile_out())

    def boom(s, u):
        raise llm.LLMError("RateLimitError: insufficient_quota")

    monkeypatch.setitem(llm.MOCKS, "score", boom)
    a = agent()
    a.build_profile()
    with pytest.raises(llm.LLMError, match="insufficient_quota"):
        a.score_sample()


def test_reply_limits_are_enforced_in_code():
    pb = problem()
    plan = schedule(solve_method(pb, "M1", best_alone(pb)), pb)
    visit = next(s.place_id for d in plan.days for s in d.stops if s.role == "visit")
    reply = MemberReply(accept=False, add=["a1_1", "made_up", "a1_2", "a1_3", "a0_1"], remove=[visit, "not_in_plan"], give_up="", reason="")
    clean = clean_reply(reply, plan, ["a1_1", "a1_2", "a1_3", "a0_1"])
    assert clean.add == ["a1_1", "a1_2", "a1_3"] and clean.remove == [visit]
    failed = clean_reply(llm.LLMError("x"), plan, [])
    assert not failed.accept and failed.add == []
    accepted = clean_reply(MemberReply(accept=True, add=["a1_1"], remove=[], give_up="", reason=""), plan, ["a1_1"])
    assert accepted.add == []


class FakePrep:
    def __init__(self):
        self.profiles = profiles()
        self.profiles[0].must_see = ["a0_1"]


def history(**replies):
    return [{"round": 1, "sat": {"m0": 0.5, "m1": 0.6, "m2": 0.9}, "replies": replies}]


def reply(add=(), remove=(), accept=False):
    return MemberReply(accept=accept, add=list(add), remove=list(remove), give_up="", reason="")


def test_mediator_rules():
    h = history(m0=reply(add=["a1", "a2", "a3"]), m1=reply(remove=["a0_1", "b1"]), m2=reply(accept=True))
    dec = MediatorDecision(include=["a1", "a2", "a3", "invented"], exclude=["a0_1", "b1", "a1"], note="",
                           weights=[MemberWeight(member_id="m0", weight=5.0), MemberWeight(member_id="m1", weight=0.0)])
    weights = {"m0": 1 / 3, "m1": 1 / 3, "m2": 1 / 3}
    include, exclude, new = enforce_rules(dec, h, FakePrep(), weights)
    assert include == ["a2", "a3"]                   # invented id gone; a1 is in both lists, so it is ignored
    assert exclude == ["b1"]                         # a must see cannot be excluded; nobody asked to remove a1
    assert sum(new.values()) == pytest.approx(1)
    assert new["m0"] > new["m2"] > new["m1"]         # clamped to the allowed range, then renormalised


def test_at_most_two_requests_from_one_member_are_granted():
    h = history(m0=reply(add=["a1", "a2", "a3"]), m1=reply(accept=True), m2=reply(accept=True))
    dec = MediatorDecision(include=["a1", "a2", "a3"], exclude=[], note="", weights=[])
    include, _, new = enforce_rules(dec, h, FakePrep(), {"m0": 1 / 3, "m1": 1 / 3, "m2": 1 / 3})
    assert include == ["a1", "a2"] and sum(new.values()) == pytest.approx(1)


def test_gave_in_counts_acceptance_below_the_mean_and_refused_requests():
    h = history(m0=reply(accept=True), m1=reply(add=["x"]), m2=reply(accept=True))
    h[0]["decision"] = {"include": [], "exclude": []}
    counts = gave_in(h)
    assert counts["m0"] == 1 and counts["m1"] == 1 and counts["m2"] == 0


def test_negotiation_stops_when_everyone_accepts(monkeypatch):
    prep = prepare(TRIP, toy_context())
    monkeypatch.setitem(llm.MOCKS, "reply", lambda s, u: reply(accept=True))
    plan = negotiate(prep)
    assert plan.log["status"] == "agreed" and plan.log["rounds"] == 1


def test_negotiation_grants_requests_and_records_them(monkeypatch):
    monkeypatch.setattr(cfg().negotiation, "max_drop", 1.0)
    prep = prepare(TRIP, toy_context())
    asked = {}

    def ask(s, u):
        import re
        ids = re.findall(r"^([ar]\d_\d) \|", u.split("Places you may add", 1)[1], re.M)
        asked["id"] = ids[0]
        return reply(add=ids[:1])

    def grant(s, u):
        return MediatorDecision(include=[asked["id"]], exclude=[], note="ok",
                                weights=[MemberWeight(member_id=m, weight=1 / 3) for m in ("m0", "m1", "m2")])

    monkeypatch.setitem(llm.MOCKS, "reply", ask)
    monkeypatch.setitem(llm.MOCKS, "mediator", grant)
    plan = negotiate(prep, max_rounds=2)
    granted = plan.log["transcript"][0]["decision"]["include"]
    assert plan.log["rounds"] >= 1 and granted
    if plan.log["status"] != "fallback to M3":
        assert set(granted) <= {s.place_id for d in plan.days for s in d.stops}


def asks_and_mediates(monkeypatch, counter=None):
    """Every agent asks for its first candidate, and the mediator grants whatever was asked."""
    import re
    asked = []

    def ask(s, u):
        ids = re.findall(r"^([ar]\d_\d) \|", u.split("Places you may add", 1)[1], re.M)
        asked.extend(ids[:1])
        return reply(add=ids[:1])

    def grant(s, u):
        if counter is not None:
            counter.append(1)
        return MediatorDecision(include=list(dict.fromkeys(asked)), exclude=[], note="ok",
                                weights=[MemberWeight(member_id=m, weight=1 / 3) for m in ("m0", "m1", "m2")])

    monkeypatch.setitem(llm.MOCKS, "reply", ask)
    monkeypatch.setitem(llm.MOCKS, "mediator", grant)


def test_the_final_plan_is_one_the_agents_reviewed(monkeypatch):
    monkeypatch.setattr(cfg().negotiation, "max_drop", 1.0)
    prep = prepare(TRIP, toy_context())
    mediator_calls = []
    asks_and_mediates(monkeypatch, mediator_calls)
    plan = negotiate(prep, max_rounds=3)
    assert plan.log["status"] in ("max rounds", "no further change")
    reviews = len(plan.log["transcript"])
    assert len(mediator_calls) <= reviews and "decision" not in plan.log["transcript"][-1] or plan.log["status"] == "no further change"
    if plan.log["status"] == "max rounds":
        assert reviews == 3 and len(mediator_calls) == 2


def test_a_request_the_group_cannot_afford_is_refused_and_recorded(monkeypatch):
    monkeypatch.setattr(cfg().negotiation, "max_drop", -1.0)         # nothing is affordable
    prep = prepare(TRIP, toy_context())
    asks_and_mediates(monkeypatch)
    plan = negotiate(prep, max_rounds=2)
    first = plan.log["transcript"][0]["decision"]
    assert first["refused"] and not first["include"] and plan.log["status"] == "no further change"
    assert plan.log["rounds"] == 1


def test_negotiation_stops_early_when_no_agent_answers(monkeypatch):
    prep = prepare(TRIP, toy_context())

    def boom(s, u):
        raise llm.LLMError("down")

    called = {"n": 0}

    def mediator(s, u):
        called["n"] += 1
        return MediatorDecision(include=[], exclude=[], note="", weights=[])

    monkeypatch.setitem(llm.MOCKS, "reply", boom)
    monkeypatch.setitem(llm.MOCKS, "mediator", mediator)
    plan = negotiate(prep)
    assert called["n"] == 0 and plan.log["rounds"] == 1


def test_plan_lines_name_every_stop():
    pb = problem()
    plan = schedule(solve_method(pb, "M1", best_alone(pb)), pb)
    text = plan_lines(plan, P)
    assert "lunch" in text and "dinner" in text and plan.hotel_id not in text


def test_agents_never_change_utilities_arrays_in_place():
    pb = problem()
    before = {m: u.copy() for m, u in pb.util.items()}
    solve_method(pb, "M1", best_alone(pb))
    assert all(np.array_equal(before[m], pb.util[m]) for m in before)


def test_the_reply_prompt_names_the_language_of_the_wishes(monkeypatch):
    monkeypatch.setitem(llm.MOCKS, "profile", lambda s, u: profile_out(language="Arabic"))
    pb = problem()
    a = agent()
    a.build_profile()
    plan = schedule(solve_method(pb, "M1", best_alone(pb)), pb)
    a.m.member_id = pb.profiles[0].member_id
    a.profile.member_id = pb.profiles[0].member_id
    job, _ = a.reply_job(plan, {p.member_id: 0.5 for p in pb.profiles}, [], pb)
    assert 'Write "reason" in Arabic' in job["user"]
