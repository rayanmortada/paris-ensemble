"""Inviting a group: seats, personal links, private views, and how the group's plan is picked."""
import numpy as np
import pytest

from src import store
from src.planner.scoring import pick_best
from src.schemas import MemberInput, MemberReply, Plan, Profile
from ui import views


@pytest.fixture
def trip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FEEDBACK_DATASET_REPO", raising=False)
    store.create_trip("t", "visitor", "2026-12-04", 3, 3)
    store.create_invite("t")
    store.add_person("t", "m1", "organizer")
    return "t"


def answers(name="Omar", member_id=""):
    return MemberInput(member_id=member_id, name=name, wishes="I want old churches and long lunches.", budget_eur=900)


def test_an_invite_opens_only_its_trip(trip):
    invite = store.invite_of(trip)
    assert len(invite) >= 20
    assert store.trip_of_invite(invite)["trip_id"] == trip
    assert store.trip_of_invite("guessed") is None


def test_each_friend_gets_a_free_seat_and_a_personal_link_until_the_trip_is_full(trip):
    first = store.join_trip(trip, answers("Omar"))
    second = store.join_trip(trip, answers("Lina"))
    assert first != second
    assert store.get_person(first)["member_id"] == "m2" and store.get_person(second)["member_id"] == "m3"
    assert store.get_person(first)["role"] == "traveler"
    assert store.join_trip(trip, answers("Karim")) is None           # m1 stays reserved for the organizer
    assert {m.name for m in store.get_members(trip)} == {"Omar", "Lina"}
    assert store.get_person("guessed") is None


def test_removing_a_friend_frees_the_seat_but_never_removes_the_organizer(trip):
    token = store.join_trip(trip, answers("Omar"))
    store.save_member(trip, answers("Sara", "m1"))
    store.remove_traveler(trip, "m2")
    assert store.get_person(token) is None and [m.name for m in store.get_members(trip)] == ["Sara"]
    assert store.get_person(store.join_trip(trip, answers("Lina")))["member_id"] == "m2"
    store.remove_traveler(trip, "m1")
    assert any(p["role"] == "organizer" for p in store.people_of(trip))


def test_two_travelers_cannot_share_a_name_but_you_can_keep_your_own(trip):
    store.join_trip(trip, answers("Omar"))
    assert store.name_taken(trip, " omar ")
    assert not store.name_taken(trip, "Omar", "m2")
    assert not store.name_taken(trip, "Lina")


def test_access_keys_are_never_exported(trip):
    assert "people" not in store.TABLES and "invites" not in store.TABLES


def plan(sat) -> Plan:
    return Plan(hotel_id="h", days=[], cost_per_person=500, satisfaction=sat)


PLANS = {"A": plan({"m1": 0.9, "m2": 0.6}), "B": plan({"m1": 0.8, "m2": 0.75})}


def rating(label, score, name="x"):
    return {"plan_label": label, "member_name": name, "wishes_respected": score, "fair": score, "realistic": score, "clear_explanation": score}


def test_the_most_votes_win():
    label, why = pick_best(PLANS, [{"preferred_label": "A"}, {"preferred_label": "A"}, {"preferred_label": "B"}], [])
    assert label == "A" and "2 of 3" in why


def test_tied_votes_go_to_the_better_rated_plan():
    label, why = pick_best(PLANS, [{"preferred_label": "A"}, {"preferred_label": "B"}], [rating("A", 3), rating("B", 4)])
    assert label == "B" and "rated higher" in why


def test_tied_votes_and_ratings_go_to_the_fairer_plan():
    label, why = pick_best(PLANS, [{"preferred_label": "A"}, {"preferred_label": "B"}], [rating("A", 4), rating("B", 4)])
    assert label == "B" and "fairest" in why                         # 0.75 at worst against 0.60


def test_no_votes_goes_to_the_fairer_plan():
    label, why = pick_best(PLANS, [], [])
    assert label == "B" and "Nobody voted" in why


def profile(mid, name):
    return Profile(member_id=mid, name=name, wishes="secret wishes of " + name, theme_weights={}, budget_eur=500 if mid == "m1" else 1500)


class NoPlaces:
    by_id = {}


def explained() -> Plan:
    return Plan(hotel_id="h", days=[], cost_per_person=450, satisfaction={"m1": 0.9, "m2": 0.6},
                log={"explanations": {"m1": "You get the Louvre, Sara.", "m2": "You get the Marais, Omar."}})


def test_the_plan_page_shows_your_own_fit_and_only_numbers_for_the_others():
    me, other = profile("m1", "Sara"), profile("m2", "Omar")
    card = views.fit_card(explained(), me)
    assert "You get the Louvre" in card and "90%" in card and "60%" in card
    assert "Marais" not in card and "Omar" not in card
    assert "Louvre" not in views.fit_card(explained(), other)
    assert "Louvre" not in views.fit_card(explained(), None) and "Marais" not in views.fit_card(explained(), None)


def test_the_budget_shown_is_yours_not_the_lowest_in_the_group():
    rich = profile("m2", "Omar")
    tiles = views.kpis(explained(), rich)
    assert "1,500" in tiles and "lowest budget" not in tiles and "500 EUR" not in tiles.replace("1,500 EUR", "")


def test_the_negotiation_hides_who_said_what_except_your_own_agent():
    me, other = profile("m1", "Sara"), profile("m2", "Omar")
    log = [{"round": 1, "sat": {"m1": 0.9, "m2": 0.6},
            "replies": {"m1": MemberReply(accept=True, add=[], remove=[], give_up="", reason="I love this plan.").model_dump(),
                        "m2": MemberReply(accept=False, add=["x"], remove=[], give_up="", reason="As a vegetarian I want more.").model_dump()},
            "decision": {"include": ["x"], "exclude": [], "refused": [], "weights": {"m1": 0.5, "m2": 0.5}, "note": "Omar is the least satisfied, and Sara accepts."}}]
    text = views.transcript(Plan(hotel_id="h", days=[], cost_per_person=1, log={"transcript": log, "status": "agreed"}), NoPlaces, [me, other], me)
    assert "I love this plan" in text
    assert "vegetarian" not in text and "Omar" not in text and "Sara" not in text
    assert "Traveler 1 is the least satisfied" in text and "60%" not in text


def test_a_shared_note_never_names_anyone(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")
    from src.pipeline import prepare
    from tests.toy_city import TRIP, toy_context
    prep = prepare(TRIP, toy_context())
    names = {p.name for p in prep.profiles}
    assert not any(name in note for note in prep.notes for name in names)
    assert np.isfinite(list(prep.s_star.values())).all()
