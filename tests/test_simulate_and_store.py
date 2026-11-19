import numpy as np
import pytest

from src.eval.experiment import metric_row
from src.eval.metrics import gini
from src.eval.simulate import INTERESTS, KINDS, draw_weights, make_groups, write_wishes
from src.schemas import MemberInput
from tests.toy_city import PLACES as P


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")


def test_group_shapes():
    rng = np.random.default_rng(1)
    assert len(draw_weights("family", rng)) == 3
    assert len(draw_weights("minority", rng)) == 4
    assert 3 <= len(draw_weights("friends", rng)) <= 4
    assert 4 <= len(draw_weights("colleagues", rng)) <= 5


def test_the_outsider_really_differs():
    rng = np.random.default_rng(2)
    for _ in range(20):
        w = draw_weights("minority", rng)
        base = np.mean(w[:3], axis=0)
        assert w[3] @ base / (np.linalg.norm(w[3]) * np.linalg.norm(base)) < 0.5


def test_same_seed_same_groups_and_all_types():
    a, b = make_groups(P, seed=3, per_type=2), make_groups(P, seed=3, per_type=2)
    assert [g.members[0].w for g in a] == [g.members[0].w for g in b]
    assert [g.kind for g in a[::2]] == list(KINDS) and len(a) == 8
    assert all(sum(m.w) == pytest.approx(1) for g in a for m in g.members)


def test_true_utility_is_stable_bounded_and_respects_diet():
    m = make_groups(P, seed=4, per_type=1)[0].members[0]
    m.vegetarian = True
    u = m.true_utility(P)
    assert (u >= 0).all() and (u <= 5).all() and np.array_equal(u, m.true_utility(P))
    assert u[P.index["r0_2"]] == 0                # marked not vegetarian friendly


def test_a_passion_lifts_only_some_attractions_and_is_part_of_true_taste():
    m = make_groups(P, seed=4, per_type=1)[0].members[0]
    assert m.interest in INTERESTS
    lift = m.passion(P)
    assert lift.max() > 0 and lift.max() <= 2 and (lift >= 0).all()
    assert (lift[P.idx("restaurant")] == 0).all() and (lift[P.idx("hotel")] == 0).all()
    with_passion = m.true_utility(P)
    m.interest = ""
    assert (m.passion(P) == 0).all()
    assert (with_passion >= m.true_utility(P)).all() and (with_passion > m.true_utility(P)).any()


def test_wishes_are_written_for_everyone():
    groups = make_groups(P, seed=5, per_type=1)
    assert write_wishes(groups, P) == 0
    assert all(g.wishes[m.member_id] for g in groups for m in g.members)
    assert "I must see" not in groups[0].wishes.get("none", "")


def test_metrics_row_uses_true_utilities(monkeypatch):
    from src.pipeline import prepare, run_method
    from tests.toy_city import toy_context
    from src.eval.experiment import true_setup
    group = make_groups(P, seed=6, per_type=1)[3]        # a minority group
    write_wishes([group], P)
    trip = group.trip()
    prep = prepare(trip, toy_context())
    true_u, s_true = true_setup(group, prep, P)
    plan = run_method(prep, "M1", explain=False)
    row, members = metric_row(group, "M1", plan, prep, true_u, s_true, P)
    assert 0 <= row["min_sat"] <= row["mean_sat"] <= 1.0001
    assert row["gini"] == pytest.approx(gini([m["true_sat"] for m in members]))
    assert "minority_gap" in row and sum(m["is_outsider"] for m in members) == 1
    assert row["feasible"] and row["n_violations"] == 0


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FEEDBACK_DATASET_REPO", raising=False)
    from src import store as module
    return module


def test_trip_members_ratings_roundtrip(store):
    store.create_trip("t1", "v1", "2026-12-04", 3, 2)
    assert store.get_trip("t1")["status"] == "draft"
    store.save_member("t1", MemberInput(member_id="m1", name="Sara", wishes="art", budget_eur=900, vegetarian=True, pace="normal"))
    store.save_member("t1", MemberInput(member_id="m1", name="Sara", wishes="art and food", budget_eur=950, pace="normal"))
    members = store.get_members("t1")
    assert len(members) == 1 and members[0].wishes == "art and food" and members[0].budget_eur == 950
    store.save_rating("t1", "Sara", "A", 5, 4, 3, 2)
    store.save_rating("t1", "Sara", "A", 1, 1, 1, 1)            # rating again replaces
    assert len(store.get_ratings("t1")) == 1 and store.get_ratings("t1")[0]["fair"] == 1
    store.save_pref("t1", "Sara", "B", "nice")
    assert store.get_prefs("t1")[0]["preferred_label"] == "B"


def test_daily_cap_ignores_drafts(store):
    store.create_trip("t2", "v2", "2026-12-04", 2, 2)
    assert store.trips_today() == 0
    store.update_trip("t2", status="running")
    assert store.trips_today() == 1 and store.trips_by_visitor("v2") == 1


def test_plan_versions_and_excel(store):
    import openpyxl
    from src.schemas import Plan
    import io
    store.create_trip("t3", "v3", "2026-12-04", 2, 2)
    store.save_plan("t3", "A", "M1", 1, Plan(hotel_id="h0", days=[], cost_per_person=100), 1.0)
    store.save_plan("t3", "A", "M1", 2, Plan(hotel_id="h1", days=[], cost_per_person=200), 1.0)
    assert store.latest_plan("t3", "A")["plan"].hotel_id == "h1"
    assert store.first_plan("t3", "A")["plan"].hotel_id == "h0"
    book = openpyxl.load_workbook(io.BytesIO(store.export_excel()))
    assert set(book.sheetnames) == set(store.TABLES)


def test_table_names_are_checked(store):
    with pytest.raises(ValueError):
        store.read_table("visits; DROP TABLE trips")
