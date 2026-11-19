"""Metrics, geography, name matching and the strict schema rule."""
import numpy as np
import pytest
from openai.lib._pydantic import to_strict_json_schema

import src.schemas as sc
from src.data.geo import haversine_km, leg
from src.data.places import Places, fold
from scripts.check_places import DESCRIPTIONS, NAMES, closed_days, reaches
from src.eval.metrics import bootstrap_ci, gini, holm, sacrificed
from tests.toy_city import THEMES, place


def test_gini():
    assert gini([1, 1, 1]) == pytest.approx(0)
    assert gini([0, 0, 1]) == pytest.approx(2 / 3)
    assert gini([0, 0, 0]) == 0


def test_sacrificed_member():
    assert sacrificed(np.array([[0.2, 0.3], [0.5, 0.6]]))
    assert not sacrificed(np.array([[0.2, 0.7], [0.5, 0.6]]))
    assert not sacrificed(np.array([[0.2, 0.3]]))


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_bootstrap_interval_contains_the_mean():
    lo, hi = bootstrap_ci([0.5, 0.6, 0.7, 0.8])
    assert lo < 0.65 < hi


def test_louvre_to_eiffel_is_about_3_km():
    assert 3.0 < float(haversine_km(48.8606, 2.3376, 48.8584, 2.2945)) < 3.4


def test_walk_or_metro():
    a = place("a", "attraction", 0, lat=48.8606, lon=2.3376)
    near = place("b", "attraction", 0, lat=48.8660, lon=2.3376)    # 0.6 km straight, 0.8 km with detour
    far = place("c", "attraction", 0, lat=48.8584, lon=2.2945)
    assert leg(a, far)[1:] == ("metro", 2.55)
    assert leg(a, near)[1:] == ("walk", 0.0)
    assert leg(a, place("d", "attraction", 0, lat=48.8700, lon=2.3376))[1] == "walk"
    assert leg(a, place("d", "attraction", 0, lat=48.8700, lon=2.3376), mobility=True)[1] == "metro"


def strict_ok(schema) -> bool:
    """False if any object has free keys, which OpenAI strict mode rejects at call time."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" and schema.get("additionalProperties", False) is not False:
            return False
        return all(strict_ok(v) for v in schema.values())
    if isinstance(schema, list):
        return all(strict_ok(v) for v in schema)
    return True


@pytest.mark.parametrize("model", [sc.ProfileOut, sc.ScoreBatch, sc.MemberReply, sc.MediatorDecision,
                                   sc.Explanations, sc.PlannerOut, sc.PersonaText, sc.EnrichBatch])
def test_llm_output_models_are_strict(model):
    assert strict_ok(to_strict_json_schema(model))


def test_the_strict_check_catches_a_dict_field():
    class Bad(sc.BaseModel):
        weights: dict[str, float]
    assert not strict_ok(to_strict_json_schema(Bad))


def test_fold():
    assert fold("Sacré-Cœur") == "sacre coeur"
    assert fold("Musée d'Orsay") == "musee d orsay"


@pytest.fixture
def named(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")      # embedding fallback uses stable random vectors
    items = [place("eiffel", "attraction", 0, name="Tour Eiffel", name_en="Eiffel Tower"),
             place("louvre", "attraction", 0, name="Musée du Louvre", name_en="Louvre Museum"),
             place("orsay", "attraction", 0, name="Musée d'Orsay", name_en="Orsay Museum"),
             place("sacre", "attraction", 0, name="Basilique du Sacré-Cœur"),
             place("resto", "restaurant", 0, name="Louvre Bistro")]
    return Places(items, np.random.default_rng(0).normal(size=(5, 1536)))


@pytest.mark.parametrize("written, expected", [
    ("tour eiffel", "eiffel"), ("the eiffel tower", "eiffel"), ("louvre", "louvre"),
    ("orsay museum", "orsay"), ("sacre coeur", "sacre"),
])
def test_names_match_in_french_or_english(named, written, expected):
    assert named.match_names([written]) == [expected]


def test_unknown_name_is_left_out(named):
    assert named.match_names(["Disneyland"]) == []


def test_closed_days_are_read_from_a_note():
    assert closed_days("Closed Mondays and Tuesdays.") == [0, 1] and closed_days("Free. Closed Sundays") == [6]
    assert closed_days("Open Wednesday to Saturday afternoons.") == []


@pytest.fixture(scope="module")
def paris():
    return Places.load()


@pytest.mark.parametrize("written, expected", NAMES)
def test_a_written_name_reaches_the_right_real_place_or_none(paris, written, expected):
    row = paris.name_hit(written)
    if expected is None or written in DESCRIPTIONS:
        assert row is None, paris.items[row].name
    else:
        assert row is not None and reaches(paris.items[row], expected), None if row is None else paris.items[row].name


def test_all_themes_are_in_the_toy_data():
    assert len(THEMES) == 10


def test_prompts_all_load_and_fill():
    from src.prompts import load
    for name, fields in {"p1_profile": ["wishes"], "p7_wishes": ["interest", "top_theme", "next_two", "bottom_two", "budget_words", "pace", "needs", "must_see"]}.items():
        prompt = load(name)
        assert prompt.version.startswith(name) and prompt.system
        prompt.fill_user(**{f: "x" for f in fields})
    assert "Satisfaction" in load("p3_reply").system.format(name="Sara")


def test_every_curated_attraction_is_complete():
    import yaml
    from src.config import ROOT
    from src.data.hours import is_valid
    curated = yaml.safe_load((ROOT / "data" / "curated_attractions.yaml").read_text())
    assert len(curated) >= 150
    for key, c in curated.items():
        assert c.get("name") and c.get("duration", 0) > 0 and c.get("price", -1) >= 0, key
        assert c.get("source") in ("web", "free", "estimate"), key
        assert (c["price"] == 0) == (c["source"] == "free") or c["name"] in ("Parc de Bagatelle",), key
        if c.get("hours"):
            assert is_valid(c["hours"]), key
    for closed in ("Q178065", "Q323767", "Q614450"):              # Pompidou, Montparnasse deck, Palais de la Decouverte
        assert closed not in curated


def test_escaping_of_visitor_text():
    from ui.views import esc, safe_url
    assert "<script" not in esc("<script>alert(1)</script>")
    assert safe_url("javascript:alert(1)") is None
    assert safe_url("example.com") == "https://example.com"
    assert safe_url("https://a.fr; https://b.fr") == "https://a.fr"
