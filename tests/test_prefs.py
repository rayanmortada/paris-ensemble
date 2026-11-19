import numpy as np
import pytest
from scipy.stats import spearmanr

from src.config import cfg
from src.prefs.model import PrefModel, place_features
from src.schemas import Feedback, Profile

N = 600


class Synthetic:
    def __init__(self, seed=1):
        r = np.random.default_rng(seed)
        self.themes = r.integers(0, 6, (N, 10)).astype(float)
        self.kind = r.choice(["attraction", "restaurant", "hotel"], N)
        self.price = r.uniform(0, 80, N)
        self.dist_centre = r.uniform(0, 6, N)
        self.indoor = r.choice([0.0, 0.5, 1.0], N)
        self.ids = [f"p{i}" for i in range(N)]
        self.index = {p: i for i, p in enumerate(self.ids)}


PLACES = Synthetic()
X = place_features(PLACES, np.random.default_rng(3).normal(size=(N, 64)))


def traveler(r):
    w = r.dirichlet(np.full(10, 0.5))
    truth = np.clip(PLACES.themes @ w + r.normal(0, 0.3, N), 0, 5)
    read = r.dirichlet(30 * w + 0.1)                  # the profile the agent read, a bit off
    profile = Profile(member_id="m", name="m", wishes="", budget_eur=800,
                      theme_weights=dict(zip(cfg().themes, read.tolist())))
    rows = r.choice(N, 60, replace=False)
    scores = {PLACES.ids[i]: float(np.clip(np.rint(truth[i] + r.normal(0, 0.5)), 0, 5)) for i in rows}
    model = PrefModel(PLACES, X)
    model.fit(profile, scores)
    return model, truth, read


def test_model_is_not_worse_than_the_profile_alone():
    r = np.random.default_rng(2)
    prior, ridge = [], []
    for _ in range(10):
        model, truth, read = traveler(r)
        prior.append(spearmanr(PLACES.themes @ read, truth).statistic)
        ridge.append(spearmanr(model.predict("m"), truth).statistic)
    assert np.mean(ridge) >= np.mean(prior) - 0.02


def test_added_place_goes_up_and_removed_place_goes_down():
    model, _, _ = traveler(np.random.default_rng(4))
    before = model.predict("m")
    low, high = PLACES.ids[int(np.argmin(before))], PLACES.ids[int(np.argmax(before))]
    model.update("m", Feedback(member_id="m", place_id=low, action="add"))
    model.update("m", Feedback(member_id="m", place_id=high, action="remove"))
    after = model.predict("m")
    assert after[PLACES.index[low]] > before[PLACES.index[low]]
    assert after[PLACES.index[high]] < before[PLACES.index[high]]


def test_surprising_feedback_does_not_bend_the_whole_model():
    r = np.random.default_rng(5)
    model, truth, _ = traveler(r)
    start = spearmanr(model.predict("m"), truth).statistic
    for _ in range(3):                                # worst case: always where the model is most wrong
        pred = model.predict("m")
        model.update("m", Feedback(member_id="m", place_id=PLACES.ids[int(np.argmax(truth - pred))], action="add"))
        model.update("m", Feedback(member_id="m", place_id=PLACES.ids[int(np.argmin(truth - pred))], action="remove"))
    assert spearmanr(model.predict("m"), truth).statistic >= start - 0.03


def test_the_line_sets_the_scale_and_the_agent_counts_directly():
    model, _, read = traveler(np.random.default_rng(6))
    slope, _ = model.line["m"]
    assert slope >= cfg().model.slope_floor
    rows, y = model.data["m"]
    pure = model.line["m"][1] + slope * (PLACES.themes @ read)
    share = cfg().model.own_score_share
    assert model.base("m")[rows] == pytest.approx((1 - share) * pure[rows] + share * np.asarray(y))
    others = np.setdiff1d(np.arange(N), rows)
    assert model.base("m")[others] == pytest.approx(pure[others])


def test_feedback_reaches_similar_places_but_not_much_of_the_rest():
    model, _, _ = traveler(np.random.default_rng(7))
    before = model.predict("m")
    place = PLACES.ids[int(np.argmax(before))]
    model.update("m", Feedback(member_id="m", place_id=place, action="remove"))
    change = np.abs(model.predict("m") - before)
    i = PLACES.index[place]
    assert change[i] == pytest.approx(cfg().model.nudge_limit)
    others = np.delete(change, i)
    assert others.mean() < 0.2 and others.max() > 0
