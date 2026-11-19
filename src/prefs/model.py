"""Per member preference model, fitted to the scores of that member's agent.

It predicts from the profile. A straight line from profile score to agent score sets the scale, and what the agent said about
a place also counts directly for a share. A richer ridge on place embeddings did worse on real LLM scores: 60 noisy scores
cannot carry that many features (scripts/design_checks/compare_real_models.py). Feedback is cleaner, so it does train a ridge
on the embeddings, and removing a museum also lowers the museums most like it.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import cfg


def place_features(places, embeddings: np.ndarray, dims: int | None = None) -> np.ndarray:
    """Shared by every member; computed once when the app or the runner starts."""
    kinds = np.array(["attraction", "restaurant", "hotel"])
    base = np.c_[
        PCA(dims or cfg().model.pca_dims, random_state=7).fit_transform(embeddings),
        places.themes,
        np.log1p(places.price),
        (places.kind[:, None] == kinds).astype(float),
        places.dist_centre,
        places.indoor,                       # 1, 0, or 0.5 when unknown
    ]
    return StandardScaler().fit_transform(base)


class PrefModel:
    """One instance per trip; holds one model per member."""

    def __init__(self, places, X: np.ndarray):
        self.X, self.themes, self.index = X, places.themes, places.index
        self.weights, self.data, self.line, self.feedback, self.fixes = {}, {}, {}, {}, {}

    def prior(self, mid: str) -> np.ndarray:
        """Profile score: theme weights times theme scores, 0..5."""
        return self.themes @ self.weights[mid]

    def fit(self, profile, scores: dict[str, float]) -> None:
        mid = profile.member_id
        self.weights[mid] = np.array([profile.theme_weights[t] for t in cfg().themes])
        rows, y = [self.index[p] for p in scores], [float(v) for v in scores.values()]
        self.data[mid] = (rows, y)
        prior = self.prior(mid)[rows]
        slope = max(np.polyfit(prior, y, 1)[0], cfg().model.slope_floor) if np.std(prior) > 0 else cfg().model.slope_floor
        self.line[mid] = (float(slope), float(np.mean(y) - slope * np.mean(prior)))
        self.feedback[mid], self.fixes[mid] = ([], []), None

    def base(self, mid: str) -> np.ndarray:
        slope, intercept = self.line[mid]
        share = cfg().model.own_score_share
        u = intercept + slope * self.prior(mid)
        rows, y = self.data[mid]
        u[rows] = (1 - share) * u[rows] + share * np.asarray(y)
        return u

    def update(self, mid: str, fb) -> None:
        """One feedback action. The target moves at most nudge_limit from the current prediction,
        so one surprising action cannot bend the whole model."""
        m = cfg().model
        i = self.index[fb.place_id]
        now = float(self.predict(mid)[i])
        rows, targets = self.feedback[mid]
        rows.append(i)
        targets.append(float(np.clip(getattr(m.feedback_targets, fb.action), now - m.nudge_limit, now + m.nudge_limit)))
        self.fixes[mid] = Ridge(alpha=m.feedback_alpha, fit_intercept=False).fit(self.X[rows], np.asarray(targets) - self.base(mid)[rows])

    def predict(self, mid: str) -> np.ndarray:
        u = self.base(mid)
        rows, targets = self.feedback[mid]
        if rows:
            u = u + self.fixes[mid].predict(self.X)
            u[rows] = targets                       # what the person said about a place stands as said
        return np.clip(u, 0, 5)
