"""Compares preference model designs on synthetic travelers. No API calls.

Scenario A: true taste is fully explained by the 10 themes; embeddings carry nothing useful.
Scenario B: part of the true taste is NOT in the themes but is visible in the embeddings, and the
            profile read from the text is noisier. This is the case the learned model exists for.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

N, SAMPLE, TRAVELERS = 600, 60, 40
ALPHAS_OLD = [0.1, 1, 10, 100]
ALPHAS_NEW = [1, 10, 100, 1000, 10000]


def make_places(seed):
    r = np.random.default_rng(seed)
    themes = r.integers(0, 6, (N, 10)).astype(float)
    emb = r.normal(size=(N, 64))
    emb[:, :8] *= 3                          # real embeddings have a few strong semantic directions
    other = np.c_[np.log1p(r.uniform(0, 80, N)), r.choice([0.0, 1.0], (N, 3)), r.uniform(0, 6, N), r.choice([0, .5, 1], N)]
    return themes, emb, other


def features(themes, emb, other, dims):
    return StandardScaler().fit_transform(np.c_[PCA(dims, random_state=7).fit_transform(emb), themes, other])


def traveler(themes, emb, r, scenario):
    w = r.dirichlet(np.full(10, 0.5))
    truth = themes @ w
    if scenario == "B":
        v = r.normal(size=8)
        hidden = emb[:, :8] @ v
        truth = truth + 0.8 * (hidden - hidden.mean()) / hidden.std()
    truth = np.clip(truth + r.normal(0, 0.3, N), 0, 5)
    read = r.dirichlet((30 if scenario == "A" else 10) * w + 0.1)   # the profile the agent read
    return truth, read


def fit_old(X, match, rows, y, w):
    Z = np.c_[X, (match - match.mean()) / (match.std() + 1e-9)]
    return np.clip(RidgeCV(alphas=ALPHAS_OLD).fit(Z[rows], y, sample_weight=w).predict(Z), 0, 5)


def fit_residual(X, match, rows, y, w):
    """Ridge learns only the correction to the profile score, so strong shrinkage falls back on the profile."""
    model = RidgeCV(alphas=ALPHAS_NEW).fit(X[rows], np.asarray(y) - match[rows], sample_weight=w)
    return np.clip(match + model.predict(X), 0, 5)


def run(scenario):
    themes, emb, other = make_places(1)
    X = {d: features(themes, emb, other, d) for d in (8, 16, 32)}
    r = np.random.default_rng(2)
    res = {}

    def rec(name, pred, truth):
        res.setdefault(name, []).append(spearmanr(pred, truth).statistic)

    for _ in range(TRAVELERS):
        truth, read = traveler(themes, emb, r, scenario)
        match = themes @ read
        rows = list(r.choice(N, SAMPLE, replace=False))
        y = list(np.clip(np.rint(truth[rows] + r.normal(0, 0.5, SAMPLE)), 0, 5))
        w = [1.0] * SAMPLE
        rec("profile only (no training)", match, truth)
        rec("old: ridge on raw scores, PCA 32", fit_old(X[32], match, rows, y, w), truth)
        for d in (8, 16, 32):
            rec(f"residual ridge, PCA {d}", fit_residual(X[d], match, rows, y, w), truth)
        # three feedback rounds on the residual model with PCA 16: the plan's rule, weight 3
        fr, fy, fw = list(rows), list(y), list(w)
        pred = fit_residual(X[16], match, fr, fy, fw)
        for _round in range(3):
            order = np.argsort(pred)                         # what the model would pick or drop
            top = order[-40:]
            bottom = order[:40]
            best = bottom[np.argmax(truth[bottom])]          # a place the person loves that the model ranks low
            worst = top[np.argmin(truth[top])]               # a place the person dislikes that the model ranks high
            fr += [int(best), int(worst)]
            fy += [5.0, 0.0]
            fw += [3.0, 3.0]
            pred = fit_residual(X[16], match, fr, fy, fw)
        rec("residual ridge, PCA 16, after 3 feedback rounds", pred, truth)
    print(f"\nScenario {scenario} ({TRAVELERS} synthetic travelers, Spearman with the true taste)")
    for name, vals in res.items():
        print(f"  {name:50s} {np.mean(vals):.3f} (sd {np.std(vals):.3f})")


if __name__ == "__main__":
    run("A")
    run("B")
