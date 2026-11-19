"""Worst case feedback: the traveler always adds a place the model ranks low and removes one it ranks high."""
import numpy as np
from scipy.stats import spearmanr
from compare_feedback import ROUNDS, TRAVELERS, fit
from compare_models import N, SAMPLE, features, make_places, traveler

RULES = {
    "5 / 0, weight 3, new alpha":               (5, 0, 3, False, None),
    "5 / 0, weight 3, keep alpha":              (5, 0, 3, True, None),
    "5 / 0, weight 1, keep alpha":              (5, 0, 1, True, None),
    "nudge by at most 2, weight 3, keep alpha": (5, 0, 3, True, 2.0),
    "nudge by at most 1, weight 3, keep alpha": (5, 0, 3, True, 1.0),
}
for scenario in "AB":
    themes, emb, other = make_places(1)
    X = features(themes, emb, other, 16)
    print(f"\nScenario {scenario}, worst case feedback: Spearman with the truth, rounds 0 to {ROUNDS}")
    for name, (t_add, t_rem, weight, keep, nudge) in RULES.items():
        r = np.random.default_rng(2)
        rho = np.zeros(ROUNDS + 1)
        for _ in range(TRAVELERS):
            truth, read = traveler(themes, emb, r, scenario)
            match = themes @ read
            rows = list(r.choice(N, SAMPLE, replace=False))
            y = list(np.clip(np.rint(truth[rows] + r.normal(0, 0.5, SAMPLE)), 0, 5))
            w = [1.0] * SAMPLE
            pred, alpha0 = fit(X, match, rows, y, w)
            rho[0] += spearmanr(pred, truth).statistic
            for k in range(1, ROUNDS + 1):
                order = np.argsort(pred)
                best = order[:40][np.argmax(truth[order[:40]])]
                worst = order[-40:][np.argmin(truth[order[-40:]])]
                for i, target in ((int(best), float(t_add)), (int(worst), float(t_rem))):
                    if nudge is not None:
                        target = float(np.clip(target, pred[i] - nudge, pred[i] + nudge))
                    rows.append(i)
                    y.append(target)
                    w.append(float(weight))
                pred, _ = fit(X, match, rows, y, w, alpha=alpha0 if keep else None)
                rho[k] += spearmanr(pred, truth).statistic
        print(f"  {name:44s} {' '.join(f'{v:.3f}' for v in rho / TRAVELERS)}")
