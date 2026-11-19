"""Compares ways to learn from feedback, with feedback following the plan's rule:
the plan is the model's top 12 places; the traveler removes the plan place they like least (if below 2.5)
and adds the best place missing from the plan. No API calls.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV

from compare_models import ALPHAS_NEW, N, SAMPLE, features, make_places, traveler

TRAVELERS, ROUNDS, PLAN = 40, 3, 12


def fit(X, match, rows, y, w, alpha=None):
    target = np.asarray(y) - match[rows]
    if alpha is None:
        model = RidgeCV(alphas=ALPHAS_NEW).fit(X[rows], target, sample_weight=w)
        return np.clip(match + model.predict(X), 0, 5), float(model.alpha_)
    model = Ridge(alpha=alpha).fit(X[rows], target, sample_weight=w)
    return np.clip(match + model.predict(X), 0, 5), alpha


# name -> (target for add, target for remove, weight, keep the first alpha, nudge limit or None)
RULES = {
    "no model update (hard constraints only)": None,
    "plan rule: 5 / 0, weight 3, new alpha":   (5, 0, 3, False, None),
    "5 / 0, weight 1, new alpha":              (5, 0, 1, False, None),
    "4 / 1, weight 1, new alpha":              (4, 1, 1, False, None),
    "4 / 1, weight 1, keep alpha":             (4, 1, 1, True, None),
    "5 / 0, weight 3, keep alpha":             (5, 0, 3, True, None),
    "nudge by at most 1, weight 1, keep alpha": (5, 0, 1, True, 1.0),
    "nudge by at most 1, weight 3, keep alpha": (5, 0, 3, True, 1.0),
    "nudge by at most 2, weight 3, keep alpha": (5, 0, 3, True, 2.0),
}


def quality(pred, truth):
    """True utility of the model's top places, relative to the best possible."""
    top = np.argsort(-pred)[:PLAN]
    return truth[top].sum() / np.sort(truth)[-PLAN:].sum()


def run(scenario):
    themes, emb, other = make_places(1)
    X = features(themes, emb, other, 16)
    results = {name: {"rho": np.zeros(ROUNDS + 1), "q": np.zeros(ROUNDS + 1)} for name in RULES}
    for name, rule in RULES.items():
        r = np.random.default_rng(2)                 # the same travelers for every rule
        for _ in range(TRAVELERS):
            truth, read = traveler(themes, emb, r, scenario)
            match = themes @ read
            rows = list(r.choice(N, SAMPLE, replace=False))
            y = list(np.clip(np.rint(truth[rows] + r.normal(0, 0.5, SAMPLE)), 0, 5))
            w = [1.0] * SAMPLE
            pred, alpha0 = fit(X, match, rows, y, w)
            banned, forced = set(), set()
            results[name]["rho"][0] += spearmanr(pred, truth).statistic
            results[name]["q"][0] += quality(pred, truth)
            for k in range(1, ROUNDS + 1):
                ranked = [i for i in np.argsort(-pred) if i not in banned]
                plan = list(forced) + [i for i in ranked if i not in forced][:PLAN - len(forced)]
                worst = min(plan, key=lambda i: truth[i])
                missing = [i for i in range(N) if i not in plan and i not in banned]
                best = max(missing, key=lambda i: truth[i])
                actions = [(int(best), "add")]
                if truth[worst] < 2.5:
                    actions.append((int(worst), "remove"))
                for i, action in actions:
                    (forced if action == "add" else banned).add(i)
                    forced.discard(i) if action == "remove" else banned.discard(i)
                if rule is not None:
                    t_add, t_rem, weight, keep, nudge = rule
                    for i, action in actions:
                        target = float(t_add if action == "add" else t_rem)
                        if nudge is not None:
                            target = float(np.clip(target, pred[i] - nudge, pred[i] + nudge))
                        rows.append(i)
                        y.append(target)
                        w.append(float(weight))
                    pred, _ = fit(X, match, rows, y, w, alpha=alpha0 if keep else None)
                results[name]["rho"][k] += spearmanr(pred, truth).statistic
                # the plan after this round: forced places stay, banned stay out
                ranked = [i for i in np.argsort(-pred) if i not in banned]
                final = list(forced) + [i for i in ranked if i not in forced][:PLAN - len(forced)]
                results[name]["q"][k] += truth[final].sum() / np.sort(truth)[-PLAN:].sum()
    print(f"\nScenario {scenario}: {TRAVELERS} synthetic travelers. Spearman with the truth over all places, "
          f"and true quality of the top {PLAN} (1.0 = the best possible), rounds 0 to {ROUNDS}")
    for name in RULES:
        rho = results[name]["rho"] / TRAVELERS
        q = results[name]["q"] / TRAVELERS
        print(f"  {name:44s} rho {' '.join(f'{v:.3f}' for v in rho)}   quality {' '.join(f'{v:.3f}' for v in q)}")


if __name__ == "__main__":
    run("A")
    run("B")
