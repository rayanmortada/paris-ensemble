"""Which preference model works on real LLM scores? Everything is cached, so a rerun costs nothing new.

  python scripts/design_checks/compare_real_models.py

The first designs were chosen on synthetic scores on the same scale as the profile. Real scores are not, and the result
turned around, so the options are compared again here on real scores. It uses separate development groups, not the 40
test groups of the experiments, so the choice cannot flatter the final results.
Each option is judged by its accuracy against hidden true taste and by the true satisfaction of the plan it leads to.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import ndcg_score  # noqa: E402

from src.config import cfg  # noqa: E402
from src.eval.simulate import load_groups, make_groups, save_groups, write_wishes  # noqa: E402
from src.llm import embed  # noqa: E402
from src.pipeline import load_context, prepare  # noqa: E402
from src.planner.optimizer import best_alone, solve_method  # noqa: E402
from src.planner.scheduler import schedule  # noqa: E402
from src.planner.scoring import satisfaction  # noqa: E402

DEV_SEED, DEV_PER_TYPE = 2024, 3
DEV_FILE = ROOT / "results" / "dev_groups.json"
LIGHT, HEAVY = cfg().model.first_design_alphas, [100, 1000, 10000, 100000]


def anchor(prior, rows, y, floor=1.0):
    """A straight line from profile score to agent score. The slope never goes below the floor, so the profile keeps its say."""
    slope, _ = np.polyfit(prior[rows], y, 1)
    slope = max(slope, floor)
    return np.mean(y) - slope * np.mean(prior[rows]) + slope * prior, slope


def blend(pred, rows, y, beta):
    out = pred.copy()
    out[rows] = (1 - beta) * pred[rows] + beta * y          # what the agent said about a place also counts directly
    return out


def two_signals(prior, sim, rows, y):
    """A small regression per traveler: agent score from profile score and closeness to the wishes. Weights cannot go negative."""
    def scale(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    F = np.c_[scale(prior), scale(sim)]
    mean_f, mean_y = F[rows].mean(axis=0), np.mean(y)
    coef, _ = nnls(F[rows] - mean_f, np.asarray(y) - mean_y)
    return mean_y + (F - mean_f) @ coef, coef


def options(prior, sim, X, rows, y) -> dict[str, np.ndarray]:
    y = np.asarray(y, float)
    q, _ = anchor(prior, rows, y)
    heavy = q + RidgeCV(alphas=HEAVY).fit(X[rows], y - q[rows]).predict(X)
    two, _ = two_signals(prior, sim, rows, y)
    out = {"profile only": prior,
           "first design: fixed slope, light correction": prior + RidgeCV(alphas=LIGHT).fit(X[rows], y - prior[rows]).predict(X),
           "fitted slope, heavy correction": heavy,
           "fitted slope, own scores 0.25": blend(q, rows, y, 0.25),
           "fitted slope, own scores 0.50": blend(q, rows, y, 0.5),
           "two signals": two,
           "two signals, own scores 0.25": blend(two, rows, y, 0.25),
           "two signals, own scores 0.50": blend(two, rows, y, 0.5)}
    return out


def main() -> None:
    ctx = load_context()
    P = ctx.places
    if DEV_FILE.exists():
        groups = load_groups(DEV_FILE)
    else:
        groups = make_groups(P, seed=DEV_SEED, per_type=DEV_PER_TYPE)
        failed = write_wishes(groups, P)
        if failed:
            sys.exit(f"{failed} wishes failed. Run again: finished ones are cached.")
        save_groups(groups, DEV_FILE)
    att, res = P.idx("attraction"), P.idx("restaurant")
    both = np.r_[att, res]
    acc, plans, notes, weights = {}, {}, [], []
    for number, group in enumerate(groups, 1):
        prep = prepare(group.trip(), ctx)
        members = {m.member_id: m for m in group.members}
        truths = {"with passion": {}, "themes only": {}}
        for mid, m in members.items():
            truths["with passion"][mid] = m.true_utility(P)
            interest, m.interest = m.interest, ""
            truths["themes only"][mid] = m.true_utility(P)
            m.interest = interest
        stars = {}
        for label, true_u in truths.items():
            pb = prep.pb.copy()
            pb.util = true_u
            stars[label] = best_alone(pb)
        by_option = {}
        for profile in prep.profiles:
            mid = profile.member_id
            rows, y = prep.model.data[mid]
            rows = list(rows)
            prior = prep.model.prior(mid)
            truth = truths["with passion"][mid]
            plain = truths["themes only"][mid]
            passion = members[mid].passion(P)
            hit = passion[att] > 0
            scored = np.isin(att, rows)
            notes.append({"profile vs true taste, on the scored places": spearmanr(prior[rows], truth[rows]).statistic,
                          "agent scores vs true taste, on the scored places": spearmanr(y, truth[rows]).statistic,
                          "fitted slope": anchor(prior, rows, np.asarray(y, float))[1],
                          "share of the passion's places that the agent scored": (hit & scored).sum() / max(hit.sum(), 1),
                          "mean agent score, passion places": float(np.mean([s for r, s in zip(rows, y) if passion[r] > 0.5] or [np.nan])),
                          "mean agent score, other attractions": float(np.mean([s for r, s in zip(rows, y) if passion[r] <= 0.5 and P.kind[r] == "attraction"] or [np.nan]))})
            wish = embed([profile.wishes])[0]
            sim = P.unit @ (wish / np.linalg.norm(wish))
            weights.append(two_signals(prior, sim, rows, y)[1])
            for name, pred in options(prior, sim, ctx.X, rows, y).items():
                pred = np.clip(pred, 0, 5)
                by_option.setdefault(name, {})[mid] = pred
                acc.setdefault(name, []).append((spearmanr(pred[att], truth[att]).statistic, ndcg_score(truth[att][None, :], pred[att][None, :], k=20),
                                                 spearmanr(pred[att], plain[att]).statistic, ndcg_score(plain[att][None, :], pred[att][None, :], k=20),
                                                 spearmanr(pred[both], truth[both]).statistic))
        for name, util in by_option.items():
            pb = prep.pb.copy()
            pb.util = util
            plan = schedule(solve_method(pb, "M1", {}), pb)
            a = list(satisfaction(plan, truths["with passion"], stars["with passion"], P).values())
            b = list(satisfaction(plan, truths["themes only"], stars["themes only"], P).values())
            plans.setdefault(name, []).append((np.mean(a), np.min(a), np.mean(b), np.min(b)))
        print(f"  group {number} of {len(groups)} done", flush=True)
    print(f"\n{len(notes)} travelers in {len(groups)} development groups\n")
    for key in notes[0]:
        print(f"  {key:52} {np.nanmean([n[key] for n in notes]):6.3f}")
    print(f"  {'two signals: mean weight of profile, of closeness':52} {np.mean(weights, axis=0).round(3)}")
    print(f"\n{'':54}{'with passion':^36}{'themes only':^36}")
    print(f"{'model':54}{'Spearman':>9}{'NDCG@20':>9}{'plan mean':>10}{'plan min':>9}{'Spearman':>9}{'NDCG@20':>9}{'plan mean':>10}{'plan min':>9}")
    for name in acc:
        a, s = np.nanmean(acc[name], axis=0), np.mean(plans[name], axis=0)
        print(f"{name:54}{a[0]:9.3f}{a[1]:9.3f}{s[0]:10.3f}{s[1]:9.3f}{a[2]:9.3f}{a[3]:9.3f}{s[2]:10.3f}{s[3]:9.3f}")


if __name__ == "__main__":
    main()
