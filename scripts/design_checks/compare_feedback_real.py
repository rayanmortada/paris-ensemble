"""How strongly should feedback spread to similar places? M1 on the development groups, so no new LLM calls.

  python scripts/design_checks/compare_feedback_real.py

Compares keeping only the hard constraints (removed stays out, added stays in) with also updating the model,
for several ridge strengths. Higher alpha means feedback spreads less.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from run_experiments import simulated_feedback, timed_run  # noqa: E402
from src.config import cfg  # noqa: E402
from src.eval.experiment import metric_row, true_setup  # noqa: E402
from src.eval.simulate import load_groups  # noqa: E402
from src.pipeline import apply_feedback, clone_prepared, load_context, prepare  # noqa: E402

ROUNDS = 3
ARMS = {"hard constraints only": None, "model, alpha 10": 10, "model, alpha 100": 100, "model, alpha 1000": 1000, "model, alpha 10000": 10000}


def main() -> None:
    ctx = load_context()
    P = ctx.places
    non_hotel = np.r_[P.idx("attraction"), P.idx("restaurant")]
    results = {arm: [[] for _ in range(ROUNDS + 1)] for arm in ARMS}
    groups = load_groups(ROOT / "results" / "dev_groups.json")
    for number, group in enumerate(groups, 1):
        prep = prepare(group.trip(), ctx)
        true_u, s_true = true_setup(group, prep, P)
        for arm, alpha in ARMS.items():
            if alpha:
                cfg().model.feedback_alpha = alpha
            state = clone_prepared(prep)
            for rnd in range(ROUNDS + 1):
                plan = timed_run(state, "M1")
                row, _ = metric_row(group, "M1", plan, state, true_u, s_true, P)
                rho = np.mean([spearmanr(state.model.predict(m)[non_hotel], true_u[m][non_hotel]).statistic for m in true_u])
                results[arm][rnd].append((rho, row["mean_sat"], row["min_sat"]))
                if rnd < ROUNDS:
                    apply_feedback(state, simulated_feedback(plan, group, true_u, P), update_model=alpha is not None)
        print(f"  group {number} of {len(groups)} done", flush=True)
    print(f"\n{'':26}{'Spearman':^24}{'plan mean':^24}{'plan min':^24}")
    print(f"{'after round':26}" + "".join(f"{'0':>8}{'1':>8}{'3':>8}" for _ in range(3)))
    for arm, by_round in results.items():
        cells = [np.mean(by_round[r], axis=0) for r in (0, 1, 3)]
        print(f"{arm:26}" + "".join(f"{c[k]:8.3f}" for k in range(3) for c in cells))


if __name__ == "__main__":
    main()
