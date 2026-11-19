"""How much may one round of negotiation cost the group? On the development groups only, never the test groups.

  python scripts/design_checks/compare_negotiation.py

Results append to results/dev_negotiation.jsonl, so a stopped run continues where it was.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

from run_experiments import timed_run  # noqa: E402
from src.config import cfg  # noqa: E402
from src.eval.experiment import metric_row, true_setup  # noqa: E402
from src.eval.simulate import load_groups  # noqa: E402
from src.pipeline import clone_prepared, load_context, prepare  # noqa: E402

OUT = ROOT / "results" / "dev_negotiation.jsonl"
ARMS = {"M1": None, "M2": None, "M3": None, "M4, may lose 0.02": 0.02, "M4, may lose 0.05": 0.05, "M4, no limit": 1.0}


def main() -> None:
    ctx = load_context()
    done = [json.loads(line) for line in OUT.read_text().splitlines()] if OUT.exists() else []
    seen = {(r["group_id"], r["arm"]) for r in done}
    for group in load_groups(ROOT / "results" / "dev_groups.json"):
        if all((group.group_id, arm) in seen for arm in ARMS):
            continue
        prep = prepare(group.trip(), ctx)
        true_u, s_true = true_setup(group, prep, ctx.places)
        for arm, drop in ARMS.items():
            if (group.group_id, arm) in seen:
                continue
            if drop is not None:
                cfg().negotiation.max_drop = drop
            plan = timed_run(clone_prepared(prep), arm[:2])
            row, _ = metric_row(group, arm[:2], plan, prep, true_u, s_true, ctx.places)
            row.update(arm=arm, pred_min=min(plan.satisfaction.values()),
                       forced=sum(len((h.get("decision") or {}).get("include", [])) for h in plan.log.get("transcript", [])),
                       refused=sum(len((h.get("decision") or {}).get("refused", [])) for h in plan.log.get("transcript", [])))
            done.append(row)
            with open(OUT, "a") as f:
                f.write(json.dumps(row) + "\n")
        print(group.group_id, "done", flush=True)
    print(f"\n{'arm':20}{'mean':>7}{'min':>7}{'gini':>7}{'calls':>7}{'sec':>7}{'forced':>8}{'refused':>8}  status")
    for arm in ARMS:
        rows = [r for r in done if r["arm"] == arm]
        status = {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows} if s}
        print(f"{arm:20}{np.mean([r['mean_sat'] for r in rows]):7.3f}{np.mean([r['min_sat'] for r in rows]):7.3f}{np.mean([r['gini'] for r in rows]):7.3f}"
              f"{np.mean([r['llm_calls'] or 0 for r in rows]):7.1f}{np.mean([r['seconds'] or 0 for r in rows]):7.1f}"
              f"{np.mean([r['forced'] for r in rows]):8.1f}{np.mean([r['refused'] for r in rows]):8.1f}  {status}")
    print(f"\n{'by group type':20}" + "".join(f"{k:>18}" for k in ("friends", "family", "colleagues", "minority")))
    for arm in ARMS:
        cells = []
        for kind in ("friends", "family", "colleagues", "minority"):
            rows = [r for r in done if r["arm"] == arm and r["kind"] == kind]
            cells.append(f"{np.mean([r['mean_sat'] for r in rows]):9.3f}{np.mean([r['min_sat'] for r in rows]):9.3f}")
        print(f"{arm:20}" + "".join(cells))


if __name__ == "__main__":
    main()
