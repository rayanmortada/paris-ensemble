"""Runs the experiments of the report. Finished work is skipped, so a run resumes after a crash.

  python scripts/run_experiments.py groups               write the simulated groups (P7)
  python scripts/run_experiments.py E1 --out results/E1  main comparison of M1 to M5, plus E2 model accuracy
  python scripts/run_experiments.py E3 --out results/E3  feedback: no feedback, hard constraints, hard constraints + model
  add --limit 4 for the pilot: the first group of each type
"""
import argparse
import csv
import json
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src import llm  # noqa: E402
from src.config import cfg  # noqa: E402
from src.eval.experiment import metric_row, model_accuracy, true_setup  # noqa: E402
from src.eval.simulate import KINDS, load_groups, make_groups, save_groups, write_wishes  # noqa: E402
from src.pipeline import apply_feedback, clone_prepared, load_context, prepare, run_method  # noqa: E402
from src.schemas import Feedback  # noqa: E402


@contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError(f"over the {seconds} s limit")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_info(out: Path, **extra) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        commit = ""
    build = json.loads((ROOT / "data" / "build_info.json").read_text())
    (out / "run_info.json").write_text(json.dumps({
        "time": time.strftime("%Y-%m-%d %H:%M"), "git_commit": commit, "place_data": build,
        "mock_llm": llm.os.getenv("LLM_MOCK") == "1", "model_fast": llm.MODEL_FAST, "model_strong": llm.MODEL_STRONG,
        "config": (ROOT / "config.yaml").read_text(),
        "prompts": {p.name: p.read_text().splitlines()[0] for p in sorted((ROOT / "prompts").glob("*.md"))}, **extra}, indent=1))


def pick(groups, limit: int | None):
    if not limit:
        return groups
    by_kind = {k: [g for g in groups if g.kind == k] for k in KINDS}
    picked = []
    i = 0
    while len(picked) < limit:
        for k in KINDS:
            if i < len(by_kind[k]) and len(picked) < limit:
                picked.append(by_kind[k][i])
        i += 1
    return picked


def timed_run(prep, method, **kw):
    before = llm.stats()
    plan = run_method(prep, method, explain=False, **kw)
    after = llm.stats()
    plan.log.update(llm_calls=after["calls"] - before["calls"], input_tokens=after["input_tokens"] - before["input_tokens"],
                    output_tokens=after["output_tokens"] - before["output_tokens"])
    return plan


def cmd_groups(args) -> None:
    ctx = load_context()
    groups = make_groups(ctx.places)
    failed = write_wishes(groups, ctx.places)
    save_groups(groups)
    print(f"{len(groups)} groups, {sum(len(g.members) for g in groups)} travelers, {failed} failed wishes; saved results/sim_groups.json")
    if failed:
        sys.exit("Rerun: finished wishes are cached, so only the failed ones cost anything.")


def cmd_e1(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plans_path, e2_path = out / "plans.jsonl", out / "e2.jsonl"
    done = {(r["group_id"], r["method"]) for r in read_jsonl(plans_path)}
    done_e2 = {r["group_id"] for r in read_jsonl(e2_path)}
    ctx = load_context()
    run_info(out, command=" ".join(sys.argv))
    methods = args.methods.split(",")
    limit_s = cfg().experiment.pair_timeout_s
    for group in pick(load_groups(), args.limit):
        todo = [m for m in methods if (group.group_id, m) not in done]
        if not todo and group.group_id in done_e2:
            continue
        print(f"{group.group_id}: {todo}", flush=True)
        try:
            prep = prepare(group.trip(), ctx)
            true_u, s_true = true_setup(group, prep, ctx.places)
        except Exception as e:
            for m in todo:
                append_jsonl(plans_path, {"group_id": group.group_id, "kind": group.kind, "method": m, "run_status": f"failed in prepare: {e!r}"})
            continue
        for method in todo:
            try:
                with time_limit(limit_s):
                    plan = timed_run(prep, method)
                row, members = metric_row(group, method, plan, prep, true_u, s_true, ctx.places)
                append_jsonl(plans_path, {**row, "run_status": "ok", "members": members, "plan": plan.model_dump(mode="json")})
                if method != "M5" and plan.log["violations"]:
                    sys.exit(f"The validator found problems in {method}, group {group.group_id}: {plan.log['violations']}. That is a bug, not a result.")
            except Exception as e:
                append_jsonl(plans_path, {"group_id": group.group_id, "kind": group.kind, "method": method, "run_status": f"failed: {e!r}"})
                print(f"  {method} failed: {e!r}", flush=True)
        if group.group_id not in done_e2:
            for row in model_accuracy(group, prep, true_u, ctx.places):
                append_jsonl(e2_path, row)
    rows = [r for r in read_jsonl(plans_path)]
    ok = [{k: v for k, v in r.items() if k not in ("members", "plan")} for r in rows if r.get("run_status") == "ok"]
    write_csv(out / "metrics.csv", ok)
    write_csv(out / "members.csv", [m for r in rows if r.get("run_status") == "ok" for m in r["members"]])
    write_csv(out / "e2.csv", read_jsonl(e2_path))
    failed = [r for r in rows if r.get("run_status") != "ok"]
    print(f"done: {len(ok)} plans ok, {len(failed)} failed. Results in {out}")


def simulated_feedback(plan, group, true_u, places) -> list[Feedback]:
    """Each member removes the plan attraction they like least (if below 2.5) and adds their best missing one."""
    in_plan = {s.place_id for d in plan.days for s in d.stops if s.role == "visit"}
    actions = []
    for m in group.members:
        u = true_u[m.member_id]
        if in_plan:
            worst = min(in_plan, key=lambda pid: u[places.index[pid]])
            if u[places.index[worst]] < 2.5 and worst not in m.must_see:
                actions.append(Feedback(member_id=m.member_id, place_id=worst, action="remove"))
        missing = [i for i in places.idx("attraction") if places.ids[i] not in in_plan]
        if missing:
            best = max(missing, key=lambda i: u[i])
            actions.append(Feedback(member_id=m.member_id, place_id=places.ids[best], action="add"))
    return actions


def cmd_e3(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "e3.jsonl"
    done = {(r["group_id"], r["method"], r["arm"], r["round"]) for r in read_jsonl(path)}
    ctx = load_context()
    run_info(out, command=" ".join(sys.argv))
    groups = pick(load_groups(), args.limit)
    subset = {g.group_id for g in pick(groups, args.m4_groups)}
    P = ctx.places
    non_hotel = np.r_[P.idx("attraction"), P.idx("restaurant")]
    for group in groups:
        prep = prepare(group.trip(), ctx)
        true_u, s_true = true_setup(group, prep, P)
        for method in ("M1", "M4"):
            if method == "M4" and group.group_id not in subset:
                continue
            start = None                                # both arms begin from the same plan, because M4 does not repeat exactly
            for arm, update in (("hard constraints", False), ("hard constraints + model", True)):
                if all((group.group_id, method, arm, r) in done for r in range(args.rounds + 1)):
                    continue
                state = clone_prepared(prep)
                for rnd in range(args.rounds + 1):
                    plan = timed_run(state, method) if rnd or start is None else start
                    start = start or plan
                    row, _ = metric_row(group, method, plan, state, true_u, s_true, P)
                    rho = np.mean([spearmanr(state.model.predict(m)[non_hotel], true_u[m][non_hotel]).statistic for m in true_u])
                    if (group.group_id, method, arm, rnd) not in done:
                        append_jsonl(path, {"group_id": group.group_id, "kind": group.kind, "method": method, "arm": arm, "round": rnd,
                                            "mean_sat": row["mean_sat"], "min_sat": row["min_sat"], "gini": row["gini"], "spearman": float(rho),
                                            "feasible": row["feasible"], "relaxed": row["relaxed"], "dropped": row["dropped"]})
                    if rnd < args.rounds:
                        apply_feedback(state, simulated_feedback(plan, group, true_u, P), update_model=update)
        print(f"{group.group_id} done", flush=True)
    write_csv(out / "e3.csv", read_jsonl(path))
    print(f"Results in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["groups", "E1", "E3"])
    parser.add_argument("--out", default="results/run")
    parser.add_argument("--methods", default="M1,M2,M3,M4,M5")
    parser.add_argument("--limit", type=int, help="pilot: only this many groups, spread over the types")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--m4-groups", type=int, default=10)
    args = parser.parse_args()
    {"groups": cmd_groups, "E1": cmd_e1, "E3": cmd_e3}[args.command](args)
