"""Stress test on the real place file: many random trips across the year, with random tastes, no LLM and no cost.
Every plan must pass the validator. Also reports how often the rules had to be relaxed and how long solving takes.

  python scripts/check_plans.py --trips 60
"""
import argparse
import collections
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import cfg  # noqa: E402
from src.pipeline import load_context  # noqa: E402
from src.planner.optimizer import Problem, best_alone, solve_method  # noqa: E402
from src.planner.scheduler import schedule  # noqa: E402
from src.planner.validator import validate  # noqa: E402
from src.schemas import MemberInput, Profile, TripRequest  # noqa: E402


def random_trip(ctx, rng, workers, limit):
    P = ctx.places
    themes = cfg().themes
    n = int(rng.integers(2, 6))
    days = int(rng.integers(2, 5))
    start = date(2026, 11, 1) + timedelta(days=int(rng.integers(0, 365)))
    profiles, util = [], {}
    for k in range(n):
        w = rng.dirichlet(np.full(len(themes), 0.5))
        mid = f"m{k + 1}"
        util[mid] = np.clip(P.themes @ w + rng.normal(0, 0.5, len(P.items)), 0, 5)
        profiles.append(Profile(member_id=mid, name=mid, wishes="", theme_weights=dict(zip(themes, w.tolist())),
                                budget_eur=float(rng.choice([450, 900, 1600]) * days / 3),
                                vegetarian=bool(rng.random() < 0.2), mobility=bool(rng.random() < 0.1),
                                pace=str(rng.choice(["relaxed", "normal", "packed"]))))
    attractions = P.idx("attraction")
    include = [P.ids[i] for i in rng.choice(attractions, int(rng.integers(0, 3)), replace=False)]
    trip = TripRequest(trip_id="check", start_date=start, days=days,
                       members=[MemberInput(member_id=p.member_id, name=p.name, wishes="x") for p in profiles])
    return Problem(trip=trip, profiles=profiles, util=util, places=P, neighbours=ctx.neighbours, include=include,
                   time_limit_s=limit, workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    ctx = load_context()
    rng = np.random.default_rng(args.seed)
    violations, relaxed, dropped, weekdays = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    times, stats = [], collections.defaultdict(list)
    bad = []
    for t in range(args.trips):
        pb = random_trip(ctx, rng, args.workers, args.limit)
        started = time.time()
        s_star = best_alone(pb)
        for method in ("M1", "M3"):
            sel = solve_method(pb, method, s_star)
            plan = schedule(sel, pb)
            found = validate(plan, pb)
            over = "budget" in sel.relaxed
            for v in found:
                if not (v.code == "V7" and over):
                    violations[v.code] += 1
                    bad.append((t, method, str(pb.trip.start_date), v.model_dump()))
            if over:
                stats["over_budget_ratio"].append(plan.cost_per_person / min(p.budget_eur for p in pb.profiles))
            for r in sel.relaxed:
                relaxed[r.split(" request ")[0]] += 1
            for d in plan.dropped:
                dropped[d.reason] += 1
            stats["visits_per_day"].append(sum(s.role == "visit" for day in plan.days for s in day.stops) / pb.trip.days)
            stats["travel_per_day"].append(sum(s.travel_min for day in plan.days for s in day.stops) / pb.trip.days)
            stats["last_visit_end"].append(max(max((s.end_min for s in day.stops if s.role == "visit"), default=0) for day in plan.days))
            for day in plan.days:
                weekdays[day.date.strftime("%a")] += 1
        times.append(time.time() - started)
    print(f"{args.trips} random trips, 2 methods each, dates across a year, workers={args.workers}")
    print("validator problems that are not an announced budget overrun:", dict(violations) or "none")
    for b in bad[:12]:
        print("  ", b)
    print("rules relaxed:", dict(relaxed) or "none")
    if stats["over_budget_ratio"]:
        print(f"over budget plans: {len(stats['over_budget_ratio'])}, cost / budget median {np.median(stats['over_budget_ratio']):.2f}, worst {max(stats['over_budget_ratio']):.2f}")
    print("places dropped by the scheduler:", dict(dropped) or "none")
    print(f"visits a day {np.mean(stats['visits_per_day']):.2f}, travel a day {np.mean(stats['travel_per_day']):.0f} min, "
          f"latest visit end {max(stats['last_visit_end']) // 60}:{max(stats['last_visit_end']) % 60:02d}")
    print(f"seconds per trip (S* for all members + 2 methods): median {np.median(times):.1f}, worst {max(times):.1f}")
    print("days per weekday:", dict(weekdays))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
