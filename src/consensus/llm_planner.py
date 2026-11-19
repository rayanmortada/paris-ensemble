"""M5, the baseline: one LLM call writes the whole plan, and we check it without repairing it."""
from datetime import timedelta

from src.data.hours import day_intervals
from src.llm import MODEL_STRONG, call
from src.planner.optimizer import candidates, pace_limits
from src.planner.scheduler import simulate_fixed
from src.prompts import load
from src.schemas import PlannerOut


def _clock(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def llm_plan(prep) -> "Plan":  # noqa: F821
    pb, P = prep.pb, prep.pb.places
    prompt = load("p6_planner")
    dates = [pb.trip.start_date + timedelta(days=d) for d in range(pb.trip.days)]
    A, R, H = candidates(pb)

    def hours(place) -> str:
        parts = []
        for day in dates:
            spans = day_intervals(place.opening_hours, day)
            parts.append(f"{day:%a} " + (",".join(f"{_clock(s)}-{_clock(e)}" for s, e in spans) or "closed"))
        return "; ".join(parts)

    lines = []
    for i in [*A, *R, *H]:
        p = P.items[i]
        minutes_here = p.duration_min if p.kind == "attraction" else 0
        lines.append(f"{p.id} | {p.name} | {p.kind} | {p.zone} | {p.lat:.4f},{p.lon:.4f} | {p.price_eur:.0f} | {hours(p)} | {minutes_here}")
    members = "\n".join(
        f"{p.name} | {p.budget_eur:.0f} | {', '.join(x for x in ('vegetarian' if p.vegetarian else '', 'limited mobility' if p.mobility else '') if x) or 'none'} | {p.pace} | {p.wishes}"
        for p in prep.profiles)
    max_visits, max_minutes = pace_limits(prep.profiles)
    out = call(PlannerOut, prompt.fill_system(days=pb.trip.days, date=f"{dates[0]:%A %d %B %Y}", max_visits=max_visits, max_minutes=max_minutes),
               prompt.fill_user(dates=", ".join(f"{d:%a %d %b}" for d in dates), member_lines=members,
                                candidate_lines="\n".join(lines)),
               purpose="llm_planner", model=MODEL_STRONG, effort="medium", version=prompt.version)

    # Places that do not exist cannot be timed: they are counted as V1 violations and removed. Nothing else is touched.
    unknown = []

    def known(pid: str, d: int | None) -> bool:
        if pid in P.by_id:
            return True
        unknown.append(("V1", d, pid, "the LLM named a place that does not exist"))
        return False

    hotel_id = out.hotel_id
    if not known(hotel_id, None) or P.by_id[hotel_id].kind != "hotel":
        hotel_id = P.ids[H[0]]                     # needed to compute times; the violation stays on record
    day_orders = []
    for d, day in enumerate(out.days):
        day_orders.append({
            "place_ids": [x for x in day.place_ids if known(x, d)],
            "lunch_after": day.lunch_after,
            "lunch_id": day.lunch_id if known(day.lunch_id, d) else None,
            "dinner_id": day.dinner_id if known(day.dinner_id, d) else None,
        })
    plan = simulate_fixed(hotel_id, day_orders, pb)
    plan.log.update(unknown_ids=unknown, reasoning=out.reasoning)
    return plan
