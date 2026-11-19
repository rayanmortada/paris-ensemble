"""Satisfaction: how good a plan is for one member compared with the best they could get alone."""
import numpy as np

from src.config import cfg, minutes
from src.data.hours import fits
from src.planner.optimizer import pace_limits


def not_doable(plan, pb) -> list[tuple[int, str]]:
    """Stops a group could not really do: closed at that time, too late in the day, past the pace it asked for, or a repeat.
    They earn nothing. Otherwise a plan that overfills its days would look like the one that makes everyone happiest.
    Only M5 can have any, because it is not repaired; the others are built inside these rules."""
    C, P = cfg().day, pb.places
    max_visits, max_minutes = pace_limits(pb.profiles)
    out, seen = [], set()
    for d, day in enumerate(plan.days):
        visits, visit_minutes = 0, 0
        for s in day.stops:
            if s.role == "hotel" or s.place_id not in P.index:
                continue
            bad = s.place_id in seen or not fits(P.by_id[s.place_id].opening_hours, day.date, s.start_min, s.end_min)
            seen.add(s.place_id)
            if s.role == "visit":
                visits += 1
                visit_minutes += s.end_min - s.start_min
                bad = bad or s.end_min > minutes(C.end) + 15 or visits > max_visits or visit_minutes > max_minutes
            elif s.role == "dinner":
                bad = bad or s.start_min > minutes(C.dinner_latest)
            if bad:
                out.append((d, s.place_id))
    return out


def day_rows(plan, d: int, places) -> list[int]:
    skip = {tuple(x) for x in plan.log.get("not_counted") or []}
    return [places.index[s.place_id] for s in plan.days[d].stops
            if s.role != "hotel" and s.place_id in places.index and (d, s.place_id) not in skip]


def plan_rows(plan, places) -> list[int]:
    return [row for d in range(len(plan.days)) for row in day_rows(plan, d, places)]


def satisfaction(plan, util: dict[str, np.ndarray], s_star: dict[str, float], places) -> dict[str, float]:
    """0 to 1. Capped at 1, because S* can be slightly low when the solver stops on its time limit."""
    rows = plan_rows(plan, places)
    return {m: float(min(util[m][rows].sum() / s_star[m], 1.0)) for m in util}


def daily_satisfaction(plan, util, s_star, places) -> dict[str, list[float]]:
    """Per member and day: that day's utility / (S* / number of days)."""
    out = {m: [] for m in util}
    n_days = max(len(plan.days), 1)
    for d in range(len(plan.days)):
        rows = day_rows(plan, d, places)
        for m in util:
            out[m].append(float(util[m][rows].sum() / (s_star[m] / n_days)))
    return out


def pick_best(plans: dict, prefs: list[dict], ratings: list[dict]) -> tuple[str, str]:
    """The group's plan: most votes, then the better average rating, then the plan fairest to the group by the model's estimate.
    Returns its label and the reason in plain words."""
    votes = {label: sum(p["preferred_label"] == label for p in prefs) for label in plans}

    def rating(label: str) -> float:
        rows = [r for r in ratings if r["plan_label"] == label]
        return sum(r["wishes_respected"] + r["fair"] + r["realistic"] + r["clear_explanation"] for r in rows) / max(len(rows), 1)

    def fairness(label: str) -> tuple[float, float]:
        sat = list(plans[label].satisfaction.values()) or [0.0]
        return min(sat), sum(sat) / len(sat)

    labels = sorted(plans)
    most = max(votes.values())
    leaders = [label for label in labels if votes[label] == most]
    if most and len(leaders) == 1:
        return leaders[0], f"{most} of {len(prefs)} travelers preferred it."
    if most:
        best = max(leaders, key=lambda x: (rating(x), fairness(x)))
        if rating(best) > min(rating(x) for x in leaders):
            return best, "The votes were tied, so this is the plan the group rated higher."
        return max(leaders, key=fairness), "The votes and the ratings were tied, so this is the plan fairest to the whole group."
    return max(labels, key=fairness), "Nobody voted, so this is the plan fairest to the whole group."
