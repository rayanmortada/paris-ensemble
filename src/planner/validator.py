"""One checker for every plan, whatever method made it."""
from src.config import cfg, minutes
from src.data.geo import leg
from src.data.hours import fits
from src.planner.optimizer import pace_limits
from src.schemas import Plan, Violation


def validate(plan: Plan, pb) -> list[Violation]:
    C, P = cfg().day, pb.places
    mobility = any(p.mobility for p in pb.profiles)
    vegetarian = any(p.vegetarian for p in pb.profiles)
    max_visits, max_minutes = pace_limits(pb.profiles)
    avoid = {pid for p in pb.profiles for pid in p.avoid}
    out, seen = [], set()

    def add(code, day, pid, detail):
        out.append(Violation(code=code, day=day, place_id=pid, detail=detail))

    for code, day, pid, detail in plan.log.get("unknown_ids", []):     # the LLM named places that do not exist
        add(code, day, pid, detail)
    if len(plan.days) != pb.trip.days:
        add("V9", None, None, f"{len(plan.days)} days instead of {pb.trip.days}")
    if plan.hotel_id not in P.index or P.items[P.index[plan.hotel_id]].kind != "hotel":
        add("V1", None, plan.hotel_id, "unknown hotel")
        return out
    for d, day in enumerate(plan.days):
        prev, prev_end = P.items[P.index[plan.hotel_id]], minutes(C.start)
        for s in day.stops:
            if s.place_id not in P.index:
                add("V1", d, s.place_id, "unknown place id")
                continue
            p = P.items[P.index[s.place_id]]
            if s.role != "hotel":
                if s.place_id in seen:
                    add("V2", d, s.place_id, "used twice in the trip")
                seen.add(s.place_id)
                if not fits(p.opening_hours, day.date, s.start_min, s.end_min):
                    add("V3", d, s.place_id, "outside opening hours or closed that day")
                if s.place_id in avoid:
                    add("V8", d, s.place_id, "a member asked to avoid this place")
                if vegetarian and s.role in ("lunch", "dinner") and p.veg_friendly is not True:
                    add("V8", d, s.place_id, "vegetarian options not confirmed")
                if mobility and p.wheelchair == "no":
                    add("V8", d, s.place_id, "not wheelchair accessible")
            if s.start_min < prev_end + leg(prev, p, mobility)[0] - 1:
                add("V4", d, s.place_id, "starts before the previous stop ends plus travel")
            prev, prev_end = p, s.end_min
        visits = [s for s in day.stops if s.role == "visit"]
        lunch = [s for s in day.stops if s.role == "lunch"]
        dinner = [s for s in day.stops if s.role == "dinner"]
        if visits and max(s.end_min for s in visits) > minutes(C.end) + 15:
            add("V5", d, None, "last visit ends after the end of the day")
        if len(dinner) == 1 and dinner[0].start_min > minutes(C.dinner_latest):
            add("V5", d, None, "dinner starts too late")
        if len(lunch) != 1 or not minutes(C.lunch_from) <= lunch[0].start_min <= minutes(C.lunch_to):
            add("V6", d, None, "no lunch, or lunch outside its window")
        if len(dinner) != 1:
            add("V6", d, None, "no dinner")
        minutes_seen = sum(s.end_min - s.start_min for s in visits)
        if len(visits) > max_visits or minutes_seen > max_minutes:
            add("V10", d, None, f"{len(visits)} visits and {minutes_seen} minutes, the group's pace allows {max_visits} and {max_minutes}")
    budget = min(p.budget_eur for p in pb.profiles)
    if plan.cost_per_person > budget + 0.5:
        add("V7", None, None, f"{plan.cost_per_person:.0f} EUR per person, lowest budget {budget:.0f} EUR")
    return out
