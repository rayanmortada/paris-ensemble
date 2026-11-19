"""Turns the optimizer's selection into timed days, using real opening hours on the real dates."""
import math
from datetime import date, timedelta

import numpy as np

from src.config import cfg, minutes
from src.data.geo import leg
from src.data.hours import earliest_start
from src.schemas import DayPlan, Dropped, Plan, Selection, Stop


def next_five(m: int) -> int:
    """Times on a plan are rounded up to 5 minutes: travel is an estimate, and 09:47 would suggest false precision."""
    return (m + 4) // 5 * 5


def order_visits(hotel, visits, mobility):
    """Nearest neighbour from the hotel, then 2-opt on travel minutes."""
    def length(seq):
        pts = [hotel, *seq]
        return sum(leg(a, b, mobility)[0] for a, b in zip(pts, pts[1:]))

    left, order, cur = list(visits), [], hotel
    while left:
        nxt = min(left, key=lambda p: leg(cur, p, mobility)[0])
        order.append(nxt)
        left.remove(nxt)
        cur = nxt
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order) + 1):
                new = order[:i] + order[i:j][::-1] + order[j:]
                if length(new) < length(order):
                    order, improved = new, True
    return order


def simulate_day(day: date, hotel, seq, lunch, mobility):
    """Times for one day in a fixed order: (stops, misfits, waiting), or None if lunch cannot start in its window."""
    C = cfg().day
    t, prev, stops, misfits, waiting = minutes(C.start), hotel, [], [], 0
    for p in seq:
        travel, mode, _ = leg(prev, p, mobility)
        arrive = t + travel
        if p is lunch:
            start = earliest_start(p.opening_hours, day, next_five(max(arrive, minutes(C.lunch_from))), C.lunch_min)
            if start is None or start > minutes(C.lunch_to):
                return None
            end, role = start + C.lunch_min, "lunch"
        else:
            start = earliest_start(p.opening_hours, day, next_five(arrive), p.duration_min)
            if start is None:
                misfits.append(p)
                continue
            end, role = start + p.duration_min, "visit"
        waiting += start - arrive
        stops.append(Stop(place_id=p.id, role=role, arrive_min=arrive, start_min=start, end_min=end,
                          travel_min=travel, mode=mode))
        t, prev = end, p
    return stops, misfits, waiting


def best_lunch(day: date, hotel, order, lunch, mobility):
    """Lunch at the position with the fewest misfits, then the least waiting.
    Lunch first always works: the optimizer only picks lunch places open by 12:00, so we can wait."""
    tries = [simulate_day(day, hotel, order[:k] + [lunch] + order[k:], lunch, mobility)
             for k in range(len(order) + 1)]
    return min((r for r in tries if r is not None), key=lambda r: (len(r[1]), r[2]))


def plan_day(day: date, hotel, visits, lunch, dinner, mobility, group_u, index):
    C = cfg().day
    order, dropped = order_visits(hotel, visits, mobility), []
    while True:
        stops, misfits, _ = best_lunch(day, hotel, order, lunch, mobility)
        if misfits:                                    # try the misfit at every other position first
            p = misfits[0]
            rest = [q for q in order if q is not p]
            fixed = [o for o in (rest[:k] + [p] + rest[k:] for k in range(len(rest) + 1))
                     if not best_lunch(day, hotel, o, lunch, mobility)[1]]
            if fixed:
                order = fixed[0]
                continue
            reason = "closed at arrival"
            if not best_lunch(day, hotel, [p], lunch, mobility)[1]:   # it fits alone, so the day is too full
                p, reason = min(order, key=lambda q: group_u[index[q.id]]), "day too long"
            order.remove(p)
            dropped.append(Dropped(place_id=p.id, reason=reason))
            continue
        if max((s.end_min for s in stops if s.role == "visit"), default=0) > minutes(C.end):
            p = min(order, key=lambda q: group_u[index[q.id]])
            order.remove(p)
            dropped.append(Dropped(place_id=p.id, reason="day too long"))
            continue
        break
    by_id = {p.id: p for p in [lunch, *order]}
    travel, mode, _ = leg(by_id[stops[-1].place_id], dinner, mobility)
    arrive = stops[-1].end_min + travel
    start = earliest_start(dinner.opening_hours, day, next_five(max(arrive, minutes(C.dinner_at))), C.dinner_min)
    if start is None:                                  # the optimizer checked 19:30; a late arrival can still miss it
        start = next_five(max(arrive, minutes(C.dinner_at)))
    stops.append(Stop(place_id=dinner.id, role="dinner", arrive_min=arrive, start_min=start,
                      end_min=start + C.dinner_min, travel_min=travel, mode=mode))
    travel, mode, _ = leg(dinner, hotel, mobility)
    back = stops[-1].end_min + travel
    stops.append(Stop(place_id=hotel.id, role="hotel", arrive_min=back, start_min=back, end_min=back,
                      travel_min=travel, mode=mode))
    return stops, dropped


def cost_breakdown(plan: Plan, pb) -> dict[str, float]:
    """Cost per person: the group shares rooms two by two; everything else is individual."""
    C, P, n = cfg(), pb.places, len(pb.profiles)
    out = {"hotel": P.items[P.index[plan.hotel_id]].price_eur * len(plan.days) * math.ceil(n / 2) / n
           if plan.hotel_id in P.index else 0.0,
           "tickets": 0.0, "meals": 0.0, "breakfast": C.breakfast_eur * len(plan.days), "metro": 0.0}
    for day in plan.days:
        for s in day.stops:
            price = P.items[P.index[s.place_id]].price_eur if s.place_id in P.index else 0.0
            if s.role == "visit":
                out["tickets"] += price
            elif s.role in ("lunch", "dinner"):
                out["meals"] += price
            if s.mode == "metro":
                out["metro"] += C.travel.ticket_eur
    return {k: round(v, 2) for k, v in out.items()}


def finish(plan: Plan, pb) -> Plan:
    plan.cost_breakdown = cost_breakdown(plan, pb)
    plan.cost_per_person = round(sum(plan.cost_breakdown.values()), 2)
    return plan


def schedule(sel: Selection, pb) -> Plan:
    P = pb.places
    mobility = any(p.mobility for p in pb.profiles)
    group_u = np.mean([pb.util[p.member_id] for p in pb.profiles], axis=0)
    hotel = P.items[P.index[sel.hotel_id]]
    days, dropped = [], []
    for d, ds in enumerate(sel.days):
        day = pb.trip.start_date + timedelta(days=d)
        stops, lost = plan_day(day, hotel, [P.items[P.index[i]] for i in ds.attractions],
                               P.items[P.index[ds.lunch_id]], P.items[P.index[ds.dinner_id]],
                               mobility, group_u, P.index)
        days.append(DayPlan(date=day, zones=ds.zones, stops=stops))
        dropped += lost
    plan = Plan(hotel_id=sel.hotel_id, days=days, cost_per_person=0.0, dropped=dropped,
                log={"relaxed": sel.relaxed, **sel.log})
    return finish(plan, pb)


def simulate_fixed(hotel_id: str, day_orders: list[dict], pb) -> Plan:
    """Times in exactly the given order, with no reordering and nothing dropped (the M5 path).
    Unknown ids must be removed before this. The validator then reports whatever breaks."""
    C, P = cfg().day, pb.places
    mobility = any(p.mobility for p in pb.profiles)
    hotel = P.items[P.index[hotel_id]]
    days = []
    for d, order in enumerate(day_orders):
        day = pb.trip.start_date + timedelta(days=d)
        t, prev, stops = minutes(C.start), hotel, []
        visits = [P.items[P.index[i]] for i in order["place_ids"]]
        lunch_after = max(0, min(order.get("lunch_after", len(visits) // 2), len(visits)))
        seq = [(p, "visit") for p in visits[:lunch_after]]
        if order.get("lunch_id"):
            seq.append((P.items[P.index[order["lunch_id"]]], "lunch"))
        seq += [(p, "visit") for p in visits[lunch_after:]]
        if order.get("dinner_id"):
            seq.append((P.items[P.index[order["dinner_id"]]], "dinner"))
        for p, role in seq:
            travel, mode, _ = leg(prev, p, mobility)
            arrive = t + travel
            duration = {"lunch": C.lunch_min, "dinner": C.dinner_min}.get(role, p.duration_min)
            floor = {"lunch": minutes(C.lunch_from), "dinner": minutes(C.dinner_at)}.get(role, 0)
            start = earliest_start(p.opening_hours, day, next_five(max(arrive, floor)), duration)
            if start is None:
                start = next_five(max(arrive, floor))   # no valid time: keep it, the validator flags it
            stops.append(Stop(place_id=p.id, role=role, arrive_min=arrive, start_min=start,
                              end_min=start + duration, travel_min=travel, mode=mode))
            t, prev = start + duration, p
        travel, mode, _ = leg(prev, hotel, mobility)
        stops.append(Stop(place_id=hotel.id, role="hotel", arrive_min=t + travel, start_min=t + travel,
                          end_min=t + travel, travel_min=travel, mode=mode))
        days.append(DayPlan(date=day, zones=sorted({P.items[P.index[s.place_id]].zone for s in stops}), stops=stops))
    return finish(Plan(hotel_id=hotel_id, days=days, cost_per_person=0.0), pb)
