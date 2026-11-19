"""CP-SAT model: one hotel, attractions and meals per day, 1 or 2 neighbouring zones per day."""
import math
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np
from ortools.sat.python import cp_model

from src.config import cfg, minutes
from src.data.hours import earliest_start
from src.schemas import DaySelection, Profile, Selection, TripRequest

RELAX_STEPS = ["coverage", "min attractions", "budget"]


@dataclass
class Problem:
    trip: TripRequest
    profiles: list[Profile]
    util: dict[str, np.ndarray]            # member_id -> utility 0..5 per place, in Places order
    places: object                         # Places (src/data/places.py)
    neighbours: dict[int, list[int]]       # zone -> neighbouring zones
    include: list[str] = field(default_factory=list)   # oldest first; relaxation drops the newest
    exclude: set[str] = field(default_factory=set)
    relaxed: list[str] = field(default_factory=list)
    time_limit_s: int | None = None
    workers: int | None = None
    budget_cap: int | None = None          # set when no plan fits the budget: the cheapest workable cost + 5%

    def copy(self) -> "Problem":
        return replace(self, include=list(self.include), exclude=set(self.exclude),
                       relaxed=list(self.relaxed))


def pace_limits(profiles) -> tuple[int, int]:
    """Visits and visit minutes per day. The most relaxed member sets both."""
    C = cfg()
    return (min(getattr(C.pace_max_attractions, p.pace) for p in profiles),
            min(getattr(C.pace_max_visit_minutes, p.pace) for p in profiles))


def visitable(p, day: date) -> bool:
    """The visit fits inside the opening hours and inside the planning day."""
    C = cfg().day
    start = earliest_start(p.opening_hours, day, minutes(C.start), p.duration_min)
    return start is not None and start + p.duration_min <= minutes(C.end)


def lunch_ok(r, day: date) -> bool:
    C = cfg().day
    start = earliest_start(r.opening_hours, day, minutes(C.lunch_from), C.lunch_min)
    return start is not None and start <= minutes(C.lunch_must_open_by)


def dinner_ok(r, day: date) -> bool:
    C = cfg().day
    if r.category == "cafe":
        return False
    start = earliest_start(r.opening_hours, day, minutes(C.dinner_at), C.dinner_min)
    return start is not None and start <= minutes(C.dinner_latest)


def candidates(pb: Problem) -> tuple[list[int], list[int], list[int]]:
    """Top places by mean utility plus every member's own favourites, so nobody is cut early."""
    o, P = cfg().optimizer, pb.places
    mids = [p.member_id for p in pb.profiles]
    mean_u = np.mean([pb.util[m] for m in mids], axis=0)
    banned = {P.index[i] for i in pb.exclude}
    forced = {P.index[i] for i in pb.include}

    def pick(kind: str, top: int, per_member: int) -> list[int]:
        idx = P.idx(kind)
        if kind == "restaurant" and any(p.vegetarian for p in pb.profiles):    # chosen before ranking, so they fill the list
            idx = idx[[P.items[i].veg_friendly is True for i in idx]]
        chosen = set(idx[np.argsort(-mean_u[idx])[:top]].tolist())
        for m in mids:
            chosen |= set(idx[np.argsort(-pb.util[m][idx])[:per_member]].tolist())
        chosen |= {i for i in forced if P.kind[i] == kind}
        return sorted(chosen - banned)

    A = pick("attraction", o.top_attractions, o.top_attractions_per_member)
    R = pick("restaurant", o.top_restaurants, o.top_restaurants_per_member)
    hotels = P.idx("hotel")
    n_m = len(mids)
    share = P.price[hotels] * pb.trip.days * math.ceil(n_m / 2) / n_m
    budget = min(p.budget_eur for p in pb.profiles)
    ok = hotels[share <= o.hotel_share_max * budget]
    if len(ok) == 0:                                   # nothing affordable: take the cheapest
        return A, R, sorted(hotels[np.argsort(share)[: o.top_hotels]].tolist())
    # the best star level the budget allows, central first, and always a few cheap ones so the budget can bind
    stars = np.array([P.items[i].stars or 0 for i in ok])
    ranked = ok[np.lexsort((P.dist_centre[ok], -stars))]
    H = set(ranked[: o.top_hotels - 5].tolist()) | set(hotels[np.argsort(share)[:5]].tolist())
    return A, R, sorted(H)


def build(pb: Problem, objective: str, *, member: str | None = None,
          weights: dict[str, float] | None = None, s_star: dict[str, float] | None = None,
          floor: int | None = None):
    """objective: 'alone' (best plan for one member), 'sum' (M1, M3, M2 stage 2),
    'min' (M2 stage 1), 'weighted' (M4). floor: minimum satisfaction x1000 for every member."""
    C, P = cfg(), pb.places
    A, R, H = candidates(pb)
    D = range(pb.trip.days)
    dates = [pb.trip.start_date + timedelta(days=d) for d in D]
    K = sorted(pb.neighbours)
    mids = [p.member_id for p in pb.profiles]
    zone = {i: P.items[i].zone for i in A + R + H}
    m = cp_model.CpModel()

    x = {(i, d): m.new_bool_var(f"x{i}_{d}") for i in A for d in D}
    lu = {(i, d): m.new_bool_var(f"l{i}_{d}") for i in R for d in D}
    di = {(i, d): m.new_bool_var(f"n{i}_{d}") for i in R for d in D}
    z = {(k, d): m.new_bool_var(f"z{k}_{d}") for k in K for d in D}
    h = {j: m.new_bool_var(f"h{j}") for j in H}
    u = {k: m.new_bool_var(f"u{k}") for k in K}

    # 1, 2: each place at most once; one hotel; one lunch and one dinner per day
    for i in A:
        m.add(sum(x[i, d] for d in D) <= 1)
    for i in R:
        m.add(sum(lu[i, d] + di[i, d] for d in D) <= 1)
    m.add_exactly_one(list(h.values()))
    for d in D:
        m.add_exactly_one([lu[i, d] for i in R])
        m.add_exactly_one([di[i, d] for i in R])

    # 3: pace, at most one long visit a day, and a cap per category so a day is not four churches
    max_a, max_minutes = pace_limits(pb.profiles)
    min_a = 1 if "min attractions" in pb.relaxed else C.optimizer.min_attractions_per_day
    categories = {P.items[i].category for i in A}
    long_visits = [i for i in A if P.items[i].duration_min >= C.long_visit_minutes]
    for d in D:
        count = sum(x[i, d] for i in A)
        m.add(count >= min_a)
        m.add(count <= max_a)
        m.add(sum(P.items[i].duration_min * x[i, d] for i in A) <= max_minutes)
        m.add(sum(x[i, d] for i in long_visits) <= 1)
        for cat in categories:                  # what a traveler asked for is not counted: this rule is ours, the request is theirs
            limit = getattr(C.category_max_per_day, cat, C.category_max_default)
            m.add(sum(x[i, d] for i in A if P.items[i].category == cat and P.ids[i] not in pb.include) <= limit)

    # 4: proximity: a day uses 1 or 2 neighbouring zones, and its places lie in them
    for d in D:
        for i in A:
            m.add(x[i, d] <= z[zone[i], d])
        for i in R:
            m.add(lu[i, d] <= z[zone[i], d])
            m.add(di[i, d] <= z[zone[i], d])
        m.add(sum(z[k, d] for k in K) >= 1)
        m.add(sum(z[k, d] for k in K) <= 2)
        for k1 in K:
            for k2 in K:
                if k1 < k2 and k2 not in pb.neighbours[k1]:
                    m.add(z[k1, d] + z[k2, d] <= 1)

    # 5: the day fits between start and end (visits + overhead + lunch + a second zone)
    day_len = minutes(C.day.end) - minutes(C.day.start)
    for d in D:
        m.add(sum((P.items[i].duration_min + C.day.stop_overhead_min) * x[i, d] for i in A)
              + C.day.lunch_min + C.day.extra_zone_min * (sum(z[k, d] for k in K) - 1) <= day_len)

    # 6: real opening hours on the real dates: closed days, split services, seasons
    for d, day in zip(D, dates):
        for i in A:
            if not visitable(P.items[i], day):
                m.add(x[i, d] == 0)
        for i in R:
            if not lunch_ok(P.items[i], day):
                m.add(lu[i, d] == 0)
            if not dinner_ok(P.items[i], day):
                m.add(di[i, d] == 0)

    # 7: budget per person in whole euros: hotel share + tickets + meals + breakfast + metro.
    # Every leg of a day (visits + 3: lunch, dinner, back to the hotel) is counted as a metro ride,
    # so this is an upper bound and the scheduled plan can never cost more than the optimizer thinks.
    n_m, nights = len(mids), pb.trip.days
    rooms = math.ceil(n_m / 2)
    hotel_share = {j: math.ceil(P.items[j].price_eur * nights * rooms / n_m) for j in H}
    ride = math.ceil(C.travel.ticket_eur)
    cost = (sum(hotel_share[j] * h[j] for j in H)
            + sum(math.ceil(P.items[i].price_eur) * x[i, d] for i in A for d in D)
            + sum(math.ceil(P.items[i].price_eur) * (lu[i, d] + di[i, d]) for i in R for d in D)
            + sum(ride * (sum(x[i, d] for i in A) + 3) for d in D)
            + math.ceil(C.breakfast_eur) * nights)
    if objective != "cheapest":
        m.add(cost <= (pb.budget_cap or int(min(p.budget_eur for p in pb.profiles))))

    # 8: coverage, and the hotel sits in or next to a zone the trip uses
    for k in K:
        m.add(u[k] <= sum(z[k, d] for d in D))
    cover = pb.trip.days if "coverage" in pb.relaxed else pb.trip.days + 1
    m.add(sum(u.values()) >= min(cover, len(K)))
    for j in H:
        m.add(h[j] <= sum(u[k] for k in [zone[j], *pb.neighbours[zone[j]]]))

    # 9: diet and mobility needs of any member apply to the whole group
    if any(p.vegetarian for p in pb.profiles):
        for i in R:
            if P.items[i].veg_friendly is not True:
                for d in D:
                    m.add(lu[i, d] == 0)
                    m.add(di[i, d] == 0)
    if any(p.mobility for p in pb.profiles):
        for i in A:
            if P.items[i].wheelchair == "no":
                for d in D:
                    m.add(x[i, d] == 0)
        for i in R:
            if P.items[i].wheelchair == "no":
                for d in D:
                    m.add(lu[i, d] == 0)
                    m.add(di[i, d] == 0)

    # 10: requests (excluded places never became candidates, so exclude wins over include)
    in_a, in_r = set(A), set(R)
    for pid in pb.include:
        i = P.index[pid]
        if i in in_a:
            m.add(sum(x[i, d] for d in D) == 1)
        elif i in in_r:
            m.add(sum(lu[i, d] + di[i, d] for d in D) == 1)

    # objective
    def total(coef):
        return (sum(int(coef[i]) * x[i, d] for i in A for d in D)
                + sum(int(coef[i]) * (lu[i, d] + di[i, d]) for i in R for d in D))

    def utility(mid):                       # x100
        return total(np.rint(100 * pb.util[mid]))

    def satisfaction(mid):                  # x1000
        return total(np.rint(1000 * pb.util[mid] / s_star[mid]))

    # Places are what the group scored; a better hotel is a small tie-break, so spare budget is not simply unused.
    stars = sum((P.items[j].stars or 0) * h[j] for j in H)
    if floor is not None:
        for mid in mids:
            m.add(satisfaction(mid) >= floor)
    if objective == "cheapest":
        m.minimize(cost)
    elif objective == "alone":
        m.maximize(utility(member) + 30 * stars)
    elif objective == "sum":
        m.maximize(sum(utility(mid) for mid in mids) + 30 * len(mids) * stars)
    elif objective == "min":
        t = m.new_int_var(0, 1000, "t")
        for mid in mids:
            m.add(t <= satisfaction(mid))
        m.maximize(t)
    elif objective == "weighted":
        m.maximize(sum(round(1000 * weights[mid]) * satisfaction(mid) for mid in mids) + 2000 * stars)
    else:
        raise ValueError(objective)
    return m, {"x": x, "lu": lu, "di": di, "z": z, "h": h, "A": A, "R": R, "H": H, "D": D, "K": K}


def run(pb: Problem, objective: str, first_only: bool = False, **kw) -> Selection | None:
    """One CP-SAT solve. None when no feasible plan was found in the time limit. first_only stops at the first plan found."""
    model, v = build(pb, objective, **kw)
    o = cfg().optimizer
    solver = cp_model.CpSolver()
    solver.parameters.stop_after_first_solution = first_only
    solver.parameters.max_time_in_seconds = pb.time_limit_s or o.time_limit_s
    solver.parameters.num_workers = pb.workers or o.workers
    solver.parameters.random_seed = o.seed
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    P, val = pb.places, solver.value
    days = [DaySelection(
        attractions=[P.ids[i] for i in v["A"] if val(v["x"][i, d])],
        lunch_id=next(P.ids[i] for i in v["R"] if val(v["lu"][i, d])),
        dinner_id=next(P.ids[i] for i in v["R"] if val(v["di"][i, d])),
        zones=[k for k in v["K"] if val(v["z"][k, d])],
    ) for d in v["D"]]
    return Selection(hotel_id=next(P.ids[j] for j in v["H"] if val(v["h"][j])), days=days,
                     status=solver.status_name(status), objective=solver.objective_value,
                     relaxed=list(pb.relaxed),
                     log={"bound": solver.best_objective_bound, "seconds": round(solver.wall_time, 2)})


def solve_relaxed(pb: Problem, objective: str, **kw) -> Selection:
    """Solve. If no plan exists, change as little as possible and record every change in Selection.relaxed:
    relax coverage, then the attraction minimum, then the budget, undo what turned out not to be needed,
    and put back as many of the travelers' requests as still fit, oldest first."""
    sel = run(pb, objective, **kw)
    if sel is not None:
        return sel

    def possible() -> Selection | None:             # only whether a plan exists: the full solve comes once, at the end
        return run(pb, objective, first_only=True, **kw)

    requests, pb.include = list(pb.include), []
    sel = possible()
    if sel is None:
        for step in RELAX_STEPS:
            pb.relaxed.append(step)
            if step == "budget":
                cheapest = run(pb, "cheapest", **{k: v for k, v in kw.items() if k == "member"})
                if cheapest is None:
                    raise RuntimeError("no feasible plan even after every relaxation")
                pb.budget_cap = math.ceil(cheapest.objective * 1.05)
            sel = possible()
            if sel is not None:
                break
        else:
            raise RuntimeError("no feasible plan even after every relaxation")
        for step in [s for s in pb.relaxed if s != "budget"]:      # was this step really needed?
            pb.relaxed.remove(step)
            attempt = possible()
            if attempt is None:
                pb.relaxed.append(step)
            else:
                sel = attempt
    for pid in requests:
        pb.include.append(pid)
        attempt = possible()
        if attempt is None:
            pb.include.pop()
            pb.relaxed.append(f"dropped request {pid}")
        else:
            sel = attempt
    sel = run(pb, objective, **kw) or sel
    sel.relaxed = list(pb.relaxed)
    sel.log["budget_cap"] = pb.budget_cap
    return sel


def selection_utility(sel: Selection, util, places) -> float:
    rows = [places.index[i] for day in sel.days for i in [*day.attractions, day.lunch_id, day.dinner_id]]
    return float(util[rows].sum())


def best_alone(pb: Problem) -> dict[str, float]:
    """S*: the best utility each member could get alone under the group's constraints."""
    out = {}
    for p in pb.profiles:
        sel = solve_relaxed(pb.copy(), "alone", member=p.member_id)
        out[p.member_id] = max(selection_utility(sel, pb.util[p.member_id], pb.places), 1e-6)
    return out


def solve_method(pb: Problem, method: str, s_star: dict[str, float],
                 weights: dict[str, float] | None = None) -> Selection:
    pb = pb.copy()
    if method == "M1":
        return solve_relaxed(pb, "sum")
    if method == "M2":
        first = solve_relaxed(pb, "min", s_star=s_star)
        sel = solve_relaxed(pb, "sum", s_star=s_star, floor=int(first.objective))
        sel.log["min_satisfaction"] = first.objective / 1000
        return sel
    if method == "M3":
        base = solve_relaxed(pb, "sum")          # applies any relaxation the group needs anyway
        o = cfg().optimizer
        tau = o.fairness_threshold
        while tau > 0:
            sel = run(pb, "sum", s_star=s_star, floor=int(1000 * tau))
            if sel is not None:
                sel.log["tau"] = tau
                return sel
            tau = round(tau - o.fairness_step, 2)
        base.log["tau"] = 0.0
        return base
    if method == "M4":
        return solve_relaxed(pb, "weighted", s_star=s_star, weights=weights)
    raise ValueError(method)
