"""M4: traveler agents object to a plan, a mediator turns that into requests and weights, the optimizer solves again.

A request is granted only if the group can afford it, one request at a time: nobody's gain may quietly use up what everyone
had. The final plan is always one the agents have seen. Without these two rules, forced places piled up round after round,
satisfaction fell every round, and the plan that came out had been reviewed by no one.
"""
from src.agents.mediator import decide, enforce_rules
from src.agents.traveler import clean_reply
from src.config import cfg
from src.llm import LLMError, call_many
from src.planner.optimizer import solve_method
from src.planner.scheduler import schedule
from src.planner.scoring import satisfaction


def _ids(plan) -> set[tuple[int, str]]:
    return {(d, s.place_id) for d, day in enumerate(plan.days) for s in day.stops if s.role != "hotel"}


def _jsonable(history: list[dict]) -> list[dict]:
    return [{**h, "replies": {m: r.model_dump() for m, r in h["replies"].items()}} for h in history]


def _grant(pb, prep, plan, weights, proposed, include, exclude, floors):
    """Tries the new weights, then each request, keeping what the group can afford. Returns what was kept and refused."""
    def affordable(candidate) -> bool:
        sat = list(satisfaction(candidate, pb.util, prep.s_star, pb.places).values())
        return sum(sat) / len(sat) >= floors[0] and min(sat) >= floors[1]

    attempt = schedule(solve_method(pb, "M4", prep.s_star, weights=proposed), pb)
    if affordable(attempt):
        plan, weights = attempt, proposed
    granted_in, granted_out, refused = [], [], []
    for pid, wanted in [(p, True) for p in include] + [(p, False) for p in exclude]:
        trial = pb.copy()
        if wanted:
            trial.include = list(dict.fromkeys([*trial.include, pid]))
        else:
            trial.exclude = {*trial.exclude, pid}
        attempt = schedule(solve_method(trial, "M4", prep.s_star, weights=weights), trial)
        present = any(s.place_id == pid for day in attempt.days for s in day.stops)
        if present == wanted and affordable(attempt):
            pb.include, pb.exclude, plan = trial.include, trial.exclude, attempt
            (granted_in if wanted else granted_out).append(pid)
        else:
            refused.append(pid)
    return plan, weights, granted_in, granted_out, refused


def negotiate(prep, max_rounds: int | None = None) -> "Plan":  # noqa: F821
    C = cfg().negotiation
    max_rounds = max_rounds or C.max_rounds
    pb = prep.pb.copy()
    mids = [p.member_id for p in prep.profiles]
    weights = {m: 1 / len(mids) for m in mids}
    plan = schedule(solve_method(pb, "M1", prep.s_star), pb)              # round 1 reviews the plan of M1
    start = list(satisfaction(plan, pb.util, prep.s_star, pb.places).values())
    floors = (sum(start) / len(start) - C.max_drop, min(start) - C.max_drop)
    history, status, rnd = [], "max rounds", 0
    for rnd in range(1, max_rounds + 1):
        sat = satisfaction(plan, pb.util, prep.s_star, pb.places)
        built = [agent.reply_job(plan, sat, history, pb) for agent in prep.agents]
        answers = call_many([job for job, _ in built])
        replies = {a.m.member_id: clean_reply(r, plan, cand) for a, r, (_, cand) in zip(prep.agents, answers, built)}
        entry = {"round": rnd, "sat": sat, "replies": replies, "plan_cost": plan.cost_per_person}
        history.append(entry)
        if all(r.accept for r in replies.values()):
            status = "agreed"
            break
        if all(isinstance(a, LLMError) for a in answers):                 # nobody answered: nothing to mediate
            status = "agents did not answer"
            break
        if rnd == max_rounds:                                             # the last plan is one the agents have reviewed
            break
        decision = decide(history, weights, prep, max_rounds)
        include, exclude, proposed = enforce_rules(decision, history, prep, weights)
        new, weights, granted_in, granted_out, refused = _grant(pb, prep, plan, weights, proposed, include, exclude, floors)
        entry["decision"] = {"include": granted_in, "exclude": granted_out, "refused": refused, "weights": weights, "note": decision.note}
        if _ids(new) == _ids(plan):                                       # nothing moved: more rounds would repeat
            status = "no further change"
            break
        plan = new
    sat = satisfaction(plan, pb.util, prep.s_star, pb.places)
    if status != "agreed" and min(sat.values()) < cfg().optimizer.fairness_threshold:
        plan = schedule(solve_method(prep.pb, "M3", prep.s_star), prep.pb)
        status = "fallback to M3"
    plan.log.update(status=status, rounds=rnd, transcript=_jsonable(history))
    return plan
