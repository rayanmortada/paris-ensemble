"""The one path from a group's wishes to a checked plan. The app and the experiments both use it."""
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.agents.traveler import TravelerAgent, plan_lines
from src.config import ROOT, cfg
from src.consensus.llm_planner import llm_plan
from src.consensus.negotiation import negotiate
from src.data.places import Places
from src.llm import MODEL_STRONG, LLMError, call
from src.planner.optimizer import Problem, best_alone, solve_method
from src.planner.scheduler import schedule
from src.planner.scoring import not_doable, satisfaction
from src.planner.validator import validate
from src.prefs.model import PrefModel, place_features
from src.prompts import load
from src.schemas import Explanations, Feedback, Plan, Profile, TripRequest

METHOD_NAMES = {"M1": "Average", "M2": "Least misery", "M3": "Fairness rule", "M4": "Agent negotiation", "M5": "Single LLM planner"}


@dataclass
class Context:
    places: Places
    X: np.ndarray                               # place_features, computed once
    neighbours: dict[int, list[int]]
    build_info: dict = field(default_factory=dict)


def load_context() -> Context:
    places = Places.load()
    zones = json.loads((ROOT / "data" / "zones.json").read_text())
    info_path = ROOT / "data" / "build_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return Context(places, place_features(places, places.embeddings),
                   {int(k): v for k, v in zones["neighbours"].items()}, info)


@dataclass
class Prepared:
    trip: TripRequest
    agents: list[TravelerAgent]
    profiles: list[Profile]
    model: PrefModel
    pb: Problem
    s_star: dict[str, float]
    notes: list[str] = field(default_factory=list)


def _utility_for_everyone(model: PrefModel, profiles: list[Profile]) -> dict[str, np.ndarray]:
    return {p.member_id: model.predict(p.member_id) for p in profiles}


def prepare(trip: TripRequest, ctx: Context, *, app: bool = False, progress: Callable[[str], None] = lambda _: None) -> Prepared:
    agents = [TravelerAgent(m, ctx.places) for m in trip.members]
    progress("Reading everyone's wishes")
    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        profiles = list(pool.map(lambda a: a.build_profile(), agents))
        progress("Each traveler's agent is scoring places")
        scores = list(pool.map(lambda a: a.score_sample(), agents))
    progress("Learning each traveler's taste")
    model = PrefModel(ctx.places, ctx.X)
    for profile, sc in zip(profiles, scores):
        model.fit(profile, sc)
    pb = Problem(trip=trip, profiles=profiles, util=_utility_for_everyone(model, profiles), places=ctx.places,
                 neighbours=ctx.neighbours,
                 include=list(dict.fromkeys(pid for p in profiles for pid in p.must_see)),
                 exclude={pid for p in profiles for pid in p.avoid})
    if app:
        pb.time_limit_s = cfg().app.time_limit_s
        pb.workers = cfg().app.workers
    progress("Finding the best possible trip for each traveler")
    s_star = best_alone(pb)
    notes = []
    if any(p.vegetarian for p in profiles):
        notes.append("Someone in the group is vegetarian, so every meal is at a place OpenStreetMap lists with vegetarian options. Menus change, so check before you go.")
    if any(p.mobility for p in profiles):
        notes.append("Someone in the group has limited mobility. Places known to be inaccessible are left out, but most Paris metro stations have no lift, so plan on buses or taxis for the longer trips.")
    return Prepared(trip, agents, profiles, model, pb, s_star, notes)


def clone_prepared(prep: Prepared) -> Prepared:
    """An independent copy: shares the read-only place data, copies everything feedback changes."""
    other = copy.copy(prep)
    other.pb = prep.pb.copy()
    other.pb.util = dict(prep.pb.util)
    other.s_star = dict(prep.s_star)
    other.profiles = [p.model_copy(deep=True) for p in prep.profiles]
    other.pb.profiles = other.profiles
    other.agents = [copy.copy(a) for a in prep.agents]
    for agent, profile in zip(other.agents, other.profiles):
        agent.profile = profile
    other.model = PrefModel(prep.pb.places, prep.model.X)
    other.model.weights = dict(prep.model.weights)
    other.model.line = dict(prep.model.line)
    other.model.fixes = dict(prep.model.fixes)
    other.model.data = dict(prep.model.data)
    other.model.feedback = {m: (list(a), list(b)) for m, (a, b) in prep.model.feedback.items()}
    return other


def explain_plan(plan: Plan, prep: Prepared) -> dict[str, str]:
    """P5. It never sees how the plan was made, so the blind comparison stays blind."""
    prompt = load("p5_explain")
    members = "\n".join(f"{p.member_id} | {p.name} | {p.language} | {plan.satisfaction.get(p.member_id, 0):.2f} | {p.wishes}" for p in prep.profiles)
    stops = plan_lines(plan, prep.pb.places)
    if any(p.vegetarian for p in prep.profiles):      # the names alone do not show it, and it told a vegetarian the opposite
        stops += "\nEvery lunch and dinner above is at a place listed with vegetarian options."
    try:
        out = call(Explanations, prompt.system, prompt.fill_user(plan_lines=stops, member_lines=members),
                   purpose="explain", model=MODEL_STRONG, effort="low", version=prompt.version)
    except LLMError:
        return {}
    return {x.member_id: x.text for x in out.items}


def set_apart_agent_requests(plan: Plan, own_requests: list[str]) -> None:
    """A group only hears about what its members asked for. What their agents asked for while negotiating goes in the
    negotiation log: on the plan it would puzzle people, and give away which of the two blind plans was negotiated."""
    relaxed = plan.log.get("relaxed") or []
    agents = [r for r in relaxed if r.startswith("dropped request ") and r.removeprefix("dropped request ") not in own_requests]
    plan.log["relaxed"] = [r for r in relaxed if r not in agents]
    plan.log["agent_requests_dropped"] = [r.removeprefix("dropped request ") for r in agents]


def run_method(prep: Prepared, method: str, *, explain: bool = True, max_rounds: int | None = None) -> Plan:
    started = time.time()
    if method == "M4":
        plan = negotiate(prep, max_rounds)
    elif method == "M5":
        plan = llm_plan(prep)
    else:
        plan = schedule(solve_method(prep.pb, method, prep.s_star), prep.pb)
    plan.method = method
    set_apart_agent_requests(plan, prep.pb.include)
    plan.log["not_counted"] = not_doable(plan, prep.pb)
    plan.satisfaction = satisfaction(plan, prep.pb.util, prep.s_star, prep.pb.places)
    plan.log["violations"] = [v.model_dump() for v in validate(plan, prep.pb)]
    plan.log["notes"] = prep.notes
    plan.log["fallback_scores"] = sum(a.fallback_scores for a in prep.agents)
    plan.log["seconds"] = round(time.time() - started, 1)
    if explain:
        plan.log["explanations"] = explain_plan(plan, prep)
    return plan


def plan_trip(trip: TripRequest, method: str, ctx: Context, *, explain: bool = True) -> Plan:
    return run_method(prepare(trip, ctx), method, explain=explain)


def apply_feedback(prep: Prepared, actions: list[Feedback], *, update_model: bool = True) -> None:
    """Removed places stay out and added places stay in; when asked, the member's model learns from the action too."""
    names = prep.pb.places.by_id
    for fb in actions:
        if update_model:
            prep.model.update(fb.member_id, fb)
        if fb.action == "remove":
            prep.pb.exclude.add(fb.place_id)
        elif fb.action == "add":
            prep.pb.include = list(dict.fromkeys([*prep.pb.include, fb.place_id]))
        profile = next(p for p in prep.profiles if p.member_id == fb.member_id)
        verb = {"remove": "removed", "add": "added", "up": "liked", "down": "disliked"}[fb.action]
        profile.notes.append(f"{verb} {names[fb.place_id].name}")
    if update_model:
        prep.pb.util = _utility_for_everyone(prep.model, prep.profiles)
        prep.s_star = best_alone(prep.pb)
