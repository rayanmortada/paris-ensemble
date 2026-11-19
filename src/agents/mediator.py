"""The mediator: reads objections and returns place ids and member weights, never a plan."""
from collections import Counter

from src.config import cfg
from src.llm import MODEL_STRONG, LLMError, call
from src.prompts import load
from src.schemas import MediatorDecision, MemberWeight


def gave_in(history: list[dict]) -> Counter:
    """Times each member accepted while below the group mean, or had a request refused."""
    counts = Counter()
    for h in history:
        mean = sum(h["sat"].values()) / len(h["sat"])
        d = h.get("decision") or {}
        for mid, r in h["replies"].items():
            if r.accept and h["sat"][mid] < mean:
                counts[mid] += 1
            counts[mid] += sum(1 for x in r.add if d and x not in d.get("include", []))
            counts[mid] += sum(1 for x in r.remove if d and x not in d.get("exclude", []))
    return counts


def decide(history: list[dict], weights: dict[str, float], prep, max_rounds: int) -> MediatorDecision:
    """P4. If the call fails, the requests of this round wait and the weights stay as they are."""
    C = cfg().negotiation
    prompt = load("p4_mediator")
    P, n = prep.pb.places, len(prep.profiles)
    conceded = gave_in(history)
    latest = history[-1]
    member_lines = []
    for p in prep.profiles:
        mid = p.member_id
        trail = ", ".join(f"{h['sat'][mid]:.2f}" for h in history)
        accepted = "yes" if latest["replies"][mid].accept else "no"
        member_lines.append(f"{mid} | {p.name} | {trail} | {accepted} | {conceded[mid]} | {weights[mid]:.2f}")
    name = lambda pid: P.by_id[pid].name if pid in P.by_id else pid  # noqa: E731
    requests = []
    for p in prep.profiles:
        r = latest["replies"][p.member_id]
        if r.accept:
            requests.append(f"{p.name}: accepts")
        else:
            add = ", ".join(f"{x} ({name(x)})" for x in r.add)
            rem = ", ".join(f"{x} ({name(x)})" for x in r.remove)
            requests.append(f'{p.name}: add [{add}] remove [{rem}]; gives up "{r.give_up}"; says "{r.reason}"')
    last = history[-2].get("decision") if len(history) > 1 else None
    must = sorted({pid for p in prep.profiles for pid in p.must_see})
    try:
        return call(MediatorDecision, prompt.fill_system(max_requests=C.max_requests_per_member,
                                                         low=f"{C.weight_min_factor / n:.2f}",
                                                         high=f"{C.weight_max_factor / n:.2f}"),
                    prompt.fill_user(round=latest["round"], max_rounds=max_rounds, member_lines="\n".join(member_lines),
                                     must_see=", ".join(must) or "none", requests="\n".join(requests),
                                     last_decision=str(last) if last else "none"),
                    purpose="mediator", model=MODEL_STRONG, effort="medium", version=prompt.version)
    except LLMError:
        return MediatorDecision(include=[], exclude=[], note="The mediator did not answer, so the plan stays as it is.",
                                weights=[MemberWeight(member_id=m, weight=w) for m, w in weights.items()])


def enforce_rules(dec: MediatorDecision, history: list[dict], prep, weights: dict[str, float]):
    """Only ids asked for this round; never a must see excluded; no member steers alone; weights in range."""
    C, replies = cfg().negotiation, history[-1]["replies"]
    asked_add = {pid: mid for mid, r in replies.items() for pid in r.add}
    asked_remove = {pid: mid for mid, r in replies.items() for pid in r.remove}
    protected = {pid for p in prep.profiles for pid in p.must_see}
    used, include, exclude = Counter(), [], []
    for pid in dec.include:
        mid = asked_add.get(pid)
        if mid and pid not in dec.exclude and pid not in include and used[mid, "in"] < C.max_requests_per_member:
            include.append(pid)
            used[mid, "in"] += 1
    for pid in dec.exclude:
        mid = asked_remove.get(pid)
        if mid and pid not in protected and pid not in include and pid not in exclude and used[mid, "out"] < C.max_requests_per_member:
            exclude.append(pid)
            used[mid, "out"] += 1
    n = len(weights)
    asked = {x.member_id: x.weight for x in dec.weights}
    clamped = {m: min(max(asked.get(m, 1 / n), C.weight_min_factor / n), C.weight_max_factor / n) for m in weights}
    total = sum(clamped.values())
    return include, exclude, {m: v / total for m, v in clamped.items()}
