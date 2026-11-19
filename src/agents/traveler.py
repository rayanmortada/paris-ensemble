"""One traveler agent: reads the wishes into a profile, scores a sample of places, answers proposed plans."""
import hashlib
from datetime import timedelta

import numpy as np

from src.config import cfg
from src.llm import LLMError, call, call_many, embed
from src.planner.optimizer import visitable
from src.prompts import load
from src.schemas import MemberInput, MemberReply, Profile, ProfileOut, ScoreBatch


def plan_lines(plan, places) -> str:
    lines = []
    for d, day in enumerate(plan.days, 1):
        for s in day.stops:
            if s.role != "hotel" and s.place_id in places.index:
                lines.append(f"Day {d} ({day.date:%a %d %b}) | {s.role} | {places.by_id[s.place_id].name} | {s.place_id}")
    return "\n".join(lines)


class TravelerAgent:
    def __init__(self, member: MemberInput, places):
        self.m, self.places, self.profile = member, places, None
        self.fallback_scores = 0

    def build_profile(self) -> Profile:
        prompt = load("p1_profile")
        out = call(ProfileOut, prompt.system, prompt.fill_user(wishes=self.m.wishes), purpose="profile", version=prompt.version)
        themes = cfg().themes
        imp = np.array([getattr(out.theme_importance, t) for t in themes], float).clip(0, 5)
        w = imp / imp.sum() if imp.sum() > 0 else np.full(len(imp), 1 / len(imp))
        must, avoid, unmatched = [], [], []
        for names, target in ((out.must_see_names, must), (out.avoid_names, avoid)):
            for name in names:
                found = self.places.match_names([name])
                target.extend(found)
                if not found:
                    unmatched.append(name)
        budget = self.m.budget_eur if self.m.budget_eur is not None else out.budget_eur
        self.profile = Profile(
            member_id=self.m.member_id, name=self.m.name, wishes=self.m.wishes,
            theme_weights=dict(zip(themes, w.tolist())), budget_eur=float(budget if budget else 800),
            vegetarian=self.m.vegetarian or out.vegetarian, mobility=self.m.mobility or out.mobility,
            must_see=list(dict.fromkeys(must)), avoid=list(dict.fromkeys(avoid)),
            pace=self.m.pace or out.pace, unmatched=unmatched, language=out.language)
        return self.profile

    def sample_places(self) -> list[str]:
        """The places closest to the wishes plus a random share, so the sample also holds places they would dislike."""
        a = cfg().agents
        seed = int(hashlib.sha256(f"{self.m.member_id}|{self.m.wishes}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        wish = embed([self.m.wishes])[0]
        sim = self.places.unit @ (wish / np.linalg.norm(wish))
        picked = []
        for kind, near, rand in (("attraction", a.sample_near_attractions, a.sample_random_attractions),
                                 ("restaurant", a.sample_near_restaurants, a.sample_random_restaurants)):
            idx = self.places.idx(kind)
            order = idx[np.argsort(-sim[idx])]
            picked += order[:near].tolist()
            rest = order[near:]
            if len(rest):
                picked += rng.choice(rest, min(rand, len(rest)), replace=False).tolist()
        ids = list(dict.fromkeys([self.places.ids[i] for i in picked] + self.profile.must_see))
        rng.shuffle(ids)                          # position in a batch must not bias the score
        return ids

    def score_sample(self) -> dict[str, float]:
        prompt = load("p2_score")
        p, size = self.profile, cfg().agents.score_batch
        ids = self.sample_places()
        batches = [ids[i:i + size] for i in range(0, len(ids), size)]
        top = sorted(p.theme_weights, key=lambda t: -p.theme_weights[t])[:3]
        jobs = [{"schema": ScoreBatch, "system": prompt.system, "purpose": "score", "version": prompt.version,
                 "user": prompt.fill_user(wishes=p.wishes, top_themes=", ".join(top), budget=f"{p.budget_eur:.0f}",
                                          vegetarian="yes" if p.vegetarian else "no",
                                          mobility="yes" if p.mobility else "no", pace=p.pace,
                                          lines=self.places.lines(b))} for b in batches]
        results = call_many(jobs)
        failed = [res for res in results if isinstance(res, LLMError)]
        if len(failed) * 2 > len(batches):        # a plan from mostly neutral scores would only look fine
            raise failed[0]
        scores = {}
        for batch, res in zip(batches, results):
            got = {} if isinstance(res, LLMError) else {s.id: s.score for s in res.scores}
            for pid in batch:
                if pid not in got:
                    self.fallback_scores += 1
                scores[pid] = float(np.clip(got.get(pid, 2), 0, 5))      # neutral 2 when missing, and counted
        return scores

    def candidates(self, plan, pb) -> list[str]:
        """The places this member may ask to add: their best predicted attractions missing from the plan."""
        P, a = pb.places, cfg().agents
        in_plan = {s.place_id for d in plan.days for s in d.stops}
        dates = [pb.trip.start_date + timedelta(days=d) for d in range(pb.trip.days)]
        util = pb.util[self.m.member_id]
        out = []
        for i in P.idx("attraction")[np.argsort(-util[P.idx("attraction")])]:
            pid = P.ids[i]
            if pid not in in_plan and pid not in pb.exclude and any(visitable(P.items[i], day) for day in dates):
                out.append(pid)
            if len(out) == a.candidates_in_reply:
                break
        return out

    def reply_job(self, plan, sat: dict[str, float], history: list[dict], pb) -> tuple[dict, list[str]]:
        prompt = load("p3_reply")
        p, P, mid = self.profile, pb.places, self.m.member_id
        names = {x.member_id: x.name for x in pb.profiles}
        cand = self.candidates(plan, pb)
        util = pb.util[mid]
        cand_lines = "\n".join(f"{c} | {P.by_id[c].name} | {P.by_id[c].category} | {util[P.index[c]]:.1f}" for c in cand)
        past = []
        for h in history:
            r = h["replies"].get(mid)
            d = h.get("decision") or {}
            if r:
                got_add = [x for x in r.add if x in d.get("include", [])]
                got_rem = [x for x in r.remove if x in d.get("exclude", [])]
                past.append(f"Round {h['round']}: asked add {r.add} remove {r.remove}; granted add {got_add} remove {got_rem}")
        mean = sum(sat.values()) / len(sat)
        job = {"schema": MemberReply, "purpose": "reply", "version": prompt.version,
               "system": prompt.fill_system(name=p.name),
               "user": prompt.fill_user(wishes=p.wishes, notes="; ".join(p.notes) or "none",
                                        plan_lines=plan_lines(plan, P),
                                        all_members=", ".join(f"{names[m]} {v:.2f}" for m, v in sat.items()),
                                        own=f"{sat[mid]:.2f}", mean=f"{mean:.2f}",
                                        candidate_lines=cand_lines, history="\n".join(past) or "none yet", language=p.language)}
        return job, cand


def clean_reply(r, plan, candidates: list[str]) -> MemberReply:
    """The prompt states the limits, and code enforces them again."""
    if isinstance(r, LLMError):
        return MemberReply(accept=False, add=[], remove=[], give_up="", reason="(no answer from this agent)")
    k = cfg().agents.max_changes_per_reply
    in_plan = {s.place_id for d in plan.days for s in d.stops if s.role != "hotel"}
    r.add = list(dict.fromkeys(p for p in r.add if p in candidates))[:k]
    r.remove = list(dict.fromkeys(p for p in r.remove if p in in_plan))[:k]
    if r.accept:
        r.add, r.remove = [], []
    return r
