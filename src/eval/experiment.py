"""One row of metrics per group and method, always measured against the hidden true utilities."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import ndcg_score

from src.config import cfg
from src.eval.metrics import gini, sacrificed
from src.planner.optimizer import best_alone
from src.planner.scoring import daily_satisfaction, satisfaction
from src.prefs.model import place_features

CODES = [f"V{i}" for i in range(1, 11)]


def true_setup(group, prep, places):
    """True utilities and the true S*: the best each simulated traveler could get alone."""
    true_u = {m.member_id: m.true_utility(places) for m in group.members}
    pb = prep.pb.copy()
    pb.util = true_u
    pb.include, pb.exclude = list(prep.pb.include), set(prep.pb.exclude)
    return true_u, best_alone(pb)


def metric_row(group, method, plan, prep, true_u, s_true, places) -> tuple[dict, list[dict]]:
    sat = satisfaction(plan, true_u, s_true, places)
    s = np.array(list(sat.values()))
    daily = daily_satisfaction(plan, true_u, s_true, places)
    violations = plan.log.get("violations", [])
    zones = {places.by_id[stop.place_id].zone for d in plan.days for stop in d.stops
             if stop.role != "hotel" and stop.place_id in places.by_id}
    row = {"group_id": group.group_id, "kind": group.kind, "method": method, "n_members": len(s),
           "mean_sat": float(s.mean()), "min_sat": float(s.min()), "gap": float(s.max() - s.min()), "gini": gini(s),
           "sacrificed": sacrificed(np.array([daily[m] for m in sat])),
           "feasible": len(violations) == 0, "n_violations": len(violations),
           "travel_min_per_day": sum(st.travel_min for d in plan.days for st in d.stops) / max(len(plan.days), 1),
           "zones": len(zones), "n_stops": sum(st.role == "visit" for d in plan.days for st in d.stops),
           "cost": plan.cost_per_person, "dropped": len(plan.dropped), "relaxed": len(plan.log.get("relaxed") or []),
           "pred_mean_sat": float(np.mean(list(plan.satisfaction.values()))) if plan.satisfaction else None,
           "rounds": plan.log.get("rounds"), "status": plan.log.get("status"), "seconds": plan.log.get("seconds"),
           "llm_calls": plan.log.get("llm_calls"), "input_tokens": plan.log.get("input_tokens"),
           "output_tokens": plan.log.get("output_tokens"), "fallback_scores": plan.log.get("fallback_scores"),
           "not_counted": len(plan.log.get("not_counted") or []),
           "agent_requests_dropped": len(plan.log.get("agent_requests_dropped") or [])}
    for code in CODES:
        row[code] = sum(v["code"] == code for v in violations)
    outsider = group.outsider
    if outsider:
        rest = [v for m, v in sat.items() if m != outsider]
        row["minority_gap"] = float(sat[outsider] - np.mean(rest))
    members = [{"group_id": group.group_id, "kind": group.kind, "method": method, "member_id": m, "true_sat": v,
                "pred_sat": plan.satisfaction.get(m), "is_outsider": m == outsider, "daily": [round(x, 3) for x in daily[m]]}
               for m, v in sat.items()]
    return row, members


def model_accuracy(group, prep, true_u, places, dims_options=(8, 16, 32)) -> list[dict]:
    """E2: how well each design predicts true taste, for every traveler of a group."""
    rows = []
    idx = np.r_[places.idx("attraction"), places.idx("restaurant")]
    att = places.idx("attraction")
    features = {d: place_features(places, places.embeddings, d) for d in dims_options}
    alphas = cfg().model.first_design_alphas
    for agent, profile in zip(prep.agents, prep.profiles):
        mid = profile.member_id
        truth = true_u[mid]
        scored, y = prep.model.data[mid]
        scored, y = list(scored), np.asarray(y)
        prior = prep.model.prior(mid)
        wish = agent.places.unit @ _unit(_wish_vector(profile))
        preds = {"chosen model": prep.model.predict(mid), "profile only": prior, "wish similarity": wish}
        # the rejected designs: plain ridge on the raw scores, and a ridge correction to the profile with a fixed slope
        Z = np.c_[features[32], (prior - prior.mean()) / (prior.std() + 1e-9)]
        preds["plain ridge, PCA 32"] = RidgeCV(alphas=[0.1, 1, 10, 100]).fit(Z[scored], y).predict(Z)
        for d in dims_options:
            correction = RidgeCV(alphas=alphas).fit(features[d][scored], y - prior[scored]).predict(features[d])
            preds[f"first design, PCA {d}"] = np.clip(prior + correction, 0, 5)
        for name, pred in preds.items():
            rows.append({"group_id": group.group_id, "member_id": mid, "model": name,
                         "spearman": float(spearmanr(pred[idx], truth[idx]).statistic),
                         "ndcg20": float(ndcg_score(truth[att][None, :], pred[att][None, :], k=20))})
    return rows


def _unit(v):
    return v / np.linalg.norm(v)


def _wish_vector(profile):
    from src.llm import embed
    return embed([profile.wishes])[0]
