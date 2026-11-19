"""Figures and tables for the report, made only from result files, so every number can be traced.

  python scripts/make_figures.py --e1 results/E1 --e3 results/E3 --db data_local/study.db

One accent marks our method (M4) and the others stay grey: the question is always how M4 compares.
Colours where series need names were run through the palette validator (colour blindness and contrast).
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from src.eval.metrics import bootstrap_ci, holm  # noqa: E402

METHODS = ["M1", "M2", "M3", "M4", "M5"]
NAMES = {"M1": "Average", "M2": "Least misery", "M3": "Fairness rule", "M4": "Agent negotiation", "M5": "Single LLM"}
INK, MUTED, GRID = "#171717", "#68655D", "#E7E4DD"
ACCENT, OTHERS = "#C8912F", "#A8A39A"           # one highlight, the rest quiet
SERIES = ["#3F51B5", "#C8912F", "#00897B", "#C2452D"]
VIOLATIONS = {"V1": "Place does not exist", "V2": "Place used twice", "V3": "Closed at that time", "V4": "No time to travel",
              "V5": "Day runs too late", "V6": "Missing or badly timed meal", "V7": "Over budget", "V8": "Ignores a need or an avoid", "V9": "Wrong number of days", "V10": "Too much for the group's pace"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "figure.dpi": 130, "savefig.bbox": "tight", "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("  wrote", name)


def colour(method: str) -> str:
    return ACCENT if method == "M4" else OTHERS


def present(df: pd.DataFrame) -> list[str]:
    return [m for m in METHODS if m in set(df["method"])]


def fig_satisfaction(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
    methods = present(df)[::-1]
    for ax, col, title in zip(axes, ("mean_sat", "min_sat"), ("Average member", "Least satisfied member")):
        for y, m in enumerate(methods):
            values = df.loc[df.method == m, col]
            lo, hi = bootstrap_ci(values)
            ax.plot([lo, hi], [y, y], color=colour(m), lw=2, solid_capstyle="round")
            ax.scatter(values.mean(), y, s=52, color=colour(m), zorder=3, edgecolor="white", linewidth=1.2)
            ax.text(hi + 0.015, y, f"{values.mean():.2f}", va="center", color=INK, fontsize=9.5)
        ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Satisfaction by method, mean with 95% interval over groups", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.supxlabel("True satisfaction (1 is the best trip that member could have alone)", fontsize=10, color=MUTED)
    fig.tight_layout()
    save(fig, out, "fig1_satisfaction")


def fig_distribution(members, out):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    methods = present(members)[::-1]
    rng = np.random.default_rng(7)
    for y, m in enumerate(methods):
        values = members.loc[members.method == m, "true_sat"].to_numpy()
        ax.scatter(values, y + rng.uniform(-0.17, 0.17, len(values)), s=9, color=colour(m), alpha=0.55, linewidth=0)
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        ax.plot([q1, q3], [y - 0.3, y - 0.3], color=INK, lw=2, solid_capstyle="round")
        ax.scatter(med, y - 0.3, s=26, color=INK, zorder=3)
    ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("True satisfaction. Each pale dot is one simulated traveler; the black bar is the middle half and its dot the median")
    ax.set_title("Who is left behind: every traveler under every method")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig2_distribution")


def fig_gini(df, out):
    kinds = [k for k in ("friends", "family", "colleagues", "minority") if k in set(df["kind"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(9.4, 2.9), sharey=True, squeeze=False)
    methods = present(df)
    for ax, kind in zip(axes[0], kinds):
        means = [df.loc[(df.kind == kind) & (df.method == m), "gini"].mean() for m in methods]
        ax.bar(range(len(methods)), means, color=[colour(m) for m in methods], width=0.68, edgecolor="white", linewidth=1)
        for x, v in enumerate(means):
            ax.text(x, v + 0.004, f"{v:.2f}", ha="center", color=INK, fontsize=8.5)
        ax.set_xticks(range(len(methods)), methods)
        ax.set_title(kind.capitalize())
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Mean Gini (lower is fairer)")
    fig.suptitle("Inequality between members, by type of group", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.04, "  ".join(f"{m} {NAMES[m]}" for m in methods), color=MUTED, fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig3_gini_by_type")


def fig_minority(members, out):
    data = members[members.kind == "minority"]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 3))
    methods = present(data)[::-1]
    for y, m in enumerate(methods):
        rows = data[data.method == m]
        outsider, others = rows.loc[rows.is_outsider, "true_sat"].mean(), rows.loc[~rows.is_outsider, "true_sat"].mean()
        ax.plot([outsider, others], [y, y], color=GRID, lw=2.5, zorder=1)
        ax.scatter(outsider, y, s=60, color=SERIES[3], zorder=3, edgecolor="white", linewidth=1.2, label="The outsider" if y == 0 else None)
        ax.scatter(others, y, s=60, color=SERIES[0], zorder=3, edgecolor="white", linewidth=1.2, label="The other three" if y == 0 else None)
        ax.text(max(outsider, others) + 0.015, y, f"gap {others - outsider:+.2f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Mean true satisfaction, groups of three alike and one outsider")
    ax.set_title("Is one member sacrificed for the majority?")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, borderaxespad=0)
    save(fig, out, "fig4_minority")


def fig_violations(df, out):
    codes = [c for c in VIOLATIONS if c in df.columns]
    table = df.groupby("method")[codes].mean().reindex(present(df)).fillna(0)
    table.round(3).to_csv(out / "table_violations_per_plan.csv")
    if "M5" not in table.index:
        return
    others = table.drop(index="M5").to_numpy().sum()
    rows = table.loc["M5"]
    rows = rows[rows > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * max(len(rows), 2) + 1.2))
    ax.barh(range(len(rows)), rows.values, color=OTHERS, height=0.6, edgecolor="white", linewidth=1)
    ax.set_yticks(range(len(rows)), [f"{VIOLATIONS[c]} ({c})" for c in rows.index])
    top = rows.max() if len(rows) else 1
    for y, v in enumerate(rows.values):
        ax.text(v + top * 0.015, y, f"{v:.1f}", va="center", color=INK, fontsize=9.5)
    feasible = df.groupby("method")["feasible"].mean()
    note = "M1 to M4 had no problems at all." if others == 0 else f"M1 to M4 also had problems ({others:.2f} per plan across them), a bug to investigate."
    ax.set_xlabel(f"Problems per plan. {note}")
    ax.set_title(f"What goes wrong when one LLM plans alone ({feasible.get('M5', 0):.0%} of its plans pass every check)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig5_violations")


def fig_feedback(e3, out):
    methods = [m for m in ("M1", "M4") if m in set(e3["method"])]
    measures = [("spearman", "Model accuracy (Spearman)"), ("mean_sat", "Mean true satisfaction"), ("min_sat", "Lowest true satisfaction")]
    arms = list(e3["arm"].unique())
    means = e3.groupby(["method", "arm", "round"])[["spearman", "mean_sat", "min_sat"]].mean()
    fig, axes = plt.subplots(len(methods), 3, figsize=(10, 2.5 * len(methods) + 0.5), squeeze=False, sharex=True, sharey="col")
    for c, (col, label) in enumerate(measures):
        gap = max(means[col].max() - means[col].min(), 0.01) * 0.1      # smallest distance between two end labels
        for r, m in enumerate(methods):
            ax = axes[r][c]
            ends = []
            for k, arm in enumerate(arms):
                series = means.loc[(m, arm), col]
                ax.plot(series.index, series.values, color=SERIES[k], lw=2, marker="o", ms=5, label=arm)
                ends.append((series.values[-1], series.index[-1]))
            ends.sort()
            lift = max(gap - (ends[-1][0] - ends[0][0]), 0) / 2         # spread evenly only when they would touch
            for (value, x), shift in zip(ends, (-lift, lift)):
                ax.text(x + 0.08, value + shift, f"{value:.2f}", va="center", color=INK, fontsize=8.5)
            if r == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(NAMES[m], fontsize=10, color=INK)
            if r == len(methods) - 1:
                ax.set_xlabel("Feedback rounds")
            ax.set_xticks(sorted(e3["round"].unique()))
            ax.set_xlim(-0.1, e3["round"].max() + 0.5)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower right", bbox_to_anchor=(0.99, 0.99), ncol=len(arms), fontsize=9, borderaxespad=0)
    fig.tight_layout()
    save(fig, out, "fig6_feedback")


def fig_cost(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    methods = present(df)[::-1]
    for ax, col, label in zip(axes, ("llm_calls", "seconds"), ("LLM calls per plan", "Seconds per plan")):
        values = [df.loc[df.method == m, col].mean() for m in methods]
        ax.barh(range(len(methods)), values, color=[colour(m) for m in methods], height=0.6, edgecolor="white", linewidth=1)
        top = max(values) or 1
        for y, v in enumerate(values):
            ax.text(v + top * 0.02, y, f"{v:.1f}", va="center", color=INK, fontsize=9.5)
        ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
        ax.set_xlim(0, top * 1.18)
        ax.set_title(label)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("Not counted: the shared step that every method needs (profiles and place scores)", fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, out, "fig7_cost")


def fig_map(plans, out):
    from src.pipeline import load_context
    places = load_context().places
    rows = {(r["group_id"], r["method"]): r for r in plans if r.get("run_status") == "ok"}
    groups = sorted({g for g, m in rows if m == "M4" and (g, "M1") in rows})
    if not groups:
        return
    # the group where negotiation changed the least satisfied member the most
    group = max(groups, key=lambda g: abs(rows[g, "M4"]["min_sat"] - rows[g, "M1"]["min_sat"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True, sharey=True)
    for ax, m in zip(axes, ("M1", "M4")):
        plan = rows[group, m]["plan"]
        for d, day in enumerate(plan["days"]):
            pts = [places.by_id[s["place_id"]] for s in day["stops"] if s["role"] != "hotel"]
            ax.plot([p.lon for p in pts], [p.lat for p in pts], color=SERIES[d % 4], lw=1.8, marker="o", ms=5.5, label=f"Day {d + 1}", zorder=2)
        hotel = places.by_id[plan["hotel_id"]]
        ax.scatter(hotel.lon, hotel.lat, marker="s", s=52, color=INK, zorder=3, label="Hotel")
        ax.set_title(f"{NAMES[m]}, lowest satisfaction {rows[group, m]['min_sat']:.2f}")
        ax.set_aspect(1 / np.cos(np.radians(48.86)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle(f"One group, {group}, under two methods (picked as the group where they differ most)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig8_plans_on_a_map")


def tables(df, out):
    rows = []
    for m in present(df):
        d = df[df.method == m]
        row = {"method": m, "name": NAMES[m], "groups": len(d)}
        for col in ("mean_sat", "min_sat", "gap", "gini"):
            lo, hi = bootstrap_ci(d[col])
            row.update({col: round(d[col].mean(), 3), f"{col}_median": round(d[col].median(), 3),
                        f"{col}_iqr": round(d[col].quantile(0.75) - d[col].quantile(0.25), 3),
                        f"{col}_ci_low": round(lo, 3), f"{col}_ci_high": round(hi, 3)})
        row.update({"sacrificed_share": round(d["sacrificed"].mean(), 3), "feasible_share": round(d["feasible"].mean(), 3),
                    "violations_per_plan": round(d["n_violations"].mean(), 2), "travel_min_per_day": round(d["travel_min_per_day"].mean(), 1),
                    "zones": round(d["zones"].mean(), 2), "cost": round(d["cost"].mean(), 1), "llm_calls": round(d["llm_calls"].mean(), 1),
                    "seconds": round(d["seconds"].mean(), 1)})
        if "minority_gap" in d:
            row["minority_gap"] = round(d["minority_gap"].mean(), 3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "table_main_results.csv", index=False)

    tests = []
    for col in ("mean_sat", "min_sat", "gini"):
        pivot = df.pivot(index="group_id", columns="method", values=col)
        found = []
        for other in ("M1", "M2", "M3", "M5"):
            if {other, "M4"} <= set(pivot.columns):
                pair = pivot[["M4", other]].dropna()
                diff = pair["M4"] - pair[other]
                p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
                found.append({"measure": col, "compared_with": other, "groups": len(pair), "median_difference_M4_minus_other": round(diff.median(), 4), "p": p})
        if found:
            for f, adj in zip(found, holm([f["p"] for f in found])):
                f["p_holm"] = adj
            tests += found
    frame = pd.DataFrame(tests)
    frame.to_csv(out / "table_wilcoxon_m4.csv", index=False)
    print(frame.to_string(index=False))


def study_tables(db: Path, out: Path) -> None:
    con = sqlite3.connect(db)
    ratings = pd.read_sql_query("SELECT r.*, p.method FROM ratings r JOIN (SELECT trip_id, label, method FROM plans WHERE version = 1) p ON r.trip_id = p.trip_id AND r.plan_label = p.label", con)
    prefs = pd.read_sql_query("SELECT f.*, p.method FROM prefs f JOIN (SELECT trip_id, label, method FROM plans WHERE version = 1) p ON f.trip_id = p.trip_id AND f.preferred_label = p.label", con)
    actions = pd.read_sql_query("SELECT action, COUNT(*) AS n FROM actions GROUP BY action", con)
    con.close()
    if ratings.empty:
        print("  no ratings yet")
        return
    summary = ratings.groupby("method")[["wishes_respected", "fair", "realistic", "clear_explanation"]].agg(["mean", "count"]).round(2)
    summary.to_csv(out / "table_user_study_ratings.csv")
    prefs.groupby("method").size().rename("preferred_by").to_csv(out / "table_user_study_preference.csv")
    actions.to_csv(out / "table_user_study_actions.csv", index=False)
    print("  user study:", ratings["trip_id"].nunique(), "groups,", ratings["member_name"].nunique(), "names")
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", default="results/E1")
    parser.add_argument("--e3", default="results/E3")
    parser.add_argument("--db", default="data_local/study.db")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    e1, e3, db = Path(args.e1), Path(args.e3), Path(args.db)
    if (e1 / "metrics.csv").exists():
        df = pd.read_csv(e1 / "metrics.csv")
        members = pd.read_csv(e1 / "members.csv")
        print(f"E1: {df['group_id'].nunique()} groups, {len(df)} plans")
        info = json.loads((e1 / "run_info.json").read_text()) if (e1 / "run_info.json").exists() else {}
        if info.get("mock_llm"):
            print("  WARNING: this run used the offline stand-in, so its numbers mean nothing.")
        tables(df, out)
        fig_satisfaction(df, out)
        fig_distribution(members, out)
        fig_gini(df, out)
        fig_minority(members, out)
        fig_violations(df, out)
        fig_cost(df, out)
        plans_file = e1 / "plans.jsonl"
        if plans_file.exists():
            fig_map([json.loads(line) for line in plans_file.read_text().splitlines()], out)
    if (e1 / "e2.csv").exists():
        e2 = pd.read_csv(e1 / "e2.csv")
        table = e2.groupby("model")[["spearman", "ndcg20"]].agg(["mean", "std"]).round(3).sort_values(("spearman", "mean"), ascending=False)
        table.to_csv(out / "table_model_accuracy.csv")
        print(table.to_string())
    if (e3 / "e3.csv").exists():
        fig_feedback(pd.read_csv(e3 / "e3.csv"), out)
    if db.exists():
        study_tables(db, out)


if __name__ == "__main__":
    main()
