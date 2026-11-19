"""The first real OpenAI test. Nothing is mocked and nothing comes from the cache, so it costs a little money.

  python scripts/smoke_llm.py          the key, the models, embeddings, and real answers for wishes, scoring and enrichment
  python scripts/smoke_llm.py --trip   one real trip through all five methods; run it after build_places.py --stage enrich

The exit code is 0 only if every check passed. Everything the LLM answered is saved in results/smoke/ to be read.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ["LLM_CACHE"] = "off"

import numpy as np  # noqa: E402
from openai import APIConnectionError, AuthenticationError, OpenAI  # noqa: E402

from build_places import enrich_line  # noqa: E402
from src import llm  # noqa: E402
from src.agents.traveler import TravelerAgent  # noqa: E402
from src.config import cfg  # noqa: E402
from src.data.places import fold  # noqa: E402
from src.eval.simulate import describe, make_groups  # noqa: E402
from src.pipeline import METHOD_NAMES, clone_prepared, load_context, prepare, run_method  # noqa: E402
from src.prompts import load  # noqa: E402
from src.schemas import EnrichBatch, MemberInput, PersonaText, TripRequest  # noqa: E402
from ui.examples import EXAMPLES  # noqa: E402

OUT = ROOT / "results" / "smoke"
failed = 0


def check(name: str, ok: bool, detail: object = "") -> bool:
    global failed
    failed += not ok
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail != "" else ""))
    return ok


def finish(started: float, tokens_before: dict) -> None:
    now = llm.stats()
    print(f"\n{now['calls'] - tokens_before['calls']} chat calls, {now['input_tokens'] - tokens_before['input_tokens']} tokens in, "
          f"{now['output_tokens'] - tokens_before['output_tokens']} tokens out, {time.time() - started:.0f} s. "
          "The per-call log is in results/llm_calls.csv, and the OpenAI usage page shows the exact cost.")
    print(f"{failed} checks failed. Read the answers in results/smoke/." if failed else "All checks passed. Read the answers in results/smoke/.")
    sys.exit(1 if failed else 0)


def save(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str))


def place_id(places, word: str) -> str:
    return next(p.id for p in places.items if word in fold(f"{p.name} {p.name_local or ''} {p.name_en or ''}"))


def member(name: str, wishes: str) -> MemberInput:
    """Only the text: no form fields, so everything found was found by the LLM."""
    return MemberInput(member_id=name.lower(), name=name, wishes=wishes)


def step_key_and_models() -> None:
    try:
        available = {m.id for m in OpenAI().models.list()}
    except AuthenticationError as e:
        reason = re.sub(r"sk-[\w*-]+", "sk-***", e.body.get("message", "") if isinstance(e.body, dict) else "")
        sys.exit(f"OpenAI refused this key ({reason}) Put a working key in OPENAI_API_KEY in .env.")
    except APIConnectionError:
        sys.exit("Could not reach OpenAI. Check the internet connection.")
    check("the key works", True, f"{len(available)} models visible")
    missing = [m for m in (llm.MODEL_FAST, llm.MODEL_STRONG, llm.EMBED_MODEL) if not check(f"this key can use {m}", m in available)]
    if missing:
        print("Models this key can use:", ", ".join(sorted(m for m in available if m.startswith("gpt-"))))
        sys.exit("Choose others with MODEL_FAST, MODEL_STRONG or EMBED_MODEL in .env, then run this again.")


def step_embeddings() -> None:
    a, b, c = llm.embed([f"Eiffel Tower {time.time()}", "Tour Eiffel, the iron tower of Paris", "a vegetarian lunch menu"])
    check(f"embeddings have {llm.EMBED_DIM} numbers", a.shape == (llm.EMBED_DIM,), a.shape)
    near, far = [float(a @ x / np.linalg.norm(a) / np.linalg.norm(x)) for x in (b, c)]
    check("embeddings put similar texts closer", near > far, f"{near:.2f} against {far:.2f}")


def step_wishes(ctx) -> None:
    P = ctx.places
    wishes = {m["name"]: m["wishes"] for group in EXAMPLES.values() for m in group}
    cases = {"French": "J'adore l'art et je voudrais absolument voir le musée du Louvre. Je suis végétarien et mon budget est de 600 euros pour tout le voyage.",
             "Arabic": "أحب الفن والمتاحف وأريد زيارة متحف اللوفر. أنا نباتي وميزانيتي 600 يورو للرحلة كلها."}
    out = {}
    for name in ("Sara", "Samir", "Lina", "Karim", "French", "Arabic"):
        out[name] = TravelerAgent(member(name, wishes.get(name) or cases[name]), P).build_profile()
    save("profiles.json", {k: v.model_dump() for k, v in out.items()})
    top = lambda p: max(p.theme_weights, key=p.theme_weights.get)     # noqa: E731
    check("Sara: art comes first", top(out["Sara"]) == "art", top(out["Sara"]))
    check("Sara: must see Orsay", place_id(P, "orsay") in out["Sara"].must_see, out["Sara"].must_see)
    check("Samir: limited mobility is understood", out["Samir"].mobility)
    check("Samir: must see Les Invalides", place_id(P, "invalides") in out["Samir"].must_see, out["Samir"].must_see)
    check("Lina: vegetarian is understood", out["Lina"].vegetarian)
    check("Karim: the Eiffel Tower goes to avoid", place_id(P, "eiffel") in out["Karim"].avoid, out["Karim"].avoid)
    check("Karim: no invented vegetarian or mobility need", not out["Karim"].vegetarian and not out["Karim"].mobility)
    louvre = place_id(P, "louvre")
    for name in ("French", "Arabic"):
        p = out[name]
        check(f"{name} wishes: the Louvre, vegetarian, 600 EUR", louvre in p.must_see and p.vegetarian and p.budget_eur == 600,
              f"must see {p.must_see}, vegetarian {p.vegetarian}, budget {p.budget_eur}")


def step_scoring(ctx) -> None:
    wishes = next(m["wishes"] for group in EXAMPLES.values() for m in group if m["name"] == "Sara")
    agent = TravelerAgent(member("Sara", wishes), ctx.places)
    agent.build_profile()
    scores = agent.score_sample()
    save("scores_sara.json", {ctx.places.by_id[k].name: v for k, v in sorted(scores.items(), key=lambda kv: -kv[1])})
    check("every place in the sample got a score", agent.fallback_scores == 0, f"{len(scores)} scores, {agent.fallback_scores} missing")
    values = np.array(list(scores.values()))
    check("the scores use the scale", len(set(values.tolist())) >= 3 and values.std() >= 0.5, f"values {sorted(set(values.tolist()))}, spread {values.std():.2f}")
    orsay = place_id(ctx.places, "orsay")
    check("Sara's must see Orsay scores high", scores.get(orsay, 0) >= 4, scores.get(orsay))


def step_enrichment() -> None:
    base = json.loads((ROOT / "data" / "places_base.json").read_text())
    by_word = {w: next(p for p in base if w in fold(p["name"])) for w in ("louvre", "luxembourg gardens")}
    picks = [*by_word.values(), next(p for p in base if p["kind"] == "restaurant")]
    prompt = load("p8_enrich")
    out = llm.call(EnrichBatch, prompt.system, prompt.fill_user(lines="\n".join(enrich_line(p) for p in picks)), purpose="enrich", version=prompt.version)
    save("enrichment.json", out.model_dump())
    got = {e.id: e for e in out.places}
    check("enrichment copies every id exactly", set(got) == {p["id"] for p in picks}, sorted(got))
    if set(got) != {p["id"] for p in picks}:
        return
    check("themes stay between 0 and 5", all(0 <= v <= 5 for e in out.places for v in e.themes.model_dump().values()))
    check("the Louvre is about art", got[by_word["louvre"]["id"]].themes.art >= 4)
    check("the Luxembourg Gardens are about nature", got[by_word["luxembourg gardens"]["id"]].themes.nature >= 3)
    check("the Luxembourg Gardens are outdoors", got[by_word["luxembourg gardens"]["id"]].indoor is False)
    check("the restaurant has a price tier", got[picks[2]["id"]].price_tier in (1, 2, 3), got[picks[2]["id"]].price_tier)


def step_strong_model(ctx) -> None:
    traveler = make_groups(ctx.places, per_type=1)[0].members[0]
    prompt = load("p7_wishes")
    out = llm.call(PersonaText, prompt.system, prompt.fill_user(**describe(traveler, ctx.places)), purpose="persona", model=llm.MODEL_STRONG, version=f"smoke {time.time()}")
    save("simulated_wishes.json", {"given": describe(traveler, ctx.places), "written": out.wishes})
    check(f"{llm.MODEL_STRONG} answers in the required format", len(out.wishes) > 40, out.wishes[:140])


def step_trip(ctx) -> None:
    group = EXAMPLES["Friends with different tastes"]
    start = date.fromisoformat(cfg().experiment.start_date)
    trip = TripRequest(trip_id="smoke", start_date=start, days=3, members=[
        MemberInput(member_id=f"m{i}", name=m["name"], wishes=m["wishes"], budget_eur=m["budget"], vegetarian=m["veg"], mobility=m["mob"], pace=m["pace"])
        for i, m in enumerate(group)])
    started = time.time()
    prep = prepare(trip, ctx, app=True, progress=lambda text: print("  ...", text))
    print(f"  prepared in {time.time() - started:.0f} s")
    check("no scoring call failed", sum(a.fallback_scores for a in prep.agents) == 0)
    out = {"profiles": [p.model_dump() for p in prep.profiles], "best_alone": prep.s_star, "plans": {}}
    for method in METHOD_NAMES:
        before, started = llm.stats(), time.time()
        plan = run_method(clone_prepared(prep), method)
        after = llm.stats()
        names = {p.member_id: p.name for p in prep.profiles}
        codes = [v["code"] for v in plan.log["violations"]]
        print(f"\n{method} {METHOD_NAMES[method]}: {time.time() - started:.0f} s, {after['calls'] - before['calls']} calls, cost {plan.cost_per_person:.0f} EUR per person")
        print("  satisfaction:", {names[m]: round(v, 2) for m, v in plan.satisfaction.items()})
        print("  relaxed:", plan.log.get("relaxed") or "nothing", "| dropped:", [d.reason for d in plan.dropped] or "nothing", "| problems:", codes or "none")
        for d, day in enumerate(plan.days, 1):
            print(f"  day {d}:", ", ".join(ctx.places.by_id[s.place_id].name for s in day.stops if s.role == "visit" and s.place_id in ctx.places.by_id))
        out["plans"][method] = plan.model_dump(mode="json")
        in_plan = {s.place_id for day in plan.days for s in day.stops}
        if method != "M5":
            check(f"{method} passes every check of the validator", not codes, codes or "")
            wanted = [pid for p in prep.profiles for pid in p.must_see]
            missing = [ctx.places.by_id[pid].name for pid in wanted if pid not in in_plan]
            check(f"{method} keeps every must see, or says what it dropped", not missing or bool(plan.log.get("relaxed")), missing or "")
        check(f"{method} explains the plan to everyone", set(plan.log.get("explanations", {})) == set(names))
        if method == "M4":
            print("  negotiation:", plan.log["status"], "after", plan.log["rounds"], "rounds")
            for entry in plan.log["transcript"]:
                for m, r in entry["replies"].items():
                    print(f"    round {entry['round']} {names[m]}: {'accepts' if r['accept'] else 'objects'}; add {len(r['add'])}, remove {len(r['remove'])}. {r['reason'][:110]}")
                if entry.get("decision"):
                    print(f"    mediator: {entry['decision']['note'][:140]}")
            answered = all(not r["reason"].startswith("(") for entry in plan.log["transcript"] for r in entry["replies"].values())
            check("every traveler agent answered", answered)
    save("trip.json", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", action="store_true")
    args = parser.parse_args()
    if os.getenv("LLM_MOCK") == "1":
        sys.exit("LLM_MOCK=1 is set: this test only makes sense with the real API.")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        sys.exit("No key found. Put your key after OPENAI_API_KEY= in the .env file.")
    started, before = time.time(), llm.stats()
    ctx = load_context()
    if args.trip:
        if ctx.build_info.get("enrichment") != "openai":
            sys.exit("The place data was built by the offline stand-in. Run scripts/build_places.py --stage enrich first.")
        step_trip(ctx)
    else:
        step_key_and_models()
        step_embeddings()
        step_enrichment()
        step_wishes(ctx)
        step_scoring(ctx)
        step_strong_model(ctx)
    finish(started, before)


if __name__ == "__main__":
    main()
