"""Planning runs in background threads, so a phone that locks or a page that reloads never loses the work.
Results go to the database; the page only polls."""
import random
import threading
import time
import traceback
from collections import OrderedDict
from datetime import date

from src import store
from src.config import cfg
from src.llm import LLMError
from src.pipeline import Context, Prepared, apply_feedback, clone_prepared, prepare, run_method
from src.schemas import Feedback, TripRequest


def friendly_error(e: Exception) -> str:
    text = repr(e)
    if "api_key" in text or "AuthenticationError" in text or "Incorrect API key" in text:
        return "The OpenAI key is missing or wrong, so nothing could be planned. Check OPENAI_API_KEY."
    if "insufficient_quota" in text or "RateLimitError" in text:
        return "OpenAI refused the request because of a spending or rate limit. Wait a little, or check the limits on the OpenAI account."
    if "model" in text and ("does not exist" in text or "not found" in text):
        return "The OpenAI model name in the settings is not available on this key. Check MODEL_FAST and MODEL_STRONG."
    if isinstance(e, LLMError):
        return "The AI service did not give a usable answer. Please try again."
    if "no feasible plan" in text:
        return "No workable plan exists for these dates and budgets, even after relaxing the rules. Try a higher budget or fewer must see places."
    return "Something went wrong while planning. Please try again."


class Jobs:
    """One instance for the whole server, shared by all sessions."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.lock = threading.Lock()
        self.status: dict[str, dict] = {}
        self.prepared: OrderedDict[str, Prepared] = OrderedDict()     # latest state of each trip, after feedback
        self.base: OrderedDict[str, Prepared] = OrderedDict()         # as first prepared, for comparing methods
        self.slots = threading.Semaphore(3)                           # free hosting has little CPU

    def _start(self, key: str, fn, *args) -> None:
        with self.lock:
            if self.status.get(key, {}).get("state") == "running":
                return
            self.status[key] = {"state": "running", "steps": [], "error": None, "started": time.time()}
        threading.Thread(target=self._guard, args=(key, fn, args), daemon=True).start()

    def _guard(self, key: str, fn, args) -> None:
        try:
            with self.slots:
                fn(key, *args)
            self.status[key]["state"] = "done"
        except Exception as e:
            traceback.print_exc()
            self.status[key].update(state="failed", error=friendly_error(e))

    def step(self, key: str, text: str) -> None:
        self.status[key]["steps"].append(text)

    def get(self, key: str) -> dict:
        return self.status.get(key, {"state": "idle", "steps": [], "error": None})

    def _trip(self, trip_id: str) -> TripRequest:
        row = store.get_trip(trip_id)
        return TripRequest(trip_id=trip_id, start_date=date.fromisoformat(row["start_date"]), days=row["days"],
                           members=store.get_members(trip_id))

    def _remember(self, trip_id: str, prep: Prepared) -> None:
        with self.lock:
            self.prepared[trip_id] = prep
            self.base.setdefault(trip_id, clone_prepared(prep))
            for table in (self.prepared, self.base):
                table.move_to_end(trip_id)
                while len(table) > 12:
                    table.popitem(last=False)

    def get_prepared(self, trip_id: str, key: str | None = None) -> Prepared:
        """From memory, or rebuilt after a restart. The LLM cache makes a rebuild free if the cache survived."""
        with self.lock:
            if trip_id in self.prepared:
                return self.prepared[trip_id]
        prep = prepare(self._trip(trip_id), self.ctx, app=True, progress=(lambda t: self.step(key, t)) if key else (lambda _: None))
        self._remember(trip_id, prep)
        return prep

    # ---------- plan a trip: M1 and M4, shown blind as A and B ----------

    def start_plan(self, trip_id: str) -> None:
        store.update_trip(trip_id, status="running", error=None)
        self._start(f"plan:{trip_id}", self._plan, trip_id)

    def _plan(self, key: str, trip_id: str) -> None:
        try:
            prep = prepare(self._trip(trip_id), self.ctx, app=True, progress=lambda t: self.step(key, t))
            self._remember(trip_id, prep)
            base = self.base[trip_id]
            self.step(key, "Building the first plan")
            first = run_method(base, "M1")
            self.step(key, "Building the second plan")
            second = run_method(base, "M4", max_rounds=cfg().app.max_rounds)
            labels = ["A", "B"]
            random.Random(trip_id).shuffle(labels)
            store.save_plan(trip_id, labels[0], "M1", 1, first, first.log["seconds"])
            store.save_plan(trip_id, labels[1], "M4", 1, second, second.log["seconds"])
            store.update_trip(trip_id, status="ready", labels=f"{labels[0]}=M1,{labels[1]}=M4")
            self.step(key, "Done")
            store.backup(force=True)
        except Exception as e:
            store.update_trip(trip_id, status="failed", error=friendly_error(e))
            raise

    # ---------- feedback: change one plan and solve again with the same method ----------

    def start_replan(self, trip_id: str, label: str, member_name: str, actions: list[Feedback]) -> None:
        self._start(f"replan:{trip_id}", self._replan, trip_id, label, member_name, actions)

    def _replan(self, key: str, trip_id: str, label: str, member_name: str, actions: list[Feedback]) -> None:
        prep = self.get_prepared(trip_id, key)
        row = store.latest_plan(trip_id, label)
        self.step(key, "Learning from your changes")
        apply_feedback(prep, actions)
        for fb in actions:
            store.save_action(trip_id, member_name, label, fb.action, fb.place_id)
        self.step(key, "Solving the plan again")
        plan = run_method(prep, row["method"], max_rounds=cfg().app.max_rounds)
        store.save_plan(trip_id, label, row["method"], row["version"] + 1, plan, plan.log["seconds"])
        self.step(key, "Done")

    # ---------- show all five methods side by side for this group ----------

    def start_compare(self, trip_id: str) -> None:
        self._start(f"compare:{trip_id}", self._compare, trip_id)

    def _compare(self, key: str, trip_id: str) -> None:
        self.get_prepared(trip_id, key)
        base = self.base[trip_id]
        for method, label in (("M2", "x2"), ("M3", "x3"), ("M5", "x5")):
            if store.latest_plan(trip_id, label):
                continue
            self.step(key, f"Running {method}")
            plan = run_method(base, method, explain=False)
            store.save_plan(trip_id, label, method, 1, plan, plan.log["seconds"])
        self.step(key, "Done")
