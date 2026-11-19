"""Everything the app saves: SQLite under $DATA_DIR, an Excel export, and a private Hugging Face backup.

Free hosting wipes its disk on restart, so when FEEDBACK_DATASET_REPO and HF_TOKEN are set, the database is pushed to a
private Hugging Face dataset (at most every 2 minutes, and always after a rating) and restored at start. Without those
two settings it simply stays local.
"""
import io
import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT
from src.schemas import MemberInput, Plan

_lock = threading.Lock()
_restored = False
_last_backup = 0.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits  (visitor_id TEXT PRIMARY KEY, ts TEXT, nickname TEXT, consent INTEGER);
CREATE TABLE IF NOT EXISTS trips   (trip_id TEXT PRIMARY KEY, visitor_id TEXT, ts TEXT, start_date TEXT, days INTEGER,
                                    n_members INTEGER, status TEXT, error TEXT, labels TEXT);
CREATE TABLE IF NOT EXISTS members (trip_id TEXT, member_id TEXT, name TEXT, wishes TEXT, budget_eur REAL, vegetarian INTEGER,
                                    mobility INTEGER, pace TEXT, ts TEXT, PRIMARY KEY (trip_id, member_id));
CREATE TABLE IF NOT EXISTS plans   (plan_id INTEGER PRIMARY KEY AUTOINCREMENT, trip_id TEXT, label TEXT, method TEXT, version INTEGER,
                                    plan_json TEXT, seconds REAL, ts TEXT);
CREATE TABLE IF NOT EXISTS ratings (trip_id TEXT, member_name TEXT, plan_label TEXT, wishes_respected INTEGER, fair INTEGER,
                                    realistic INTEGER, clear_explanation INTEGER, ts TEXT, PRIMARY KEY (trip_id, member_name, plan_label));
CREATE TABLE IF NOT EXISTS prefs   (trip_id TEXT, member_name TEXT, preferred_label TEXT, comment TEXT, ts TEXT, PRIMARY KEY (trip_id, member_name));
CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY AUTOINCREMENT, trip_id TEXT, member_name TEXT, plan_label TEXT, action TEXT,
                                    place_id TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS invites (invite TEXT PRIMARY KEY, trip_id TEXT);
CREATE TABLE IF NOT EXISTS people  (token TEXT PRIMARY KEY, trip_id TEXT, member_id TEXT, role TEXT, ts TEXT);
"""
TABLES = ("visits", "trips", "members", "plans", "ratings", "prefs", "actions")     # invites and people hold access keys, so they are never exported


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR") or ROOT / "data_local")


def db_path() -> Path:
    return data_dir() / "study.db"


def _hf() -> tuple[str, str] | None:
    repo, token = os.getenv("FEEDBACK_DATASET_REPO", "").strip(), os.getenv("HF_TOKEN", "").strip()
    return (repo, token) if repo and token else None


def _restore_once() -> None:
    global _restored
    if _restored:
        return
    _restored = True
    if db_path().exists() or not _hf():
        return
    repo, token = _hf()
    try:
        from huggingface_hub import hf_hub_download
        data_dir().mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=repo, repo_type="dataset", filename="study.db", token=token, local_dir=str(data_dir()))
        print(f"restored study.db from {repo}")
    except Exception as e:
        print(f"no backup restored ({e!r}), starting empty")


def _connect() -> sqlite3.Connection:
    data_dir().mkdir(parents=True, exist_ok=True)
    _restore_once()
    con = sqlite3.connect(db_path(), timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _write(sql: str, params: tuple = ()) -> None:
    with _lock:
        con = _connect()
        try:
            con.execute(sql, params)
            con.commit()
        finally:
            con.close()


def _read(sql: str, params: tuple = ()) -> list[dict]:
    con = _connect()
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def backup(force: bool = False) -> None:
    """Push the database to the private dataset. Never lets a backup problem reach the visitor."""
    global _last_backup
    if not _hf() or not db_path().exists():
        return
    if not force and time.time() - _last_backup < 120:
        return
    _last_backup = time.time()
    repo, token = _hf()

    def push():
        try:
            from huggingface_hub import HfApi
            HfApi(token=token).upload_file(path_or_fileobj=str(db_path()), path_in_repo="study.db", repo_id=repo, repo_type="dataset")
        except Exception as e:
            print(f"backup failed: {e!r}")

    threading.Thread(target=push, daemon=True).start()


def save_visit(visitor_id: str, nickname: str, consent: bool) -> None:
    _write("INSERT OR REPLACE INTO visits VALUES (?,?,?,?)", (visitor_id, _now(), nickname, int(consent)))


def create_trip(trip_id: str, visitor_id: str, start_date: str, days: int, n_members: int) -> None:
    _write("INSERT INTO trips VALUES (?,?,?,?,?,?,?,?,?)", (trip_id, visitor_id, _now(), start_date, days, n_members, "draft", None, None))
    backup()


def create_invite(trip_id: str) -> str:
    """The one link for the whole group. It opens this trip only."""
    invite = secrets.token_urlsafe(16)
    _write("INSERT INTO invites VALUES (?,?)", (invite, trip_id))
    return invite


def invite_of(trip_id: str) -> str | None:
    rows = _read("SELECT invite FROM invites WHERE trip_id = ?", (trip_id,))
    return rows[0]["invite"] if rows else None


def trip_of_invite(invite: str) -> dict | None:
    rows = _read("SELECT trip_id FROM invites WHERE invite = ?", (invite,))
    return get_trip(rows[0]["trip_id"]) if rows else None


def add_person(trip_id: str, member_id: str, role: str) -> str:
    """A personal link: whoever holds it is this traveler, with no password. Only they see their answers."""
    token = secrets.token_urlsafe(16)
    _write("INSERT INTO people VALUES (?,?,?,?,?)", (token, trip_id, member_id, role, _now()))
    return token


def get_person(token: str) -> dict | None:
    rows = _read("SELECT * FROM people WHERE token = ?", (token,))
    return rows[0] if rows else None


def people_of(trip_id: str) -> list[dict]:
    return _read("SELECT * FROM people WHERE trip_id = ? ORDER BY member_id", (trip_id,))


def join_trip(trip_id: str, m: MemberInput) -> str | None:
    """Takes the next free seat and saves the answers in one step, so two friends finishing together cannot share a seat.
    Returns the new personal link, or None when the trip is full."""
    token = secrets.token_urlsafe(16)
    with _lock:
        con = _connect()
        try:
            n = con.execute("SELECT n_members FROM trips WHERE trip_id = ?", (trip_id,)).fetchone()["n_members"]
            taken = {r["member_id"] for r in con.execute("SELECT member_id FROM people WHERE trip_id = ?", (trip_id,))}
            taken |= {r["member_id"] for r in con.execute("SELECT member_id FROM members WHERE trip_id = ?", (trip_id,))}
            seat = next((f"m{k}" for k in range(1, n + 1) if f"m{k}" not in taken), None)
            if seat is None:
                return None
            con.execute("INSERT INTO people VALUES (?,?,?,?,?)", (token, trip_id, seat, "traveler", _now()))
            con.execute("INSERT OR REPLACE INTO members VALUES (?,?,?,?,?,?,?,?,?)",
                        (trip_id, seat, m.name, m.wishes, m.budget_eur, int(m.vegetarian), int(m.mobility), m.pace, _now()))
            con.commit()
        finally:
            con.close()
    backup()
    return token


def remove_traveler(trip_id: str, member_id: str) -> None:
    _write("DELETE FROM members WHERE trip_id = ? AND member_id = ?", (trip_id, member_id))
    _write("DELETE FROM people WHERE trip_id = ? AND member_id = ? AND role != 'organizer'", (trip_id, member_id))


def name_taken(trip_id: str, name: str, member_id: str | None = None) -> bool:
    """Ratings are stored by name, so two travelers on a trip cannot share one."""
    return any(m.name.strip().lower() == name.strip().lower() and m.member_id != member_id for m in get_members(trip_id))


def get_trip(trip_id: str) -> dict | None:
    rows = _read("SELECT * FROM trips WHERE trip_id = ?", (trip_id,))
    return rows[0] if rows else None


def update_trip(trip_id: str, **fields) -> None:
    keys = ", ".join(f"{k} = ?" for k in fields)
    _write(f"UPDATE trips SET {keys} WHERE trip_id = ?", (*fields.values(), trip_id))


def trips_today() -> int:
    return _read("SELECT COUNT(*) AS n FROM trips WHERE ts >= ? AND status != 'draft'", (_now()[:10],))[0]["n"]


def trips_by_visitor(visitor_id: str) -> int:
    return _read("SELECT COUNT(*) AS n FROM trips WHERE visitor_id = ? AND status != 'draft'", (visitor_id,))[0]["n"]


def save_member(trip_id: str, m: MemberInput) -> None:
    _write("INSERT OR REPLACE INTO members VALUES (?,?,?,?,?,?,?,?,?)",
           (trip_id, m.member_id, m.name, m.wishes, m.budget_eur, int(m.vegetarian), int(m.mobility), m.pace, _now()))
    backup()


def get_members(trip_id: str) -> list[MemberInput]:
    return [MemberInput(member_id=r["member_id"], name=r["name"], wishes=r["wishes"], budget_eur=r["budget_eur"],
                        vegetarian=bool(r["vegetarian"]), mobility=bool(r["mobility"]), pace=r["pace"])
            for r in _read("SELECT * FROM members WHERE trip_id = ? ORDER BY member_id", (trip_id,))]


def save_plan(trip_id: str, label: str, method: str, version: int, plan: Plan, seconds: float) -> None:
    _write("INSERT INTO plans (trip_id, label, method, version, plan_json, seconds, ts) VALUES (?,?,?,?,?,?,?)",
           (trip_id, label, method, version, plan.model_dump_json(), seconds, _now()))
    backup()


def latest_plan(trip_id: str, label: str) -> dict | None:
    rows = _read("SELECT * FROM plans WHERE trip_id = ? AND label = ? ORDER BY version DESC, plan_id DESC LIMIT 1", (trip_id, label))
    if not rows:
        return None
    rows[0]["plan"] = Plan.model_validate_json(rows[0]["plan_json"])
    return rows[0]


def first_plan(trip_id: str, label: str) -> dict | None:
    rows = _read("SELECT * FROM plans WHERE trip_id = ? AND label = ? ORDER BY version ASC, plan_id ASC LIMIT 1", (trip_id, label))
    if not rows:
        return None
    rows[0]["plan"] = Plan.model_validate_json(rows[0]["plan_json"])
    return rows[0]


def save_rating(trip_id: str, member_name: str, label: str, wishes_respected: int, fair: int, realistic: int, clear: int) -> None:
    _write("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?,?,?,?)", (trip_id, member_name, label, wishes_respected, fair, realistic, clear, _now()))


def save_pref(trip_id: str, member_name: str, preferred: str, comment: str) -> None:
    _write("INSERT OR REPLACE INTO prefs VALUES (?,?,?,?,?)", (trip_id, member_name, preferred, comment, _now()))
    backup(force=True)


def get_prefs(trip_id: str) -> list[dict]:
    return _read("SELECT * FROM prefs WHERE trip_id = ?", (trip_id,))


def get_ratings(trip_id: str) -> list[dict]:
    return _read("SELECT * FROM ratings WHERE trip_id = ?", (trip_id,))


def save_action(trip_id: str, member_name: str, label: str, action: str, place_id: str) -> None:
    _write("INSERT INTO actions (trip_id, member_name, plan_label, action, place_id, ts) VALUES (?,?,?,?,?,?)",
           (trip_id, member_name, label, action, place_id, _now()))


def counts() -> dict[str, int]:
    return {t: _read(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] for t in TABLES}


def read_table(name: str) -> list[dict]:
    if name not in TABLES:
        raise ValueError(name)
    return _read(f"SELECT * FROM {name}")


def export_excel() -> bytes:
    import pandas as pd
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for t in TABLES:
            frame = pd.DataFrame(read_table(t))
            if t == "plans" and not frame.empty:
                frame["plan_json"] = frame["plan_json"].str.slice(0, 30000)    # Excel limits a cell to 32,767 characters
            frame.to_excel(writer, sheet_name=t, index=False)
    backup(force=True)
    return buffer.getvalue()


def as_json(rows: list[dict]) -> str:
    return json.dumps(rows, ensure_ascii=False, default=str)
