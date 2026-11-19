"""Every OpenAI call goes through this file: structured output, cache, cost log, mock mode."""
import csv
import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from src.config import ROOT

MODEL_FAST = os.getenv("MODEL_FAST") or "gpt-5.4-mini"
MODEL_STRONG = os.getenv("MODEL_STRONG") or "gpt-5.5"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "text-embedding-3-small"
EMBED_DIM = 1536

CACHE_DB = ROOT / "cache" / "llm_cache.sqlite"
CALL_LOG = ROOT / "results" / "llm_calls.csv"
MOCKS: dict[str, Callable[[str, str], BaseModel]] = {}   # purpose -> fake answer, filled by tests

_lock = threading.Lock()
_client: OpenAI | None = None
_stats = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0}


def stats() -> dict[str, int]:
    """Totals since the process started. Take the difference around a piece of work to measure it."""
    with _lock:
        return dict(_stats)


class LLMError(RuntimeError):
    pass


def _openai() -> OpenAI:
    global _client
    if _client is None:
        # The SDK itself retries rate limits, timeouts and server errors with backoff.
        _client = OpenAI(timeout=60, max_retries=2)
    return _client


def _db() -> sqlite3.Connection:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_DB)
    con.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS embeds (key TEXT PRIMARY KEY, vec BLOB)")
    return con


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _log(purpose: str, model: str, usage, cached: bool, seconds: float) -> None:
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _stats["cached" if cached else "calls"] += 1
        _stats["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        _stats["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        new = not CALL_LOG.exists()
        with open(CALL_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["ts", "purpose", "model", "input_tokens", "output_tokens", "cached", "seconds"])
            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), purpose, model,
                             getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0),
                             int(cached), round(seconds, 2)])


def call(schema: type[BaseModel], system: str, user: str, *, purpose: str,
         model: str | None = None, effort: str = "low", version: str = "v1") -> BaseModel:
    """One structured call with cache lookup and cost log. Raises LLMError when no usable answer comes back."""
    model = model or MODEL_FAST
    if os.getenv("LLM_MOCK") == "1":
        if purpose not in MOCKS:
            from src.dev_mock import install        # the offline stand-in, for tests and local trials only
            install()
        with _lock:
            _stats["calls"] += 1
        return MOCKS[purpose](system, user)
    use_cache = os.getenv("LLM_CACHE", "on") != "off"
    key = _key(model, effort, version, schema.__name__,
               json.dumps(schema.model_json_schema(), sort_keys=True), system, user)
    if use_cache:
        with _lock:
            con = _db()
            row = con.execute("SELECT value FROM calls WHERE key = ?", (key,)).fetchone()
            con.close()
        if row:
            _log(purpose, model, None, True, 0.0)
            return schema.model_validate_json(row[0])
    out, last_error = None, None
    for _ in range(2):                       # one more try on a refusal or a broken answer
        started = time.time()
        try:
            resp = _openai().responses.parse(
                model=model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                text_format=schema,
                reasoning={"effort": effort},
                timeout=60 if effort == "low" else 300,     # a whole plan from the strong model can take minutes
            )
        except Exception as e:               # the SDK has already retried transient errors
            last_error = e
            continue
        _log(purpose, model, resp.usage, False, time.time() - started)
        out = resp.output_parsed
        if out is not None:
            break
    if out is None:
        raise LLMError(f"{purpose}: no usable answer ({last_error!r})")
    if use_cache:
        with _lock:
            con = _db()
            con.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (key, out.model_dump_json()))
            con.commit()
            con.close()
    return out


def call_many(jobs: list[dict], max_workers: int = 8) -> list[BaseModel | LLMError]:
    """Runs call(**job) in parallel threads, keeps the order, and returns LLMError for failed jobs."""
    def safe(job):
        try:
            return call(**job)
        except Exception as e:
            return LLMError(str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(safe, jobs))


def embed(texts: list[str], model: str | None = None) -> np.ndarray:
    """Embeddings as float32 rows, cached per text."""
    model = model or EMBED_MODEL
    if os.getenv("LLM_MOCK") == "1":                # stable fake vectors, one per text
        return np.stack([np.random.default_rng(int(_key(t)[:8], 16)).normal(size=EMBED_DIM)
                         for t in texts]).astype("float32")
    keys = [_key(model, t) for t in texts]
    found: dict[str, np.ndarray] = {}
    with _lock:
        con = _db()
        for start in range(0, len(keys), 500):
            part = keys[start:start + 500]
            marks = ",".join("?" * len(part))
            for k, v in con.execute(f"SELECT key, vec FROM embeds WHERE key IN ({marks})", part):
                found[k] = np.frombuffer(v, dtype="float32")
        con.close()
    missing = [i for i, k in enumerate(keys) if k not in found]
    for start in range(0, len(missing), 100):
        batch = missing[start:start + 100]
        resp = _openai().embeddings.create(model=model, input=[texts[i] for i in batch])
        with _lock:
            con = _db()
            for i, item in zip(batch, resp.data):
                vec = np.array(item.embedding, dtype="float32")
                found[keys[i]] = vec
                con.execute("INSERT OR REPLACE INTO embeds VALUES (?, ?)", (keys[i], vec.tobytes()))
            con.commit()
            con.close()
    return np.stack([found[k] for k in keys])
