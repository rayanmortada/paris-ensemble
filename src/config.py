"""Loads config.yaml once. Use cfg().travel.walk_kmh; dicts with number keys stay dicts."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _ns(obj):
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


@lru_cache
def cfg() -> SimpleNamespace:
    return _ns(yaml.safe_load((ROOT / "config.yaml").read_text()))


def minutes(hhmm: str) -> int:
    hours, mins = hhmm.split(":")
    return int(hours) * 60 + int(mins)
