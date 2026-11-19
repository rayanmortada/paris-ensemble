"""Opening hours in OSM syntax, resolved for a real date, so seasons, weekdays and split services are right."""
from datetime import date, datetime, time, timedelta
from functools import lru_cache

from opening_hours import OpeningHours, ParserError, State

NOT_WALK_IN = ("rendez-vous", "rdv", "appointment", "réservation obligatoire")


def is_valid(spec: str | None) -> bool:
    """True if the string parses and is open at some point in a typical year."""
    if not spec:
        return False
    try:
        return any(day_intervals(spec, date(2026, month, 15) + timedelta(days=offset))
                   for month in range(1, 13) for offset in range(7))
    except (ParserError, ValueError):
        return False


@lru_cache(maxsize=200_000)
def day_intervals(spec: str, day: date) -> tuple[tuple[int, int], ...]:
    """Open intervals on one date, in minutes after midnight, clipped to that date and merged."""
    start = datetime.combine(day, time.min)
    end = start + timedelta(days=1)
    merged: list[list[int]] = []
    for a, b, state, comment in OpeningHours(spec).intervals(start, end):
        if state != State.OPEN or any(word in (comment or "").lower() for word in NOT_WALK_IN):
            continue
        s = max(0, int((a - start).total_seconds() // 60)) if a else 0
        e = min(1440, int((b - start).total_seconds() // 60)) if b else 1440
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        elif e > s:
            merged.append([s, e])
    return tuple((s, e) for s, e in merged)


def earliest_start(spec: str, day: date, arrive: int, duration: int) -> int | None:
    """First start at or after `arrive` that fits `duration` minutes inside one open interval, or None."""
    for s, e in day_intervals(spec, day):
        start = max(arrive, s)
        if start + duration <= e:
            return start
    return None


def fits(spec: str, day: date, start: int, end: int) -> bool:
    return any(s <= start and end <= e for s, e in day_intervals(spec, day))
