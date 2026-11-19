"""Distances and travel times."""
import numpy as np

from src.config import cfg


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def leg(a, b, mobility: bool = False) -> tuple[int, str, float]:
    """Minutes, mode and cost per person to go from place a to place b."""
    t = cfg().travel
    km = float(haversine_km(a.lat, a.lon, b.lat, b.lon)) * t.detour_factor
    if km <= (t.walk_max_km_mobility if mobility else t.walk_max_km):
        return round(km / t.walk_kmh * 60), "walk", 0.0
    return round(t.metro_overhead_min + km / t.metro_kmh * 60), "metro", t.ticket_eur
