"""Downloads attractions, restaurants and hotels inside Paris from the Overpass API (free, no key).
Run from the repo root: python -m src.data.fetch_osm
Finished categories are kept, so a rerun only fetches what is missing. Use --force to fetch everything again.
"""
import json
import sys
import time

import requests

from src.config import ROOT

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
HEADERS = {"User-Agent": "group-trip-planner/0.1 (university course project)"}
AREA = 'area["name"="Paris"]["admin_level"="8"]["boundary"="administrative"]->.paris;'
QUERIES = {
    "attractions": (
        'nwr["tourism"~"^(museum|attraction|gallery|viewpoint|zoo|aquarium)$"]["name"](area.paris);'
        'nwr["historic"~"^(monument|castle|memorial|palace)$"]["name"](area.paris);'
        'nwr["amenity"="place_of_worship"]["wikidata"](area.paris);'
        'nwr["leisure"~"^(park|garden)$"]["name"]["wikidata"](area.paris);'
        'nwr["landuse"="cemetery"]["name"]["wikidata"](area.paris);'
    ),
    "extras": (                                  # squares, neighbourhoods, markets, shops, bridges, canals
        'nwr["place"~"^(square|quarter|neighbourhood|suburb)$"]["name"]["wikidata"](area.paris);'
        'nwr["shop"~"^(department_store|mall)$"]["name"]["wikidata"](area.paris);'
        'nwr["amenity"="marketplace"]["name"](area.paris);'
        'nwr["man_made"="bridge"]["name"]["wikidata"](area.paris);'
        'nwr["waterway"="canal"]["name"]["wikidata"](area.paris);'
    ),
    "restaurants": 'nwr["amenity"~"^(restaurant|cafe)$"]["name"]["cuisine"](area.paris);',
    "hotels": 'nwr["tourism"="hotel"]["name"](area.paris);',
}


def fetch(kind: str) -> list[dict]:
    query = f"[out:json][timeout:300];{AREA}({QUERIES[kind]});out center tags;"
    last_error = None
    for attempt in range(3):
        for url in ENDPOINTS:
            try:
                resp = requests.post(url, data={"data": query}, headers=HEADERS, timeout=400)
                resp.raise_for_status()
                elements = resp.json()["elements"]
            except (requests.RequestException, ValueError, KeyError) as e:
                last_error = e
                print(f"  {kind}: {url} failed ({e!r}), trying the next server")
                time.sleep(10)
                continue
            if not elements:                      # a busy server can answer with an empty result
                last_error = RuntimeError("empty result")
                print(f"  {kind}: {url} returned nothing, trying the next server")
                continue
            rows = []
            for el in elements:
                centre = el.get("center", {})
                lat, lon = el.get("lat", centre.get("lat")), el.get("lon", centre.get("lon"))
                if lat is not None and lon is not None:
                    rows.append({"osm_id": f"osm:{el['type']}/{el['id']}", "lat": lat, "lon": lon,
                                 "tags": el.get("tags", {})})
            return rows
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"Overpass failed for {kind}: {last_error!r}")


if __name__ == "__main__":
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    force = "--force" in sys.argv
    for kind in QUERIES:
        path = out / f"osm_{kind}.json"
        if path.exists() and not force:
            print(f"{kind}: already downloaded")
            continue
        rows = fetch(kind)
        path.write_text(json.dumps(rows, ensure_ascii=False))
        print(f"{kind}: {len(rows)}")
        time.sleep(15)
