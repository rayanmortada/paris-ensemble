"""Builds the place file in two stages.

  python scripts/build_places.py --stage base     free, no key: OSM, Wikidata, Wikipedia, Commons, zones, sampling
  python scripts/build_places.py --stage enrich   OpenAI: theme scores, restaurant price tiers, embeddings

The base stage needs data/raw from `python -m src.data.fetch_osm`. Its result, data/places_base.json, is committed,
so the enrich stage can run anywhere. Every download is cached under data/raw.
"""
import argparse
import json
import random
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from src.config import cfg  # noqa: E402
from src.data.hours import day_intervals, is_valid  # noqa: E402
from src.data.wiki import fetch_entities, fetch_extracts, fetch_image_info  # noqa: E402
from src.llm import EMBED_MODEL, MODEL_FAST, LLMError, call_many, embed  # noqa: E402
from src.prompts import load  # noqa: E402
from src.schemas import EnrichBatch, Place  # noqa: E402

RAW = ROOT / "data" / "raw"
OPEN_AIR = {"neighbourhood", "square", "bridge", "canal", "street"}
NOT_A_MEAL = {"coffee_shop", "bubble_tea", "ice_cream", "tea", "juice", "donut", "cake", "dessert", "pastry", "frozen_yogurt", "chocolate"}
KEEP_TAGS = ("tourism", "historic", "amenity", "leisure", "cuisine", "stars", "fee", "outdoor_seating")


def category(tags: dict) -> str:
    for key, value, cat in (("tourism", "museum", "museum"), ("tourism", "gallery", "gallery"), ("tourism", "zoo", "zoo"),
                            ("tourism", "aquarium", "zoo"), ("amenity", "place_of_worship", "worship"),
                            ("leisure", "park", "park"), ("leisure", "garden", "park"), ("landuse", "cemetery", "cemetery"),
                            ("place", "square", "square"), ("place", "quarter", "neighbourhood"),
                            ("place", "neighbourhood", "neighbourhood"), ("place", "suburb", "neighbourhood"),
                            ("shop", "department_store", "shopping"), ("shop", "mall", "shopping"),
                            ("amenity", "marketplace", "market"), ("man_made", "bridge", "bridge"), ("waterway", "canal", "canal"),
                            ("historic", "castle", "palace"), ("historic", "palace", "palace"),
                            ("historic", "monument", "monument"), ("historic", "memorial", "monument"),
                            ("tourism", "viewpoint", "viewpoint"), ("tourism", "attraction", "attraction")):
        if tags.get(key) == value:
            return cat
    return "attraction"


def address(tags: dict) -> str | None:
    street = " ".join(x for x in (tags.get("addr:housenumber"), tags.get("addr:street")) if x)
    return ", ".join(x for x in (street, tags.get("addr:postcode")) if x) or None


def wheelchair(tags: dict) -> str:
    value = tags.get("wheelchair", "unknown")
    return {"designated": "yes", "partial": "limited"}.get(value, value if value in ("yes", "limited", "no") else "unknown")


def confirmed_vegetarian(tags: dict) -> bool:
    """Only what OSM says. An LLM cannot know a menu from a name: it called a sushi bar friendly and two Indian places not."""
    return (tags.get("diet:vegetarian") in ("yes", "only", "limited") or tags.get("diet:vegan") in ("yes", "only", "limited")
            or bool({"vegetarian", "vegan"} & {x.strip() for x in tags.get("cuisine", "").split(";")}))


def usable_hours(tag: str | None) -> str | None:
    """Sunset rules are dropped: the library has no location, so it assumes a fixed sunset all year."""
    if not tag or "sunset" in tag or "sunrise" in tag:
        return None
    tag = tag.replace("–", "-")
    for candidate in (tag, tag.replace(" : ", " ")):      # a common typo that is otherwise a clean rule
        if is_valid(candidate):
            return candidate
    return None


def arrondissement(tags: dict) -> str | None:
    m = re.fullmatch(r"750(\d\d)", tags.get("addr:postcode", ""))
    if not m:
        return None
    n = int(m.group(1))
    return f"{n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'} arrondissement"


def clean_text(text: str) -> str:
    """Wikipedia intros carry pronunciation guides that read badly in a card."""
    text = re.sub(r"\s*\[[^\]]*\]", "", text)
    text = re.sub(r"\s*\((?:French|UK|US|English|pronounced|lit\.|also)[^()]*(?:\([^()]*\)[^()]*)*\)", "", text)
    text = re.sub(r"\s*\((?=[^()]*(?:French:|ə))[^()]*\)", "", text)          # asides holding a pronunciation or a French name
    text = re.sub(r"\s*\(\s*[,;:][^()]*\)", "", text)                          # what is left of one
    text = re.sub(r"\(\s*;?\s*\)", "", text)
    text = re.sub(r"^(The [^,.]{3,40}), (is|are) ", r"\1 \2 ", text)
    return re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()


def cached(name: str, fetch, keys: list[str]) -> dict:
    """Fetch only what the cache lacks."""
    path = RAW / name
    data = json.loads(path.read_text()) if path.exists() else {}
    missing = [k for k in keys if k not in data]
    if missing:
        fresh = fetch(missing)
        data.update(fresh)
        for k in missing:
            data.setdefault(k, None)                # remember misses so reruns do not ask again
        path.write_text(json.dumps(data, ensure_ascii=False))
    return data


def stage_base() -> None:
    curated = yaml.safe_load((ROOT / "data" / "curated_attractions.yaml").read_text())
    rows = [r for name in ("osm_attractions.json", "osm_extras.json") for r in json.loads((RAW / name).read_text())]
    by_qid, by_osm = {}, {r["osm_id"]: r for r in rows}
    for r in rows:
        q = r["tags"].get("wikidata", "").split(";")[0].strip()
        if q:
            if q in by_qid:
                by_qid[q] = {**by_qid[q], "tags": {**r["tags"], **by_qid[q]["tags"]}}
            else:
                by_qid[q] = r
    missing = [k for k in curated if k not in by_qid and k not in by_osm]
    if missing:
        sys.exit(f"Curated places not found in the OSM data: {missing}")

    qids = [k for k in curated if k.startswith("Q")]
    ent = cached("wikidata.json", fetch_entities, qids)
    titles = [ent[q]["enwiki"] for q in qids if ent.get(q) and ent[q].get("enwiki")]
    extracts = cached("extracts.json", fetch_extracts, titles)
    files = [ent[q]["image_file"] for q in qids if ent.get(q) and ent[q].get("image_file")]
    images = cached("images.json", fetch_image_info, files)

    attractions = []
    for key, c in curated.items():
        raw = by_qid.get(key) or by_osm[key]
        tags, e = raw["tags"], ent.get(key) or {}
        cat = c.get("cat") or category(tags)
        img = images.get(e.get("image_file") or "") or {}
        if c.get("hours"):
            hours, source = c["hours"], "curated"
        elif c.get("open_air") or cat in OPEN_AIR:
            hours, source = "24/7", "open_air"
        else:
            hours, source = usable_hours(tags.get("opening_hours")), "osm"
        if not hours or not is_valid(hours):
            sys.exit(f"No usable hours for {c['name']} ({key}). Add verified hours to the curated list, or remove the place.")
        attractions.append({
            "id": key if key.startswith("osm:") else f"wd:{key}", "name": c["name"], "name_local": tags.get("name"),
            "name_en": tags.get("name:en") or e.get("label_en"), "kind": "attraction", "category": cat,
            "lat": raw["lat"], "lon": raw["lon"], "duration_min": c["duration"], "price_eur": float(c["price"]),
            "price_source": c["source"], "opening_hours": hours, "hours_source": source,
            "wheelchair": c.get("wheelchair") or wheelchair(tags), "wikidata": key if key.startswith("Q") else None,
            "sitelinks": e.get("sitelinks", 0), "note": c.get("note", ""),
            "description": c.get("text") or clean_text(extracts.get(e.get("enwiki") or "") or "") or e.get("short") or c.get("note") or c["name"],
            "website": tags.get("website") or tags.get("contact:website") or e.get("website"),
            "address": address(tags), "image_url": (img.get("url") or "").split("?")[0] or None, "image_credit": img.get("credit"),
            "image_page": img.get("page"), "osm_tags": {k: tags[k] for k in KEEP_TAGS if k in tags},
        })
    print(f"attractions: {len(attractions)}, with real hours {sum(a['hours_source'] == 'osm' for a in attractions)}, "
          f"curated {sum(a['hours_source'] == 'curated' for a in attractions)}, "
          f"open air {sum(a['hours_source'] == 'open_air' for a in attractions)}, "
          f"with photo {sum(bool(a['image_url']) for a in attractions)}, "
          f"with text {sum(a['description'] != a['name'] for a in attractions)}")

    # zones from attraction coordinates
    C = cfg()

    def to_km(lat, lon):
        return np.c_[np.asarray(lon, float) * 111.32 * np.cos(np.radians(48.86)), np.asarray(lat, float) * 110.57]

    km = KMeans(n_clusters=C.places.n_zones, n_init=20, random_state=7).fit(
        to_km([a["lat"] for a in attractions], [a["lon"] for a in attractions]))
    centres = km.cluster_centers_
    dist = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    neighbours = {k: [j for j in range(C.places.n_zones) if j != k and dist[k, j] < C.places.neighbour_km]
                  for k in range(C.places.n_zones)}
    for k in neighbours:                                # no zone is left isolated: its nearest zone is a neighbour
        nearest = min((j for j in neighbours if j != k), key=lambda j: dist[k, j])
        if nearest not in neighbours[k]:
            neighbours[k].append(nearest)
            neighbours[nearest].append(k)
    neighbours = {k: sorted(set(v)) for k, v in neighbours.items()}
    for a, z in zip(attractions, km.labels_):
        a["zone"] = int(z)

    rng = random.Random(C.experiment.seed)
    test_days = [date(2026, 12, 7) + timedelta(days=i) for i in range(7)]

    def serves(hours: str, start: int, end: int) -> bool:
        return any(any(s <= start and end <= e for s, e in day_intervals(hours, d)) for d in test_days)

    # restaurants: real opening hours only, so every lunch and dinner in a plan is checked against real data
    south, west, north, east = C.bbox
    pool = []
    seen = set()
    for r in json.loads((RAW / "osm_restaurants.json").read_text()):
        tags = r["tags"]
        cuisines = {x.strip() for x in tags.get("cuisine", "").split(";")}
        hours = usable_hours(tags.get("opening_hours"))
        key = (tags.get("name"), round(r["lat"], 3))
        if not hours or cuisines <= NOT_A_MEAL or key in seen or not (south <= r["lat"] <= north and west <= r["lon"] <= east):
            continue
        if not (serves(hours, 12 * 60, 13 * 60 + 15) or serves(hours, 19 * 60 + 30, 21 * 60)):
            continue
        seen.add(key)
        pool.append({**r, "hours": hours})
    zones = km.predict(to_km([r["lat"] for r in pool], [r["lon"] for r in pool]))
    restaurants = []
    used_names = set()                                  # a chain appears once, so a trip cannot repeat it
    for k in range(C.places.n_zones):
        here = [r for r, z in zip(pool, zones) if z == k]
        rng.shuffle(here)
        veg = [r for r in here if confirmed_vegetarian(r["tags"])][:15]    # enough choice for a vegetarian in every zone
        ids = {r["osm_id"] for r in veg}
        rest = [r for r in here if r["osm_id"] not in ids]
        chosen = []
        for r in veg + rest:
            name = r["tags"]["name"].strip().lower()
            if name not in used_names and len(chosen) < C.places.restaurants_per_zone:
                used_names.add(name)
                chosen.append(r)
        for r in chosen:
            tags = r["tags"]
            cuisine = tags.get("cuisine", "").replace(";", ", ").replace("_", " ")
            where = arrondissement(tags)
            is_cafe = tags.get("amenity") == "cafe"
            restaurants.append({
                "id": r["osm_id"], "name": tags["name"], "name_local": tags["name"], "kind": "restaurant",
                "category": "cafe" if is_cafe else "restaurant", "lat": r["lat"], "lon": r["lon"], "zone": k,
                "duration_min": C.day.lunch_min, "price_eur": 0.0, "price_source": "tier", "opening_hours": r["hours"],
                "hours_source": "osm", "wheelchair": wheelchair(tags), "cuisine": cuisine,
                "veg_friendly": True if confirmed_vegetarian(tags) else False if tags.get("diet:vegetarian") == "no" else None,
                "description": " ".join(x for x in (cuisine.capitalize(), "café" if is_cafe else "restaurant",
                                                    f"in the {where}" if where else "") if x),
                "website": tags.get("website") or tags.get("contact:website"), "address": address(tags),
                "osm_tags": {k2: tags[k2] for k2 in KEEP_TAGS if k2 in tags},
            })

    # hotels: a star rating only, because the price estimate depends on it
    pool = []
    for r in json.loads((RAW / "osm_hotels.json").read_text()):
        tags = r["tags"]
        m = re.match(r"\d", tags.get("stars", ""))
        if m and south <= r["lat"] <= north and west <= r["lon"] <= east:
            pool.append({**r, "stars": min(5, max(1, int(m.group()))) })
    zones = km.predict(to_km([r["lat"] for r in pool], [r["lon"] for r in pool]))
    hotels = []
    for k in range(C.places.n_zones):
        here = [r for r, z in zip(pool, zones) if z == k]
        rng.shuffle(here)
        by_stars = {s: [r for r in here if r["stars"] == s] for s in range(1, 6)}
        picked = []
        while len(picked) < C.places.hotels_per_zone and any(by_stars.values()):
            for s in range(1, 6):                       # a range of star levels in every zone
                if by_stars[s] and len(picked) < C.places.hotels_per_zone:
                    picked.append(by_stars[s].pop())
        for r in picked:
            tags = r["tags"]
            where = arrondissement(tags)
            hotels.append({
                "id": r["osm_id"], "name": tags["name"], "name_local": tags["name"], "kind": "hotel", "category": "hotel",
                "lat": r["lat"], "lon": r["lon"], "zone": k, "duration_min": 0,
                "price_eur": float(C.hotel_eur_by_stars[r["stars"]]), "price_source": "stars", "stars": r["stars"],
                "opening_hours": "24/7", "hours_source": "open_air", "wheelchair": wheelchair(tags),
                "description": f"{r['stars']} star hotel" + (f" in the {where}" if where else ""),
                "website": tags.get("website") or tags.get("contact:website"), "address": address(tags), "osm_tags": {},
            })

    print(f"restaurants: {len(restaurants)}, confirmed vegetarian {sum(r['veg_friendly'] is True for r in restaurants)}, "
          f"cafes {sum(r['category'] == 'cafe' for r in restaurants)}; hotels: {len(hotels)}")
    print("zone: attractions, restaurants, hotels, neighbours")
    for k in range(C.places.n_zones):
        print(f"  {k}: {sum(a['zone'] == k for a in attractions)}, {sum(r['zone'] == k for r in restaurants)}, "
              f"{sum(h['zone'] == k for h in hotels)}, {neighbours[k]}  e.g. {[a['name'] for a in attractions if a['zone'] == k][:3]}")
    (ROOT / "data" / "places_base.json").write_text(json.dumps(attractions + restaurants + hotels, ensure_ascii=False))
    lon = centres[:, 0] / (111.32 * np.cos(np.radians(48.86)))
    lat = centres[:, 1] / 110.57
    (ROOT / "data" / "zones.json").write_text(json.dumps(
        {"centres": [[float(a), float(b)] for a, b in zip(lat, lon)], "neighbours": neighbours}))
    print("saved data/places_base.json and data/zones.json")


def enrich_line(p: dict) -> str:
    text = p.get("cuisine") if p["kind"] == "restaurant" else p["description"][:300]
    tags = json.dumps(p.get("osm_tags", {}), ensure_ascii=False)
    return f"{p['id']} | {p['name']} | {p['category']} | {text} | {tags}"


def stage_enrich() -> None:
    base = json.loads((ROOT / "data" / "places_base.json").read_text())
    prompt = load("p8_enrich")
    C = cfg()
    todo = [p for p in base if p["kind"] != "hotel"]
    batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]
    started = time.time()
    results = call_many([{"schema": EnrichBatch, "system": prompt.system,
                          "user": prompt.fill_user(lines="\n".join(enrich_line(p) for p in b)),
                          "purpose": "enrich", "version": prompt.version} for b in batches], max_workers=6)
    enriched, failed = {}, 0
    for res in results:
        if isinstance(res, LLMError):
            failed += 1
            print("batch failed:", res)
            continue
        enriched.update({e.id: e for e in res.places})
    missing = [p["id"] for p in todo if p["id"] not in enriched]
    if missing:
        print(f"retrying {len(missing)} places one batch at a time")
        by_id = {p["id"]: p for p in todo}
        retry = [[by_id[i] for i in missing[k:k + 10]] for k in range(0, len(missing), 10)]
        for res in call_many([{"schema": EnrichBatch, "system": prompt.system,
                               "user": prompt.fill_user(lines="\n".join(enrich_line(p) for p in b)),
                               "purpose": "enrich", "version": prompt.version + "-retry"} for b in retry], max_workers=4):
            if not isinstance(res, LLMError):
                enriched.update({e.id: e for e in res.places})
    still = [p["id"] for p in todo if p["id"] not in enriched]
    if still:
        sys.exit(f"Enrichment failed for {len(still)} places, not writing the place file: {still[:10]}")

    places = []
    for p in base:
        p = {k: v for k, v in p.items() if k != "osm_tags"}
        if p["kind"] == "hotel":
            p["themes"] = {t: 0 for t in C.themes}
            p["indoor"] = True
        else:
            e = enriched[p["id"]]
            p["themes"] = {t: int(np.clip(getattr(e.themes, t), 0, 5)) for t in C.themes}
            p["indoor"] = e.indoor
            if p["kind"] == "restaurant":
                tier = e.price_tier if e.price_tier in (1, 2, 3) else 2
                if p["category"] == "cafe":
                    tier = min(tier, 2)
                p["price_tier"] = tier
                p["price_eur"] = float(C.meal_eur_by_tier[tier])
        places.append(Place(**p).model_dump())
    print(f"enriched {len(todo)} places in {time.time() - started:.0f} s; {failed} batches failed and were retried")

    def embed_text(p: dict) -> str:
        top = sorted(p["themes"], key=lambda t: -p["themes"][t])[:3]
        return f"{p['name']}. {p['category']}. {p['description']} Main themes: {', '.join(top)}."

    vectors = embed([embed_text(p) for p in places])
    (ROOT / "data" / "paris_places.json").write_text(json.dumps(places, ensure_ascii=False))
    np.save(ROOT / "data" / "embeddings.npy", vectors)
    import os
    mock = os.getenv("LLM_MOCK") == "1"
    (ROOT / "data" / "build_info.json").write_text(json.dumps({
        "enrichment": "mock" if mock else "openai", "enrich_model": "fake" if mock else MODEL_FAST,
        "embed_model": "fake" if mock else EMBED_MODEL, "built": time.strftime("%Y-%m-%d %H:%M"),
        "counts": {k: sum(p["kind"] == k for p in places) for k in ("attraction", "restaurant", "hotel")}}, indent=2))
    print(f"saved data/paris_places.json, data/embeddings.npy, data/build_info.json ({'FAKE LLM' if mock else 'real OpenAI'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["base", "enrich"], required=True)
    args = parser.parse_args()
    if args.stage == "base":
        stage_base()
    else:
        stage_enrich()
