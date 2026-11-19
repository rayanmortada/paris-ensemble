"""Free facts about places: Wikidata (popularity, website, photo), Wikipedia (description), Commons (photo credit)."""
import html
import re
import time

import requests

HEADERS = {"User-Agent": "group-trip-planner/0.1 (university course project)"}
WIKIDATA = "https://www.wikidata.org/w/api.php"
WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
COMMONS = "https://commons.wikimedia.org/w/api.php"


def _get(url: str, params: dict) -> dict:
    for attempt in range(5):
        try:
            resp = requests.get(url, params={**params, "format": "json", "formatversion": 2},
                                headers=HEADERS, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"status {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == 4:
                raise
            print(f"  retrying {url}: {e!r}")
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_entities(qids: list[str]) -> dict[str, dict]:
    """qid -> sitelinks count, English and French labels and Wikipedia titles, description, photo file, website."""
    out = {}
    for start in range(0, len(qids), 50):
        data = _get(WIKIDATA, {"action": "wbgetentities", "ids": "|".join(qids[start:start + 50]),
                               "props": "sitelinks|claims|labels|descriptions", "languages": "en|fr"})
        for qid, e in data.get("entities", {}).items():
            if "missing" in e:
                continue
            claims = e.get("claims", {})

            def first(prop):
                try:
                    return claims[prop][0]["mainsnak"]["datavalue"]["value"]
                except (KeyError, IndexError):
                    return None

            links = e.get("sitelinks", {})
            out[qid] = {
                "sitelinks": len(links),
                "enwiki": links.get("enwiki", {}).get("title"),
                "frwiki": links.get("frwiki", {}).get("title"),
                "label_en": e.get("labels", {}).get("en", {}).get("value"),
                "label_fr": e.get("labels", {}).get("fr", {}).get("value"),
                "short": e.get("descriptions", {}).get("en", {}).get("value"),
                "image_file": first("P18"),
                "website": first("P856"),
                "instance_of": [c["mainsnak"]["datavalue"]["value"]["id"] for c in claims.get("P31", [])
                                if c["mainsnak"].get("datavalue")],
            }
        time.sleep(0.5)
    return out


def fetch_extracts(titles: list[str]) -> dict[str, str]:
    """English Wikipedia title -> its first sentences, plain text."""
    out = {}
    for start in range(0, len(titles), 20):
        data = _get(WIKIPEDIA, {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                "exlimit": 20, "redirects": 1, "titles": "|".join(titles[start:start + 20])})
        query = data.get("query", {})
        renamed = {r["to"]: r["from"] for r in query.get("redirects", [])}
        renamed.update({n["to"]: n["from"] for n in query.get("normalized", [])})
        for page in query.get("pages", []):
            text = (page.get("extract") or "").strip()
            title = page.get("title")
            if text and title:
                out[renamed.get(title, title)] = first_sentences(text)
        time.sleep(0.5)
    return out


def first_sentences(text: str, limit: int = 3, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[a-z\)\]0-9]{2}[.!?])\s+(?=[A-Z])", text)
    out = ""
    for part in parts[:limit]:
        if out and len(out) + len(part) > max_chars:
            break
        out = f"{out} {part}".strip()
    return out[:max_chars + 80]


def fetch_image_info(files: list[str], width: int = 640) -> dict[str, dict]:
    """Commons file name -> thumbnail URL, author and licence, for correct attribution."""
    out = {}
    for start in range(0, len(files), 40):
        batch = files[start:start + 40]
        data = _get(COMMONS, {"action": "query", "prop": "imageinfo", "iiprop": "url|extmetadata",
                              "iiurlwidth": width, "iiextmetadatafilter": "Artist|LicenseShortName",
                              "titles": "|".join(f"File:{f}" for f in batch)})
        query = data.get("query", {})
        renamed = {n["to"]: n["from"] for n in query.get("normalized", [])}
        for page in query.get("pages", []):
            infos = page.get("imageinfo")
            if not infos:
                continue
            meta = infos[0].get("extmetadata", {})
            artist = html.unescape(re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))).strip()
            licence = meta.get("LicenseShortName", {}).get("value", "").strip()
            title = renamed.get(page["title"], page["title"]).removeprefix("File:")
            out[title] = {"url": infos[0].get("thumburl") or infos[0].get("url"),
                          "page": infos[0].get("descriptionurl"),
                          "credit": ", ".join(x for x in (artist[:80], licence) if x)}
        time.sleep(0.5)
    return out
