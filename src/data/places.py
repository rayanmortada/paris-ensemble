"""The place file in memory: Place objects plus the numpy arrays every module uses."""
import json
import unicodedata

import numpy as np
from rapidfuzz import fuzz, utils

from src.config import ROOT, cfg
from src.data.geo import haversine_km
from src.llm import embed
from src.schemas import Place


ARTICLES = frozenset("the a an and of in at on to le la les de du des et au aux un une paris".split())
# Words many places share: they cannot identify a place on their own.
GENERIC = frozenset("""museum musee gallery galerie garden gardens jardin jardins park parc church eglise cathedral cathedrale basilica
    basilique palace palais castle chateau square place avenue street rue boulevard bridge pont market marche tower tour monument memorial
    cemetery cimetiere fountain fontaine gate porte house maison hotel centre center national grand grande petit petite saint sainte
    notre royal art arts history histoire modern moderne contemporary city ville""".split())


def fold(text: str) -> str:
    """Lowercase, no accents, no punctuation: 'Sacré-Cœur' becomes 'sacre coeur'."""
    text = text.replace("œ", "oe").replace("Œ", "Oe").replace("æ", "ae").replace("Æ", "Ae")
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return utils.default_process(plain)


def name_words(text: str) -> list[str]:
    return [w for w in fold(text).split() if len(w) > 1 and w not in ARTICLES]


def same_word(a: str, b: str) -> bool:
    """Spelling slips are fine, except in short words, where one letter makes another word."""
    return a == b or (min(len(a), len(b)) > 4 and fuzz.ratio(a, b) >= cfg().places.name_min_word)


class Places:
    def __init__(self, items: list[Place], embeddings: np.ndarray):
        self.items = items
        self.ids = [p.id for p in items]
        self.index = {pid: i for i, pid in enumerate(self.ids)}
        self.by_id = dict(zip(self.ids, items))
        self.kind = np.array([p.kind for p in items])
        self.price = np.array([p.price_eur for p in items], float)
        self.themes = np.array([[p.themes[t] for t in cfg().themes] for p in items], float)
        self.indoor = np.array([0.5 if p.indoor is None else float(p.indoor) for p in items])
        self.lat = np.array([p.lat for p in items])
        self.lon = np.array([p.lon for p in items])
        self.dist_centre = haversine_km(self.lat, self.lon, *cfg().centre)
        self.embeddings = embeddings
        self.unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._names = []                          # (folded name, its words, row) for every known name of every attraction
        for i in self.idx("attraction"):
            for name in {items[i].name, items[i].name_en, items[i].name_local} - {None}:
                self._names.append((fold(name), name_words(name), int(i)))

    @classmethod
    def load(cls, folder=ROOT / "data") -> "Places":
        items = [Place(**p) for p in json.loads((folder / "paris_places.json").read_text())]
        return cls(items, np.load(folder / "embeddings.npy"))

    def idx(self, kind: str) -> np.ndarray:
        return np.flatnonzero(self.kind == kind)

    def lines(self, ids: list[str]) -> str:
        """Compact prompt lines: id | name | kind | price | strong themes | description."""
        out = []
        for pid in ids:
            p = self.by_id[pid]
            strong = ", ".join(f"{t} {p.themes[t]}" for t in cfg().themes if p.themes[t] >= 3)
            label = p.cuisine if p.kind == "restaurant" and p.cuisine else p.category
            out.append(f"{p.id} | {p.name} | {label} | {p.price_eur:.0f} EUR | {strong} | {p.description[:220]}")
        return "\n".join(out)

    def name_hit(self, written: str) -> int | None:
        """The attraction a written name means, or None. It compares words, not whole strings: an overall similarity
        score calls 'the Catacombs' and 'Basilica of the Sacred Heart' a match, because they share the word 'the'."""
        C = cfg().places
        text, words = fold(written), name_words(written)
        key = [w for w in words if w not in GENERIC]
        best, best_score = None, None
        for name, known, row in self._names:
            distinctive = [w for w in known if w not in GENERIC]
            if key:
                fit = all(any(same_word(w, k) for k in known) for w in key)
            else:                                 # only shared words, as in Grand Palais: they must be the whole name
                fit = len(words) > 1 and not distinctive and all(any(same_word(w, k) for k in known) for w in words)
            if not fit and fuzz.ratio(text, name) < C.name_min_whole:
                continue
            score = (min(len(key) / max(len(distinctive), 1), 1), fuzz.token_sort_ratio(text, name), self.items[row].sitelinks)
            if best_score is None or score > best_score:
                best, best_score = row, score
        return best

    def match_names(self, names: list[str]) -> list[str]:
        """Place ids for names a traveler wrote: by words first, then by embeddings for descriptions like
        'Napoleon's tomb'. Unmatched names are left out; callers tell the user."""
        found = []
        for name in names:
            row = self.name_hit(name)
            if row is not None:
                found.append(self.ids[row])
                continue
            q = embed([name])[0]
            att = self.idx("attraction")
            cos = self.unit[att] @ (q / np.linalg.norm(q))
            if cos.max() >= cfg().places.name_min_cos:
                found.append(self.ids[int(att[cos.argmax()])])
        return list(dict.fromkeys(found))
