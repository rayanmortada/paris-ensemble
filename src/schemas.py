"""Shared types. Internal models first, then the models the LLM must fill (no dict fields there)."""
from datetime import date
from typing import Literal

from pydantic import BaseModel

Kind = Literal["attraction", "restaurant", "hotel"]
Pace = Literal["relaxed", "normal", "packed"]


# ---------- internal models ----------

class Place(BaseModel):
    id: str                              # "osm:node/123456"
    name: str                            # display name, English where one exists
    name_local: str | None = None        # OSM name, usually French
    name_en: str | None = None           # used for name matching
    kind: Kind
    category: str = ""                   # museum, park, worship, restaurant, cafe, hotel, ...
    lat: float
    lon: float
    zone: int                            # set by k-means
    themes: dict[str, int]               # 10 themes, each 0..5
    duration_min: int                    # visit length (meals use config values)
    price_eur: float                     # ticket, meal per person, or room per night
    price_source: str = "estimate"       # web, free, estimate, stars (hotels), tier (restaurants)
    price_tier: int | None = None        # restaurants, 1..3
    stars: int | None = None             # hotels
    opening_hours: str                   # OSM syntax, resolved per date by src/data/hours.py
    hours_source: str = "curated"        # osm, curated (checked on the web), open_air
    wheelchair: Literal["yes", "limited", "no", "unknown"] = "unknown"
    veg_friendly: bool | None = None     # True only if OSM says so; a vegetarian group eats only at those
    indoor: bool | None = None
    wikidata: str | None = None
    sitelinks: int = 0
    cuisine: str | None = None
    description: str = ""
    note: str = ""                       # shown to travelers
    website: str | None = None
    address: str | None = None
    image_url: str | None = None
    image_credit: str | None = None
    image_page: str | None = None


class MemberInput(BaseModel):            # what the form collects
    member_id: str
    name: str
    wishes: str
    budget_eur: float | None = None      # per person for the whole trip
    vegetarian: bool = False
    mobility: bool = False
    pace: Pace | None = None


class Profile(BaseModel):                # built by the traveler agent
    member_id: str
    name: str
    wishes: str
    theme_weights: dict[str, float]      # sums to 1
    budget_eur: float
    vegetarian: bool = False
    mobility: bool = False
    must_see: list[str] = []             # place ids after name matching
    avoid: list[str] = []
    pace: Pace = "normal"
    notes: list[str] = []                # added by feedback, e.g. "removed 3 museums"
    unmatched: list[str] = []            # names they wrote that are not in our Paris list
    language: str = "English"            # of their wishes; replies to the group use it


class TripRequest(BaseModel):
    trip_id: str
    start_date: date
    days: int                            # 2..4
    members: list[MemberInput]


class DaySelection(BaseModel):           # optimizer output for one day
    attractions: list[str]
    lunch_id: str
    dinner_id: str
    zones: list[int]


class Selection(BaseModel):              # optimizer output
    hotel_id: str
    days: list[DaySelection]
    status: str                          # OPTIMAL or FEASIBLE
    objective: float
    relaxed: list[str] = []
    log: dict = {}


class Stop(BaseModel):
    place_id: str
    role: Literal["visit", "lunch", "dinner", "hotel"]
    arrive_min: int
    start_min: int
    end_min: int
    travel_min: int
    mode: Literal["walk", "metro", "none"]


class DayPlan(BaseModel):
    date: date
    zones: list[int]
    stops: list[Stop]


class Dropped(BaseModel):
    place_id: str
    reason: str                          # "closed at arrival", "day too long", ...


class Plan(BaseModel):
    method: str = ""                     # M1..M5
    hotel_id: str
    days: list[DayPlan]
    cost_per_person: float
    cost_breakdown: dict[str, float] = {}  # hotel, tickets, meals, breakfast, metro
    satisfaction: dict[str, float] = {}  # member_id -> 0..1
    dropped: list[Dropped] = []
    log: dict = {}                       # rounds, calls, seconds, transcript, violations


class Violation(BaseModel):
    code: str                            # V1..V9
    day: int | None
    place_id: str | None
    detail: str


class Feedback(BaseModel):
    member_id: str
    place_id: str
    action: Literal["remove", "add", "up", "down"]


# ---------- LLM output models (strict mode: no dict fields) ----------

class ThemeScores(BaseModel):
    art: int
    history: int
    architecture: int
    food: int
    nature: int
    views: int
    shopping: int
    nightlife: int
    family: int
    local_life: int


class ProfileOut(BaseModel):             # P1
    theme_importance: ThemeScores        # 0..5, normalised to weights in code
    budget_eur: float | None
    vegetarian: bool
    mobility: bool
    must_see_names: list[str]
    avoid_names: list[str]
    pace: Pace
    language: str                        # of the wishes, named in English


class PlaceScore(BaseModel):
    id: str
    score: int                           # 0..5, clipped in code


class ScoreBatch(BaseModel):             # P2
    scores: list[PlaceScore]


class MemberReply(BaseModel):            # P3
    accept: bool
    add: list[str]                       # at most 3, cut in code
    remove: list[str]
    give_up: str
    reason: str


class MemberWeight(BaseModel):
    member_id: str
    weight: float


class MediatorDecision(BaseModel):       # P4
    include: list[str]
    exclude: list[str]
    weights: list[MemberWeight]          # renormalised to sum 1 in code
    note: str


class Explanation(BaseModel):
    member_id: str
    text: str


class Explanations(BaseModel):           # P5
    items: list[Explanation]


class PlannedDay(BaseModel):
    place_ids: list[str]                 # attractions in visiting order
    lunch_after: int                     # how many attractions come before lunch
    lunch_id: str
    dinner_id: str


class PlannerOut(BaseModel):             # P6
    hotel_id: str
    days: list[PlannedDay]
    reasoning: str


class PersonaText(BaseModel):            # P7
    wishes: str


class EnrichedPlace(BaseModel):
    id: str
    themes: ThemeScores
    price_tier: int | None               # restaurants, 1..3
    indoor: bool | None


class EnrichBatch(BaseModel):            # P8
    places: list[EnrichedPlace]
