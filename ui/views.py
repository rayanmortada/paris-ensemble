"""HTML pieces and the map. Everything a visitor or an LLM wrote is escaped before it reaches the page."""
import base64
import html
import re
from datetime import date
from functools import lru_cache

import folium
from branca.element import MacroElement
from jinja2 import Template

from src.config import cfg
from src.data.hours import day_intervals
from src.pipeline import METHOD_NAMES
from src.schemas import Plan

GOLD_INK = "#8C7340"
DAY_COLOURS = ["#3F51B5", "#C8912F", "#00897B", "#C2452D"]      # checked for colour blindness; the numbers carry identity too
DAY_TEXT = ["#FFFFFF", "#171717", "#FFFFFF", "#FFFFFF"]
CATEGORY_LABELS = {
    "museum": "Museum", "gallery": "Gallery", "worship": "Place of worship", "park": "Park", "cemetery": "Cemetery",
    "neighbourhood": "Neighbourhood", "square": "Square", "bridge": "Bridge", "canal": "Canal", "street": "Street",
    "shopping": "Shopping", "market": "Market", "monument": "Monument", "palace": "Palace", "zoo": "Zoo or aquarium",
    "viewpoint": "Viewpoint", "attraction": "Attraction", "cruise": "River cruise", "restaurant": "Restaurant", "cafe": "Café", "hotel": "Hotel",
}
_ICONS = {
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "link": '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "food": '<path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"/><path d="M7 2v20"/><path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"/>',
    "bed": '<path d="M2 4v16"/><path d="M2 8h18a2 2 0 0 1 2 2v10"/><path d="M2 17h20"/><path d="M6 8v9"/>',
    "train": '<path d="M8 3.1V7a4 4 0 0 0 8 0V3.1"/><path d="m9 15-1-1"/><path d="m15 15 1-1"/><path d="M9 19c-2.8 0-5-2.2-5-5v-4a8 8 0 0 1 16 0v4c0 2.8-2.2 5-5 5Z"/><path d="m8 19-2 3"/><path d="m16 19 2 3"/>',
    "route": '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/>',
    "chat": '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>',
    "walk": '<path d="M4 16v-2.38C4 11.5 2.97 10.5 3 8c.03-2.72 1.49-6 4.5-6C9.37 2 10 3.8 10 5.5c0 3.11-2 5.66-2 8.68V16a2 2 0 1 1-4 0Z"/><path d="M20 20v-2.38c0-2.12 1.03-3.12 1-5.62-.03-2.72-1.49-6-4.5-6C14.63 6 14 7.8 14 9.5c0 3.11 2 5.66 2 8.68V20a2 2 0 1 0 4 0Z"/><path d="M16 17h4"/><path d="M4 13h4"/>',
    "refresh": '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    "wallet": '<path d="M19 7V4a1 1 0 0 0-1-1H5a2 2 0 0 0 0 4h15a1 1 0 0 1 1 1v4h-3a2 2 0 0 0 0 4h3a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1"/><path d="M3 5v14a2 2 0 0 0 2 2h15a1 1 0 0 0 1-1v-4"/>',
    "coffee": '<path d="M17 8h1a4 4 0 1 1 0 8h-1"/><path d="M3 8h14v9a4 4 0 0 1-4 4H7a4 4 0 0 1-4-4Z"/><line x1="6" y1="2" x2="6" y2="4"/><line x1="10" y1="2" x2="10" y2="4"/><line x1="14" y1="2" x2="14" y2="4"/>',
    "down": '<polyline points="6 9 12 15 18 9"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
}


@lru_cache(maxsize=256)
def icon(name: str, size: int = 15, cls: str = "", color: str = "#68655D") -> str:
    """An embedded image: Streamlit's HTML sanitizer removes inline SVG but keeps images."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')
    data = base64.b64encode(svg.encode()).decode()
    return f'<img class="pe-i {cls}" src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" alt=""/>'


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def clock(m: int) -> str:
    return f"{m // 60 % 24:02d}:{m % 60:02d}"


def span(m: int) -> str:
    if m < 60:
        return f"{m} min"
    return f"{m // 60} h {m % 60:02d}" if m % 60 else f"{m // 60} h"


def money(value: float) -> str:
    return f"{value:,.0f} EUR"


def safe_url(url: str | None) -> str | None:
    if not url:
        return None
    url = url.split(";")[0].strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"https://{url}" if "." in url and " " not in url else None


def price_line(p) -> str:
    if p.kind == "hotel":
        return f"about {p.price_eur:.0f} EUR a night per room"
    if p.kind == "restaurant":
        return f"about {p.price_eur:.0f} EUR a meal"
    if p.price_eur == 0:
        return "free"
    return f"{p.price_eur:g} EUR" if p.price_source == "web" else f"about {p.price_eur:g} EUR"


def hours_line(p, day: date) -> str:
    if p.hours_source == "open_air":
        return "Open air, always accessible."
    spans = day_intervals(p.opening_hours, day)
    if not spans:
        return f"Closed on {day:%A}."
    text = ", ".join(f"{clock(s)} to {clock(e) if e < 1440 else '24:00'}" for s, e in spans)
    source = {"osm": "from OpenStreetMap", "curated": "usual hours"}.get(p.hours_source, "")
    return f"Open {text} on {day:%A} ({source})."


def place_more(p, day: date, show_hours: bool = True) -> str:
    parts = []
    if p.image_url:
        credit = f'Photo: <a href="{esc(p.image_page)}" target="_blank" rel="noopener">{esc(p.image_credit or "Wikimedia Commons")}</a>'
        parts.append(f'<img class="pe-photo" src="{esc(p.image_url)}" alt="{esc(p.name)}" loading="lazy" referrerpolicy="no-referrer"/><div class="pe-credit">{credit}</div>')
    if p.description and p.description != p.name:
        parts.append(f"<p>{esc(p.description)}</p>")
    if p.note:
        parts.append(f'<div class="pe-tip">{esc(p.note)}</div>')
    if p.kind == "restaurant":
        veg = {True: "Vegetarian options, according to OpenStreetMap.", False: "No vegetarian options, according to OpenStreetMap.",
               None: "Vegetarian options not confirmed."}[p.veg_friendly]
        parts.append(f'<p class="pe-small">{esc(veg)}</p>')
    if show_hours:
        parts.append(f'<p class="pe-small">{esc(hours_line(p, day))}</p>')
    if p.address:
        parts.append(f'<p class="pe-small">{esc(p.address)}</p>')
    links = [f'<a class="pe-link" href="https://www.google.com/maps/search/?api=1&query={p.lat:.6f},{p.lon:.6f}" target="_blank" rel="noopener">{icon("pin", 14)} Map</a>']
    site = safe_url(p.website)
    if site:
        links.append(f'<a class="pe-link" href="{esc(site)}" target="_blank" rel="noopener">{icon("link", 14)} Website</a>')
    parts.append(f'<div class="pe-links">{"".join(links)}</div>')
    return f'<div class="pe-more">{"".join(parts)}</div>'


def _travel(s, target: str) -> str:
    if s.travel_min <= 0:
        return ""
    how = "by metro" if s.mode == "metro" else "on foot"
    return f'<div class="pe-travel">{icon("train" if s.mode == "metro" else "walk", 14)} About {span(s.travel_min)} {how}{target}</div>'


def day_block(day, index: int, places, is_open: bool = False) -> str:
    colour = DAY_COLOURS[index % len(DAY_COLOURS)]
    text_colour = DAY_TEXT[index % len(DAY_TEXT)]
    rows, previous_end, number = [], None, 0
    visits = sum(s.role == "visit" for s in day.stops)
    travel = sum(s.travel_min for s in day.stops)
    stops = day.stops
    for k, s in enumerate(stops):
        p = places.by_id.get(s.place_id)
        last = k == len(stops) - 1
        rail_end = "" if last else '<span class="pe-rail-line"></span>'
        if s.role == "hotel":
            rows.append(f'<div class="pe-stop"><div class="pe-rail"><span class="pe-dot soft">{icon("bed", 13)}</span></div>'
                        f'<div class="pe-stop-body">{_travel(s, "")}<div class="pe-meta" style="padding:10px 0 4px">Back at the hotel around {clock(s.arrive_min)}</div></div></div>')
            continue
        number += 1
        free = ""
        ready = (previous_end or 0) + s.travel_min
        gap = s.start_min - ready
        if previous_end is not None and gap >= 40:
            free = f'<div class="pe-free">{icon("clock", 14, color=GOLD_INK)} Free time, about {span(gap // 5 * 5)}</div>'
        role_icon = {"lunch": icon("food", 14), "dinner": icon("food", 14)}.get(s.role, "")
        kind = {"lunch": "Lunch", "dinner": "Dinner"}.get(s.role, CATEGORY_LABELS.get(p.category, "")) if p else ""
        cuisine = f" · {esc(p.cuisine.capitalize())}" if p and p.cuisine else ""
        name = esc(p.name) if p else esc(s.place_id)
        length = span(s.end_min - s.start_min)
        price = price_line(p) if p else ""
        rows.append(
            f'<div class="pe-stop"><div class="pe-rail"><span class="pe-dot" style="background:{colour};color:{text_colour}">{number}</span>{rail_end}</div>'
            f'<div class="pe-stop-body">{_travel(s, "")}{free}'
            f'<details class="pe-place"><summary><div class="pe-place-top"><span class="pe-time">{clock(s.start_min)} · {length}</span>{icon("down", 15, "pe-chev")}</div>'
            f'<div class="pe-name">{name}</div><div class="pe-meta">{role_icon} {esc(kind)}{cuisine} · {esc(price)}</div></summary>'
            f'{place_more(p, day.date) if p else ""}</details></div></div>')
        previous_end = s.end_min
    title = f"Day {index + 1} · {day.date:%a %-d %b}"
    return (f'<details class="pe-day"{" open" if is_open else ""}><summary><span class="pe-day-title"><span class="pe-day-dot" style="background:{colour}"></span>{title}</span>'
            f'<span style="display:flex;align-items:center;gap:10px"><span class="pe-day-meta">{visits} visits · {span(travel)} of travel</span>{icon("down", 18, "pe-chev")}</span></summary>'
            f'<div class="pe-stops">{"".join(rows)}</div></details>')


def kpis(plan: Plan, me=None) -> str:
    """Only the viewer's own budget: the lowest one in the group belongs to someone else."""
    days = max(len(plan.days), 1)
    travel = sum(s.travel_min for d in plan.days for s in d.stops) / days
    visits = sum(s.role == "visit" for d in plan.days for s in d.stops)
    low = min(plan.satisfaction.values()) if plan.satisfaction else 0
    if me:
        within = plan.cost_per_person <= me.budget_eur
        foot = f"{'within' if within else 'over'} your budget of {me.budget_eur:,.0f} EUR"
    else:
        foot = "estimated"

    def tile(ic, label, value, note):
        return (f'<div class="pe-kpi"><div class="pe-kpi-label">{icon(ic, 14)} {label}</div><div class="pe-kpi-value">{value}</div>'
                f'<div class="pe-kpi-foot">{note}</div></div>')

    return ('<div class="pe-grid4" style="margin:6px 0 16px">'
            + tile("wallet", "Cost per person", f"{plan.cost_per_person:,.0f} EUR", foot)
            + tile("route", "Travel per day", span(int(travel // 5 * 5)), "walking and metro, estimated")
            + tile("pin", "Places to visit", str(visits), f"over {days} days")
            + tile("scale", "Lowest match", f"{low:.0%}", "the traveler this plan suits least")
            + "</div>")


def hotel_card(plan: Plan, places, n_members: int, day: date) -> str:
    h = places.by_id.get(plan.hotel_id)
    if not h:
        return ""
    stars = "★" * (h.stars or 0)
    rooms = (n_members + 1) // 2
    site = safe_url(h.website)
    links = [f'<a class="pe-link" href="https://www.google.com/maps/search/?api=1&query={h.lat:.6f},{h.lon:.6f}" target="_blank" rel="noopener">{icon("pin", 14)} Map</a>']
    if site:
        links.append(f'<a class="pe-link" href="{esc(site)}" target="_blank" rel="noopener">{icon("link", 14)} Website</a>')
    address = f'<div class="pe-meta">{esc(h.address)}</div>' if h.address else ""
    return (f'<div class="pe-card" style="margin:0 0 14px;display:flex;gap:14px;align-items:flex-start">'
            f'<div class="pe-feature-icon" style="margin:0;flex:none">{icon("bed", 20, color=GOLD_INK)}</div><div style="min-width:0">'
            f'<div class="pe-small">Where you stay</div><div class="pe-name">{esc(h.name)} <span style="color:var(--pe-gold)">{stars}</span></div>'
            f'{address}<div class="pe-meta">{esc(price_line(h))} (estimate from its star rating), {rooms} room{"s" if rooms > 1 else ""} for {n_members}. Check real prices before booking.</div>'
            f'<div class="pe-links">{"".join(links)}</div></div></div>')


def fit_card(plan: Plan, me=None) -> str:
    """How the plan fits the viewer, in words. The rest of the group only as anonymous numbers."""
    values = list(plan.satisfaction.values()) or [0.0]
    rows = ""
    if me:
        value = plan.satisfaction.get(me.member_id, 0)
        text = (plan.log.get("explanations") or {}).get(me.member_id, "")
        body = f"<p>{esc(text)}</p>" if text else ""
        rows = (f'<div class="pe-fit"><div class="pe-fit-top"><span>You</span><span class="pe-fit-pct">{value:.0%}</span></div>'
                f'<div class="pe-bar"><span style="width:{value * 100:.0f}%"></span></div>{body}</div>')
    group = (f'<div class="pe-fit"><div class="pe-fit-top"><span>The whole group</span><span class="pe-fit-pct">{min(values):.0%} to {max(values):.0%}</span></div>'
             f'<p>The match of the least and the most satisfied traveler. Other people\'s answers stay private.</p></div>')
    return (f'<div class="pe-card" style="margin-bottom:14px"><div class="pe-h3">How well it fits you</div>'
            f'<p class="pe-small" style="margin:0 0 10px">The model\'s estimate of how close this plan comes to the best trip you could have alone. 100% is that best trip.</p>{rows}{group}</div>')


def cost_card(plan: Plan) -> str:
    labels = {"hotel": "Hotel share", "tickets": "Tickets", "meals": "Lunches and dinners", "breakfast": "Breakfasts", "metro": "Metro"}
    rows = "".join(f'<div class="pe-cost"><span>{labels[k]}</span><b>{v:,.0f} EUR</b></div>' for k, v in plan.cost_breakdown.items())
    return (f'<div class="pe-card" style="margin-bottom:14px"><div class="pe-h3">Cost per person</div>{rows}'
            f'<div class="pe-cost"><span><b>Total</b></span><b>{plan.cost_per_person:,.0f} EUR</b></div>'
            f'<p class="pe-small" style="margin-top:8px">Estimates: hotels from their star rating, meals from a price level, tickets at full adult price, sharing two to a room. Getting to Paris is not included.</p></div>')


def notes_block(plan: Plan, places) -> str:
    notes = list(plan.log.get("notes") or [])
    relaxed = [r for r in plan.log.get("relaxed") or []]
    for r in relaxed:
        if r == "budget":
            notes.append("No plan fits the lowest budget in the group. This is close to the cheapest plan we can make with these places, and it goes over that budget.")
        elif r == "coverage":
            notes.append("This plan covers fewer parts of the city than we aim for, to stay within the other limits.")
        elif r == "min attractions":
            notes.append("Some days have only one visit, to stay within the other limits.")
        elif r.startswith("dropped request "):
            pid = r.removeprefix("dropped request ")
            notes.append(f"{places.by_id[pid].name if pid in places.by_id else pid} was requested but could not fit.")
    for d in plan.dropped:
        name = places.by_id[d.place_id].name if d.place_id in places.by_id else d.place_id
        notes.append(f"{name} was left out: {d.reason}.")
    return "".join(f'<div class="pe-note">{esc(n)}</div>' for n in notes)


class _NoLeafletPrefix(MacroElement):
    """Keeps the required OpenStreetMap credit, drops the small flag and library link before it, and quiets the
    map's colours so the numbered routes stand out."""
    _template = Template("""{% macro script(this, kwargs) %}
        {{ this._parent.get_name() }}.attributionControl.setPrefix(false);
        document.head.insertAdjacentHTML("beforeend", "<style>.leaflet-tile-pane{filter:grayscale(.75) brightness(1.04) contrast(.9)}</style>");
    {% endmacro %}""")


def plan_map(plan: Plan, places) -> folium.Map:
    centre = cfg().centre
    m = folium.Map(location=centre, zoom_start=12, tiles="OpenStreetMap", scrollWheelZoom=False, control_scale=False)
    m.add_child(_NoLeafletPrefix())
    points = []
    hotel = places.by_id.get(plan.hotel_id)
    if hotel:
        points.append([hotel.lat, hotel.lon])
        folium.Marker([hotel.lat, hotel.lon], tooltip=f"Hotel: {hotel.name}", icon=folium.DivIcon(
            icon_size=(30, 30), icon_anchor=(15, 15),
            html='<div style="width:30px;height:30px;border-radius:50%;background:#171717;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.35);'
                 'display:flex;align-items:center;justify-content:center;color:#fff;font:700 15px sans-serif">H</div>')).add_to(m)
    for d, day in enumerate(plan.days):
        colour = DAY_COLOURS[d % len(DAY_COLOURS)]
        ink = DAY_TEXT[d % len(DAY_TEXT)]
        line, n = [], 0
        for s in day.stops:
            p = places.by_id.get(s.place_id)
            if not p or s.role == "hotel":
                continue
            n += 1
            line.append([p.lat, p.lon])
            points.append([p.lat, p.lon])
            folium.Marker([p.lat, p.lon], tooltip=f"Day {d + 1}, stop {n}: {html.escape(p.name)}", icon=folium.DivIcon(
                icon_size=(24, 24), icon_anchor=(12, 12),
                html=f'<div style="width:24px;height:24px;border-radius:50%;background:{colour};border:1.5px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.3);'
                     f'display:flex;align-items:center;justify-content:center;color:{ink};font:700 11px sans-serif">{n}</div>')).add_to(m)
        if len(line) > 1:
            folium.PolyLine(line, color=colour, weight=2.5, opacity=0.75).add_to(m)
    if points:
        m.fit_bounds(points, padding=(24, 24))
    return m


def map_legend(plan: Plan) -> str:
    items = "".join(f'<span class="pe-chip"><span class="pe-day-dot" style="background:{DAY_COLOURS[d % 4]};width:10px;height:10px"></span>Day {d + 1}</span>'
                    for d in range(len(plan.days)))
    return f'<div style="margin:8px 0 12px">{items}<span class="pe-chip"><span class="pe-day-dot" style="background:#171717;width:10px;height:10px"></span>Hotel</span></div>'


def transcript(plan: Plan, places, profiles, me=None) -> str:
    """The negotiation as the viewer may see it: their own agent in full, the others without names or reasons."""
    log = plan.log.get("transcript") or []
    me_id = me.member_id if me else None
    others = [p for p in profiles if p.member_id != me_id]
    names = {p.member_id: f"Traveler {k}" for k, p in enumerate(others, 1)}
    if me:
        names[me_id] = "You"

    def place(pid):
        return places.by_id[pid].name if pid in places.by_id else pid

    def anonymous(text: str) -> str:
        for p in others:
            text = re.sub(rf"\b{re.escape(p.name)}\b", names[p.member_id], text)
        return re.sub(rf"\b{re.escape(me.name)}\b", "you", text) if me else text

    out = []
    for h in log:
        lines = []
        for mid, r in h["replies"].items():
            who = esc(names.get(mid, "A traveler"))
            mine = mid == me_id
            pct = f' ({h["sat"].get(mid, 0):.0%})' if mine else ""
            reason = f" {esc(r['reason'])}" if mine else ""
            if r["accept"]:
                lines.append(f"<p><b>{who}</b>{pct} {'accept' if mine else 'accepts'}.{reason}</p>")
            else:
                ask = []
                if r["add"]:
                    ask.append("add " + ", ".join(esc(place(x)) for x in r["add"]))
                if r["remove"]:
                    ask.append("remove " + ", ".join(esc(place(x)) for x in r["remove"]))
                lines.append(f"<p><b>{who}</b>{pct} {'ask' if mine else 'asks'} to {' and '.join(ask) or 'change the plan'}.{reason}</p>")
        d = h.get("decision")
        if d:
            granted = []
            if d["include"]:
                granted.append("in: " + ", ".join(esc(place(x)) for x in d["include"]))
            if d["exclude"]:
                granted.append("out: " + ", ".join(esc(place(x)) for x in d["exclude"]))
            refused = ", ".join(esc(place(x)) for x in d.get("refused") or [])
            cost = f"<br/>Not carried out, because it did not fit or cost the group too much: {refused}." if refused else ""
            lines.append(f'<div class="pe-mediator"><b>Mediator:</b> {esc(anonymous(d["note"]))}<br/>{"; ".join(granted) or "no places changed"}.{cost}</div>')
        out.append(f'<div class="pe-round"><h4>Round {h["round"]}</h4>{"".join(lines)}</div>')
    status = {"agreed": "Everyone accepted.", "max rounds": "The rounds ran out before everyone accepted, so the plan is the one they reviewed last.",
              "no further change": "The plan stopped changing, so the negotiation ended.",
              "fallback to M3": "No agreement, and someone stayed too low, so the fairness rule decided the final plan.",
              "agents did not answer": "The agents did not answer, so this plan is the starting plan with no negotiation."}.get(plan.log.get("status"), "")
    dropped = ", ".join(esc(place(x)) for x in plan.log.get("agent_requests_dropped") or [])
    could_not = f'<div class="pe-note">Asked for, but could not fit with everything else: {dropped}.</div>' if dropped else ""
    return "".join(out) + (f'<div class="pe-note">{esc(status)}</div>' if status else "") + could_not


def compare_table(rows: list[dict]) -> str:
    """rows: label, method, plan. M4 is highlighted as our method."""
    head = "<tr><th>Method</th><th class='num'>Average match</th><th class='num'>Lowest match</th><th class='num'>Gap</th><th class='num'>Cost</th><th class='num'>Travel a day</th><th class='num'>Rule problems</th></tr>"
    body = []
    for r in rows:
        plan = r["plan"]
        values = list(plan.satisfaction.values()) or [0]
        days = max(len(plan.days), 1)
        travel = sum(s.travel_min for d in plan.days for s in d.stops) / days
        problems = len(plan.log.get("violations", []))
        body.append(f"<tr class=\"{'ours' if r['method'] == 'M4' else ''}\"><td>{esc(METHOD_NAMES[r['method']])}</td><td class='num'>{sum(values) / len(values):.0%}</td>"
                    f"<td class='num'>{min(values):.0%}</td><td class='num'>{max(values) - min(values):.0%}</td><td class='num'>{plan.cost_per_person:,.0f} EUR</td>"
                    f"<td class='num'>{span(int(travel))}</td><td class='num'>{problems}</td></tr>")
    return f'<div class="pe-scroll"><table class="pe-table">{head}{"".join(body)}</table></div>'


def plan_as_text(plan: Plan, places, trip_name: str = "Paris trip") -> str:
    """A plain itinerary to save or share."""
    lines = [trip_name, ""]
    hotel = places.by_id.get(plan.hotel_id)
    if hotel:
        lines += [f"Hotel: {hotel.name} ({hotel.stars or '?'} stars), {hotel.address or 'see map'}", ""]
    for i, day in enumerate(plan.days, 1):
        lines.append(f"Day {i}, {day.date:%A %d %B %Y}")
        for s in day.stops:
            p = places.by_id.get(s.place_id)
            if not p or s.role == "hotel":
                continue
            what = {"lunch": "Lunch: ", "dinner": "Dinner: "}.get(s.role, "")
            lines.append(f"  {clock(s.start_min)}  {what}{p.name} ({price_line(p)})")
        lines.append("")
    lines += [f"Estimated cost per person: {plan.cost_per_person:,.0f} EUR",
              "Times, travel and prices are estimates. Check opening hours and book tickets on the official websites."]
    return "\n".join(lines)
