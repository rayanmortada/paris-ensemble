"""The pages of the app: password, home, trip, travelers, planning, compare and rate, results, admin."""
import hmac
import os
import secrets
import time
import uuid
from datetime import date, timedelta
from urllib.parse import quote, urlsplit

import streamlit as st
from streamlit_folium import st_folium

from src import store
from src.config import cfg
from src.jobs import Jobs, friendly_error
from src.pipeline import METHOD_NAMES, load_context
from src.planner.scoring import pick_best
from src.schemas import Feedback, MemberInput
from ui import theme, views
from ui.examples import EXAMPLES
from ui.views import esc, icon

PACE_LABELS = {"relaxed": "Relaxed", "normal": "Normal", "packed": "Packed"}
RATING_QUESTIONS = [("wishes_respected", "My wishes are respected"), ("fair", "It is fair to the whole group"),
                    ("realistic", "It looks realistic to do"), ("clear", "The explanation is clear")]
METHOD_TEXT = {
    "M1": "One calculation that picks the places with the highest total score for the group. Nobody negotiates.",
    "M4": "Each traveler's AI agent looked at the plan and objected or accepted. A mediator turned that into requests and weights, and the optimizer solved the plan again each round.",
}


@st.cache_resource(show_spinner=False)
def resources():
    ctx = load_context()
    return ctx, Jobs(ctx)


def ss():
    return st.session_state


# ---------- shared pieces ----------

def mock_active() -> bool:
    return os.getenv("LLM_MOCK") == "1"


def test_banner(ctx=None) -> None:
    if mock_active():
        st.html('<div class="pe-banner"><b>Test mode.</b> The AI answers on this page come from an offline stand-in made of simple rules, not from OpenAI. Plans and explanations here say nothing about real quality.</div>')
    elif ctx and ctx.build_info.get("enrichment") == "mock":
        st.html('<div class="pe-banner"><b>Test data.</b> The place scores were built by the offline stand-in. Run <code>python scripts/build_places.py --stage enrich</code> with your OpenAI key before using these plans.</div>')


def key_missing() -> bool:
    return not mock_active() and not os.getenv("OPENAI_API_KEY", "").strip()


def _nav_buttons(suffix: str) -> None:
    mine = ss().get("my_token")
    width = "stretch" if suffix == "m" else "content"
    if mine and st.query_params.get("me") != mine and st.button("My trip", key=f"nav_trip_{suffix}", width=width):
        st.query_params.clear()
        st.query_params["me"] = mine
        st.rerun()
    if st.button("New trip", key=f"nav_new_{suffix}", width=width):
        st.query_params.clear()
        ss().page = "setup"
        st.rerun()
    if st.button("Log out", key=f"nav_out_{suffix}", width=width):
        ss().clear()
        st.query_params.clear()
        st.rerun()


def topbar() -> None:
    """Friends who came through a link only see the brand: new trips need the password."""
    with st.container(horizontal=True, key="topbar", vertical_alignment="center", gap="small"):
        st.html(theme.brand())
        if not ss().get("authed"):
            return
        st.space("stretch")
        with st.container(horizontal=True, key="nav_wide", width="content", gap="small", vertical_alignment="center"):
            _nav_buttons("w")
        with st.container(key="nav_narrow", width="content"):          # on phones the same buttons sit in a menu
            with st.popover("", icon=":material/menu:"):
                _nav_buttons("m")


def stepper(current: int) -> None:
    parts = []
    for n in range(1, 6):
        cls = "on" if n == current else "done" if n < current else ""
        mark = icon("check", 15, color="#FFFFFF") if n < current else str(n)
        parts.append(f'<div class="pe-step {cls}">{mark}</div>')
        if n < 5:
            parts.append('<div class="pe-step-line"></div>')
    st.html(f'<div class="pe-steps">{"".join(parts)}</div>')


def footer() -> None:
    st.html('<div class="pe-footer">A university project on group trip planning with AI agents. '
            'Place data <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">© OpenStreetMap contributors</a>; '
            'descriptions from Wikipedia; photos from Wikimedia Commons, credited on each place. '
            'Times, travel and prices are estimates: check opening hours and book on the official websites before you go.</div>')


# ---------- password ----------

def gate() -> None:
    theme.skyline_background()
    passcode = os.getenv("APP_PASSCODE", "").strip()
    st.html(f'<div class="pe-hero" style="padding-bottom:18px"><div style="margin:0 auto 10px">{theme.emblem_img(46)}</div>'
            f'<h1 class="pe-h1" style="font-size:clamp(32px,5vw,46px)">{theme.APP_NAME}</h1>'
            f'<p class="pe-sub" style="margin-top:8px">Plan a trip to Paris that the whole group can agree on.</p></div>')
    with st.container(key="narrow"):
        with st.container(key="card_gate"):
            if not passcode:
                st.error("This app has no password set (APP_PASSCODE), so it stays locked.")
                return
            locked = ss().get("locked_until", 0) - time.time()
            with st.form("gate", border=False):
                code = st.text_input("Password", type="password")
                nickname = st.text_input("Your first name or a nickname", max_chars=40)
                agree = st.checkbox("I agree that what I write and my ratings are stored anonymously for a university project. "
                                    "I will not write personal details in my wishes.")
                go = st.form_submit_button("Enter", type="primary", width="stretch")
            if go:
                if locked > 0:
                    st.error(f"Too many wrong attempts. Try again in {int(locked) + 1} seconds.")
                elif not hmac.compare_digest(code.encode(), passcode.encode()):
                    time.sleep(1)
                    ss().fails = ss().get("fails", 0) + 1
                    if ss().fails >= 5:
                        ss().locked_until = time.time() + 60
                        ss().fails = 0
                    st.error("That password is not right.")
                elif not nickname.strip():
                    st.error("Please enter a first name or a nickname.")
                elif not agree:
                    st.error("Please agree to continue.")
                else:
                    ss().authed = True
                    ss().visitor_id = uuid.uuid4().hex[:12]
                    ss().nickname = nickname.strip()
                    store.save_visit(ss().visitor_id, ss().nickname, agree)
                    st.rerun()


# ---------- home ----------

def home() -> None:
    ctx, _ = resources()
    test_banner(ctx)
    st.html('<div class="pe-hero"><p class="pe-eyebrow">Paris, together</p><h1 class="pe-h1">Your group trip<br/>to Paris</h1>'
            '<p class="pe-sub">Everyone says what they want in their own words. An AI agent speaks for each person, and a checked plan comes out that is fair to the whole group.</p></div>')
    with st.container(horizontal=True, horizontal_alignment="center", key="cta_home"):
        if st.button("Plan our trip", type="primary", icon=":material/route:", key="home_start"):
            ss().page = "setup"
            st.rerun()
    if key_missing():
        st.html('<div class="pe-banner">The OpenAI key is not set yet (OPENAI_API_KEY), so planning will not work until it is added.</div>')

    def card(ic, title, text):
        return f'<div class="pe-card"><div class="pe-feature-icon">{icon(ic, 21, color=views.GOLD_INK)}</div><div class="pe-h3" style="font-size:16px">{title}</div><p>{text}</p></div>'

    st.html('<div class="pe-section"><h2 class="pe-h2 pe-center" style="margin-bottom:22px">Everything for a good group trip</h2><div class="pe-grid4">'
            + card("chat", "An agent for each traveler", "Write what you want in plain words. Your agent reads it and defends it when the group decides.")
            + card("shield", "Plans that really work", "Real opening hours on your dates, travel time and your budget are checked by code, not left to the AI.")
            + card("scale", "Fair to everyone", "See how well the plan suits you and the least happy traveler, so nobody is quietly sacrificed. Your answers stay private.")
            + card("refresh", "Learns from you", "Remove a place or add one, and the plan is solved again around what you changed.")
            + "</div></div>")

    def step(n, title, text):
        return f'<div class="pe-card"><div class="pe-num">{n}</div><div class="pe-h3" style="font-size:16px">{title}</div><p>{text}</p></div>'

    st.html('<div class="pe-section"><h2 class="pe-h2 pe-center" style="margin:30px 0 22px">How it works</h2><div class="pe-grid3">'
            + step(1, "Invite your group", "Choose the dates and send everyone one link. Each person fills in their own part, on their own phone, and only they see it.")
            + step(2, "Compare two plans", "When everyone has answered, two complete plans are made, with a map, times and costs. Each person rates them and votes, without knowing how each was made.")
            + step(3, "Get the best plan", "The plan the group prefers wins. See how it was made, change what you like, and take the itinerary with you.")
            + "</div></div>")
    footer()


# ---------- trip setup ----------

def setup() -> None:
    ctx, _ = resources()
    test_banner(ctx)
    stepper(1)
    with st.container(key="narrow"):
        st.html('<div class="pe-page-title"><h2 class="pe-h2" style="font-family:Satoshi,sans-serif;font-weight:700;font-size:28px">1. Your trip</h2><p class="pe-sub" style="margin-top:4px;font-size:16px">When are you going, and how many of you?</p></div>')
        with st.container(key="card_setup"):
            today = date.today()
            friday = today + timedelta(days=(4 - today.weekday()) % 7 or 7)
            with st.form("setup", border=False):
                start = st.date_input("First day", value=friday, min_value=today, max_value=today + timedelta(days=500), format="DD/MM/YYYY")
                days = st.segmented_control("Number of days", [2, 3, 4], default=3, key="setup_days")
                people = st.segmented_control("Number of travelers", [2, 3, 4, 5], default=3, key="setup_people")
                go = st.form_submit_button("Next", type="primary", width="stretch", icon=":material/arrow_forward:")
            if go:
                trip_id = secrets.token_urlsafe(6)
                store.create_trip(trip_id, ss().visitor_id, start.isoformat(), int(days or 3), int(people or 3))
                store.create_invite(trip_id)
                ss().my_token = store.add_person(trip_id, "m1", "organizer")
                st.query_params.clear()
                st.query_params["me"] = ss().my_token
                st.rerun()
    footer()


# ---------- who is looking ----------

def is_organizer(person: dict) -> bool:
    return person["role"] == "organizer"


def public_base() -> str:
    """The address friends should open. PUBLIC_URL wins, for hosts where the app sees a different address than visitors."""
    url = os.getenv("PUBLIC_URL", "").strip() or st.context.url or ""
    return url.split("?")[0].replace("/~/+/", "/").rstrip("/") + "/"


def invalid_link() -> None:
    with st.container(key="narrow"):
        with st.container(key="card_invalid"):
            st.html('<div class="pe-h3">This link does not work</div><p class="pe-small" style="margin:0">Ask the person who invited you to send it again.</p>')


# ---------- travelers ----------

def _lowest_sensible_budget(days: int) -> int:
    """Roughly what the cheapest possible trip costs per person."""
    C = cfg()
    hotel = min(C.hotel_eur_by_stars.values()) / 2
    meals = 2 * min(C.meal_eur_by_tier.values())
    return int((hotel + meals + C.breakfast_eur + 3 * C.travel.ticket_eur) * days)


def begin_planning(trip_id: str) -> str | None:
    """Starts the plans, unless a limit says no. Returns the reason in that case."""
    trip = store.get_trip(trip_id)
    C = cfg().app
    if trip["status"] != "draft":
        return None
    if key_missing():
        return "The OpenAI key is not set (OPENAI_API_KEY), so planning cannot start."
    if store.trips_today() >= C.max_trips_per_day:
        return "The daily limit of trips for this app has been reached. Please try again tomorrow."
    if store.trips_by_visitor(trip["visitor_id"]) >= C.max_trips_per_visitor:
        return f"The organizer has planned {C.max_trips_per_visitor} trips in this visit, which is the limit."
    _, jobs = resources()
    store.update_trip(trip_id, n_members=len(store.get_members(trip_id)))
    jobs.start_plan(trip_id)
    return None


def begin_if_everyone_answered(trip_id: str) -> None:
    if len(store.get_members(trip_id)) >= store.get_trip(trip_id)["n_members"]:
        ss().plan_problem = begin_planning(trip_id)


def answers_form(trip: dict, member_id: str | None, saved: MemberInput | None, invite: str | None = None) -> None:
    """One person's own answers. With an invite, saving also takes a seat on the trip and gives them their personal link."""
    trip_id = trip["trip_id"]
    rev = ss().get("form_rev", 0)                  # a new number gives fresh widgets
    with st.form(f"member_{trip_id}_{rev}", border=False):
        name = st.text_input("Your first name", value=saved.name if saved else ss().get("nickname", ""), max_chars=30, key=f"name_{rev}")
        wishes = st.text_area("What do you want from this trip?", value=saved.wishes if saved else "", height=150, max_chars=800, key=f"wishes_{rev}",
                              placeholder="In your own words, in any language. For example: I love old churches and long lunches, and I must see Sainte-Chapelle.")
        left, right = st.columns(2)
        budget = left.number_input("Your budget (EUR, whole trip)", min_value=100, max_value=20000, step=50,
                                   value=int(saved.budget_eur) if saved and saved.budget_eur else 900, key=f"budget_{rev}")
        pace = right.selectbox("Pace", list(PACE_LABELS), format_func=PACE_LABELS.get, key=f"pace_{rev}",
                               index=list(PACE_LABELS).index(saved.pace) if saved and saved.pace else 1)
        a, b = st.columns(2)
        veg = a.checkbox("Vegetarian", value=saved.vegetarian if saved else False, key=f"veg_{rev}")
        mob = b.checkbox("Difficulty walking", value=saved.mobility if saved else False, key=f"mob_{rev}")
        agree = True
        if invite:
            agree = st.checkbox("I agree that what I write and my ratings are stored anonymously for a university project. I will not write personal details in my wishes.", key=f"agree_{rev}")
        submitted = st.form_submit_button("Save my answers" if not saved else "Save changes", type="primary", width="stretch")
    if budget < _lowest_sensible_budget(trip["days"]):
        st.html(f'<div class="pe-note">A {trip["days"]} day trip costs at least about {_lowest_sensible_budget(trip["days"])} EUR per person, even with the cheapest hotel and meals, so this budget will be hard to respect.</div>')
    if not submitted:
        return
    if not name.strip():
        st.error("Please add your first name.")
    elif len(wishes.strip()) < 15:
        st.error("Please write at least a sentence about what you want.")
    elif not agree:
        st.error("Please agree to continue.")
    elif store.name_taken(trip_id, name, member_id):
        st.error(f"Someone called {name.strip()} has already answered. If that is you, open your personal link. If not, add an initial to your name.")
    else:
        member = MemberInput(member_id=member_id or "", name=name.strip(), wishes=wishes.strip(), budget_eur=float(budget), vegetarian=veg, mobility=mob, pace=pace)
        if invite:
            token = store.join_trip(trip_id, member)
            if token is None:
                st.error("This trip already has everyone it was expecting. Ask the organizer to make room for you.")
                return
            store.save_visit(token[:12], name.strip(), True)
            st.query_params.clear()
            st.query_params["me"] = token
        else:
            store.save_member(trip_id, member)
        begin_if_everyone_answered(trip_id)
        st.rerun()


def your_part(trip: dict, person: dict, saved: MemberInput | None) -> None:
    if not saved:
        with st.container(key="card_member"):
            st.html('<div class="pe-h3">Your part</div><p class="pe-small" style="margin:0 0 6px">Only you can see what you write here.</p>')
            answers_form(trip, person["member_id"], None)
        return
    with st.container(key="card_saved"):
        st.html(f'<div class="pe-h3">{icon("check", 18, color=views.GOLD_INK)} Thanks {esc(saved.name)}, your answers are saved</div>'
                '<p class="pe-small" style="margin:0">Only you can see them. Planning starts by itself when everyone has answered, so you can close this page and come back later.</p>')
    with st.expander("Change my answers", icon=":material/edit:"):
        answers_form(trip, person["member_id"], saved)
    with st.expander("Your personal link", icon=":material/link:"):
        st.code(f"{public_base()}?me={person['token']}", language=None, wrap_lines=True)
        st.html('<p class="pe-small" style="margin:0">Save it to come back to your part and to the plans. It opens the trip as you, so keep it to yourself.</p>')


def invite_card(trip: dict, inviter: str | None) -> None:
    link = f"{public_base()}?join={store.invite_of(trip['trip_id'])}"
    first = date.fromisoformat(trip["start_date"])
    subject = "Our trip to Paris: tell me what you want"
    who = f"{inviter} here! I am" if inviter else "I am"
    text = (f"{who} planning our {trip['days']} day trip to Paris from {first:%-d %B}. Open this link and say what you want from the trip. "
            f"It takes two minutes and only you see your answers. When everyone has answered, the app makes the plans.\n\n{link}")
    with st.container(key="card_invite"):
        st.html('<div class="pe-h3">Invite your group</div><p class="pe-small" style="margin:0 0 8px">Send everyone this one link. They need no password, and it opens this trip only.</p>')
        st.code(link, language=None, wrap_lines=True)
        left, right = st.columns(2)
        left.link_button("Send by email", f"mailto:?subject={quote(subject)}&body={quote(text)}", icon=":material/mail:", width="stretch")
        right.link_button("WhatsApp", f"https://wa.me/?text={quote(text)}", icon=":material/chat:", width="stretch")
        if urlsplit(link).hostname in ("localhost", "127.0.0.1"):
            st.html('<div class="pe-note" style="margin:10px 0 0">This address only works on this computer. Friends can open the link once the app is online.</div>')


@st.fragment(run_every="4s")
def travelers_status(trip_id: str, organizer: bool) -> None:
    """Refreshes on its own, so people answering on other phones appear here. It shows who answered, never what."""
    trip = store.get_trip(trip_id)
    if trip["status"] != "draft":
        st.rerun()
    members = store.get_members(trip_id)
    n, ready = trip["n_members"], len(members)
    if ss().get("seen_ready", ready) != ready:     # the rest of the page changes only when someone new answers, so nothing moves while you use it
        ss().seen_ready = ready
        st.rerun()
    ss().seen_ready = ready
    chips = "".join(f'<span class="pe-chip ok">{icon("check", 13, color=views.GOLD_INK)} {esc(m.name)}</span>' for m in members)
    if ready < n:
        chips += f'<span class="pe-chip">Waiting for {n - ready} more</span>'
    with st.container(key="card_status"):
        st.html(f'<div class="pe-h3">{ready} of {n} travelers ready</div><div style="margin:8px 0 2px">{chips}</div>')
        if ss().get("plan_problem"):
            st.error(ss().plan_problem)
        if organizer and 2 <= ready < n and st.button(f"Plan now with these {ready}", key="start_plan", width="stretch", icon=":material/auto_awesome:"):
            ss().plan_problem = begin_planning(trip_id)
            st.rerun()


def manage_group(trip: dict, members: dict) -> None:
    trip_id = trip["trip_id"]
    low = max(2, len(members))
    friends = [m for m in members.values() if m.member_id != "m1"]
    with st.expander("Change the group", icon=":material/group:"):
        size = st.segmented_control("Number of travelers", [k for k in (2, 3, 4, 5) if k >= low], default=max(trip["n_members"], low), key=f"group_size_{trip_id}")
        if size and size != trip["n_members"]:
            store.update_trip(trip_id, n_members=int(size))
            begin_if_everyone_answered(trip_id)
            st.rerun()
        if friends:
            gone = st.selectbox("Remove a traveler, for example if they need to start again", friends, format_func=lambda m: m.name, index=None, key=f"remove_{trip_id}", placeholder="Choose")
            if gone and st.button(f"Remove {gone.name}", key=f"remove_go_{trip_id}"):
                store.remove_traveler(trip_id, gone.member_id)
                st.rerun()


def example_group(trip: dict) -> None:
    trip_id = trip["trip_id"]
    with st.expander("Try it alone with an example group", icon=":material/science:"):
        choice = st.selectbox("Example", list(EXAMPLES), key="example_choice", label_visibility="collapsed")
        st.html('<p class="pe-small" style="margin:0 0 8px">Made up travelers fill in the group, and planning starts right away. You then rate for them too.</p>')
        if st.button("Fill in this example", key="use_example"):
            if len(store.people_of(trip_id)) > 1:          # a friend joined while this page was open: their answers stay
                st.error("Someone has already joined this trip, so the example cannot replace them.")
                return
            group = EXAMPLES[choice]
            store.update_trip(trip_id, n_members=len(group))
            for k, m in enumerate(group, 1):
                store.save_member(trip_id, MemberInput(member_id=f"m{k}", name=m["name"], wishes=m["wishes"], budget_eur=float(m["budget"]),
                                                       vegetarian=m["veg"], mobility=m["mob"], pace=m["pace"]))
            begin_if_everyone_answered(trip_id)
            st.rerun()


def travelers(trip: dict, person: dict) -> None:
    ctx, _ = resources()
    test_banner(ctx)
    stepper(2)
    trip_id = trip["trip_id"]
    first = date.fromisoformat(trip["start_date"])
    members = {m.member_id: m for m in store.get_members(trip_id)}
    mine = members.get(person["member_id"])
    st.html(f'<div class="pe-page-title"><h2 class="pe-h2" style="font-family:Satoshi,sans-serif;font-weight:700;font-size:28px">2. The travelers</h2>'
            f'<p class="pe-sub" style="margin-top:4px;font-size:16px">{trip["days"]} days from {first:%A %-d %B %Y}. Each person says what they want, in their own words, on their own phone.</p></div>')
    if is_organizer(person):
        left, right = st.columns([3, 2], gap="medium")
        with left:
            your_part(trip, person, mine)
        with right:
            invite_card(trip, mine.name if mine else None)
            travelers_status(trip_id, True)
            manage_group(trip, members)
            if len(members) == (1 if mine else 0) and len(store.people_of(trip_id)) == 1:
                example_group(trip)
    else:
        with st.container(key="narrow"):
            your_part(trip, person, mine)
            travelers_status(trip_id, False)
    footer()


def join_page(trip: dict, invite: str) -> None:
    """What a friend sees when they open the invite: the trip and their own part, nothing about anyone else's answers."""
    ctx, _ = resources()
    test_banner(ctx)
    first = date.fromisoformat(trip["start_date"])
    organizer = next((m.name for m in store.get_members(trip["trip_id"]) if m.member_id == "m1"), None)
    who = f"{esc(organizer)} invited you" if organizer else "You are invited"
    with st.container(key="narrow"):
        st.html(f'<div class="pe-page-title pe-center"><p class="pe-eyebrow">{who}</p><h2 class="pe-h2">A trip to Paris</h2>'
                f'<p class="pe-sub" style="margin:6px 0 14px">{trip["days"]} days from {first:%A %-d %B %Y}. Say what you want from the trip. Only you can see your answers, and the app makes the plans when everyone has answered.</p></div>')
        if trip["status"] != "draft":
            with st.container(key="card_closed"):
                st.html('<div class="pe-h3">The plans have already been made</div><p class="pe-small" style="margin:0">Ask the organizer. They can add you to a new trip.</p>')
            return
        with st.container(key="card_member"):
            st.html('<div class="pe-h3">Your part</div>')
            answers_form(trip, None, None, invite)
        travelers_status(trip["trip_id"], False)
    footer()


# ---------- planning ----------

@st.fragment(run_every="2s")
def planning_status(trip_id: str) -> None:
    _, jobs = resources()
    trip = store.get_trip(trip_id)
    job = jobs.get(f"plan:{trip_id}")
    if trip["status"] != "running":
        st.rerun()
    if job["state"] == "idle":                  # the server restarted while this was running
        store.update_trip(trip_id, status="failed", error="The app restarted while planning. Please try again.")
        st.rerun()
    elapsed = int(time.time() - job.get("started", time.time()))
    done = job["steps"][:-1]
    current = job["steps"][-1] if job["steps"] else "Starting"
    rows = "".join(f'<div class="pe-travel" style="color:var(--pe-gold-ink)">{icon("check", 14, color=views.GOLD_INK)} {esc(s)}</div>' for s in done)
    with st.container(key="card_planning"):
        st.html(f'<div class="pe-h3">Working on your plans</div>{rows}'
                f'<div class="pe-travel" style="color:var(--pe-ink);font-weight:600">{icon("clock", 14, color="#0A0A0A")} {esc(current)}…</div>'
                f'<p class="pe-small" style="margin:12px 0 0">{elapsed} seconds so far. This usually takes 1 to 3 minutes. You can lock your phone or close this page: reopen the link and the plans will be here.</p>')


def planning(trip: dict) -> None:
    ctx, _ = resources()
    test_banner(ctx)
    theme.skyline_background()
    stepper(3)
    with st.container(key="narrow"):
        st.html('<div class="pe-page-title pe-center"><h2 class="pe-h2">Planning your trip</h2><p class="pe-sub" style="margin:4px 0 14px">Each traveler has an agent reading their wishes and scoring places.</p></div>')
        planning_status(trip["trip_id"])


def failed(trip: dict, person: dict) -> None:
    _, jobs = resources()
    stepper(3)
    with st.container(key="narrow"):
        with st.container(key="card_failed"):
            st.html(f'<div class="pe-h3">The plans could not be made</div><p class="pe-small">{esc(trip["error"] or "Something went wrong.")}</p>')
            if not is_organizer(person):
                st.html('<p class="pe-small" style="margin:0">The organizer can try again.</p>')
                return
            if st.button("Try again", type="primary", key="retry", width="stretch"):
                jobs.start_plan(trip["trip_id"])
                st.rerun()
            if st.button("Change the travelers", key="back_edit", width="stretch"):
                store.update_trip(trip["trip_id"], status="draft", error=None)
                st.rerun()


# ---------- compare and vote ----------

def my_profile(person: dict, profiles):
    return next((p for p in profiles if p.member_id == person["member_id"]), None)


def speaks_for(person: dict, profiles) -> list:
    """Whose part this person may answer: their own, and for the organizer also example travelers, who have no phone of their own."""
    own = [p for p in profiles if p.member_id == person["member_id"]]
    if not is_organizer(person):
        return own
    seated = {x["member_id"] for x in store.people_of(person["trip_id"])}
    return own + [p for p in profiles if p.member_id not in seated]


def show_plan(trip: dict, label: str, profiles, places, me=None) -> None:
    row = store.latest_plan(trip["trip_id"], label)
    plan = row["plan"]
    st.html(views.kpis(plan, me))
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st_folium(views.plan_map(plan, places), height=430, use_container_width=True, returned_objects=[],
                  key=f"map_{trip['trip_id']}_{label}_{row['version']}")
        st.html(views.map_legend(plan))
    with right:
        st.html(views.fit_card(plan, me))
    first = date.fromisoformat(trip["start_date"])
    notes = views.notes_block(plan, places)
    if me and me.unmatched:
        notes += (f'<div class="pe-note">You mentioned {esc(", ".join(me.unmatched))}. We could not match that to a place on our list, so the plan was not built around it.</div>')
    if notes:
        st.html(notes)
    st.html(views.hotel_card(plan, places, len(profiles), first) or "<div></div>")
    st.html("".join(views.day_block(day, i, places, is_open=(i == 0)) for i, day in enumerate(plan.days)))
    st.html('<p class="pe-small" style="margin:-4px 0 12px">Evenings after dinner are left free. Tap a place for its photo, hours and links.</p>')
    st.html(views.cost_card(plan))


def close_vote_if_complete(trip_id: str, profiles) -> None:
    voted = {r["member_name"] for r in store.get_prefs(trip_id)}
    if all(p.name in voted for p in profiles):
        store.update_trip(trip_id, status="decided")


def rate_form(trip: dict, person: dict, profiles) -> None:
    trip_id = trip["trip_id"]
    rev = ss().get("rate_rev", 0)                  # a new number gives an empty form for the next example traveler
    mine = speaks_for(person, profiles)
    done = {r["member_name"] for r in store.get_prefs(trip_id)}
    if not mine:
        st.html('<div class="pe-note">You did not fill in a part for this trip, so you have no vote. You can still look through both plans.</div>')
        return
    todo = [p for p in mine if p.name not in done]
    with st.container(key="card_rate"):
        st.html('<div class="pe-h3">Rate both plans</div><p class="pe-small" style="margin:0 0 8px">From 1 (not at all) to 5 (completely). Only your answers count here: nobody sees how you voted, and you find out how each plan was made once the group has decided.</p>')
        if not todo:
            st.html(f'<div class="pe-note" style="margin:0 0 10px">{icon("check", 14, color=views.GOLD_INK)} Your vote is in. You can change it until the group has decided.</div>')
        with st.form(f"rate_{trip_id}_{rev}", border=False):
            names = [p.name for p in (todo or mine)]
            who = st.selectbox("Rating for", names, key=f"rate_who_{rev}") if len(mine) > 1 else mine[0].name
            answers = {}
            for label in ("A", "B"):
                st.html(f'<div style="font-weight:700;font-size:16px;margin:12px 0 2px">Plan {label}</div>')
                for field, text in RATING_QUESTIONS:
                    answers[label, field] = st.pills(text, [1, 2, 3, 4, 5], key=f"rate_{label}_{field}_{rev}")
            preferred = st.segmented_control("Which plan would you rather go on?", ["Plan A", "Plan B"], key=f"rate_pref_{rev}")
            comment = st.text_area("Anything to add? (optional)", height=80, max_chars=600, key=f"rate_comment_{rev}")
            submitted = st.form_submit_button("Save my vote", type="primary", width="stretch")
        if submitted:
            if any(v is None for v in answers.values()) or not preferred:
                st.error("Please answer every question and choose a plan.")
            else:
                for label in ("A", "B"):
                    store.save_rating(trip_id, who, label, *(answers[label, field] for field, _ in RATING_QUESTIONS))
                store.save_pref(trip_id, who, preferred[-1], comment.strip())
                ss().rate_rev = rev + 1
                close_vote_if_complete(trip_id, profiles)
                st.toast("Thanks, your vote is saved.")
                st.rerun()


@st.fragment(run_every="4s")
def vote_status(trip_id: str, profiles, organizer: bool) -> None:
    """Who has voted, never for what. When the last vote is in, every open page moves on by itself."""
    if store.get_trip(trip_id)["status"] != "ready":
        st.rerun()
    voted = {r["member_name"] for r in store.get_prefs(trip_id)}
    chips = "".join(f'<span class="pe-chip ok">{icon("check", 13, color=views.GOLD_INK)} {esc(p.name)}</span>' if p.name in voted else f'<span class="pe-chip">{esc(p.name)}</span>' for p in profiles)
    with st.container(key="card_votes"):
        st.html(f'<div class="pe-h3" style="font-size:16px">{sum(p.name in voted for p in profiles)} of {len(profiles)} have voted</div><div style="margin:6px 0 2px">{chips}</div>'
                '<p class="pe-small" style="margin:6px 0 0">The plan the group prefers wins as soon as everyone has voted.</p>')
        if organizer and st.button("Close the vote now and pick the best plan", key="close_vote", width="stretch"):
            store.update_trip(trip_id, status="decided")
            st.rerun()


def compare(trip: dict, person: dict) -> None:
    ctx, jobs = resources()
    test_banner(ctx)
    stepper(4)
    trip_id = trip["trip_id"]
    try:
        with st.spinner("Opening your trip"):
            prep = jobs.get_prepared(trip_id)
    except Exception as e:
        st.error(friendly_error(e))
        return
    st.html('<div class="pe-page-title"><h2 class="pe-h2" style="font-family:Satoshi,sans-serif;font-weight:700;font-size:28px">4. Compare your two plans</h2>'
            '<p class="pe-sub" style="margin-top:4px;font-size:16px">Both plans were checked against opening hours on your dates and against the budget. Anything that could not be met is listed on the plan. Look through both, then rate them and vote.</p></div>')
    vote_status(trip_id, prep.profiles, is_organizer(person))
    view = st.segmented_control("View", ["Plan A", "Plan B", "Rate and vote"], default="Plan A", key=f"view_{trip_id}", label_visibility="collapsed") or "Plan A"
    if view == "Rate and vote":
        with st.container(key="narrow"):
            rate_form(trip, person, prep.profiles)
    else:
        show_plan(trip, view[-1], prep.profiles, ctx.places, my_profile(person, prep.profiles))
        st.html('<p class="pe-small pe-center" style="margin:6px 0 0">When you have seen both plans, choose <b>Rate and vote</b> at the top of the page.</p>')
    footer()


# ---------- results ----------

def results(trip: dict, person: dict) -> None:
    ctx, jobs = resources()
    test_banner(ctx)
    stepper(5)
    trip_id = trip["trip_id"]
    try:
        with st.spinner("Opening your trip"):
            prep = jobs.get_prepared(trip_id)
    except Exception as e:
        st.error(friendly_error(e))
        return
    places = ctx.places
    me = my_profile(person, prep.profiles)
    rows = {label: store.latest_plan(trip_id, label) for label in ("A", "B")}
    prefs = store.get_prefs(trip_id)
    winner, why = pick_best({label: store.first_plan(trip_id, label)["plan"] for label in ("A", "B")}, prefs, store.get_ratings(trip_id))
    st.html('<div class="pe-page-title"><h2 class="pe-h2" style="font-family:Satoshi,sans-serif;font-weight:700;font-size:28px">5. Your group\'s plan</h2></div>')
    with st.container(key="card_winner"):
        st.html(f'<div class="pe-h3">{icon("check", 18, color=views.GOLD_INK)} The best option for your group is Plan {winner}</div><p class="pe-small" style="margin:0">{esc(why)} Here is how each plan was made.</p>')
    cards = []
    for label, row in rows.items():
        votes = sum(p["preferred_label"] == label for p in prefs)
        vote_text = f'<p style="margin-top:8px"><b>{votes}</b> of {len(prefs)} preferred it.</p>' if prefs else ""
        chosen = f'<span class="pe-chip ok" style="margin:0 0 6px">{icon("check", 13, color=views.GOLD_INK)} The group\'s plan</span>' if label == winner else ""
        cards.append(f'<div class="pe-card">{chosen}<div class="pe-small">Plan {label}</div><div class="pe-h3">{esc(METHOD_NAMES[row["method"]])}</div><p>{esc(METHOD_TEXT[row["method"]])}</p>{vote_text}</div>')
    st.html(f'<div class="pe-grid2" style="margin-bottom:18px">{"".join(cards)}</div>')

    m4_label = next(label for label, row in rows.items() if row["method"] == "M4")
    first_m4 = store.first_plan(trip_id, m4_label)
    with st.expander("Read the negotiation between the agents", icon=":material/forum:"):
        st.html(views.transcript(first_m4["plan"], places, prep.profiles, me))

    st.html('<div class="pe-section"><div class="pe-h3" style="font-size:21px">The plan</div></div>')
    view = st.segmented_control("Plan", ["Plan A", "Plan B"], default=f"Plan {winner}", key=f"final_{trip_id}", label_visibility="collapsed") or f"Plan {winner}"
    label = view[-1]
    row = rows[label]
    if row["version"] > 1:
        st.html(f'<div class="pe-note">This is version {row["version"]} of Plan {label}, after changes from the group.</div>')
    show_plan(trip, label, prep.profiles, places, me)
    st.download_button("Download this itinerary", views.plan_as_text(row["plan"], places), file_name=f"paris-trip-plan-{label.lower()}.txt",
                       mime="text/plain", icon=":material/download:", key=f"dl_{label}_{row['version']}")
    edit_panel(trip, person, label, row, prep, places)
    if is_organizer(person):                       # it costs money, so it stays with the person who holds the password
        compare_panel(trip, rows)
    footer()


def edit_panel(trip: dict, person: dict, label: str, row: dict, prep, places) -> None:
    _, jobs = resources()
    mine = speaks_for(person, prep.profiles)
    if not mine:
        return
    trip_id = trip["trip_id"]
    key = f"replan:{trip_id}"
    state = jobs.get(key)
    if state["state"] == "running":
        replan_status(key)
        return
    if state["state"] == "failed":
        st.error(state["error"])
    used = row["version"] - 1
    limit = cfg().app.feedback_rounds
    st.html('<div class="pe-section"><div class="pe-h3" style="font-size:21px">Change this plan</div></div>')
    with st.container(key="card_edit"):
        if used >= limit:
            st.html(f'<p class="pe-small">You have used all {limit} rounds of changes for this plan.</p>')
            return
        st.html(f'<p class="pe-small" style="margin:0 0 8px">Remove or add places and the plan is solved again. The model also learns from what you change. Round {used + 1} of {limit}.</p>')
        names = {p.name: p for p in mine}
        who = st.selectbox("Changing for", list(names), key=f"edit_who_{label}") if len(names) > 1 else next(iter(names))
        member = names[who]
        plan = row["plan"]
        visits = [s.place_id for d in plan.days for s in d.stops if s.role == "visit" and s.place_id in places.by_id]
        agent = next(a for a in prep.agents if a.m.member_id == member.member_id)
        add_options = agent.candidates(plan, prep.pb)[:10]
        name = lambda pid: places.by_id[pid].name  # noqa: E731
        remove = st.multiselect("Remove from the plan", visits, format_func=name, key=f"edit_rm_{label}_{row['version']}")
        add = st.multiselect("Add to the plan (your best matches)", add_options, format_func=name, key=f"edit_add_{label}_{row['version']}")
        liked = st.multiselect("Places in the plan you really like (optional)", visits, format_func=name, key=f"edit_up_{label}_{row['version']}")
        if st.button("Update the plan", type="primary", width="stretch", disabled=not (remove or add or liked), key=f"edit_go_{label}"):
            actions = ([Feedback(member_id=member.member_id, place_id=p, action="remove") for p in remove]
                       + [Feedback(member_id=member.member_id, place_id=p, action="add") for p in add]
                       + [Feedback(member_id=member.member_id, place_id=p, action="up") for p in liked if p not in remove])
            jobs.start_replan(trip_id, label, who, actions)
            st.rerun()


@st.fragment(run_every="2s")
def replan_status(key: str) -> None:
    _, jobs = resources()
    state = jobs.get(key)
    if state["state"] != "running":
        st.rerun()
    current = state["steps"][-1] if state["steps"] else "Starting"
    with st.container(key="card_replan"):
        st.html(f'<div class="pe-h3">Updating your plan</div><div class="pe-travel" style="font-weight:600;color:var(--pe-ink)">{icon("clock", 14, color="#0A0A0A")} {esc(current)}…</div>')


@st.fragment(run_every="2s")
def compare_status(key: str) -> None:
    _, jobs = resources()
    state = jobs.get(key)
    if state["state"] != "running":
        st.rerun()
    current = state["steps"][-1] if state["steps"] else "Starting"
    st.html(f'<div class="pe-travel" style="font-weight:600;color:var(--pe-ink)">{icon("clock", 14, color="#0A0A0A")} {esc(current)}…</div>')


def compare_panel(trip: dict, rows: dict) -> None:
    _, jobs = resources()
    trip_id = trip["trip_id"]
    key = f"compare:{trip_id}"
    state = jobs.get(key)
    extras = {label: store.first_plan(trip_id, label) for label in ("x2", "x3", "x5")}
    st.html('<div class="pe-section"><div class="pe-h3" style="font-size:21px">All five methods for your group</div></div>')
    with st.container(key="card_compare"):
        if all(extras.values()):
            table = [{"method": r["method"], "plan": r["plan"]} for r in
                     sorted([store.first_plan(trip_id, "A"), store.first_plan(trip_id, "B"), *extras.values()], key=lambda r: r["method"])]
            st.html(views.compare_table(table))
            st.html('<p class="pe-small" style="margin:10px 0 0">Match is the model\'s estimate. Rule problems counts what our validator found: closed places, days that run over, a broken budget. Only the single LLM planner works without the optimizer and scheduler, so only it can have them.</p>')
        elif state["state"] == "running":
            compare_status(key)
        else:
            if state["state"] == "failed":
                st.error(state["error"])
            st.html('<p class="pe-small" style="margin:0 0 8px">See what the other approaches would have done for the same wishes: least misery, a fairness rule, and one LLM planning alone. This takes about a minute.</p>')
            if st.button("Compare all five methods", key="start_compare", width="stretch"):
                jobs.start_compare(trip_id)
                st.rerun()


# ---------- admin ----------

def admin() -> None:
    passcode = os.getenv("ADMIN_PASSCODE", "").strip()
    st.html(f'<div class="pe-hero" style="padding-bottom:10px"><h1 class="pe-h2">{theme.APP_NAME}: study data</h1></div>')
    with st.container(key="narrow"):
        if not passcode:
            st.error("No admin password is set (ADMIN_PASSCODE), so this page stays locked.")
            return
        if not ss().get("admin_ok"):
            with st.container(key="card_admin"):
                with st.form("admin", border=False):
                    code = st.text_input("Admin password", type="password")
                    go = st.form_submit_button("Open", type="primary", width="stretch")
                if go:
                    if hmac.compare_digest(code.encode(), passcode.encode()):
                        ss().admin_ok = True
                        st.rerun()
                    time.sleep(1)
                    st.error("That password is not right.")
            return
    counts = store.counts()
    tiles = "".join(f'<div class="pe-kpi"><div class="pe-kpi-label">{t}</div><div class="pe-kpi-value">{n}</div></div>' for t, n in counts.items() if t != "plans")
    st.html(f'<div class="pe-grid3" style="margin:10px 0 16px">{tiles}</div>')
    st.download_button("Download everything as Excel", store.export_excel(), file_name="paris-ensemble-study.xlsx", type="primary",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:")
    for table in ("trips", "ratings", "prefs", "actions", "visits"):
        with st.expander(table):
            st.dataframe(store.read_table(table), width="stretch")


# ---------- router ----------

def trip_page(person: dict) -> None:
    trip = store.get_trip(person["trip_id"])
    if not trip:
        invalid_link()
        return
    status = trip["status"]
    if status == "draft":
        travelers(trip, person)
    elif status == "running":
        planning(trip)
    elif status == "failed":
        failed(trip, person)
    elif status == "decided":
        results(trip, person)
    else:
        compare(trip, person)


def run() -> None:
    theme.inject()
    params = st.query_params
    if params.get("admin") == "1":
        admin()
        return
    if params.get("me"):                           # a personal link is the key to one person's part of one trip
        person = store.get_person(params["me"])
        topbar()
        if person:
            trip_page(person)
        else:
            invalid_link()
        return
    if params.get("join"):
        trip = store.trip_of_invite(params["join"])
        topbar()
        if trip:
            join_page(trip, params["join"])
        else:
            invalid_link()
        return
    if not ss().get("authed"):
        gate()
        return
    topbar()
    if ss().get("page") == "setup":
        setup()
    else:
        home()
