"""A real group in separate browsers: an organizer and two friends, checking the flow and, at every step, that answers stay private.

  pip install playwright && playwright install chromium
  APP_PASSCODE=test streamlit run app.py --server.port 8599        (add LLM_MOCK=1 to spend nothing)
  python scripts/e2e_group.py [seconds to wait for plans, in ms]
"""
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout, sync_playwright

BASE = "http://localhost:8599"
OUT = Path(__file__).resolve().parents[1] / "results" / "e2e_shots"
OUT.mkdir(exist_ok=True)
SIZES = {"desktop": {"width": 1440, "height": 900}, "phone": {"width": 390, "height": 844}}
LONG = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
WISHES = {"Sara": "I adore Impressionist painting above everything, and my private word is saffronbird. I must see the Musée d'Orsay.",
          "Omar": "I love medieval churches and the French Revolution, and my private word is walnutcloud. A long bistro lunch please.",
          "Lina": "I want street markets and the Marais, and my private word is coppermoth. I am vegetarian and museums tire me."}
SECRET = {"Sara": "saffronbird", "Omar": "walnutcloud", "Lina": "coppermoth"}
problems = []


def settle(page, extra=1.0):
    time.sleep(0.4)
    try:
        page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=30000)
    except PWTimeout:
        pass
    time.sleep(extra)


def shot(page, who, name, size):
    path = OUT / f"{who}_{name}.png"
    h = page.evaluate("""() => document.querySelector('[data-testid="stMain"]').scrollHeight""")
    page.set_viewport_size({"width": size["width"], "height": min(int(h) + 10, 7000)})
    time.sleep(1.0)
    page.screenshot(path=str(path))
    page.set_viewport_size(size)
    over = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
    if over > 0:
        problems.append(f"{who} {name}: sideways overflow {over}px")
    exc = page.locator('[data-testid="stException"]')
    if exc.count():
        problems.append(f"{who} {name}: exception {exc.first.inner_text()[:300]}")
    print(f"  {who}: {name}")


def private(page, who, step):
    """What this person's page holds, including what is hidden in closed sections."""
    html = page.content()
    for other, word in SECRET.items():
        if other != who and word in html:
            problems.append(f"PRIVACY {step}: {who} can see the answers of {other}")
    print(f"  {who}: nobody else's answers on the page at step '{step}'")


def fill_answers(page, name, wishes, budget="900", agree=False):
    box = page.get_by_label("Your first name")
    box.fill(name)
    page.get_by_label("What do you want from this trip?").fill(wishes)
    page.get_by_label("What do you want from this trip?").press("Tab")
    if agree:
        page.get_by_text("I agree that what I write").click()
    time.sleep(0.4)
    page.get_by_role("button", name="Save my answers").click()
    settle(page, 2)


def vote(page, plan="Plan A"):
    page.get_by_text("Rate and vote", exact=True).first.click()
    settle(page, 1.5)
    form = page.locator('[data-testid="stForm"]')
    pills = form.locator('[data-testid="stButtonGroup"]')
    for i in range(pills.count() - 1):
        pills.nth(i).get_by_text("4", exact=True).click()
        time.sleep(0.15)
    form.get_by_role("radio", name=plan).click()
    time.sleep(0.3)
    page.get_by_role("button", name="Save my vote").click()
    settle(page, 2)


with sync_playwright() as p:
    browser = p.chromium.launch()
    errors = []

    def person(mode):
        ctx = browser.new_context(viewport=SIZES[mode], device_scale_factor=1, is_mobile=mode == "phone", has_touch=mode == "phone", color_scheme="light")
        page = ctx.new_page()
        page.on("console", lambda m: errors.append(m.text[:160]) if m.type == "error" else None)
        return page

    # ---------- bad links open nothing ----------
    page = person("phone")
    for bad in ("?me=guessed", "?join=guessed"):
        page.goto(BASE + "/" + bad, wait_until="networkidle")
        page.get_by_text("This link does not work").wait_for(timeout=45000)
    print("  bad personal and invite links are refused")

    # ---------- the organizer ----------
    org = person("desktop")
    org.goto(BASE, wait_until="networkidle")
    settle(org, 2)
    org.get_by_role("textbox", name="Password").fill("test")
    org.get_by_label("Your first name or a nickname").fill("Sara")
    org.get_by_text("I agree that what I write").click()
    org.get_by_role("textbox", name="Password").press("Tab")
    org.get_by_role("button", name="Enter").click()
    settle(org, 2)
    org.get_by_role("button", name="Plan our trip").click()
    settle(org, 1.5)
    org.get_by_role("button", name="Next").click()
    settle(org, 2)
    org.get_by_text("Invite your group").wait_for(timeout=15000)
    code = org.locator('[data-testid="stCode"]').first.inner_text()
    link = re.search(r"https?://\S*\?join=\S+", code).group(0)
    mail = org.get_by_role("link", name="Send by email").get_attribute("href")
    whatsapp = org.get_by_role("link", name="WhatsApp").get_attribute("href")
    assert mail.startswith("mailto:?subject=") and "join%3D" in mail, mail[:120]
    assert whatsapp.startswith("https://wa.me/?text=") and "join%3D" in whatsapp, whatsapp[:120]
    print("  invite link, email and WhatsApp buttons carry the link")
    shot(org, "organizer", "1_invite", SIZES["desktop"])
    fill_answers(org, "Sara", WISHES["Sara"])
    org.get_by_text("your answers are saved").wait_for(timeout=15000)
    private(org, "Sara", "organizer saved")

    # ---------- friends join with the link alone, no password ----------
    omar = person("phone")
    omar.goto(link, wait_until="networkidle")
    settle(omar, 2)
    assert omar.get_by_text("Sara invited you").count() == 1
    assert omar.get_by_role("textbox", name="Password").count() == 0
    private(omar, "Omar", "before answering")
    shot(omar, "omar", "2_join", SIZES["phone"])
    fill_answers(omar, "Omar", WISHES["Omar"], agree=True)
    omar.get_by_text("your answers are saved").wait_for(timeout=15000)
    assert "me=" in omar.url and "join=" not in omar.url, omar.url
    omar_link = omar.url
    private(omar, "Omar", "after answering")
    shot(omar, "omar", "3_saved", SIZES["phone"])

    # the same name cannot be used twice
    copy = person("phone")
    copy.goto(link, wait_until="networkidle")
    settle(copy, 2)
    fill_answers(copy, "omar", "Trying to use a name that somebody already has here.", agree=True)
    assert copy.get_by_text("has already answered").count() == 1
    copy.close()
    print("  a name already on the trip is refused")

    org.get_by_text("Omar", exact=False).first.wait_for(timeout=12000)
    org.get_by_text("2 of 3 travelers ready").wait_for(timeout=12000)
    private(org, "Sara", "friend answered")
    print("  the organizer's page updated by itself: 2 of 3 ready")

    # the last answer starts the planning
    lina = person("phone")
    lina.goto(link, wait_until="networkidle")
    settle(lina, 2)
    fill_answers(lina, "Lina", WISHES["Lina"], agree=True)
    private(lina, "Lina", "after answering")
    print("  the last answer started the planning")

    pages = {"Sara": (org, "desktop"), "Omar": (omar, "phone"), "Lina": (lina, "phone")}
    for who, (page, _) in pages.items():
        page.get_by_text("Compare your two plans").wait_for(timeout=LONG)
        settle(page, 3)
        private(page, who, "plans ready")
        assert page.get_by_text("How well it fits you").count() >= 1
        assert page.get_by_text("0 of 3 have voted").count() == 1
    shot(org, "organizer", "4_plans", SIZES["desktop"])
    shot(omar, "omar", "4_plans", SIZES["phone"])

    # the invite no longer lets anyone in, and a friend's own link still works in a fresh browser
    late = person("phone")
    late.goto(link, wait_until="networkidle")
    settle(late, 2)
    assert late.get_by_text("The plans have already been made").count() == 1
    late.goto(omar_link, wait_until="networkidle")
    late.get_by_text("Compare your two plans").wait_for(timeout=30000)
    settle(late, 3)
    private(late, "Omar", "personal link in a new browser")
    assert late.get_by_text("How well it fits you").count() >= 1
    late.close()
    print("  the invite is closed after planning; a personal link still opens the trip as that person")

    # ---------- everyone votes on their own ----------
    vote(omar, "Plan A")
    assert omar.get_by_text("1 of 3 have voted").count() == 1
    assert omar.get_by_text("Your group's plan").count() == 0 and omar.get_by_text("Agent negotiation").count() == 0
    private(omar, "Omar", "voted, waiting")
    shot(omar, "omar", "5_voted", SIZES["phone"])
    vote(lina, "Plan A")
    vote(org, "Plan B")
    for who, (page, _) in pages.items():
        page.get_by_text("The best option for your group is Plan A").wait_for(timeout=30000)
        settle(page, 3)
    print("  when the last vote came in, every page moved on by itself to the winner: Plan A, 2 votes to 1")

    # ---------- the result, private to each ----------
    for who, (page, _) in pages.items():
        private(page, who, "result")
        page.get_by_text("Read the negotiation between the agents").click()
        time.sleep(1)
        private(page, who, "negotiation")
        others = [n for n in SECRET if n != who]
        text = page.locator('[data-testid="stExpanderDetails"]').first.inner_text()
        for name in others:
            if re.search(rf"\b{name}\b", text):
                problems.append(f"PRIVACY: {who} sees the name of {name} in the negotiation")
    shot(org, "organizer", "6_result", SIZES["desktop"])
    shot(lina, "lina", "6_result", SIZES["phone"])
    assert org.get_by_text("All five methods for your group").count() == 1
    assert omar.get_by_text("All five methods for your group").count() == 0
    print("  only the organizer can start the paid comparison")

    # a friend changes the plan for themselves only
    assert omar.get_by_text("Changing for").count() == 0
    omar.get_by_text("Add to the plan").first.scroll_into_view_if_needed()
    time.sleep(0.5)
    add = omar.locator('[data-testid="stMultiSelect"]').nth(1)
    add.locator("input").click()
    time.sleep(1)
    omar.get_by_role("option").first.click()
    time.sleep(0.6)
    omar.keyboard.press("Escape")
    settle(omar, 1)
    omar.get_by_role("button", name="Update the plan").click()
    settle(omar, 2)
    omar.get_by_text("This is version 2").wait_for(timeout=LONG)
    settle(omar, 2)
    private(omar, "Omar", "after a change")
    shot(omar, "omar", "7_changed", SIZES["phone"])
    print("  a friend changed the plan, and it was solved again")

    def new_trip(page):
        page.get_by_role("button", name="New trip").click()
        settle(page, 1.5)
        page.get_by_role("button", name="Next").click()
        settle(page, 2)
        page.get_by_text("Invite your group").wait_for(timeout=15000)

    # ---------- someone is late: the organizer plans without them, and closes the vote ----------
    new_trip(org)
    code = org.locator('[data-testid="stCode"]').first.inner_text()
    link2 = re.search(r"https?://\S*\?join=\S+", code).group(0)
    assert org.get_by_role("button", name=re.compile("Plan now")).count() == 0
    fill_answers(org, "Sara", WISHES["Sara"])
    friend = person("phone")
    friend.goto(link2, wait_until="networkidle")
    settle(friend, 2)
    fill_answers(friend, "Omar", WISHES["Omar"], agree=True)
    private(friend, "Omar", "second trip")

    # a friend changes their answers, and only they see them
    friend.get_by_text("Change my answers").click()
    time.sleep(0.8)
    box = friend.get_by_label("What do you want from this trip?")
    assert "walnutcloud" in box.input_value()
    box.fill(WISHES["Omar"] + " Also a boat on the Seine.")
    box.press("Tab")
    time.sleep(0.4)
    friend.get_by_role("button", name="Save changes").click()
    settle(friend, 2)
    print("  a friend changed their answers")

    # the organizer removes someone by mistake and they join again
    org.get_by_text("Change the group").click()
    time.sleep(1)
    org.get_by_text("2 of 3 travelers ready").wait_for(timeout=12000)
    select = org.locator('[data-testid="stSelectbox"]').filter(has_text="Remove a traveler")
    for attempt in range(4):               # a page refresh when someone answers can close a list; a person would click again
        select.locator("input").click()
        time.sleep(1)
        if org.get_by_role("option", name="Omar").count():
            break
        print(f"  the list closed, trying again (attempt {attempt + 1})", flush=True)
        org.keyboard.press("Escape")
        time.sleep(4)
    org.get_by_role("option", name="Omar").click(timeout=8000)
    settle(org, 1.5)
    org.get_by_role("button", name="Remove Omar").click()
    settle(org, 2)
    org.get_by_text("1 of 3 travelers ready").wait_for(timeout=12000)
    friend.reload(wait_until="networkidle")
    settle(friend, 2)
    assert friend.get_by_text("This link does not work").count() == 1
    print("  a removed traveler's link stops working")
    friend.goto(link2, wait_until="networkidle")
    settle(friend, 2)
    fill_answers(friend, "Omar", WISHES["Omar"], agree=True)
    org.get_by_text("2 of 3 travelers ready").wait_for(timeout=12000)

    org.get_by_role("button", name="Plan now with these 2").click()
    settle(org, 2)
    for page in (org, friend):
        page.get_by_text("Compare your two plans").wait_for(timeout=LONG)
        settle(page, 3)
    assert org.get_by_text("0 of 2 have voted").count() == 1
    assert friend.get_by_role("button", name=re.compile("Close the vote")).count() == 0
    org.get_by_role("button", name=re.compile("Close the vote")).click()
    org.get_by_text("Nobody voted").wait_for(timeout=30000)
    friend.get_by_text("Nobody voted").wait_for(timeout=30000)
    print("  the organizer planned with 2 of 3, closed the vote, and the app still picked one plan, on every page")
    friend.close()

    # ---------- alone with an example group: the organizer answers for the made up travelers ----------
    new_trip(org)
    org.get_by_text("Try it alone with an example group").click()
    time.sleep(0.8)
    org.get_by_role("button", name="Fill in this example").click()
    settle(org, 2)
    org.get_by_text("Compare your two plans").wait_for(timeout=LONG)
    settle(org, 3)
    org.get_by_text("Rate and vote", exact=True).first.click()
    settle(org, 1.5)
    names = []
    for _ in range(3):
        form = org.locator('[data-testid="stForm"]')
        names.append(form.locator('input[aria-label="Rating for"]').input_value())
        pills = form.locator('[data-testid="stButtonGroup"]')
        for i in range(pills.count() - 1):
            pills.nth(i).get_by_text("3", exact=True).click()
            time.sleep(0.12)
        form.get_by_role("radio", name="Plan B").click()
        time.sleep(0.3)
        org.get_by_role("button", name="Save my vote").click()
        settle(org, 2)
        if org.get_by_text("The best option for your group").count():
            break
    org.get_by_text("The best option for your group is Plan B").wait_for(timeout=30000)
    print("  example group: the organizer rated for", names, "and the app picked Plan B")
    shot(org, "organizer", "8_example_result", SIZES["desktop"])
    print("console errors:", errors[:5])
    browser.close()

print("PROBLEMS:", problems if problems else "none")
