"""The look of the app, taken from the Versailles app: warm beige, gold accents, Cinzel headings, Satoshi text,
black buttons, white rounded cards, numbered steps and a gold line drawing."""
import base64

import streamlit as st

from ui.skyline import EMBLEM, svg

APP_NAME = "Paris Ensemble"
TAGLINE = "Group trip planner"

CSS = """
:root{--pe-bg:#F1EFEA;--pe-gold:#BA9E66;--pe-gold-ink:#8C7340;--pe-gold-soft:rgba(186,158,102,.15);--pe-ink:#0A0A0A;--pe-black:#171717;
--pe-muted:#68655D;--pe-line:#E0DDD6;--pe-soft:#F7F6F2;--pe-card:#FFFFFF;--pe-warn:#8A5A12;--pe-warn-bg:#FBF3E2;--pe-bad:#9B3B2E;--pe-good:#3F6B4F}

/* Streamlit's own chrome goes away */
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],
[data-testid="stSidebarCollapsedControl"],footer,#MainMenu{display:none!important}
html,body{background:var(--pe-bg)}
[data-testid="stAppViewContainer"]{background:var(--pe-bg)}
[data-testid="stMainBlockContainer"]{max-width:1120px;padding:0 24px 96px;position:relative;z-index:1}
[data-testid="stElementContainer"]:has(> [data-testid="stHtml"]:empty){display:none}
a{color:var(--pe-gold-ink)}

/* Top bar */
.st-key-topbar{position:sticky;top:0;z-index:1000;margin:0 0 8px calc(50% - 50vw)!important;width:100vw!important;max-width:100vw!important;padding:8px max(24px,calc(50vw - 536px));
background:rgba(241,239,234,.88);-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);border-bottom:1px solid var(--pe-line)}
.pe-brand{display:flex;align-items:center;gap:10px;min-width:0}
.pe-brand .pe-emblem{width:30px;height:30px;color:var(--pe-gold);flex:none}
.pe-brand-name{font-family:Cinzel,Georgia,serif;font-size:19px;letter-spacing:.04em;line-height:1.1;color:var(--pe-ink);white-space:nowrap}
.pe-brand-tag{font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--pe-muted);white-space:nowrap}
.st-key-topbar button{min-height:34px!important;padding:2px 12px!important;font-size:13.5px!important}
.st-key-topbar p{font-size:13.5px!important}
.st-key-nav_narrow{display:none!important}

/* Text */
.pe-eyebrow{font-size:12px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--pe-gold-ink);margin:0 0 12px}
.pe-h1{font-family:Cinzel,Georgia,serif;font-weight:400;font-size:clamp(34px,6vw,60px);line-height:1.12;letter-spacing:.01em;margin:0;color:var(--pe-ink)}
.pe-h2{font-family:Cinzel,Georgia,serif;font-weight:400;font-size:clamp(25px,3.6vw,36px);line-height:1.18;margin:0 0 6px;color:var(--pe-ink)}
.pe-h3{font-size:19px;font-weight:700;margin:0 0 4px;color:var(--pe-ink)}
.pe-sub{font-size:17px;line-height:1.55;color:var(--pe-muted);margin:14px 0 0}
.pe-small{font-size:13px;line-height:1.5;color:var(--pe-muted)}
.pe-center{text-align:center}
.pe-hero{padding:clamp(34px,7vh,72px) 0 22px;max-width:760px;margin:0 auto;text-align:center}
.pe-page-title{padding:8px 0 6px}
.pe-section{margin:30px 0 10px}

/* Steps, as on the Versailles onboarding */
.pe-steps{display:flex;align-items:center;justify-content:center;gap:0;margin:20px auto 22px;max-width:520px}
.pe-step{width:32px;height:32px;flex:none;border-radius:50%;display:grid;place-items:center;font-size:14px;font-weight:600;background:#fff;color:var(--pe-muted);box-shadow:0 0 0 1px var(--pe-line)}
.pe-step.on{background:var(--pe-black);color:#fff;box-shadow:none}
.pe-step.done{background:var(--pe-gold);color:#fff;box-shadow:none}
.pe-step-line{height:1px;width:clamp(16px,6vw,56px);background:var(--pe-line)}

/* Cards */
[class*="st-key-card"]{background:var(--pe-card);border:1px solid var(--pe-line);border-radius:16px;padding:20px 22px;box-shadow:0 1px 3px rgba(23,23,23,.05)}
.st-key-narrow{max-width:560px;margin-inline:auto}
.pe-card{background:var(--pe-card);border:1px solid var(--pe-line);border-radius:16px;padding:20px 22px;box-shadow:0 1px 3px rgba(23,23,23,.05)}
.pe-grid4{display:grid;gap:14px;grid-template-columns:repeat(4,minmax(0,1fr))}
.pe-grid3{display:grid;gap:14px;grid-template-columns:repeat(3,minmax(0,1fr))}
.pe-grid2{display:grid;gap:14px;grid-template-columns:repeat(2,minmax(0,1fr))}
.pe-feature-icon{width:44px;height:44px;border-radius:12px;display:grid;place-items:center;background:var(--pe-gold-soft);color:var(--pe-gold-ink);margin-bottom:14px}
.pe-num{width:34px;height:34px;border-radius:50%;background:var(--pe-black);color:#fff;display:grid;place-items:center;font-size:14px;font-weight:600;margin-bottom:12px}
.pe-card p{margin:0;font-size:14.5px;line-height:1.55;color:var(--pe-muted)}

/* Buttons and inputs */
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-primaryFormSubmit"],[data-testid="stBaseButton-secondaryFormSubmit"]{min-height:44px;font-weight:600;border-radius:10px;transition:transform .08s ease,box-shadow .12s ease}
[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primaryFormSubmit"]:hover{box-shadow:0 4px 14px rgba(23,23,23,.18)}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-secondaryFormSubmit"]{background:#fff}
[data-testid="stWidgetLabel"] p{font-weight:500;font-size:14.5px}
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{background:#fff!important;border-radius:10px!important}
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{font-size:16px!important}   /* 16px stops iPhones zooming into a field */
.stTextArea textarea{line-height:1.5}
[data-testid="stForm"]{border:none;padding:0}

/* Segmented control and pills */
button[data-variant="segmented_control"]{min-height:40px;font-weight:600;background:#fff}
button[data-variant="segmented_control"][data-selected="true"],button[data-variant="pills"][data-selected="true"]{background:var(--pe-black)!important;border-color:var(--pe-black)!important;color:#fff!important}
button[data-variant="segmented_control"][data-selected="true"] p,button[data-variant="pills"][data-selected="true"] p{color:#fff!important}
button[data-variant="pills"]{min-width:42px;min-height:38px;justify-content:center;background:#fff}
[data-testid="stExpander"] details{background:#fff;border-radius:14px;border-color:var(--pe-line)}

/* Notices */
.pe-banner{margin:10px 0 6px;padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.5;background:var(--pe-warn-bg);border:1px solid #EBD9B4;color:var(--pe-warn)}
.pe-note{padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.5;background:var(--pe-soft);border:1px solid var(--pe-line);color:var(--pe-muted);margin:8px 0}
.pe-chip{display:inline-flex;align-items:center;gap:6px;padding:4px 11px;border-radius:999px;font-size:13px;font-weight:500;background:#fff;border:1px solid var(--pe-line);color:var(--pe-muted);margin:0 6px 6px 0;white-space:nowrap}
.pe-chip.ok{background:var(--pe-gold-soft);border-color:transparent;color:var(--pe-gold-ink)}

/* KPI tiles */
.pe-kpi{background:#fff;border:1px solid var(--pe-line);border-radius:14px;padding:14px 16px;min-width:0}
.pe-i{vertical-align:-2px;flex:none}
.pe-kpi-label{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--pe-muted);margin-bottom:4px}
.pe-kpi-value{font-size:24px;font-weight:700;letter-spacing:-.01em;color:var(--pe-ink);line-height:1.15}
.pe-kpi-foot{font-size:12px;color:var(--pe-muted);margin-top:3px}

/* Timeline, as on the Versailles itinerary */
.pe-day{background:#fff;border:1px solid var(--pe-line);border-radius:16px;margin:0 0 14px;overflow:hidden}
.pe-day>summary{list-style:none;cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:15px 18px}
.pe-day>summary::-webkit-details-marker{display:none}
.pe-day-title{display:flex;align-items:center;gap:10px;font-weight:700;font-size:16.5px;color:var(--pe-ink)}
.pe-day-dot{width:13px;height:13px;border-radius:50%;flex:none}
.pe-day-meta{font-size:13px;color:var(--pe-muted);white-space:nowrap}
.pe-chev{transition:transform .15s ease;color:var(--pe-muted);flex:none}
details[open]>summary .pe-chev{transform:rotate(180deg)}
.pe-stops{padding:2px 18px 12px}
.pe-stop{display:grid;grid-template-columns:26px minmax(0,1fr);column-gap:12px}
.pe-rail{display:flex;flex-direction:column;align-items:center}
.pe-dot{width:24px;height:24px;border-radius:50%;display:grid;place-items:center;color:#fff;font-size:11.5px;font-weight:700;flex:none;margin-top:13px}
.pe-dot.soft{background:#fff!important;color:var(--pe-muted);box-shadow:inset 0 0 0 1.5px var(--pe-line)}
.pe-rail-line{flex:1;width:1.5px;background:var(--pe-line);min-height:10px}
.pe-stop-body{min-width:0;padding-bottom:6px}
.pe-travel{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--pe-muted);padding:7px 0 0}
.pe-free{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--pe-gold-ink);padding:7px 0 0;font-weight:500}
.pe-place{border-radius:12px;transition:background .12s ease}
.pe-place>summary{list-style:none;cursor:pointer;padding:9px 10px;margin:0 -10px;border-radius:12px}
.pe-place>summary::-webkit-details-marker{display:none}
.pe-place>summary:hover{background:var(--pe-soft)}
.pe-place-top{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.pe-time{font-size:12.5px;font-weight:600;color:var(--pe-gold-ink);white-space:nowrap}
.pe-name{font-size:16px;font-weight:600;color:var(--pe-ink);line-height:1.3;margin-top:1px;overflow-wrap:anywhere}
.pe-meta{font-size:13px;color:var(--pe-muted);margin-top:2px;line-height:1.4}
.pe-more{padding:8px 0 12px;font-size:14px;line-height:1.55;color:#3D3B36;max-width:640px}
.pe-photo{width:100%;max-height:300px;object-fit:cover;border-radius:12px;display:block;background:var(--pe-soft)}
.pe-credit{font-size:11px;color:var(--pe-muted);margin:4px 0 8px}
.pe-credit a{color:var(--pe-muted)}
.pe-more p{margin:0 0 8px}
.pe-tip{background:var(--pe-gold-soft);border-radius:10px;padding:8px 11px;margin:0 0 8px;color:#5E4C24;font-size:13.5px}
.pe-links{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.pe-link{display:inline-flex;align-items:center;gap:5px;font-size:13px;font-weight:500;padding:6px 11px;border-radius:9px;border:1px solid var(--pe-line);background:#fff;color:var(--pe-ink)!important;text-decoration:none!important}
.pe-link:hover{background:var(--pe-soft)}

/* Fit bars */
.pe-fit{padding:12px 0;border-top:1px solid var(--pe-line)}
.pe-fit:first-child{border-top:none;padding-top:2px}
.pe-fit-top{display:flex;justify-content:space-between;align-items:baseline;gap:10px;font-weight:600;font-size:15px}
.pe-fit-pct{font-weight:700}
.pe-bar{height:7px;border-radius:99px;background:#ECEAE4;overflow:hidden;margin:6px 0 7px}
.pe-bar>span{display:block;height:100%;border-radius:99px;background:var(--pe-gold)}
.pe-fit p{margin:0;font-size:14px;line-height:1.5;color:var(--pe-muted)}

/* Tables */
.pe-table{width:100%;border-collapse:collapse;font-size:14px}
.pe-table th{text-align:left;font-size:12px;font-weight:600;color:var(--pe-muted);padding:8px 10px;border-bottom:1px solid var(--pe-line);white-space:nowrap}
.pe-table td{padding:9px 10px;border-bottom:1px solid var(--pe-line);color:var(--pe-ink)}
.pe-table tr:last-child td{border-bottom:none}
.pe-table .num{text-align:right;font-variant-numeric:tabular-nums}
.pe-table tr.ours td{background:var(--pe-gold-soft);font-weight:600}
.pe-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.pe-cost{display:flex;justify-content:space-between;gap:10px;font-size:14px;padding:6px 0;border-bottom:1px solid var(--pe-line);color:var(--pe-muted)}
.pe-cost b{color:var(--pe-ink);font-weight:600}
.pe-cost:last-child{border-bottom:none}

/* Negotiation log */
.pe-round{border-left:2px solid var(--pe-gold);padding:2px 0 2px 14px;margin:0 0 18px}
.pe-round h4{margin:0 0 6px;font-size:15px;font-weight:700}
.pe-round p{margin:0 0 6px;font-size:14px;line-height:1.5;color:var(--pe-muted)}
.pe-round p b{color:var(--pe-ink);font-weight:600}
.pe-mediator{background:var(--pe-soft);border-radius:10px;padding:8px 11px;font-size:13.5px;color:#3D3B36}

.pe-footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--pe-line);font-size:12.5px;line-height:1.6;color:var(--pe-muted)}
.pe-footer a{color:var(--pe-muted)}

/* Maps */
[data-testid="stCustomComponentV1"] iframe,iframe[title="streamlit_folium.st_folium"]{border-radius:14px;border:1px solid var(--pe-line)}

/* Phones */
@media (max-width:760px){
  [data-testid="stMainBlockContainer"]{padding:0 14px 72px}
  .st-key-topbar{padding:6px 14px}
  .st-key-topbar>div{flex-wrap:nowrap!important}
  .st-key-nav_wide{display:none!important}
  .st-key-nav_narrow{display:block!important;width:auto!important;flex:none!important}
  .pe-brand-tag{display:none}
  .pe-grid4{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
  .pe-grid3,.pe-grid2{grid-template-columns:minmax(0,1fr);gap:10px}
  .pe-sub{font-size:15.5px}
  .pe-kpi{padding:11px 12px}
  .pe-kpi-value{font-size:20px}
  [class*="st-key-card"],.pe-card{padding:16px}
  .pe-stops{padding:0 12px 10px}
  .pe-day>summary{padding:13px 14px}
  .pe-day-meta{display:none}
  .pe-table{font-size:13px}
}
@media print{.st-key-topbar{display:none!important}}
"""


def inject() -> None:
    st.html(f"<style>{CSS}</style>")


def skyline_background() -> None:
    """The gold line drawing along the bottom of the window, as on the Versailles pages."""
    data = base64.b64encode(svg().encode()).decode()
    st.html(f"""<style>
    [data-testid="stAppViewContainer"]{{background:var(--pe-bg) url("data:image/svg+xml;base64,{data}") no-repeat center bottom;
      background-size:max(1500px,100vw) auto;background-attachment:fixed}}
    @media (max-width:760px){{[data-testid="stAppViewContainer"]{{background-size:980px auto}}}}
    </style>""")


def emblem_img(size: int = 30) -> str:
    """An embedded image: Streamlit's HTML sanitizer removes inline SVG but keeps images."""
    svg_text = EMBLEM.replace('<svg ', f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" ').replace("currentColor", "#BA9E66")
    data = base64.b64encode(svg_text.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" alt=""/>'


def brand() -> str:
    return (f'<div class="pe-brand"><div class="pe-emblem">{emblem_img()}</div>'
            f'<div><div class="pe-brand-name">{APP_NAME}</div><div class="pe-brand-tag">{TAGLINE}</div></div></div>')
