"""Paris Ensemble: plan a group trip to Paris that everyone can agree on.

Run locally:   streamlit run app.py
Settings come from .env locally and from the app secrets on Streamlit Community Cloud.
"""
import os

import streamlit as st

st.set_page_config(page_title="Paris Ensemble", page_icon=":material/travel_explore:", layout="wide", initial_sidebar_state="collapsed")

# Secrets must be environment variables before anything in src/ is imported.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass  # no secrets file locally: settings come from .env

from ui.pages import run  # noqa: E402

run()
