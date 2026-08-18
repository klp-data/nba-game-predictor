"""NBA Game Predictor dashboard.

Run it locally from the repo root:

    streamlit run dashboard/app.py

Everything the pages read sits in dashboard/data/ and dashboard/models/, both
committed, so this runs without the Kaggle download and without data/processed.
Rebuild those files with scripts/build_dashboard_data.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = Path(__file__).resolve().parent
for p in (ROOT, DASHBOARD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

st.set_page_config(page_title="NBA Game Predictor", layout="wide")

pages = [
    st.Page("pages/matchup.py", title="Matchup", url_path="matchup", default=True),
    st.Page("pages/elo.py", title="ELO explorer", url_path="elo"),
    st.Page("pages/playoffs.py", title="Playoffs", url_path="playoffs"),
    st.Page("pages/model_report.py", title="Model report", url_path="model-report"),
]

with st.sidebar:
    st.markdown("### NBA Game Predictor")
    st.caption("Predicting NBA games, series and titles from 80 years of results.")
    st.markdown("---")
    st.caption("[Source on GitHub](https://github.com/klp-data/nba-game-predictor)")

st.navigation(pages).run()
