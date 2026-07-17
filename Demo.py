"""NBA Game Predictor — interactive demo.

Pick two teams. The model spits out a win probability and shows you which
features pushed it that way.

Run locally:
    streamlit run Demo.py
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

st.set_page_config(page_title="NBA Game Predictor", layout="centered")


# --- model + data ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODELS / "xgb_baseline.pkl")


@st.cache_data
def load_games():
    df = pd.read_parquet(DATA / "games_with_features.parquet")
    return df.sort_values("gameDate").reset_index(drop=True)


@st.cache_data
def current_teams(df):
    """Active franchises: anything that played in the last 3 seasons."""
    recent = df[df.season >= df.season.max() - 3]
    home = recent[["hometeamId", "hometeamName"]].rename(
        columns={"hometeamId": "teamId", "hometeamName": "name"})
    away = recent[["awayteamId", "awayteamName"]].rename(
        columns={"awayteamId": "teamId", "awayteamName": "name"})
    teams = (pd.concat([home, away])
               .drop_duplicates("teamId")
               .sort_values("name"))
    return teams.set_index("name")["teamId"].to_dict()


model = load_model()
df = load_games()
feature_cols = list(model.feature_names_in_)
team_map = current_teams(df)


# --- helpers -----------------------------------------------------------------
def latest_team_snapshot(team_id, df):
    """The team's most recent feature snapshot, regardless of whether they were
    home or away in that game. Returns a dict of role-neutral features."""
    games = df[(df.hometeamId == team_id) | (df.awayteamId == team_id)]
    if len(games) == 0:
        return None
    last = games.iloc[-1]
    if last.hometeamId == team_id:
        return {
            "elo": last.home_elo_pre,
            "win_rate_5": last.home_win_rate_last_5,
            "win_rate_10": last.home_win_rate_last_10,
            "win_rate_20": last.home_win_rate_last_20,
            "avg_margin_5": last.home_avg_margin_last_5,
            "avg_margin_10": last.home_avg_margin_last_10,
            "avg_margin_20": last.home_avg_margin_last_20,
            "days_rest": last.home_days_since_last_game,
            "is_b2b": last.home_is_back_to_back,
            "last_date": last.gameDate,
        }
    else:
        return {
            "elo": last.away_elo_pre,
            "win_rate_5": last.away_win_rate_last_5,
            "win_rate_10": last.away_win_rate_last_10,
            "win_rate_20": last.away_win_rate_last_20,
            "avg_margin_5": last.away_avg_margin_last_5,
            "avg_margin_10": last.away_avg_margin_last_10,
            "avg_margin_20": last.away_avg_margin_last_20,
            "days_rest": last.away_days_since_last_game,
            "is_b2b": last.away_is_back_to_back,
            "last_date": last.gameDate,
        }


def h2h_rate(home_id, away_id, df, k=5):
    """Last k meetings between the two teams. Returns share of home_id wins."""
    pair_games = df[((df.hometeamId == home_id) & (df.awayteamId == away_id)) |
                    ((df.hometeamId == away_id) & (df.awayteamId == home_id))]
    if len(pair_games) == 0:
        return 0.5
    recent = pair_games.tail(k)
    home_wins = ((recent.hometeamId == home_id) & (recent.home_win == 1)).sum() + \
                ((recent.awayteamId == home_id) & (recent.home_win == 0)).sum()
    return float(home_wins) / len(recent)


def build_feature_row(home_id, away_id, df, days_rest_home=2, days_rest_away=2):
    """Assemble the 27-feature row the model expects."""
    h = latest_team_snapshot(home_id, df)
    a = latest_team_snapshot(away_id, df)
    if h is None or a is None:
        return None
    row = {
        "home_elo_pre": h["elo"],
        "away_elo_pre": a["elo"],
        "elo_diff": h["elo"] - a["elo"],
        "h2h_home_winrate_last5": h2h_rate(home_id, away_id, df),
        "home_win_rate_last_5": h["win_rate_5"],
        "home_win_rate_last_10": h["win_rate_10"],
        "home_win_rate_last_20": h["win_rate_20"],
        "away_win_rate_last_5": a["win_rate_5"],
        "away_win_rate_last_10": a["win_rate_10"],
        "away_win_rate_last_20": a["win_rate_20"],
        "win_rate_diff_5": h["win_rate_5"] - a["win_rate_5"],
        "win_rate_diff_10": h["win_rate_10"] - a["win_rate_10"],
        "win_rate_diff_20": h["win_rate_20"] - a["win_rate_20"],
        "home_avg_margin_last_5": h["avg_margin_5"],
        "home_avg_margin_last_10": h["avg_margin_10"],
        "home_avg_margin_last_20": h["avg_margin_20"],
        "away_avg_margin_last_5": a["avg_margin_5"],
        "away_avg_margin_last_10": a["avg_margin_10"],
        "away_avg_margin_last_20": a["avg_margin_20"],
        "margin_diff_5": h["avg_margin_5"] - a["avg_margin_5"],
        "margin_diff_10": h["avg_margin_10"] - a["avg_margin_10"],
        "margin_diff_20": h["avg_margin_20"] - a["avg_margin_20"],
        "home_days_since_last_game": float(days_rest_home),
        "away_days_since_last_game": float(days_rest_away),
        "home_is_back_to_back": int(days_rest_home <= 1),
        "away_is_back_to_back": int(days_rest_away <= 1),
        "rest_diff": float(days_rest_home - days_rest_away),
    }
    return pd.DataFrame([row])[feature_cols]


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# --- UI ----------------------------------------------------------------------
st.title("NBA Game Predictor")
st.caption("Pick two teams, see the model's call. The bar chart shows which features pushed the prediction.")

# Sidebar
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "This is the XGBoost baseline from NB_03 of the project (27 features: ELO, "
        "rolling form 5/10/20, head-to-head, rest days). Trained on every NBA "
        "game 1946–2018, accuracy 64.1% on 2019+ holdout."
    )
    st.markdown(
        "**Caveat:** the model is *not* great. ~65% is the realistic ceiling on "
        "pure team-level history. To beat it you'd need real-time injury data."
    )
    st.markdown("---")
    st.markdown("[Source on GitHub](https://github.com/klp-data/nba-game-predictor)")

# Main pickers
team_names = sorted(team_map.keys())
col_h, col_a = st.columns(2)
home_name = col_h.selectbox("Home team", team_names, index=team_names.index("Celtics") if "Celtics" in team_names else 0)
away_name = col_a.selectbox("Away team", team_names, index=team_names.index("Lakers") if "Lakers" in team_names else 1)

# Optional rest days
with st.expander("Tweak rest days (advanced)"):
    col_r1, col_r2 = st.columns(2)
    rest_h = col_r1.slider("Home rest (days)", 0, 7, 2)
    rest_a = col_r2.slider("Away rest (days)", 0, 7, 2)

if home_name == away_name:
    st.warning("Pick two different teams.")
    st.stop()

home_id = team_map[home_name]
away_id = team_map[away_name]

X = build_feature_row(home_id, away_id, df, rest_h, rest_a)
if X is None or X.isna().any().any():
    st.error("Not enough data for one of these teams (probably too few recent games). Try another pair.")
    st.stop()

proba_home = float(model.predict_proba(X)[0, 1])
proba_away = 1.0 - proba_home


# Headline
col1, col2, col3 = st.columns([1, 1, 1])
col1.metric(f"P({home_name} win)", f"{proba_home:.1%}")
col2.metric(f"P({away_name} win)", f"{proba_away:.1%}")
favorite = home_name if proba_home >= 0.5 else away_name
col3.metric("Pick", favorite)

# ELO comparison
st.subheader("ELO snapshot")
elo_h = float(X.iloc[0]["home_elo_pre"])
elo_a = float(X.iloc[0]["away_elo_pre"])
col_eh, col_ea, col_ed = st.columns(3)
col_eh.metric(f"{home_name} ELO", f"{elo_h:.0f}")
col_ea.metric(f"{away_name} ELO", f"{elo_a:.0f}")
col_ed.metric("ELO diff (home - away)", f"{elo_h - elo_a:+.0f}", help="Positive = home favored on ELO alone")


# SHAP attribution
st.subheader("Why the model picked that")
st.caption("Each bar = one feature's SHAP value. Blue pushes toward home win, red pushes toward away win. "
           "Top 10 features by absolute contribution.")

explainer = get_explainer(model)
shap_values = explainer.shap_values(X)
if hasattr(shap_values, 'ndim') and shap_values.ndim == 3:   # binary classifier output
    shap_values = shap_values[:, :, 1]
shap_arr = np.asarray(shap_values)[0]

contribs = pd.DataFrame({
    "feature": feature_cols,
    "shap": shap_arr,
    "value": X.iloc[0].values,
}).assign(abs_shap=lambda d: d.shap.abs()).sort_values("abs_shap", ascending=True).tail(10)

# dark, matches the .streamlit theme
with plt.rc_context({"font.family": "monospace", "text.color": "#d5d9e0",
                     "axes.edgecolor": "#3a3f4a", "xtick.color": "#9aa0aa",
                     "ytick.color": "#d5d9e0"}):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#0a0c10")
    ax.set_facecolor("#0a0c10")
    colors = ["#4c9be8" if s > 0 else "#f2545b" for s in contribs.shap]
    labels = [f"{f} = {v:.2f}" for f, v in zip(contribs.feature, contribs.value)]
    ax.barh(labels, contribs.shap, color=colors)
    ax.axvline(0, color="#9aa0aa", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("SHAP value (→ home win)", fontsize=9)
    ax.set_title(f"Top 10 contributors: {home_name} vs {away_name}", fontsize=10)
    plt.tight_layout(pad=1.2)
st.pyplot(fig)


# small footnote
st.markdown("---")
st.caption(
    f"Feature values are each team's most recent snapshot from the historical dataset "
    f"(home team's last home game, away team's last away game). For new matchups in the upcoming "
    f"season, those snapshots are close to a team's current state. Last game date in dataset: "
    f"{df.gameDate.max().date()}."
)