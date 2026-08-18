"""Matchup page: pick two teams, get a win probability and the SHAP breakdown.

The feature row is assembled the same way Demo.py did it, but from the small
snapshot files instead of the full feature parquet.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

import charts

ELO_LIKE = {"home_elo_pre", "away_elo_pre", "elo_diff"}
COUNT_LIKE = {"home_days_since_last_game", "away_days_since_last_game",
              "home_is_back_to_back", "away_is_back_to_back", "rest_diff"}


@st.cache_resource
def load_model():
    return joblib.load(charts.MODELS / charts.meta()["model_file"])


@st.cache_data
def load_snapshots() -> pd.DataFrame:
    return pd.read_parquet(charts.DATA / "team_snapshots.parquet")


@st.cache_data
def load_h2h() -> pd.DataFrame:
    return pd.read_parquet(charts.DATA / "h2h.parquet")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


def feature_row(home, away, h2h_rate: float, rest_home: int, rest_away: int,
                feature_cols: list[str]) -> pd.DataFrame:
    """The 27 features the model wants, from two team snapshots."""
    row = {
        "home_elo_pre": home.elo,
        "away_elo_pre": away.elo,
        "elo_diff": home.elo - away.elo,
        "h2h_home_winrate_last5": h2h_rate,
        "home_days_since_last_game": float(rest_home),
        "away_days_since_last_game": float(rest_away),
        "home_is_back_to_back": int(rest_home <= 1),
        "away_is_back_to_back": int(rest_away <= 1),
        "rest_diff": float(rest_home - rest_away),
    }
    for w in (5, 10, 20):
        row[f"home_win_rate_last_{w}"] = getattr(home, f"win_rate_{w}")
        row[f"away_win_rate_last_{w}"] = getattr(away, f"win_rate_{w}")
        row[f"win_rate_diff_{w}"] = getattr(home, f"win_rate_{w}") - getattr(away, f"win_rate_{w}")
        row[f"home_avg_margin_last_{w}"] = getattr(home, f"avg_margin_{w}")
        row[f"away_avg_margin_last_{w}"] = getattr(away, f"avg_margin_{w}")
        row[f"margin_diff_{w}"] = getattr(home, f"avg_margin_{w}") - getattr(away, f"avg_margin_{w}")
    return pd.DataFrame([row])[feature_cols]


def label(feature: str, value: float) -> str:
    """Feature name and its value, rounded to something readable."""
    if feature in ELO_LIKE:
        return f"{feature} = {value:.0f}"
    if feature in COUNT_LIKE:
        return f"{feature} = {value:g}"
    return f"{feature} = {value:.2f}"


model = load_model()
snapshots = load_snapshots()
h2h = load_h2h()
feature_cols = list(model.feature_names_in_)
by_name = snapshots.set_index("teamName")
names = sorted(by_name.index)


st.title("Matchup")
st.caption(
    f"This is the XGBoost model from notebook 03. Trained with 27 features on every game "
    f"between 1946 and 2018. Performance: {charts.holdout_accuracy() * 100:.1f} % of games "
    f"are predicted right on games in 2019 and later."
)

col_home, col_away = st.columns(2)
home_name = col_home.selectbox("Home team", names, index=names.index("Celtics"))
away_name = col_away.selectbox("Away team", names, index=names.index("Lakers"))

with st.expander("Rest days"):
    st.caption("Rest days before games is an interactive parameter that influences the "
               "win prediction noticeably.")
    col_r1, col_r2 = st.columns(2)
    rest_home = col_r1.slider("Home rest, days", 0, 7, 2)
    rest_away = col_r2.slider("Away rest, days", 0, 7, 2)

if home_name == away_name:
    st.warning("Pick two different teams.")
    st.stop()

home = by_name.loc[home_name]
away = by_name.loc[away_name]
pair = h2h[(h2h.home_id == home.teamId) & (h2h.away_id == away.teamId)]
h2h_rate = float(pair.home_winrate_last5.iloc[0]) if len(pair) else 0.5

X = feature_row(home, away, h2h_rate, rest_home, rest_away, feature_cols)
p_home = float(model.predict_proba(X)[0, 1])

with st.container(border=True):
    st.subheader("The call")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"P({home_name} win)", f"{p_home * 100:.1f} %")
    m2.metric(f"P({away_name} win)", f"{(1 - p_home) * 100:.1f} %")
    m3.metric("Model pick", home_name if p_home >= 0.5 else away_name)

    st.plotly_chart(
        charts.elo_bars(home_name, home.elo, away_name, away.elo),
        width="stretch", config={"displayModeBar": False})
    st.caption(f"ELO difference, home minus away: {home.elo - away.elo:+.0f}. "
               f"(100 is the size of home court advantage)")

with st.container(border=True):
    st.subheader("What pushed it there")
    st.caption("Each bar is one feature's SHAP value for this matchup, the ten largest "
               "by size. Blue pushes toward the home team, red toward the away team.")

    shap_values = load_explainer(model).shap_values(X)
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:                  # binary classifier, take the positive class
        shap_arr = shap_arr[:, :, 1]
    contrib = (pd.DataFrame({"feature": feature_cols,
                             "shap": shap_arr[0],
                             "value": X.iloc[0].to_numpy()})
               .assign(size=lambda d: d.shap.abs())
               .sort_values("size")
               .tail(10))
    st.plotly_chart(
        charts.shap_bars([label(f, v) for f, v in zip(contrib.feature, contrib.value)],
                         contrib.shap.tolist()),
        width="stretch", config={"displayModeBar": False})

st.caption(charts.data_note())
