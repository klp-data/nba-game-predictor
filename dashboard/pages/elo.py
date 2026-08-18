"""ELO explorer: team ratings over time, in team colours."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import team_colors

DEFAULT_TEAMS = ["Celtics", "Lakers", "Spurs", "Knicks", "Thunder"]
DEFAULT_RANGE = 20      # seasons shown on first load
MAX_POINTS = 2000       # per team, before the line gets thinned out


@st.cache_data
def load_history() -> pd.DataFrame:
    return pd.read_parquet(charts.DATA / "elo_history.parquet")


@st.cache_data
def load_champions() -> pd.DataFrame:
    return pd.read_csv(charts.DATA / "champions.csv")


def thin(frame: pd.DataFrame, max_points: int = MAX_POINTS) -> pd.DataFrame:
    """Keep at most max_points games, evenly spaced across the range.

    Picking every nth game, not a rolling average: smoothing would flatten the
    peaks, and the peaks are the interesting part of an ELO curve.
    """
    if len(frame) <= max_points:
        return frame
    keep = np.unique(np.linspace(0, len(frame) - 1, max_points).astype(int))
    return frame.iloc[keep]


history = load_history()
champions = load_champions()
franchises = sorted(history.franchise.unique())
seasons = (int(history.season.min()), int(history.season.max()))


st.title("ELO explorer")
st.caption(
    "ELO rating of every team after every game since 1946. Everyone starts at 1500. "
    "A strong team sits around 1600 to 1750. The flat stretches are the off-season"
)

col_teams, col_range = st.columns([2, 3])
picked = col_teams.multiselect("Teams", franchises, default=DEFAULT_TEAMS)
lo, hi = col_range.slider(
    "Seasons", seasons[0], seasons[1],
    value=(max(seasons[0], seasons[1] - DEFAULT_RANGE + 1), seasons[1]))
mark_titles = st.toggle("Mark title seasons", value=True)

if not picked:
    st.warning("Pick at least one team.")
    st.stop()

window = history[(history.season >= lo) & (history.season <= hi)]

fig = go.Figure()
thinned_any = False
for name in picked:
    team = window[window.franchise == name].sort_values("gameDate")
    if team.empty:
        continue
    shown = thin(team)
    thinned_any = thinned_any or len(shown) < len(team)
    fig.add_trace(go.Scattergl(
        x=shown.gameDate, y=shown.elo, mode="lines", name=name,
        line=dict(color=team_colors.color(name), width=1.4),
        customdata=shown.season,
        hovertemplate=(f"{name}<br>%{{x|%Y-%m-%d}}, season %{{customdata}}"
                       "<br>ELO %{y:.0f}<extra></extra>"),
    ))

    if mark_titles:
        won = champions[(champions.champion == name)
                        & (champions.season.between(lo, hi))].season
        ends = (team[team.season.isin(won)]
                .sort_values("gameDate").groupby("season").last().reset_index())
        if len(ends):
            fig.add_trace(go.Scattergl(
                x=ends.gameDate, y=ends.elo, mode="markers", name=f"{name} title",
                marker=dict(color=team_colors.color(name), size=10,
                            line=dict(color="white", width=1.5)),
                showlegend=False, customdata=ends.season,
                hovertemplate=(f"{name} won the title<br>season %{{customdata}}"
                               "<br>ELO %{y:.0f}<extra></extra>"),
            ))

fig.add_hline(y=1500, line_width=1, line_dash="dash", line_color=charts.MUTED)
fig.update_yaxes(title_text="ELO")
charts.style(fig, height=520, legend=True)

with st.container(border=True):
    st.subheader("Rating over time")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    note = ("Dots mark seasons a team won the title. " if mark_titles else "")
    if thinned_any:
        note += (f"Wide ranges are thinned to at most {MAX_POINTS:,} games per team, "
                 f"spread evenly across the range rather than smoothed, so the line "
                 f"keeps its peaks.")
    else:
        note += "Every game in the range is drawn."
    st.caption(note)

st.caption(charts.data_note())
