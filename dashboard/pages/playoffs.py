"""Playoffs page: the model's title odds against what actually happened."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import charts

RETRO_SEASON = 2025      # the 2026 playoffs


@st.cache_data
def load_odds() -> pd.DataFrame:
    return pd.read_parquet(charts.DATA / "playoff_odds.parquet")


def retrospective(season_odds: pd.DataFrame) -> str:
    """The 2026 write-up. Numbers come out of playoff_odds.parquet every time.

    Teams are looked up by name rather than by rank, so a rebuild that nudges the
    probabilities can never hand the wrong number to a name.
    """
    by_team = season_odds.set_index("team")
    ranked = season_odds.sort_values("p_pre", ascending=False).reset_index(drop=True)
    rank = int(ranked[ranked.team == "Knicks"].index[0]) + 1
    ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
               6: "sixth", 7: "seventh", 8: "eighth"}[rank]
    elo_rank = int(season_odds.pre_playoff_elo.rank(ascending=False, method="first")
                   [season_odds.team == "Knicks"].iloc[0])
    pct = lambda team: f"{by_team.loc[team].p_pre * 100:.1f}"
    return (
        f"Before round 1 the model had the Thunder at {pct('Thunder')} %, Spurs "
        f"{pct('Spurs')} %, Celtics {pct('Celtics')} %. The Knicks were {ordinal} at "
        f"{pct('Knicks')} %. The Knicks won the title.\n\n"
        f"The top two picks met in the West finals and the series went to seven games, "
        f"so the model was not far off there. It missed the Knicks. They went in as a "
        f"third seed with a pre-playoff ELO of "
        f"{by_team.loc['Knicks'].pre_playoff_elo:.0f}, rank {elo_rank} by ELO, and then "
        f"won thirteen playoff games in a row. ELO with K=20 needs a few weeks to catch "
        f"up to a run like that, and the playoffs are over before it does. This is the "
        f"limit of team-level history in one season: it knows who was good, not who "
        f"just got better."
    )


odds = load_odds()
seasons = sorted(odds.season.unique(), reverse=True)

st.title("Playoffs")
st.caption(
    "What the model gave each playoff team before a ball was thrown up, and how that "
    "changed as rounds resolved. The bracket is real, the series are simulated from "
    "pre-playoff ELO."
)

season = st.selectbox("Season", seasons, index=0, format_func=charts.season_label)
d = odds[odds.season == season]

with st.container(border=True):
    st.subheader("Title odds before round 1")
    st.plotly_chart(charts.playoff_pre_bars(d), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Random baseline: 6.25 %. Red is the team that went on to win it.")

with st.container(border=True):
    st.subheader("Odds by round")
    st.plotly_chart(charts.playoff_paths(d), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Only the teams that got out of round 1. A team drops to zero the "
               "moment it is eliminated, and the champion is the thick line.")

with st.container(border=True):
    st.subheader("Predicted vs. actual")
    st.plotly_chart(charts.playoff_outcome_scatter(d), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Probability before round 1 on a log scale, against how far the team "
               "actually got. Teams that won no simulation at all sit on the left edge.")

if season == RETRO_SEASON:
    st.markdown(retrospective(d))

st.caption(charts.data_note())
