"""Shared Plotly defaults and small helpers for the dashboard.

Every chart goes through ``style()`` so the template, fonts, margins and hover
boxes are the same on all four pages. Charts get no title of their own when the
page already has a subheader saying the same thing.

Also holds the paths to dashboard/data and the couple of things every page reads
out of it, so the pages themselves only load what they actually plot.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.plot_style import COLORS

DASHBOARD = Path(__file__).resolve().parent
DATA = DASHBOARD / "data"
MODELS = DASHBOARD / "models"

FONT = "sans-serif"
TEXT = "#31333f"
MUTED = "#6B6B6B"
GRID = "#e6e9ef"


# --- the small files every page touches --------------------------------------
@st.cache_data
def meta() -> dict:
    return json.loads((DATA / "meta.json").read_text(encoding="utf-8"))


def data_note() -> str:
    """The one line that closes every page."""
    return f"Data through {meta()['last_game_date']}."


@st.cache_data
def calibration() -> pd.DataFrame:
    return pd.read_csv(DATA / "calibration.csv", comment="#")


def holdout_accuracy() -> float:
    """Accuracy on the 2019+ holdout, recovered from the calibration bins.

    Every bin sits entirely on one side of 0.5, so the model's call is the same
    for every game in a bin and the bins carry enough to add the accuracy back up.
    Beats writing the number into the page by hand and watching it go stale.
    """
    cal = calibration()
    correct = (cal.n * cal.mean_obs.where(cal.mean_pred >= 0.5, 1 - cal.mean_obs)).sum()
    return correct / cal.n.sum()


# --- plotly ------------------------------------------------------------------
def style(fig: go.Figure, height: int = 380, legend: bool = False,
          top_margin: int = 10) -> go.Figure:
    """The house style. Call it on every figure, last."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT, size=13, color=TEXT),
        margin=dict(l=10, r=10, t=top_margin, b=10),
        height=height,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    title_text="", font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     title_font=dict(size=12), tickfont=dict(size=12))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     title_font=dict(size=12), tickfont=dict(size=12))
    return fig


def percent_axis(fig: go.Figure, axis: str = "x", decimals: int = 0) -> go.Figure:
    fmt = f".{decimals}%"
    if axis == "x":
        fig.update_xaxes(tickformat=fmt)
    else:
        fig.update_yaxes(tickformat=fmt)
    return fig


def shap_bars(labels: list[str], values: list[float]) -> go.Figure:
    """Horizontal SHAP bars, blue toward a home win, red toward an away win."""
    colors = [COLORS["primary"] if v > 0 else COLORS["secondary"] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        hovertemplate="%{y}<br>SHAP %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color=MUTED)
    fig.update_xaxes(title_text="SHAP value, positive means it favours the home team")
    fig.update_yaxes(automargin=True)
    return style(fig, height=40 * len(labels) + 80)


def elo_bars(home_name: str, home_elo: float, away_name: str, away_elo: float) -> go.Figure:
    """Two bars, ELO of the two teams. Kept deliberately small."""
    lo = min(home_elo, away_elo) - 120
    fig = go.Figure(go.Bar(
        x=[home_elo, away_elo], y=[home_name, away_name], orientation="h",
        marker_color=[COLORS["primary"], MUTED],
        text=[f"{home_elo:.0f}", f"{away_elo:.0f}"], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>ELO %{x:.0f}<extra></extra>",
    ))
    fig.update_xaxes(range=[lo, max(home_elo, away_elo) + 60], showticklabels=False,
                     showgrid=False)
    fig.update_yaxes(automargin=True)
    return style(fig, height=130)


# --- playoffs ----------------------------------------------------------------
STATE_COLS = ["p_pre", "p_after_r1", "p_after_r2", "p_after_r3"]
STATE_LABELS = ["Before playoffs", "After R1", "After R2", "After conf. finals"]
ROUND_LABELS = {1: "Lost in round 1", 2: "Lost in round 2", 3: "Lost conf. finals",
                4: "Lost the finals", 5: "Champion"}
P_FLOOR = 1e-4      # left edge of the log axis, for teams that never won a sim


def season_label(season: int) -> str:
    """2025 becomes 2025-26. Seasons are stored by their starting year."""
    return f"{season}-{str(season + 1)[-2:]}"


def playoff_pre_bars(odds: pd.DataFrame) -> go.Figure:
    """Championship probability before round 1, all 16 teams, champion in red."""
    from src.plot_style import COLORS as C
    d = odds.sort_values("p_pre")
    colors = [C["secondary"] if champ else C["primary"] for champ in d.is_champion]
    fig = go.Figure(go.Bar(
        x=d.p_pre, y=d.team, orientation="h", marker_color=colors,
        text=[f"{p:.1%}" for p in d.p_pre], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.2%} before round 1<extra></extra>",
    ))
    fig.update_xaxes(range=[0, d.p_pre.max() * 1.18])
    fig.update_yaxes(automargin=True)
    percent_axis(fig)
    return style(fig, height=520)


def playoff_paths(odds: pd.DataFrame) -> go.Figure:
    """How the odds moved for the teams that got out of round 1."""
    import team_colors
    fig = go.Figure()
    for row in odds[odds.round_reached >= 2].sort_values("p_pre", ascending=False).itertuples():
        values = [getattr(row, c) for c in STATE_COLS]
        fig.add_trace(go.Scatter(
            x=STATE_LABELS, y=values, mode="lines+markers", name=row.team,
            line=dict(color=team_colors.color(row.franchise),
                      width=3.5 if row.is_champion else 1.6),
            marker=dict(size=9 if row.is_champion else 6),
            hovertemplate=f"{row.team}<br>%{{x}}<br>%{{y:.1%}}<extra></extra>",
        ))
    percent_axis(fig, axis="y")
    fig.update_yaxes(rangemode="tozero")
    return style(fig, height=440, legend=True)


def playoff_outcome_scatter(odds: pd.DataFrame, title: str | None = None) -> go.Figure:
    """What the model said against what happened, for all 16 teams.

    The x axis is log scaled because the probabilities span three orders of
    magnitude, and teams that never won a single simulation are pinned to the
    left edge instead of disappearing. Points at the same outcome are nudged
    apart vertically so the labels stay readable.
    """
    from src.plot_style import COLORS as C
    d = odds.copy()
    d["x"] = d.p_pre.clip(lower=P_FLOOR)
    d["y"] = d.round_reached.astype(float)
    for reached, grp in d.groupby("round_reached"):
        order = grp.p_pre.rank(method="first") - 1
        step = min(0.13, 0.5 / max(len(grp) - 1, 1))     # keep the group inside its band
        d.loc[grp.index, "y"] += (order - (len(grp) - 1) / 2) * step

    fig = go.Figure()
    for is_champ, grp in d.groupby("is_champion"):
        fig.add_trace(go.Scatter(
            x=grp.x, y=grp.y, mode="markers+text", text=grp.team,
            textposition="middle right", textfont=dict(size=11),
            marker=dict(size=14 if is_champ else 9,
                        color=C["secondary"] if is_champ else C["primary"]),
            customdata=grp[["p_pre", "round_reached"]],
            hovertemplate=("%{text}<br>%{customdata[0]:.2%} before round 1"
                           "<extra></extra>"),
            showlegend=False,
        ))
    # ticks by hand: a percent format on a log axis prints 0.01 % and 0.02 % as
    # the same 0.0 % three times over
    fig.update_xaxes(type="log", title_text="P(title) before round 1",
                     range=[-4.2, 0.15],
                     tickmode="array",
                     tickvals=[1e-4, 1e-3, 1e-2, 1e-1, 1],
                     ticktext=["0.01 %", "0.1 %", "1 %", "10 %", "100 %"])
    fig.update_yaxes(tickmode="array", tickvals=list(ROUND_LABELS),
                     ticktext=list(ROUND_LABELS.values()), range=[0.4, 5.6],
                     automargin=True)
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)))
    return style(fig, height=460, top_margin=60 if title else 10)
