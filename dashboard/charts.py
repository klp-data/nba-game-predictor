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
