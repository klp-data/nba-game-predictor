"""Model report: how well the thing actually works, and where it stops working."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import charts
from src.plot_style import COLORS

# Three eras worth marking on the walk-forward chart. Seasons are stored by their
# starting year, so 1979 is 1979-80, the year the three-point line arrived.
ERAS = [
    (1979, "Three-point line"),
    (2011, "Lockout"),
    (2020, "Covid"),
]
TOP_FEATURES = 15


@st.cache_data
def load_backtest() -> pd.DataFrame:
    return pd.read_csv(charts.DATA / "backtest_metrics.csv")


@st.cache_data
def load_odds() -> pd.DataFrame:
    return pd.read_parquet(charts.DATA / "playoff_odds.parquet")


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_csv(charts.DATA / "feature_importance.csv")


@st.cache_data
def load_seasons() -> pd.DataFrame:
    return pd.read_csv(charts.DATA / "season_summary.csv")


def headline_table(backtest: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """The headline numbers, all read back out of the shipped files."""
    acc = np.average(backtest.accuracy, weights=backtest.n_test)
    auc = np.average(backtest.auc, weights=backtest.n_test)
    ranked = odds.assign(rank=odds.groupby("season").p_pre.rank(ascending=False,
                                                               method="first"))
    champs = ranked[ranked.is_champion]
    n = len(champs)
    seasons = f"{int(odds.season.min())} to {int(odds.season.max())}"
    rows = [
        ("Out-of-sample accuracy", f"{charts.holdout_accuracy() * 100:.1f} %",
         "Single split, games from season 2019 on"),
        ("Walk-forward accuracy", f"{acc * 100:.1f} %",
         f"{len(backtest)} seasons, {int(backtest.season.min())} to "
         f"{int(backtest.season.max())}, retrained every year"),
        ("Walk-forward AUC", f"{auc:.2f}", "Steady across eight decades"),
        ("Champion is the top pick", f"{(champs['rank'] == 1).mean() * 100:.1f} %",
         f"{(champs['rank'] == 1).sum()} of {n} seasons, {seasons}"),
        ("Champion in the top 3", f"{(champs['rank'] <= 3).mean() * 100:.1f} %",
         f"{(champs['rank'] <= 3).sum()} of {n} seasons"),
        ("Champion in the top 5", f"{(champs['rank'] <= 5).mean() * 100:.1f} %",
         f"{(champs['rank'] <= 5).sum()} of {n} seasons"),
        ("Probability on the real champion", f"{champs.p_pre.mean() * 100:.1f} %",
         f"{champs.p_pre.mean() / 0.0625:.1f} times the 6.25 % random baseline"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Context"])


def mark_eras(fig: go.Figure, **kwargs) -> None:
    for season, name in ERAS:
        fig.add_vrect(x0=season - 0.5, x1=season + 0.5, fillcolor=charts.MUTED,
                      opacity=0.10, line_width=0, annotation_text=name,
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=charts.MUTED), **kwargs)


def walk_forward_figure(backtest: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Accuracy", "AUC"))
    fig.add_trace(go.Scatter(
        x=backtest.season, y=backtest.accuracy, mode="lines+markers", name="Accuracy",
        line=dict(color=COLORS["primary"], width=1.6), marker=dict(size=4),
        hovertemplate="%{x}<br>accuracy %{y:.1%}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=backtest.season, y=backtest.home_win_rate_actual, mode="lines",
        name="Always pick the home team",
        line=dict(color=charts.MUTED, width=1.2, dash="dash"),
        hovertemplate="%{x}<br>home win rate %{y:.1%}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=backtest.season, y=backtest.auc, mode="lines+markers", name="AUC",
        line=dict(color=COLORS["accent"], width=1.6), marker=dict(size=4),
        hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color=charts.MUTED, line_width=1.2,
                  row=2, col=1)
    mark_eras(fig, row=1, col=1)
    mark_eras(fig, row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(range=[0.45, 0.8], row=2, col=1)
    fig.update_xaxes(title_text="Season", row=2, col=1)
    charts.style(fig, height=560, legend=True, top_margin=50)
    fig.update_annotations(font=dict(size=13))
    return fig


def calibration_figure(cal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(color=charts.MUTED, width=1.2, dash="dash"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=cal.mean_pred, y=cal.mean_obs, mode="markers+lines", name="Observed",
        line=dict(color=COLORS["primary"], width=1.6),
        marker=dict(size=8 + 24 * cal.n / cal.n.max(), color=COLORS["primary"]),
        customdata=cal.n,
        hovertemplate=("predicted %{x:.1%}<br>actually won %{y:.1%}"
                       "<br>%{customdata:,} games<extra></extra>")))
    fig.update_xaxes(title_text="Predicted P(home win)", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title_text="Observed home win rate", tickformat=".0%", range=[0, 1])
    return charts.style(fig, height=440, legend=True)


def importance_figure(imp: pd.DataFrame) -> go.Figure:
    top = imp.head(TOP_FEATURES).sort_values("share")
    fig = go.Figure(go.Bar(
        x=top.share, y=top.feature, orientation="h", marker_color=COLORS["primary"],
        text=[f"{s:.1%}" for s in top.share], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.1%} of total gain<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, top.share.max() * 1.18])
    fig.update_yaxes(automargin=True)
    return charts.style(fig, height=520)


def home_advantage_figure(seasons: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=seasons.season, y=seasons.home_win_rate, mode="lines",
        line=dict(color=COLORS["primary"], width=1.6), name="Home win rate",
        customdata=seasons.games,
        hovertemplate="%{x}<br>%{y:.1%} of games won at home"
                      "<br>%{customdata:,} games<extra></extra>"))
    fig.add_hline(y=0.5, line_dash="dash", line_color=charts.MUTED, line_width=1.2)
    fig.update_yaxes(tickformat=".0%", title_text="Home win rate")
    fig.update_xaxes(title_text="Season")
    return charts.style(fig, height=380)


backtest = load_backtest()
odds = load_odds()

st.title("Model report")
st.caption("Overall performance of the model. ")

st.dataframe(headline_table(backtest, odds), hide_index=True, width="stretch")
st.caption("Numbers from the model in notebook 03.")

with st.container(border=True):
    st.subheader("Walk-forward performance")
    st.plotly_chart(walk_forward_figure(backtest), width="stretch",
                    config={"displayModeBar": False})
    st.caption("For every season since 1960 the model is trained on everything before "
               "it and tested on that season only. The dashed line is what you get by "
               "always picking the home team.")

with st.container(border=True):
    st.subheader("Calibration")
    st.plotly_chart(calibration_figure(charts.calibration()), width="stretch",
                    config={"displayModeBar": False})
    mid = charts.calibration().query("bin_low <= 0.65 < bin_high").iloc[0]
    st.caption("A calibration curve checks whether the models probabilities are "
               "honest. So if  70 % is expected, is 70 % home win in the end observed? "
               "Here the curve sits under the diagonal almost everywhere above 20 %. "
               "Here the model is a bit too sure of itself. Games it calls "
               f"{mid.mean_pred * 100:.0f} % are won about {mid.mean_obs * 100:.0f} % "
               "of the time. Bigger markers mean more games in the correct bin.")

with st.container(border=True):
    st.subheader("Feature importance")
    st.plotly_chart(importance_figure(load_importance()), width="stretch",
                    config={"displayModeBar": False})
    st.caption(f"Top {TOP_FEATURES} features by share of total gain.")

with st.container(border=True):
    st.subheader("Home advantage over time")
    st.plotly_chart(home_advantage_figure(load_seasons()), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Share of games won by the home team, every season since 1946. It has "
               "been falling for decades, which is part of why recent seasons are "
               "harder to predict than old ones.")

st.caption(charts.data_note())
