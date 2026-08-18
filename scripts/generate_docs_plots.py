"""Generate the 5 README PNGs into docs/.

Each plot reproduces a figure from the notebooks without modifying them.
Run from the project root:
    python scripts/generate_docs_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import plot_style, series  # noqa: E402

DATA = ROOT / "data" / "processed"
DASH_DATA = ROOT / "dashboard" / "data"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

SAVE_KW = dict(dpi=150, bbox_inches="tight")
plot_style.apply()


# -------------------------------------------------------------------------- a)
def walk_forward():
    """docs/walk_forward.png — 2x2 per-season metrics from notebook 04."""
    m = pd.read_csv(DATA / "backtest_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(m.season, m.accuracy, marker="o", color=plot_style.COLORS["primary"])
    axes[0, 0].axhline(m.home_win_rate_actual.mean(), color=plot_style.COLORS["neutral"],
                       linestyle="--", label="trivial home-win baseline")
    axes[0, 0].set_title("Accuracy per season")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()

    axes[0, 1].plot(m.season, m.auc, marker="o", color=plot_style.COLORS["accent"])
    axes[0, 1].axhline(0.5, color=plot_style.COLORS["neutral"], linestyle="--", label="random")
    axes[0, 1].set_title("AUC per season")
    axes[0, 1].set_ylabel("AUC")
    axes[0, 1].legend()

    axes[1, 0].plot(m.season, m.log_loss, marker="o", color=plot_style.COLORS["accent"])
    axes[1, 0].set_title("Log-loss per season (lower = better)")
    axes[1, 0].set_ylabel("Log-loss")
    axes[1, 0].set_xlabel("Season")

    axes[1, 1].plot(m.season, m.brier, marker="o", color=plot_style.COLORS["secondary"])
    axes[1, 1].set_title("Brier score per season (lower = better)")
    axes[1, 1].set_ylabel("Brier")
    axes[1, 1].set_xlabel("Season")

    plt.tight_layout()
    fig.savefig(DOCS / "walk_forward.png", **SAVE_KW)
    plt.close(fig)
    print("  -> walk_forward.png")


# -------------------------------------------------------------------------- b)
def bo7_amplifier():
    """docs/bo7_amplifier.png — best-of-7 amplification curve from notebook 07."""
    p_grid = np.linspace(0.4, 0.85, 100)
    series_p = [series.series_prob_closed(p) for p in p_grid]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(p_grid, p_grid, "--", color=plot_style.COLORS["neutral"],
            label="no amplification (game = series)")
    ax.plot(p_grid, series_p, color=plot_style.COLORS["primary"], linewidth=2.4,
            label="best-of-7 series probability")
    for px in [0.55, 0.60, 0.65, 0.70]:
        py = series.series_prob_closed(px)
        ax.scatter([px], [py], color=plot_style.COLORS["secondary"], zorder=5, s=42)
        ax.annotate(rf"{px:.0%} $\rightarrow$ {py:.0%}", xy=(px, py),
                    xytext=(8, -14), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Per-game win probability")
    ax.set_ylabel("Series win probability")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(DOCS / "bo7_amplifier.png", **SAVE_KW)
    plt.close(fig)
    print("  -> bo7_amplifier.png")


# -------------------------------------------------------------------------- c)
def _champion_ranks():
    """Per season: where the real champion sat in the model's list, and its odds.

    Reads dashboard/data/playoff_odds.parquet rather than the notebook 08 output,
    so this covers 41 seasons including 2026 and agrees with the dashboard.
    """
    odds = pd.read_parquet(DASH_DATA / "playoff_odds.parquet")
    out = {}
    for col in ["p_pre", "p_after_r1", "p_after_r2", "p_after_r3"]:
        ranked = odds.assign(rank=odds.groupby("season")[col].rank(ascending=False,
                                                                  method="first"))
        champs = ranked[ranked.is_champion].sort_values("season")
        out[col] = champs[["season", "rank", col]].rename(columns={col: "p"})
    return out


def bracket_backtest():
    """docs/bracket_backtest.png — 1x2, rank histogram + per-season confidence."""
    bt = _champion_ranks()["p_pre"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(bt["rank"], bins=range(1, 18),
                 color=plot_style.COLORS["primary"], edgecolor="white")
    axes[0].set_title("Real champion's rank")
    axes[0].set_xlabel("Rank in model's top list")
    axes[0].set_ylabel("Number of seasons")

    axes[1].plot(bt.season, bt.p, marker="o",
                 color=plot_style.COLORS["secondary"])
    axes[1].axhline(1 / 16, color=plot_style.COLORS["neutral"], linestyle="--",
                    label="random (1/16)")
    axes[1].set_title("P assigned to the real champion")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("P(championship)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(DOCS / "bracket_backtest.png", **SAVE_KW)
    plt.close(fig)
    print("  -> bracket_backtest.png")


# -------------------------------------------------------------------------- d)
def conditional_confidence():
    """docs/conditional_confidence.png — 1x2, how the odds sharpen by round."""
    states = _champion_ranks()
    labels = ["Pre-playoffs\n(16)", "Round 2\n(8)", "Conf. Finals\n(4)", "Finals\n(2)"]
    top1 = [(s["rank"] == 1).mean() for s in states.values()]
    top3 = [(s["rank"] <= 3).mean() for s in states.values()]
    avg_p = [s.p.mean() for s in states.values()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(labels, top1, marker="o", linewidth=2.2,
                 color=plot_style.COLORS["primary"], label="Top-1 hit rate")
    axes[0].plot(labels, top3, marker="s", linewidth=2.2,
                 color=plot_style.COLORS["accent"], label="Top-3 hit rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Hit rate")
    axes[0].set_title("Hit rate")
    axes[0].legend()
    for col in [top1, top3]:
        for i, v in enumerate(col):
            axes[0].annotate(f"{v:.0%}", (i, v), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9)

    axes[1].plot(labels, avg_p, marker="D", linewidth=2.2,
                 color=plot_style.COLORS["secondary"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Avg P(actual champion)")
    axes[1].set_title("Avg. P for the real champion")
    for i, v in enumerate(avg_p):
        axes[1].annotate(f"{v:.0%}", (i, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(DOCS / "conditional_confidence.png", **SAVE_KW)
    plt.close(fig)
    print("  -> conditional_confidence.png")


# -------------------------------------------------------------------------- e)
def playoffs_2026():
    """docs/playoffs_2026.png — the Playoffs page's third card, for the README.

    Plotly rather than matplotlib, because it is the same figure the dashboard
    draws. Needs kaleido installed for the static export.
    """
    sys.path.insert(0, str(ROOT / "dashboard"))
    import charts

    odds = pd.read_parquet(ROOT / "dashboard" / "data" / "playoff_odds.parquet")
    season = 2025                       # the 2026 playoffs
    fig = charts.playoff_outcome_scatter(
        odds[odds.season == season],
        title="2026 playoffs: title odds before round 1 against how far teams got")
    fig.write_image(DOCS / "playoffs_2026.png", width=1100, height=560, scale=2)
    print("  -> playoffs_2026.png")


if __name__ == "__main__":
    print("Generating docs/ PNGs...")
    walk_forward()
    bo7_amplifier()
    bracket_backtest()
    conditional_confidence()
    playoffs_2026()
    print("Done.")
