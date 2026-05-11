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
from src import bracket, plot_style, series  # noqa: E402

DATA = ROOT / "data" / "processed"
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
        ax.annotate(f"{px:.0%} -> {py:.0%}", xy=(px, py),
                    xytext=(8, -14), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Per-game win probability")
    ax.set_ylabel("Series win probability")
    ax.set_title("Best-of-7 amplifies any per-game edge")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(DOCS / "bo7_amplifier.png", **SAVE_KW)
    plt.close(fig)
    print("  -> bo7_amplifier.png")


# -------------------------------------------------------------------------- c)
def bracket_backtest():
    """docs/bracket_backtest.png — 1x2 from notebook 08 (rank histogram + per-season conf)."""
    bt = pd.read_csv(DATA / "bracket_backtest.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(bt.actual_champ_rank, bins=range(1, 18),
                 color=plot_style.COLORS["primary"], edgecolor="white")
    axes[0].set_title("Where did the actual champion land in our picks?")
    axes[0].set_xlabel("Rank in model's top list")
    axes[0].set_ylabel("Number of seasons")

    axes[1].plot(bt.season, bt.actual_champ_p, marker="o",
                 color=plot_style.COLORS["secondary"])
    axes[1].axhline(1 / 16, color=plot_style.COLORS["neutral"], linestyle="--",
                    label="random (1/16)")
    axes[1].set_title("Model probability for the eventual champion (pre-playoffs)")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("P(championship)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(DOCS / "bracket_backtest.png", **SAVE_KW)
    plt.close(fig)
    print("  -> bracket_backtest.png")


# -------------------------------------------------------------------------- d)
def conditional_confidence():
    """docs/conditional_confidence.png — 1x2 from notebook 09."""
    df = pd.read_csv(DATA / "conditional_predictions.csv")
    summary = df.groupby("start_round").agg(
        top1=("rank_actual", lambda r: (r == 1).mean()),
        top3=("rank_actual", lambda r: (r <= 3).mean()),
        avg_p=("p_actual", "mean"),
    )
    labels = {1: "Pre-playoffs\n(16)", 2: "Round 2\n(8)",
              3: "Conf. Finals\n(4)", 4: "Finals\n(2)"}
    x = [labels[i] for i in summary.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(x, summary.top1, marker="o", linewidth=2.2,
                 color=plot_style.COLORS["primary"], label="Top-1 hit rate")
    axes[0].plot(x, summary.top3, marker="s", linewidth=2.2,
                 color=plot_style.COLORS["accent"], label="Top-3 hit rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Hit rate")
    axes[0].set_title("Hit rate by starting knowledge level")
    axes[0].legend()
    for col in [summary.top1, summary.top3]:
        for i, v in enumerate(col):
            axes[0].annotate(f"{v:.0%}", (i, v), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9)

    axes[1].plot(x, summary.avg_p, marker="D", linewidth=2.2,
                 color=plot_style.COLORS["secondary"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Avg P(actual champion)")
    axes[1].set_title("Average probability assigned to the actual champion")
    for i, v in enumerate(summary.avg_p):
        axes[1].annotate(f"{v:.0%}", (i, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(DOCS / "conditional_confidence.png", **SAVE_KW)
    plt.close(fig)
    print("  -> conditional_confidence.png")


# -------------------------------------------------------------------------- e)
def live_2026():
    """docs/live_2026.png — top-16 championship probabilities for the 2025-26 season."""
    df = pd.read_parquet(DATA / "games_with_advanced_features.parquet")
    season = 2025
    playoffs = df[(df.season == season) & (df.gameType == "Playoffs")]

    # Pre-playoff ELO per team = each team's first ELO observed in playoffs
    home_v = playoffs[["gameDate", "hometeamId", "home_elo_pre"]].rename(
        columns={"hometeamId": "t", "home_elo_pre": "e"})
    away_v = playoffs[["gameDate", "awayteamId", "away_elo_pre"]].rename(
        columns={"awayteamId": "t", "away_elo_pre": "e"})
    preplay = (pd.concat([home_v, away_v])
                 .sort_values(["t", "gameDate"])
                 .groupby("t").first()["e"])

    # Most recent team name per ID
    names = (pd.concat([
        df[["gameDate", "hometeamId", "hometeamName"]].rename(
            columns={"hometeamId": "t", "hometeamName": "n"}),
        df[["gameDate", "awayteamId", "awayteamName"]].rename(
            columns={"awayteamId": "t", "awayteamName": "n"}),
    ]).sort_values("gameDate").drop_duplicates("t", keep="last")
       .set_index("t")["n"])

    top16 = preplay.sort_values(ascending=False).head(16)
    elos = top16.to_dict()
    team_ids = list(top16.index)

    # Standard 1v8 / 4v5 / 2v7 / 3v6 bracket (×2 for both conferences)
    seed_matchups = [(0, 7), (3, 4), (1, 6), (2, 5),
                     (8, 15), (11, 12), (9, 14), (10, 13)]

    rng = np.random.default_rng(42)
    counts = {t: 0 for t in team_ids}
    n_sim = 10000
    for _ in range(n_sim):
        r1 = [(ta if series.simulate_b07_elo(elos[ta], elos[tb], rng=rng) else tb)
              for ta, tb in [(team_ids[hi], team_ids[lo]) for hi, lo in seed_matchups]]
        r2 = []
        for i in range(0, 8, 2):
            ta, tb = r1[i], r1[i + 1]
            h, l = (ta, tb) if elos[ta] >= elos[tb] else (tb, ta)
            r2.append(h if series.simulate_b07_elo(elos[h], elos[l], rng=rng) else l)
        r3 = []
        for i in range(0, 4, 2):
            ta, tb = r2[i], r2[i + 1]
            h, l = (ta, tb) if elos[ta] >= elos[tb] else (tb, ta)
            r3.append(h if series.simulate_b07_elo(elos[h], elos[l], rng=rng) else l)
        ta, tb = r3
        h, l = (ta, tb) if elos[ta] >= elos[tb] else (tb, ta)
        champ = h if series.simulate_b07_elo(elos[h], elos[l], rng=rng) else l
        counts[champ] += 1

    probs = (pd.Series({names.get(t, str(t)): c / n_sim for t, c in counts.items()})
               .sort_values())

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [plot_style.COLORS["primary"]] * len(probs)
    if len(colors):
        colors[-1] = plot_style.COLORS["secondary"]  # highlight top pick
    ax.barh(probs.index, probs.values, color=colors)
    ax.set_xlabel("P(Championship)")
    ax.set_title(f"2025–26 NBA championship probabilities — top-16 by ELO\n"
                 f"(as of {playoffs.gameDate.max().date()}, R1 in progress)")
    for i, v in enumerate(probs.values):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
    ax.set_xlim(0, max(probs.values) * 1.18)
    plt.tight_layout()
    fig.savefig(DOCS / "live_2026.png", **SAVE_KW)
    plt.close(fig)
    print("  -> live_2026.png")


if __name__ == "__main__":
    print("Generating docs/ PNGs...")
    walk_forward()
    bo7_amplifier()
    bracket_backtest()
    conditional_confidence()
    live_2026()
    print("Done.")
