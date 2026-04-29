"""Generiert die Highlight-Plots fuer das README.

Ausfuehrung (aus dem Projekt-Root):
    python scripts/generate_highlight_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # Headless-Backend, da als Skript ohne GUI laufend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import bracket, plot_style, series  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)

plot_style.apply()


def plot_series_amplifier():
    """Plot 1: Best-of-7 verstaerkt jeden Edge."""
    p_grid = np.linspace(0.4, 0.85, 100)
    series_p = [series.series_prob_closed(p) for p in p_grid]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(p_grid, p_grid, '--', color=plot_style.COLORS['neutral'],
            label='kein Verstaerker')
    ax.plot(p_grid, series_p, color=plot_style.COLORS['primary'], linewidth=2.4,
            label='Best-of-7-Series-WSK')
    for px in [0.55, 0.60, 0.65, 0.70]:
        ax.scatter([px], [series.series_prob_closed(px)],
                   color=plot_style.COLORS['secondary'], zorder=5, s=40)
        ax.annotate(f'{px:.0%} -> {series.series_prob_closed(px):.0%}',
                    xy=(px, series.series_prob_closed(px)),
                    xytext=(6, -15), textcoords='offset points', fontsize=9)
    ax.set_xlabel('Pro-Spiel-Wahrscheinlichkeit')
    ax.set_ylabel('Series-Wahrscheinlichkeit')
    ax.set_title('Best-of-7 verstaerkt jeden Edge')
    ax.legend(loc='lower right')
    fig.savefig(ASSETS / 'series_amplifier.png')
    plt.close(fig)
    print('  -> series_amplifier.png')


def plot_conditional_hitrate():
    """Plot 2: Treffer-Quote als Funktion vom Wissensstand. Berechnet on-the-fly via src/."""
    games = pd.read_parquet(DATA / 'games_with_advanced_features.parquet')
    playoffs_all = games[games.gameType == 'Playoffs']

    rows = []
    for season in sorted(playoffs_all.season.unique()):
        po = playoffs_all[playoffs_all.season == season]
        b = bracket.build_bracket(po)
        if b is None: continue
        # ELOs
        home_v = po[['gameDate', 'hometeamId', 'home_elo_pre']].rename(
            columns={'hometeamId': 't', 'home_elo_pre': 'e'})
        away_v = po[['gameDate', 'awayteamId', 'away_elo_pre']].rename(
            columns={'awayteamId': 't', 'away_elo_pre': 'e'})
        tv = pd.concat([home_v, away_v]).sort_values(['t', 'gameDate'])
        elos = tv.groupby('t').first()['e'].to_dict()

        actual = b[b['round'] == 4].iloc[0].winner
        for sr in [1, 2, 3, 4]:
            probs = bracket.championship_probs(b, elos, n_sim=2000, seed=season * 10 + sr, start_round=sr)
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            rank = next(i for i, (t, _) in enumerate(ranked) if t == actual) + 1
            rows.append({'season': season, 'start_round': sr, 'rank_actual': rank,
                         'p_actual': probs[actual]})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / 'conditional_predictions.csv', index=False)

    summary = df.groupby('start_round').agg(
        top1=('rank_actual', lambda r: (r == 1).mean()),
        top3=('rank_actual', lambda r: (r <= 3).mean()),
        avg_p=('p_actual', 'mean'),
    )
    labels = {1: 'Pre-Playoffs\n(16 Teams)', 2: 'Round 2\n(8)',
              3: 'Conf. Finals\n(4)', 4: 'Finals\n(2)'}
    x = [labels[i] for i in summary.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, summary.top1, marker='o', linewidth=2.2,
            color=plot_style.COLORS['primary'], label='Top-1 Treffer')
    ax.plot(x, summary.top3, marker='s', linewidth=2.2,
            color=plot_style.COLORS['accent'], label='Top-3 Treffer')
    ax.plot(x, summary.avg_p, marker='D', linewidth=2.2,
            color=plot_style.COLORS['secondary'], label='Avg P(echter Champion)')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Quote / Wahrscheinlichkeit')
    ax.set_title('Wie viel Wissen braucht das Modell?')
    ax.legend()
    for col, mark in [(summary.top1, 'top'), (summary.top3, 'mid')]:
        for i, v in enumerate(col):
            ax.annotate(f'{v:.0%}', (i, v), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=9)
    fig.savefig(ASSETS / 'conditional_hitrate.png')
    plt.close(fig)
    print('  -> conditional_hitrate.png')


def plot_2023_bracket():
    """Plot 3: Pre-Playoff Championship Probabilities 2023."""
    df = pd.read_parquet(DATA / 'games_with_advanced_features.parquet')
    playoffs = df[df.gameType == 'Playoffs']

    # Pre-Playoff ELO fuer 2023
    po = playoffs[playoffs.season == 2023]
    home_v = po[['gameDate', 'hometeamId', 'home_elo_pre']].rename(
        columns={'hometeamId': 't', 'home_elo_pre': 'e'})
    away_v = po[['gameDate', 'awayteamId', 'away_elo_pre']].rename(
        columns={'awayteamId': 't', 'away_elo_pre': 'e'})
    tv = pd.concat([home_v, away_v]).sort_values(['t', 'gameDate'])
    elos = tv.groupby('t').first()['e'].to_dict()

    b = bracket.build_bracket(po)
    if b is None:
        print('  Kein 16-Team-Bracket fuer 2023 — uebersprungen.')
        return
    probs = bracket.championship_probs(b, elos, n_sim=10000)
    actual = b[b['round'] == 4].iloc[0].winner

    # JUengsten Namen pro Team-Id nehmen (sonst kommen Aliase wie SuperSonics statt Thunder raus)
    name_src = pd.concat([
        df[['gameDate', 'hometeamId', 'hometeamName']].rename(columns={'hometeamId': 'tid', 'hometeamName': 'tname'}),
        df[['gameDate', 'awayteamId', 'awayteamName']].rename(columns={'awayteamId': 'tid', 'awayteamName': 'tname'}),
    ]).sort_values('gameDate').drop_duplicates('tid', keep='last').set_index('tid')
    team_names = name_src['tname']

    view = pd.Series(probs).rename(team_names).sort_values()
    actual_name = team_names[actual]
    colors = [plot_style.COLORS['secondary'] if n == actual_name else plot_style.COLORS['primary']
              for n in view.index]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(view.index, view.values, color=colors)
    ax.set_xlabel('Pre-Playoff P(Championship)')
    ax.set_title('Saison 2023 — Title-Wahrscheinlichkeiten\n(rot = tatsaechlicher Champion: '
                 f'{actual_name})')
    for i, v in enumerate(view.values):
        ax.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=9)
    ax.set_xlim(0, max(view.values) * 1.18)
    fig.savefig(ASSETS / 'champ_probs_2023.png')
    plt.close(fig)
    print('  -> champ_probs_2023.png')


if __name__ == '__main__':
    print('Generiere Highlight-Plots...')
    plot_series_amplifier()
    plot_conditional_hitrate()
    plot_2023_bracket()
    print('Fertig.')
