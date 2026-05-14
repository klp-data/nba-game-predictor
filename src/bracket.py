"""Playoff bracket construction + Monte Carlo simulation.

I rebuild the bracket from a season's playoff games: series are unordered team
pairs, rounds come from chronological order (8 R1, 4 R2, 2 R3, 1 Finals), and
I find each round's parents by checking the lower round's actual winners.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src import series as series_mod


def build_bracket(playoffs_for_season: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Reconstruct the bracket tree from a season's playoff games."""
    df = playoffs_for_season.copy()
    df['team_pair'] = df.apply(lambda r: tuple(sorted([r.hometeamId, r.awayteamId])), axis=1)

    series_list = []
    for pair, grp in df.groupby('team_pair'):
        if len(grp) < 3:
            return None  # too short = old best-of-5 era or fragmented data
        higher = grp.hometeamId.value_counts().idxmax()
        lower = [t for t in pair if t != higher][0]
        wins_h = ((grp.hometeamId == higher) & (grp.home_win == 1)).sum() + \
                 ((grp.awayteamId == higher) & (grp.home_win == 0)).sum()
        higher_won = wins_h > (len(grp) - wins_h)
        winner = higher if higher_won else lower
        series_list.append({
            'higher': higher, 'lower': lower, 'winner': winner,
            'first_date': grp.gameDate.min(),
        })

    s = pd.DataFrame(series_list).sort_values('first_date').reset_index(drop=True)
    if len(s) != 15:
        return None  # 16-team bracket = 15 series; anything else is pre-modern

    s['round'] = [1] * 8 + [2] * 4 + [3] * 2 + [4]
    s['uid'] = range(len(s))
    s['parents'] = [[] for _ in range(len(s))]

    # for each higher round, find which two lower-round series fed it (by actual winners)
    for r in [2, 3, 4]:
        higher_round = s[s['round'] == r]
        lower_round = s[s['round'] == r - 1]
        for idx, row in higher_round.iterrows():
            here = {row.higher, row.lower}
            parents = [low.uid for _, low in lower_round.iterrows() if low.winner in here]
            if len(parents) != 2:
                return None
            s.at[idx, 'parents'] = parents
    return s


def simulate_one(bracket_rows: list[dict], team_elos: dict,
                 rng: np.random.Generator,
                 fixed_winners: Optional[dict] = None) -> int:
    """One bracket run; returns the champion's team ID. ``fixed_winners`` pins earlier rounds."""
    winners = dict(fixed_winners or {})
    for r in bracket_rows:
        if r['uid'] in winners:
            continue
        if r['round'] == 1:
            a, b = r['higher'], r['lower']
        else:
            a = winners[r['parents'][0]]
            b = winners[r['parents'][1]]
        ea, eb = team_elos[a], team_elos[b]
        higher, lower = (a, b) if ea >= eb else (b, a)
        higher_wins = series_mod.simulate_b07_elo(team_elos[higher], team_elos[lower], rng=rng)
        winners[r['uid']] = higher if higher_wins else lower
    finals_uid = next(r['uid'] for r in bracket_rows if r['round'] == 4)
    return winners[finals_uid]


def championship_probs(bracket: pd.DataFrame, team_elos: dict, n_sim: int = 10000,
                       seed: int = 42, start_round: int = 1) -> dict:
    """Per-team championship probability from ``n_sim`` bracket simulations.

    ``start_round`` controls how much of the actual bracket is pinned: 1 = simulate
    from pre-playoffs, 2 = pin R1 winners, 3 = pin R1+R2, 4 = pin everything up to finals.
    """
    rows = bracket[['uid', 'round', 'higher', 'lower', 'winner', 'parents']].to_dict('records')
    teams = list(team_elos.keys())
    pre_winners = {r['uid']: r['winner'] for r in rows if r['round'] < start_round}

    rng = np.random.default_rng(seed)
    counts = {t: 0 for t in teams}
    for _ in range(n_sim):
        champ = simulate_one(rows, team_elos, rng, fixed_winners=pre_winners)
        counts[champ] += 1
    return {t: c / n_sim for t, c in counts.items()}
