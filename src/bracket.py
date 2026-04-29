"""Playoff bracket construction and Monte-Carlo simulation.

The bracket tree is reconstructed from the actual playoff games of a season:
- Series identified via the unordered team pair
- Round derived from chronological order (8 R1, 4 R2, 2 R3, 1 Finals)
- Parent series resolved using the actual winners of each lower-round series
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src import series as series_mod


def build_bracket(playoffs_for_season: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Reconstruct a season's 16-team playoff bracket.

    Required columns in ``playoffs_for_season``:
        ``gameDate, hometeamId, awayteamId, home_win``.
    Returns ``None`` if the bracket does not have exactly 15 series
    (i.e., the season is not in modern 16-team format).
    """
    df = playoffs_for_season.copy()
    df['team_pair'] = df.apply(lambda r: tuple(sorted([r.hometeamId, r.awayteamId])), axis=1)

    series_list = []
    for pair, grp in df.groupby('team_pair'):
        if len(grp) < 3:
            return None
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
        return None

    s['round'] = [1] * 8 + [2] * 4 + [3] * 2 + [4]
    s['uid'] = range(len(s))
    s['parents'] = [[] for _ in range(len(s))]

    # Resolve parent series by checking which lower-round series fed into this one
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
    """Simulate one full bracket run. Returns the champion's team ID.

    Parameters
    ----------
    bracket_rows  : Bracket as a list of dicts (uid, round, higher, lower, winner, parents).
    team_elos     : Mapping team_id -> ELO.
    rng           : numpy random generator.
    fixed_winners : Optional mapping series_uid -> team_id. Pins outcomes of earlier
                    rounds (used for conditional predictions).
    """
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
    """Run ``n_sim`` bracket simulations and return per-team championship probabilities.

    ``start_round=1`` simulates from the pre-playoffs view. ``start_round=2`` pins the
    actual R1 winners, ``start_round=3`` pins R1+R2, and so on.
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
