"""ELO rating for NBA teams.

Standard chess-style ELO with a home-court bonus. Ratings update chronologically
so we always have the pre-game ELO available — no leakage.
"""
from __future__ import annotations

import pandas as pd

INITIAL_ELO = 1500       # standard ELO baseline
DEFAULT_K = 20           # 538's NBA default — bigger = ratings react faster to recent results
DEFAULT_HOME_ADV = 100   # roughly a +7 % home win-rate bump


def win_prob(team_elo: float, opp_elo: float, is_home: bool = True,
             home_adv: float = DEFAULT_HOME_ADV) -> float:
    """Win probability from the ELO formula.

    P(team wins) = 1 / (1 + 10^((opp - team - home_adv*sign) / 400))
    """
    delta = opp_elo - team_elo - (home_adv if is_home else -home_adv)
    return 1.0 / (1.0 + 10 ** (delta / 400))


def update(elo_winner: float, elo_loser: float, winner_was_home: bool,
           k: float = DEFAULT_K, home_adv: float = DEFAULT_HOME_ADV) -> tuple[float, float]:
    """Post-game ELO update. Returns (new_winner_elo, new_loser_elo)."""
    expected_winner = win_prob(elo_winner, elo_loser, is_home=winner_was_home, home_adv=home_adv)
    new_winner = elo_winner + k * (1.0 - expected_winner)
    new_loser = elo_loser + k * (0.0 - (1.0 - expected_winner))
    return new_winner, new_loser


def compute_history(games: pd.DataFrame, k: float = DEFAULT_K,
                    home_adv: float = DEFAULT_HOME_ADV,
                    initial: float = INITIAL_ELO) -> pd.DataFrame:
    """Walk through games chronologically and add pre-game ELOs.

    Needs ``gameDate, hometeamId, awayteamId, homeScore, awayScore`` columns.
    Returns the same dataframe with ``home_elo_pre``, ``away_elo_pre``, and
    ``elo_diff`` added.
    """
    games = games.sort_values('gameDate').reset_index(drop=True).copy()
    elos: dict = {}
    home_pre, away_pre = [], []

    for row in games.itertuples(index=False):
        rh = elos.get(row.hometeamId, initial)
        ra = elos.get(row.awayteamId, initial)
        home_pre.append(rh)
        away_pre.append(ra)

        home_won = row.homeScore > row.awayScore
        if home_won:
            elos[row.hometeamId], elos[row.awayteamId] = update(rh, ra, winner_was_home=True, k=k, home_adv=home_adv)
        else:
            elos[row.awayteamId], elos[row.hometeamId] = update(ra, rh, winner_was_home=False, k=k, home_adv=home_adv)

    games['home_elo_pre'] = home_pre
    games['away_elo_pre'] = away_pre
    games['elo_diff'] = games.home_elo_pre - games.away_elo_pre
    return games
