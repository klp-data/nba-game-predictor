"""Best-of-7 series win probabilities.

Two ways to compute them:
- ``series_prob_closed(p)`` — closed form (no home-court adjustment)
- ``simulate_series(...)`` — Monte Carlo with the NBA 2-2-1-1-1 home pattern
- ``simulate_b07_elo(...)`` — Convenience wrapper: pass two ELOs and it figures
  out the per-game win probability for you.
"""
from __future__ import annotations

from math import comb
from typing import Optional

import numpy as np

from src import elo as elo_mod


# NBA Best-of-7: the higher seed has home court in games 1, 2, 5, 7.
NBA_HOME_PATTERN = np.array([True, True, False, False, True, False, True])


def series_prob_closed(p: float) -> float:
    """P(team wins Best-of-7) given a constant per-game win probability ``p``."""
    return sum(comb(3 + k, k) * p ** 4 * (1 - p) ** k for k in range(4))


def simulate_series(p_at_home: float, p_away: float, n_sim: int = 10000,
                    rng: Optional[np.random.Generator] = None) -> float:
    """Monte-Carlo Best-of-7 with the NBA 2-2-1-1-1 home pattern.

    Parameters
    ----------
    p_at_home : Per-game probability that the higher seed wins when at home.
    p_away    : Same, when the higher seed is on the road.
    n_sim     : Number of simulations.

    Returns
    -------
    Estimated probability that the higher seed wins the series.
    """
    rng = rng or np.random.default_rng()
    probs = np.where(NBA_HOME_PATTERN, p_at_home, p_away)
    sims = rng.random((n_sim, 7)) < probs
    series_wins = 0
    for sim in sims:
        wh, wl = 0, 0
        for game_won in sim:
            if game_won: wh += 1
            else:        wl += 1
            if wh == 4: series_wins += 1; break
            if wl == 4: break
    return series_wins / n_sim


def simulate_b07_elo(higher_elo: float, lower_elo: float,
                     home_adv: float = elo_mod.DEFAULT_HOME_ADV,
                     rng: Optional[np.random.Generator] = None) -> bool:
    """Single Best-of-7 simulation driven entirely by ELO.

    Returns ``True`` iff the higher seed wins the series.
    """
    rng = rng or np.random.default_rng()
    wh = wl = 0
    for is_high_home in NBA_HOME_PATTERN:
        p = elo_mod.win_prob(higher_elo, lower_elo, is_home=bool(is_high_home), home_adv=home_adv)
        if rng.random() < p: wh += 1
        else:                wl += 1
        if wh == 4: return True
        if wl == 4: return False
    return wh > wl
