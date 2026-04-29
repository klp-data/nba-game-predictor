"""Best-of-7-Series-Wahrscheinlichkeiten.

Zwei Wege:
- `series_prob_closed(p)` — geschlossene Form (kein Heimvorteils-Effekt)
- `simulate_series(...)` — Monte Carlo mit NBA-2-2-1-1-1-Heimrecht-Pattern
- `simulate_b07_elo(...)` — Bequeme Variante: gibt zwei ELOs rein, holt sich
  pro Spiel die ELO-Wahrscheinlichkeit ab.
"""
from __future__ import annotations

from math import comb
from typing import Optional

import numpy as np

from src import elo as elo_mod


# NBA-Best-of-7: hoeherer Seed hat Heimrecht in Spielen 1, 2, 5, 7
NBA_HOME_PATTERN = np.array([True, True, False, False, True, False, True])


def series_prob_closed(p: float) -> float:
    """P(Team gewinnt Best-of-7), wenn pro Spiel Gewinn-WSK = p (kein Home-Court-Spread)."""
    return sum(comb(3 + k, k) * p ** 4 * (1 - p) ** k for k in range(4))


def simulate_series(p_at_home: float, p_away: float, n_sim: int = 10000,
                    rng: Optional[np.random.Generator] = None) -> float:
    """Monte-Carlo Best-of-7 mit NBA-Heimrechts-Pattern (2-2-1-1-1).

    Parameters
    ----------
    p_at_home : Pro-Spiel-Wahrscheinlichkeit, dass der hoehere Seed gewinnt, wenn er zuhause spielt.
    p_away    : dito, auswaerts.
    n_sim     : Anzahl Simulationen.

    Returns
    -------
    Geschaetzte Wahrscheinlichkeit, dass der hoehere Seed die Series gewinnt.
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
    """Eine einzelne Best-of-7-Simulation, ELO-basiert.

    Returns True wenn der hoehere Seed gewinnt.
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
