"""Sanity tests for src/elo.py.

Run from the project root:
    pytest tests/

These are deliberately small — the goal is to catch regressions in the ELO
math, not to prove correctness across all inputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import elo


def test_equal_ratings_no_home_advantage():
    # Two equal teams, no home court -> coin flip.
    p = elo.win_prob(1500, 1500, is_home=True, home_adv=0)
    assert abs(p - 0.5) < 1e-9


def test_home_court_helps():
    # Same skill, but home team should be favored once we add home_adv.
    p = elo.win_prob(1500, 1500, is_home=True, home_adv=100)
    assert p > 0.5


def test_400_elo_gap_is_classic_91_percent():
    # Standard ELO property: a 400-point lead -> ~91% win probability.
    p = elo.win_prob(1900, 1500, is_home=False, home_adv=0)
    assert 0.90 < p < 0.92


def test_winner_gains_loser_loses_same_amount():
    # ELO is zero-sum: total points across both teams stays constant.
    new_winner, new_loser = elo.update(1500, 1500, winner_was_home=True, k=20, home_adv=100)
    assert abs((new_winner + new_loser) - 3000) < 1e-9


def test_upset_swings_more_than_expected_win():
    # If a 1300 team beats a 1700 team at home, that should move ELO more
    # than if the 1700 team beat the 1300 team (which was expected anyway).
    upset_w, _ = elo.update(1300, 1700, winner_was_home=True, k=20)
    expected_w, _ = elo.update(1700, 1300, winner_was_home=True, k=20)
    assert (upset_w - 1300) > (expected_w - 1700)
