"""Sanity tests for the files under dashboard/data/.

Run from the project root:
    pytest tests/

The dashboard reads nothing but these files, so if one of them loses a column or
picks up a NaN the app breaks on a machine I cannot see. Rebuild them with
    python scripts/build_dashboard_data.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "dashboard" / "data"

pytestmark = pytest.mark.skipif(
    not (DATA / "meta.json").exists(),
    reason="dashboard/data not built yet, run scripts/build_dashboard_data.py")


def load(name):
    if name.endswith(".parquet"):
        return pd.read_parquet(DATA / name)
    return pd.read_csv(DATA / name, comment="#")


EXPECTED = {
    "elo_history.parquet": ["teamId", "teamName", "gameDate", "season", "elo", "franchise"],
    "team_snapshots.parquet": ["teamId", "teamName", "elo", "win_rate_5", "win_rate_10",
                               "win_rate_20", "avg_margin_5", "avg_margin_10",
                               "avg_margin_20", "days_rest", "is_b2b", "last_date"],
    "h2h.parquet": ["home_id", "away_id", "home_winrate_last5"],
    "playoff_odds.parquet": ["season", "teamId", "team", "franchise", "conference",
                             "elo_rank", "p_pre", "p_after_r1", "p_after_r2",
                             "p_after_r3", "round_reached", "is_champion",
                             "pre_playoff_elo"],
    "champions.csv": ["season", "champion", "runner_up"],
    "backtest_metrics.csv": ["season", "accuracy", "auc", "home_win_rate_actual"],
    "calibration.csv": ["bin_low", "bin_high", "mean_pred", "mean_obs", "n"],
    "feature_importance.csv": ["feature", "gain", "share"],
    "season_summary.csv": ["season", "games", "home_win_rate", "avg_total_points",
                           "avg_margin"],
}


@pytest.mark.parametrize("name,columns", sorted(EXPECTED.items()))
def test_file_loads_with_its_columns(name, columns):
    df = load(name)
    assert len(df) > 0
    missing = [c for c in columns if c not in df.columns]
    assert not missing, f"{name} is missing {missing}"


@pytest.mark.parametrize("name,columns", sorted(EXPECTED.items()))
def test_no_nans_in_key_columns(name, columns):
    df = load(name)
    bad = {c: int(df[c].isna().sum()) for c in columns if df[c].isna().any()}
    assert not bad, f"{name} has NaNs in {bad}"


def test_meta_has_what_the_app_reads():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    for key in ["last_game_date", "model_file", "n_features", "n_sims", "sim_seed",
                "build_timestamp"]:
        assert key in meta
    assert meta["n_features"] == 27


def test_model_travelled_with_the_data():
    assert (ROOT / "dashboard" / "models" / "xgb_baseline.pkl").exists()


def test_elo_history_is_two_rows_per_game():
    hist = load("elo_history.parquet")
    # every game contributes both teams, so the row count has to be even
    assert len(hist) % 2 == 0
    assert hist.elo.between(800, 2200).all()
    assert hist.franchise.nunique() == 30


def test_snapshots_cover_the_active_teams():
    snaps = load("team_snapshots.parquet")
    assert len(snaps) == 30
    assert snaps.teamId.is_unique
    assert snaps.win_rate_5.between(0, 1).all()


def test_h2h_is_every_ordered_pair():
    h2h = load("h2h.parquet")
    n = load("team_snapshots.parquet").teamId.nunique()
    assert len(h2h) == n * (n - 1)
    assert h2h.home_winrate_last5.between(0, 1).all()


def test_playoff_odds_shape():
    odds = load("playoff_odds.parquet")
    per_season = odds.groupby("season").size()
    assert (per_season == 16).all(), "every backtested season is a 16-team bracket"
    for col in ["p_pre", "p_after_r1", "p_after_r2", "p_after_r3"]:
        assert odds[col].between(0, 1).all()
    assert odds.round_reached.between(1, 5).all()
    champs = odds.groupby("season").is_champion.sum()
    assert (champs == 1).all(), "exactly one champion per season"
    # the champion is the only team that reaches round 5
    assert (odds[odds.round_reached == 5].is_champion).all()


def test_probabilities_sum_to_one_per_state():
    odds = load("playoff_odds.parquet")
    for col in ["p_pre", "p_after_r1", "p_after_r2", "p_after_r3"]:
        totals = odds.groupby("season")[col].sum()
        assert (totals - 1.0).abs().max() < 1e-9, f"{col} does not sum to 1"


def test_elo_rank_is_one_to_eight_per_conference():
    odds = load("playoff_odds.parquet")
    grouped = odds.groupby(["season", "conference"]).elo_rank
    assert grouped.min().eq(1).all()
    assert grouped.max().eq(8).all()
    assert grouped.nunique().eq(8).all()


# Conference comes from a majority vote over each half of the bracket, so it is
# worth pinning against two seasons I can check by hand.
KNOWN_CONFERENCES = {
    2025: {  # the 2026 playoffs
        "East": {"Knicks", "Celtics", "Cavaliers", "Pistons", "76ers", "Hawks",
                 "Magic", "Raptors"},
        "West": {"Thunder", "Spurs", "Nuggets", "Lakers", "Timberwolves", "Rockets",
                 "Suns", "Trail Blazers"},
    },
    2023: {  # the 2024 playoffs
        "East": {"Celtics", "Knicks", "Pacers", "Cavaliers", "76ers", "Heat",
                 "Magic", "Bucks"},
        "West": {"Mavericks", "Thunder", "Timberwolves", "Nuggets", "Clippers",
                 "Suns", "Lakers", "Pelicans"},
    },
}


@pytest.mark.parametrize("season", sorted(KNOWN_CONFERENCES))
def test_conferences_for_two_known_seasons(season):
    odds = load("playoff_odds.parquet")
    got = odds[odds.season == season]
    for conf, expected in KNOWN_CONFERENCES[season].items():
        assert set(got[got.conference == conf].franchise) == expected


def test_champions_agree_with_playoff_odds():
    champs = load("champions.csv").set_index("season").champion
    odds = load("playoff_odds.parquet")
    for season, grp in odds.groupby("season"):
        winner = grp[grp.is_champion].iloc[0].franchise
        assert champs[season] == winner, f"season {season}"


def test_feature_importance_shares_add_up():
    fi = load("feature_importance.csv")
    assert len(fi) == 27
    assert abs(fi.share.sum() - 1.0) < 1e-9


def test_calibration_bins_are_ordered():
    cal = load("calibration.csv")
    assert (cal.bin_high > cal.bin_low).all()
    assert cal.n.sum() > 8000, "the 2019+ holdout is about 8,700 games"
