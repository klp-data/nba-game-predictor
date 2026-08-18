"""Build the small derived files the dashboard reads.

Run from the repo root:

    python scripts/build_dashboard_data.py

Reads data/processed/ and models/xgb_baseline.pkl, writes ten files into
dashboard/data/ plus a copy of the model in dashboard/models/. Those two folders
are committed, so the hosted app never needs the Kaggle data or the 3.5 MB
feature parquet. Safe to re-run, every file is overwritten.

Two naming conventions in the outputs:

- ``teamName`` is the name a team carried at the time (Bullets in 1990), while
  ``franchise`` is the name it carries today (Wizards). Charts colour by
  franchise, labels use whichever reads better on the page.
- ``season`` is the starting year, so the 2026 playoffs are season 2025. The app
  displays that as 2025-26.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import bracket as bracket_mod
from src import elo as elo_mod

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT = ROOT / "dashboard" / "data"
OUT_MODELS = ROOT / "dashboard" / "models"

MODEL_FILE = "xgb_baseline.pkl"
K = 20                    # ELO step size, same as notebook 02
HOME_ADV = 100            # ELO home court bonus, same as notebook 02
HOLDOUT_SEASON = 2019     # the 2019+ holdout from notebook 03
RECENT_SEASONS = 3        # how far back a team counts as active
N_SIM = 20000             # bracket simulations per state, 4x NB08 to keep the noise down
H2H_WINDOW = 5

# Conference per current franchise. Not in the dataset, and it only decides which
# half of a bracket gets called East and which West, by majority vote.
CONFERENCE = {
    "Celtics": "East", "Nets": "East", "Knicks": "East", "76ers": "East",
    "Raptors": "East", "Bulls": "East", "Cavaliers": "East", "Pistons": "East",
    "Pacers": "East", "Bucks": "East", "Hawks": "East", "Hornets": "East",
    "Heat": "East", "Magic": "East", "Wizards": "East",
    "Nuggets": "West", "Timberwolves": "West", "Thunder": "West",
    "Trail Blazers": "West", "Jazz": "West", "Warriors": "West",
    "Clippers": "West", "Lakers": "West", "Suns": "West", "Kings": "West",
    "Mavericks": "West", "Rockets": "West", "Grizzlies": "West",
    "Pelicans": "West", "Spurs": "West",
}


def sim_seed(season: int, start_round: int) -> int:
    """Seed for one (season, state) simulation. Same convention as notebook 09."""
    return season * 10 + start_round


# --- loading -----------------------------------------------------------------
def load_games() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "games_with_features.parquet")
    return df.sort_values("gameDate").reset_index(drop=True)


def franchise_map(games: pd.DataFrame) -> pd.Series:
    """teamId -> the name the team carries in its most recent game."""
    stacked = pd.concat([
        games[["gameDate", "hometeamId", "hometeamName"]].rename(
            columns={"hometeamId": "teamId", "hometeamName": "name"}),
        games[["gameDate", "awayteamId", "awayteamName"]].rename(
            columns={"awayteamId": "teamId", "awayteamName": "name"}),
    ]).sort_values("gameDate")
    return stacked.drop_duplicates("teamId", keep="last").set_index("teamId")["name"]


# --- outputs -----------------------------------------------------------------
def build_elo_history(games: pd.DataFrame, franchises: pd.Series) -> pd.DataFrame:
    """Post-game ELO, two rows per game.

    The parquet only stores pre-game ELO, so the post-game value has to come from
    somewhere. Recomputing the whole history with src.elo would drift, because
    notebook 02 rounds every rating to one decimal and src.elo does not. So this
    replays a single update on top of the stored pre-game values, with the same
    rounding, which reproduces notebook 02 exactly. The check below proves it:
    a team's post-game ELO has to equal its pre-game ELO in its next game.
    """
    g = games
    exp_home = np.array([
        elo_mod.win_prob(h, a, is_home=True, home_adv=HOME_ADV)
        for h, a in zip(g.home_elo_pre, g.away_elo_pre)
    ])
    actual_home = g.home_win.to_numpy(dtype=float)
    home_post = np.round(g.home_elo_pre + K * (actual_home - exp_home), 1)
    away_post = np.round(g.away_elo_pre + K * ((1 - actual_home) - (1 - exp_home)), 1)

    home = pd.DataFrame({
        "teamId": g.hometeamId, "teamName": g.hometeamName,
        "gameDate": g.gameDate, "season": g.season,
        "elo": home_post, "elo_pre": g.home_elo_pre,
    })
    away = pd.DataFrame({
        "teamId": g.awayteamId, "teamName": g.awayteamName,
        "gameDate": g.gameDate, "season": g.season,
        "elo": away_post, "elo_pre": g.away_elo_pre,
    })
    hist = pd.concat([home, away], ignore_index=True).sort_values(["teamId", "gameDate"])

    # post-game ELO must equal the same team's pre-game ELO next time out
    nxt = hist.groupby("teamId")["elo_pre"].shift(-1)
    drift = (hist.elo - nxt).abs().dropna()
    print(f"  elo check: {len(drift):,} consecutive pairs, max drift {drift.max():.6f}, "
          f"mismatches {(drift > 1e-9).sum()}")

    hist["franchise"] = hist.teamId.map(franchises)
    return hist.drop(columns=["elo_pre"]).reset_index(drop=True)


def latest_snapshot(team_id: int, games: pd.DataFrame) -> dict | None:
    """The team's most recent feature row, whether they were home or away.

    Same logic as Demo.py, kept role-neutral on purpose: it is the last game the
    team played, not the last game they played at home.
    """
    played = games[(games.hometeamId == team_id) | (games.awayteamId == team_id)]
    if len(played) == 0:
        return None
    last = played.iloc[-1]
    side = "home" if last.hometeamId == team_id else "away"
    return {
        "elo": last[f"{side}_elo_pre"],
        "win_rate_5": last[f"{side}_win_rate_last_5"],
        "win_rate_10": last[f"{side}_win_rate_last_10"],
        "win_rate_20": last[f"{side}_win_rate_last_20"],
        "avg_margin_5": last[f"{side}_avg_margin_last_5"],
        "avg_margin_10": last[f"{side}_avg_margin_last_10"],
        "avg_margin_20": last[f"{side}_avg_margin_last_20"],
        "days_rest": last[f"{side}_days_since_last_game"],
        "is_b2b": last[f"{side}_is_back_to_back"],
        "last_date": last.gameDate,
    }


def active_teams(games: pd.DataFrame) -> pd.DataFrame:
    recent = games[games.season >= games.season.max() - RECENT_SEASONS]
    stacked = pd.concat([
        recent[["hometeamId", "hometeamName"]].rename(
            columns={"hometeamId": "teamId", "hometeamName": "teamName"}),
        recent[["awayteamId", "awayteamName"]].rename(
            columns={"awayteamId": "teamId", "awayteamName": "teamName"}),
    ])
    return stacked.drop_duplicates("teamId").sort_values("teamName").reset_index(drop=True)


def build_team_snapshots(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in teams.itertuples(index=False):
        snap = latest_snapshot(t.teamId, games)
        if snap is None:
            continue
        rows.append({"teamId": t.teamId, "teamName": t.teamName, **snap})
    return pd.DataFrame(rows)


def build_h2h(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Home win rate over the last five meetings, for every ordered pair.

    Same rule as Demo.py: the last five games between the two teams whoever was
    at home, and 0.5 if they have never met.
    """
    ids = teams.teamId.tolist()
    pair_key = games.hometeamId.astype(str) + "|" + games.awayteamId.astype(str)
    by_pair: dict[frozenset, list] = {}
    for key, hid, aid, hw in zip(pair_key, games.hometeamId, games.awayteamId, games.home_win):
        by_pair.setdefault(frozenset((hid, aid)), []).append(hid if hw == 1 else aid)

    rows = []
    for home_id in ids:
        for away_id in ids:
            if home_id == away_id:
                continue
            winners = by_pair.get(frozenset((home_id, away_id)), [])[-H2H_WINDOW:]
            rate = 0.5 if not winners else sum(w == home_id for w in winners) / len(winners)
            rows.append({"home_id": home_id, "away_id": away_id,
                         "home_winrate_last5": rate})
    return pd.DataFrame(rows)


def build_champions(games: pd.DataFrame, franchises: pd.Series) -> pd.DataFrame:
    """Champion and runner-up per season, from the last playoff game played.

    Names are franchise names, not era names, because the ELO explorer marks
    title seasons against franchises picked from a list of current teams.
    """
    playoffs = games[games.gameType == "Playoffs"]
    rows = []
    for season, grp in playoffs.groupby("season"):
        last = grp.sort_values("gameDate").iloc[-1]
        won = last.hometeamId if last.home_win == 1 else last.awayteamId
        lost = last.awayteamId if last.home_win == 1 else last.hometeamId
        rows.append({"season": int(season),
                     "champion": franchises[won],
                     "runner_up": franchises[lost]})
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def season_bracket(playoffs: pd.DataFrame, season: int):
    return bracket_mod.build_bracket(playoffs[playoffs.season == season])


def label_conferences(bracket: pd.DataFrame, franchises: pd.Series) -> dict:
    """teamId -> East or West, by majority vote over each half of the bracket."""
    labels = {}
    for half in bracket_mod.conference_halves(bracket):
        votes = pd.Series([CONFERENCE.get(franchises[t]) for t in half]).value_counts()
        conf = votes.index[0]
        for t in half:
            labels[t] = conf
    if len(set(labels.values())) != 2:
        raise ValueError("both halves of the bracket came out as the same conference")
    return labels


def rounds_reached(bracket: pd.DataFrame) -> dict:
    """teamId -> 1..5, where 5 is champion and 1 means lost in round 1."""
    reached = {}
    for row in bracket.itertuples(index=False):   # rows come in round order
        for t in (row.higher, row.lower):
            reached.setdefault(t, row.round)
        reached[row.winner] = row.round + 1
    return reached


def odds_for_season(playoffs: pd.DataFrame, season: int, games: pd.DataFrame,
                    franchises: pd.Series, n_sim: int = N_SIM) -> pd.DataFrame | None:
    brk = season_bracket(playoffs, season)
    if brk is None:
        return None
    elos = bracket_mod.pre_playoff_elos(playoffs[playoffs.season == season])
    teams = sorted(set(brk.higher) | set(brk.lower))
    elos = {t: elos[t] for t in teams}

    probs = {}
    for start_round in (1, 2, 3, 4):
        probs[start_round] = bracket_mod.championship_probs(
            brk, elos, n_sim=n_sim, seed=sim_seed(season, start_round),
            start_round=start_round)

    conf = label_conferences(brk, franchises)
    reached = rounds_reached(brk)
    champion = brk[brk["round"] == 4].iloc[0].winner
    era_names = (pd.concat([
        games[games.season == season][["hometeamId", "hometeamName"]].rename(
            columns={"hometeamId": "teamId", "hometeamName": "name"}),
        games[games.season == season][["awayteamId", "awayteamName"]].rename(
            columns={"awayteamId": "teamId", "awayteamName": "name"}),
    ]).drop_duplicates("teamId").set_index("teamId")["name"])

    rows = []
    for t in teams:
        rows.append({
            "season": int(season),
            "teamId": int(t),
            "team": era_names[t],
            "franchise": franchises[t],
            "conference": conf[t],
            "p_pre": probs[1][t],
            "p_after_r1": probs[2][t],
            "p_after_r2": probs[3][t],
            "p_after_r3": probs[4][t],
            "round_reached": int(reached[t]),
            "is_champion": bool(t == champion),
            "pre_playoff_elo": float(elos[t]),
        })
    out = pd.DataFrame(rows)
    out["elo_rank"] = (out.groupby("conference").pre_playoff_elo
                          .rank(ascending=False, method="first").astype(int))
    cols = ["season", "teamId", "team", "franchise", "conference", "elo_rank",
            "p_pre", "p_after_r1", "p_after_r2", "p_after_r3",
            "round_reached", "is_champion", "pre_playoff_elo"]
    return out[cols].sort_values("p_pre", ascending=False).reset_index(drop=True)


def build_playoff_odds(games: pd.DataFrame, franchises: pd.Series,
                       n_sim: int = N_SIM) -> pd.DataFrame:
    playoffs = games[games.gameType == "Playoffs"].copy()
    frames = []
    for season in sorted(playoffs.season.unique()):
        t0 = time.perf_counter()
        out = odds_for_season(playoffs, season, games, franchises, n_sim=n_sim)
        if out is None:
            continue
        frames.append(out)
        print(f"  season {season}: {time.perf_counter() - t0:5.1f}s  "
              f"top pick {out.iloc[0].team} {out.iloc[0].p_pre:.1%}")
    return pd.concat(frames, ignore_index=True)


def build_calibration(games: pd.DataFrame, model) -> pd.DataFrame:
    """Ten equal-width reliability bins on the 2019+ holdout."""
    feats = list(model.feature_names_in_)
    hold = games.dropna(subset=feats)
    hold = hold[hold.season >= HOLDOUT_SEASON]
    p = model.predict_proba(hold[feats])[:, 1]
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    rows = []
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin_low": edges[b], "bin_high": edges[b + 1],
                     "mean_pred": p[m].mean(),
                     "mean_obs": hold.home_win.to_numpy()[m].mean(),
                     "n": int(m.sum())})
    return pd.DataFrame(rows)


def build_feature_importance(model) -> pd.DataFrame:
    gain = model.get_booster().get_score(importance_type="gain")
    rows = [{"feature": f, "gain": float(gain.get(f, 0.0))}
            for f in model.feature_names_in_]
    out = pd.DataFrame(rows)
    out["share"] = out.gain / out.gain.sum()
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def build_season_summary(games: pd.DataFrame) -> pd.DataFrame:
    g = games.assign(total_points=games.homeScore + games.awayScore)
    out = g.groupby("season").agg(
        games=("gameId", "count"),
        home_win_rate=("home_win", "mean"),
        avg_total_points=("total_points", "mean"),
        avg_margin=("point_diff", "mean"),
    ).reset_index()
    return out


# --- main --------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)

    print("reading data/processed/games_with_features.parquet")
    games = load_games()
    franchises = franchise_map(games)
    model = joblib.load(MODELS / MODEL_FILE)
    teams = active_teams(games)
    print(f"  {len(games):,} games, {len(teams)} active teams, "
          f"last game {games.gameDate.max().date()}")

    written = []

    def save(name: str, frame: pd.DataFrame, header_comment: str | None = None):
        path = OUT / name
        if name.endswith(".parquet"):
            frame.to_parquet(path, index=False)
        else:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                if header_comment:
                    fh.write(f"# {header_comment}\n")
                frame.to_csv(fh, index=False)
        written.append((name, len(frame)))

    print("elo_history")
    save("elo_history.parquet", build_elo_history(games, franchises))

    print("team_snapshots")
    save("team_snapshots.parquet", build_team_snapshots(games, teams))

    print("h2h")
    save("h2h.parquet", build_h2h(games, teams))

    print("champions")
    save("champions.csv", build_champions(games, franchises))

    print(f"playoff_odds ({N_SIM:,} sims per state, seed = season * 10 + start_round)")
    save("playoff_odds.parquet", build_playoff_odds(games, franchises))

    print("backtest_metrics")
    save("backtest_metrics.csv", pd.read_csv(DATA / "backtest_metrics.csv"))

    print("calibration")
    save("calibration.csv", build_calibration(games, model),
         header_comment=f"reliability bins of models/{MODEL_FILE}, the 27-feature XGBoost "
                        f"from notebook 03, on seasons {HOLDOUT_SEASON} and later")

    print("feature_importance")
    save("feature_importance.csv", build_feature_importance(model))

    print("season_summary")
    save("season_summary.csv", build_season_summary(games))

    meta = {
        "last_game_date": str(games.gameDate.max().date()),
        "model_file": MODEL_FILE,
        "n_features": len(model.feature_names_in_),
        "n_sims": N_SIM,
        "sim_seed": "season * 10 + start_round",
        "build_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    shutil.copy(MODELS / MODEL_FILE, OUT_MODELS / MODEL_FILE)

    print("\nwritten to dashboard/data/")
    total = 0
    for name, n in written + [("meta.json", 0)]:
        size = (OUT / name).stat().st_size
        total += size
        print(f"  {name:28s} {n:>8,} rows   {size/1024:8.1f} KB")
    model_size = (OUT_MODELS / MODEL_FILE).stat().st_size
    total += model_size
    print(f"  models/{MODEL_FILE:21s} {'':>8}        {model_size/1024:8.1f} KB")
    print(f"  {'total':28s} {'':>8}        {total/1024/1024:8.2f} MB")


if __name__ == "__main__":
    main()
