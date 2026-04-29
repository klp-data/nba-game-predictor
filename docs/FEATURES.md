# Data & Features — Glossary

Explains every raw data table (from Kaggle) and every engineered feature used by the model. Written so it makes sense without deep ML background.

---

## Part 1 — The Raw Kaggle Data

### `Games.csv` (~73 000 rows, one per game)
The main dataset. Each row is one game.

| Column | Meaning |
|---|---|
| `gameId` | Unique game identifier |
| `gameDate` | Date and time |
| `hometeamId`, `awayteamId` | Numerical team IDs |
| `hometeamName`, `awayteamName` | Team names (can change historically: SuperSonics → Thunder) |
| `homeScore`, `awayScore` | Final score |
| `winner` | ID of the winning team |
| `gameType` | `Regular Season`, `Playoffs`, `Pre-Season`, etc. |
| `attendance`, `arenaName`, `officials` | Context info (not used by the model) |

### `PlayerStatistics.csv` (~1.6 M rows, one per player per game)
Player-level box scores. Very granular.

| Column | Meaning |
|---|---|
| `personId`, `firstName`, `lastName` | Player identification |
| `gameId` | Which game |
| `playerteamId` | Which team the player suited up for |
| `numMinutes` | Minutes played (NaN = did not appear) |
| `points`, `assists`, `reboundsTotal` | Box-score basics |
| `fieldGoalsAttempted`, `fieldGoalsMade` | Shooting raw counts |
| `threePointersAttempted/Made` | Three-point stats |
| `freeThrowsAttempted/Made` | Free throws |
| `steals`, `blocks`, `turnovers` | Defensive stats / ball loss |
| `plusMinusPoints` | Player's plus/minus while on court (the most useful single all-in-one indicator) |
| `comment` | DNP reason: "DNP - Coach's Decision", "DND - Injury", "NWT - Suspension" — important for availability features |

### `TeamStatistics.csv`, `TeamStatisticsExtended.csv`
Team-aggregated box scores per game. We use them indirectly — our aggregation is built up from `PlayerStatistics`.

### `Players.csv`, `TeamHistories.csv`
Static metadata (date of birth, height, team history). Not used by the current model.

### `LeagueSchedule24_25.csv`, `LeagueSchedule25_26.csv`
Schedule for the respective season (future games).

### `PlayByPlay.parquet`
Second-by-second event log. Very large; not used yet — would be a natural extension.

---

## Part 2 — Engineered Features

Every feature is constructed so it contains **no information from the current game** — only data from strictly before the game starts.

### 2.1 ELO Rating

ELO is a classic rating system from chess, adapted to basketball.

| Feature | Meaning |
|---|---|
| `home_elo_pre` | Home team's ELO **before** the game |
| `away_elo_pre` | Away team's ELO before the game |
| `elo_diff` | `home_elo_pre - away_elo_pre`. Positive = home is stronger |

How it works: every team starts at 1500. After each game the winner takes points from the loser — if an underdog wins, the swing is bigger. K-factor 20, home-court bonus 100. ELO ~1700 means strong, ~1300 means weak.

### 2.2 Rolling Form (last N games)

Form indicators across multiple windows. Resets each season.

| Feature | Meaning |
|---|---|
| `home_win_rate_last_5` | Share of wins in the home team's last 5 games (0–1) |
| `home_win_rate_last_10` | Same, last 10 games |
| `home_win_rate_last_20` | Same, last 20 games |
| `away_win_rate_last_X` | Same metrics for the away team |
| `win_rate_diff_X` | `home - away`. Who is in better form right now? |
| `home_avg_margin_last_X` | Average point differential (e.g. +5.3 = won by 5 on average) |
| `margin_diff_X` | `home_avg_margin - away_avg_margin` |

Why multiple windows? A 5-game window captures very recent form (is a star out injured?), the 20-game window captures fundamental strength.

### 2.3 Rest / Fatigue

| Feature | Meaning |
|---|---|
| `home_days_since_last_game` | Days since the home team's last game |
| `away_days_since_last_game` | Same for the away team |
| `home_is_back_to_back` | 1 if the home team also played yesterday (very tiring) |
| `away_is_back_to_back` | Same for away |
| `rest_diff` | `home_days - away_days`. Who is more rested? |

Back-to-back games statistically lead to ~5 % lower win rate.

### 2.4 Head-to-Head

| Feature | Meaning |
|---|---|
| `h2h_home_winrate_last5` | How often has the home team won the last 5 head-to-head meetings against *this specific opponent*? |

Example: Boston plays Atlanta. If Boston has won the last 5 H2H meetings → 1.0. If Boston only won one → 0.2.

### 2.5 Player Box Score (rolling, team aggregate)

Aggregated from `PlayerStatistics.csv`: per team per game we sum up the stats, then take rolling averages over the last 10 games.

| Feature | Meaning |
|---|---|
| `home_pts_roll10` | Average points scored by the home team over the last 10 games |
| `home_fg_pct_roll10` | Field goal percentage (made / attempted, 0–1) |
| `home_tp_pct_roll10` | Three-point percentage |
| `home_ft_pct_roll10` | Free-throw percentage |
| `home_ast_roll10` | Average assists |
| `home_reb_roll10` | Average rebounds |
| `home_tov_roll10` | Average turnovers |
| `home_top3_pm_roll10` | Plus/minus of the three players with the most minutes — a star-power proxy |

Plus matching `away_*` and `*_diff` variants.

### 2.6 Strength of Schedule (SoS)

**The idea:** a 7-game winning streak against tank teams is not as impressive as one against contenders. SoS quantifies that.

| Feature | Meaning |
|---|---|
| `home_sos_last10` | Average ELO of the **opponents** the home team faced in its last 10 games. Higher = tougher schedule |
| `away_sos_last10` | Same for the away team |
| `sos_last10_diff` | Who had the harder schedule? |
| `home_quality_win_rate_last10` | Share of the home team's last 10 wins that came against opponents with ELO > 1500 ("above average") |
| `home_sos_adj_margin_last10` | Point differential, adjusted for opponent strength. Formula: `margin - (1500 - opp_elo) / 25`. A +5 margin against a 1700-ELO opponent counts more than +5 against a 1300-ELO one |

### 2.7 Star Availability

Built from player data: did the team have its best players in the lineup recently?

| Feature | Meaning |
|---|---|
| `home_top5_avail_last3` | Average number of the season's top-5 players (by total minutes) that played in the home team's last 3 games. Value between 0 and 5 |
| `away_top5_avail_last3` | Same for the away team |
| `top5_avail_diff` | Who has more stars currently available? |

Detects injury waves without needing explicit injury reports — when stars suddenly disappear from recent games, this value drops.

---

## Part 3 — Target & Output Variables

| Feature | Meaning |
|---|---|
| `home_win` | Target variable (0/1): did the home team win? |
| `point_diff` | Point differential (useful for a regression-style extension) |
| `season` | NBA season as start year (2023-24 → `season=2023`) |

---

## Part 4 — How the Model Uses Them

XGBoost learns **interactions** between features automatically. For example:
- A team with high `top5_avail_diff` *and* high `home_avg_margin_last_10` *and* `home_is_back_to_back=0` is heavily favored.
- But `home_avg_margin_last_10` alone means little if `top5_avail_last3` has collapsed (stars injured).

In the backtest the most important features were:
1. `elo_diff` (clearly — relative team strength)
2. `margin_diff_20` (medium-term form difference)
3. `home_top5_avail_last3` (injury status)
4. `sos_adj_margin_last10_diff` (quality-adjusted form)
5. `home_top3_pm_roll10` (star plus/minus)

ELO carries by far the most explanatory weight — everything else is refinement on top of it.
