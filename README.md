# NBA Game Predictor

A machine-learning pipeline that predicts NBA game, series, and championship outcomes using historical data from 1947 to today.

## Highlights

| Metric | Value |
|---|---|
| Game-level accuracy (XGBoost, out-of-sample 2019+) | **64.8 %** |
| Game-level AUC | **0.705** |
| Series-level accuracy (Best-of-7) | **~74 %** |
| **Pre-playoff Top-1 champion pick (40-season backtest)** | **52.5 %** |
| Pre-playoff Top-3 hit rate | **75 %** |
| Average model confidence in the *actual* champion (before playoffs) | **34 %** *(random baseline: 6.25 %)* |

Across 40 historical seasons, the pre-playoff model placed the eventual champion in its **top 3** picks **75 %** of the time, and on the **#1 spot** **52.5 %** of the time — using only ELO ratings and the bracket structure.

## Live Demo: 2025-26 NBA Playoffs

The current playoffs are running. As of **2026-04-26**, here is the model's live view.

### Round 1 standings (so far)

| Higher Seed | ELO | Lower Seed | ELO | Series | Current Leader | Games Played |
|---|---|---|---|---|---|---|
| Thunder | 1758 | Suns | 1521 | 3-0 | Thunder | 3 |
| Spurs | 1707 | Trail Blazers | 1521 | 3-1 | Spurs | 4 |
| Celtics | 1685 | 76ers | 1513 | 3-1 | Celtics | 4 |
| Pistons | 1662 | Magic | 1540 | 2-1 | Magic | 3 |
| Nuggets | 1659 | Timberwolves | 1587 | 3-1 | Timberwolves | 4 |
| Cavaliers | 1631 | Raptors | 1516 | 2-2 | tied | 4 |
| Knicks | 1630 | Hawks | 1563 | 2-2 | tied | 4 |
| Lakers | 1625 | Rockets | 1598 | 3-1 | Lakers | 4 |

### Top-5 championship picks (top-16 ELO seeding)

| Team | P(Championship) |
|---|---|
| **Thunder** | **47.6 %** |
| Spurs | 18.2 %  |
| Celtics | 9.8 % |
| Rockets | 5.9 % |
| Pistons | 4.2 % |

OKC are the runaway favorites with ELO ~1760 — exceptional for a regular-season finisher. The model gives them nearly half of all simulated championships.

## Example: 2023-24 Season

![Pre-Playoff Title Probabilities 2023](assets/champ_probs_2023.png)

The model gave the Boston Celtics 58.1 % title probability before the playoffs began. They went on to win the Finals 4-1 against Dallas.

## Why Best-of-7 Matters

Even a modest per-game edge gets amplified disproportionately when stretched across a 7-game series:

![Series amplifier](assets/series_amplifier.png)

A 65 % per-game probability becomes an 80 % series probability. This amplification is what turns a mediocre game-level predictor into a genuinely useful championship predictor.

## Where the Uncertainty Lives

As more rounds resolve, the model's confidence in the actual champion grows:

![Conditional hit rate](assets/conditional_hitrate.png)

The biggest jump happens between *Round 2* and *Conference Finals* — once the final 4 are set, the model is highly confident about who wins it all.

## Architecture

```
nba-game-predictor/
├── src/                     # Reusable modules
│   ├── elo.py               # ELO system (pre-game ratings, win probabilities)
│   ├── series.py            # Best-of-7 simulation (analytical + Monte Carlo)
│   ├── bracket.py           # Bracket construction + championship probabilities
│   └── plot_style.py        # Publication-ready matplotlib theme
├── notebooks/               # Development journey across 10 steps
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_backtesting.ipynb
│   ├── 05_player_features.ipynb
│   ├── 06_advanced_features.ipynb
│   ├── 07_series_simulation.ipynb
│   ├── 08_bracket_simulation.ipynb
│   ├── 09_conditional_predictions.ipynb
│   └── 10_live_demo_2025.ipynb       # Live demo of the ongoing season
├── docs/
│   └── FEATURES.md                   # Detailed glossary of all data & features
├── scripts/
│   └── generate_highlight_plots.py
├── assets/                  # Highlight PNGs for README
├── data/                    # Gitignored (1.8 GB Kaggle dataset)
├── models/                  # Gitignored (trained models)
├── requirements.txt
└── README.md
```

## Pipeline

1. **EDA** — 73 000 NBA games from 1947 to 2025; home-win rate 61.6 %; scoring drift across decades.
2. **Feature Engineering** — Rolling team form (5/10/20 games), head-to-head (last 5), rest days, back-to-back detection, ELO rating (`K=20`, home advantage 100).
3. **Baseline Models** — Logistic regression as sanity check, XGBoost as main model.
4. **Walk-Forward Backtesting** — Model retrained each year on all prior seasons, tested on the next.
5. **Player Box-Score Features** — Team-aggregated FG%, 3P%, plus/minus of the top 3 minute-getters (rolling).
6. **Advanced Features** — Top-5 star availability, strength-of-schedule-adjusted form, quality wins.
7. **Series Simulation** — Best-of-7 with the NBA's 2-2-1-1-1 home-court pattern.
8. **Full Bracket Monte Carlo** — Complete playoff tree, 10 000 simulations per season, championship probability per team.
9. **Conditional Predictions** — How does the hit rate change when earlier rounds are known?
10. **Live Demo** — Predictions for the ongoing 2025-26 season.

A detailed glossary of every data table and every engineered feature lives in [`docs/FEATURES.md`](docs/FEATURES.md).

## Tech Stack

Python · pandas · scikit-learn · XGBoost · matplotlib · seaborn · Jupyter

## Setup

```bash
git clone https://github.com/klp-data/nba-game-predictor.git
cd nba-game-predictor
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download the data from [Kaggle](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores) and unpack it into `data/raw/`.

## Reproducing the Results

```bash
# Run notebooks in order (01 → 10)
jupyter lab

# Regenerate the README highlight plots
python scripts/generate_highlight_plots.py
```

## Methodological Notes

- **No data leakage:** every rolling feature uses `.shift(1)` before aggregation. ELO is updated chronologically. The walk-forward backtest only ever trains on games strictly before the test window.
- **Walk-forward instead of random splits:** sports data are time-dependent, and a random split would inflate accuracy unrealistically.
- **Series simulation:** Best-of-7 with the NBA 2-2-1-1-1 format and a home-court adjustment of roughly 7 percentage points (matching ELO home advantage 100 / 400 ≈ 7 %).
