# NBA Game Predictor

A personal machine-learning project where I try to predict the outcomes of NBA games, playoff series, and championships using historical data from [Kaggle](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores) spanning 1947 to today.

## Why basketball?

The main idea is simple: use historical statistics to predict future game outcomes. But why basketball specifically, and not tennis, football, or biathlon?

It's a team sport with a long season. Unlike individual sports such as tennis or biathlon, the result does not depend on one athlete having a good or bad day. With five players on the court and 82 regular-season games, individual variance gets averaged out. This statistical consistency makes basketball well suited for machine-learning models.

High-scoring games are also more forgiving to model. An NBA game typically ends 110–105, not 2–1. If my model is off by a couple of baskets, the predicted winner often still holds. Compare that to football, where a single goal swing flips the result entirely. More scoring events means the final score is closer to the "true" performance gap between teams, and small modeling errors do not cascade into wrong predictions.

Finally, the data is rich and easy to come by. The NBA has been tracking detailed statistics for decades, and the basketball analytics community is huge. That means I can work with clean, well-documented datasets going back to 1947.

In the following sections, I will walk you through my approach, the data, and the results. I hope you enjoy the reasoning and the conclusions.

---

## Key results

| Metric | Value | Context |
|---|---:|---|
| Out-of-sample game accuracy | **64.8 %** | Test set: 2019–2025 |
| Walk-forward backtest accuracy | **68.0 %** | 66 seasons, retrained per year |
| Walk-forward AUC | **0.71** | Stable across all eras |
| Top-3 NBA champion hit rate | **75 %** | 30/40 seasons backtested |
| Top-5 NBA champion hit rate | **92.5 %** | 37/40 seasons backtested |
| Avg. P assigned to actual champion | **34 %** | 5.5× random baseline (6.25 %) |

---

## Pipeline overview

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | EDA on 73k games — home advantage trends, era effects, missing data |
| 02 | `02_feature_engineering.ipynb` | ELO ratings, rolling form (5/10/20 games), head-to-head, rest days, back-to-backs |
| 03 | `03_baseline_model.ipynb` | Trivial baseline → Logistic Regression → XGBoost on a fixed 2019 split |
| 04 | `04_backtesting.ipynb` | Walk-forward validation: retrain model each season, predict the next |
| 05 | `05_player_features.ipynb` | Box-score aggregation (shooting %, plus/minus, turnovers) |
| 06 | `06_advanced_features.ipynb` | Star availability + strength-of-schedule features |
| 07 | `07_series_simulation.ipynb` | Best-of-7 Monte Carlo with NBA 2-2-1-1-1 home court |
| 08 | `08_bracket_simulation.ipynb` | Full bracket Monte Carlo (10k sims × 40 seasons) |
| 09 | `09_conditional_predictions.ipynb` | Probability updating as playoff rounds resolve |
| 10 | `10_live_demo_2025.ipynb` | Live championship probabilities for the ongoing 2025–26 playoffs |

---

## Results

### Walk-forward accuracy over 66 seasons
![Walk-forward performance](docs/walk_forward.png)

Accuracy is stable around 65–70 % across eight decades of NBA basketball. The dip in 2020 (Covid empty-arena games) is visible — home advantage collapsed, and the model struggled because it had been trained on eras with stronger home court.

### Best-of-7 amplifies any edge
![Best of 7 amplifier](docs/bo7_amplifier.png)

A 60 % per-game probability becomes ≈ 71 % over a best-of-7 series. This is why series-level predictions are stronger than game-level predictions.

### Where does the actual champion land in our top picks?
![Bracket backtest](docs/bracket_backtest.png)

Across 40 backtested seasons (1983–2024), the actual NBA champion was the model's top pick **52.5 %** of the time and within its top 3 picks **75 %** of the time. Random baseline: 6.25 %.

### Confidence evolves through the playoffs
![Conditional confidence](docs/conditional_confidence.png)

Average probability assigned to the actual champion: 34 % pre-playoffs → 37 % after round 1 → 45 % after the conference semis → 66 % in the finals. Most of the uncertainty lives in round 1 — once the final 8 are set, the model gets much sharper.

### Live: 2025–26 playoffs (R1 in progress)
![Live 2026](docs/live_2026.png)

Current top picks (top-16 by ELO seeded into a standard bracket):
Thunder 47.6 %, Spurs 18.2 %, Celtics 9.8 %, Rockets 5.9 %, Pistons 4.2 %.

---

## What worked and what didn't

**Worked:**
- ELO with a 100-point home-court adjustment carries the model. It captures ~35 % of total feature importance on its own and is the foundation for all downstream simulations.
- The Best-of-7 Monte Carlo amplifies any per-game edge into a sharper series prediction. Bracket-level top-3 hit rate (75 %) is much better than per-game accuracy (65 %).
- Walk-forward validation is a more honest performance estimate than a single train/test split — and reveals that modern NBA seasons are *harder* to predict (home advantage has shrunk from ~66 % in the 60s to ~55 % today).

**Didn't (honest negative findings):**
- **Player box-score features added almost nothing.** Old model 64.23 % → with 24 added features 64.86 %. ELO and rolling form already encode "this team scores well and protects the ball" indirectly — the box-score features were largely redundant.
- **Star availability and strength-of-schedule features were marginal too.** Full model with 63 features tops out at 64.82 % accuracy — within noise of the 27-feature baseline. AUC and log-loss did improve modestly, so the new features made probabilities slightly more honest, just not the binary calls.

**The ceiling lesson:** team-level historical data seems to cap out around 65 % accuracy on NBA games. Breaking past it likely requires:
- Real-time injury status and confirmed starting lineups
- Player tracking / advanced metrics not in the box score
- Hyperparameter tuning with proper nested CV (Optuna)

---

## Tech stack

`Python` · `pandas` · `numpy` · `XGBoost` · `scikit-learn` · `matplotlib` · `seaborn` · `pyarrow` · `joblib`

---

## Reproduce

```bash
git clone https://github.com/klp-data/nba-game-predictor.git
cd nba-game-predictor
pip install -r requirements.txt
# Download the Kaggle "NBA Database" dataset to data/raw/
jupyter notebook notebooks/
# Run 01 → 10 in order
```

Dataset: [NBA Database on Kaggle](https://www.kaggle.com/datasets/wyattowalsh/basketball) (Wyatt Walsh).
