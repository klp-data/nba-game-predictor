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

Current top picks (using the **real** R1 matchups + current series scores, then re-seeding by ELO for R2+):
Thunder 58.5 %, Spurs 21.6 %, Celtics 11.2 %, Pistons 3.2 %, Cavaliers 1.7 %.
Thunder jumped after going up 3-0 vs the Suns — once a team is one win away in a best-of-7, the conditional probability tilts hard.

---

## What I learned

Honestly, building the model was the easy part. Evaluating it without lying to myself turned out to be the more interesting work.

A few things stuck out:

**ELO does most of the heavy lifting.** A simple ELO rating with a 100-point home-court adjustment accounts for roughly 35% of total feature importance on its own. Everything else — rolling form, head-to-head, rest days — adds incremental value, but ELO alone gets you most of the way to a 65% accuracy model. A useful reminder that a well-chosen simple feature can outperform a long list of engineered ones.

**Walk-forward validation changed the picture.** Splitting the data into one fixed train/test cut gives a single accuracy number that looks fine but hides everything interesting. Retraining the model for every season from 1960 to 2025 and predicting only the next year revealed that modern NBA seasons are noticeably harder to predict than older ones — home advantage has shrunk from ~66% in the 1960s to ~55% today, and the model has to work harder to compensate. The single-split version of this story would have missed it entirely.

**Adding more features did not help much.** The first version of the model used 27 features (ELO, rolling form, rest, head-to-head) and reached 64.2% accuracy. Adding 24 box-score features brought it to 64.9%. Adding 12 more advanced features (star availability, strength of schedule) landed at 64.8%. Pretty much within noise on accuracy — though log-loss and AUC did improve a bit. So the new features made the *probabilities* slightly more honest without changing the binary call.

That last point is, I think, the real finding of the project: team-level historical data caps out somewhere around 65% accuracy. To break through that, the model probably needs the kind of stuff ELO and form just cannot absorb — real-time injury status, confirmed starting lineups, player tracking, advanced metrics not in the box score. Those are basically out-of-distribution events that team-level history can't predict.

There's also a methodological thing here: extending a model with more features is the obvious move, but documenting that the extensions *did not help* is actually more useful than spinning a 0.6 percentage-point gain as a breakthrough. Knowing where the ceiling is is part of knowing what the model is.

If I were to push this further the natural next steps would be hyperparameter tuning with proper nested CV (Optuna), probability calibration before feeding the per-game predictions into the bracket simulator, and a real player-level feature set built from injury reports and confirmed lineups. NB_11 is a first stab at the first two of those.

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
