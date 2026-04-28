# NBA Game Predictor

Machine-Learning-Projekt zur Vorhersage von NBA-Spielausgängen auf Basis historischer Daten (1947–heute).

## Ziel

Ein Modell entwickeln, das die Siegwahrscheinlichkeit für ein NBA-Spiel anhand von Team-Form, Head-to-Head-Bilanz, Spielerleistung und Kontext-Features berechnet — mit walk-forward Backtesting auf historischen Saisons.

## Daten

Quelle: [Historical NBA Data and Player Box Scores (Kaggle)](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)

Daten lokal in `data/raw/` ablegen (nicht im Repo, siehe `.gitignore`):
- `Games.csv`
- `PlayerStatistics.csv`
- `TeamStatistics.csv`
- `Players.csv`
- `TeamHistories.csv`
- u.a.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Projekt-Struktur

```
nba-game-predictor/
├── data/
│   ├── raw/          # Original-CSVs von Kaggle (gitignored)
│   └── processed/    # Aufbereitete Features (gitignored)
├── notebooks/        # Jupyter-Notebooks für Exploration & Analyse
├── src/              # Python-Module (Feature Engineering, Modelle)
├── models/           # Trainierte Modelle (gitignored)
├── tests/            # Unit-Tests
├── requirements.txt
└── README.md
```

## Roadmap

- [ ] Datenexploration (EDA)
- [ ] Feature Engineering (Team-Form, ELO, Head-to-Head, Spielerstats)
- [ ] Baseline-Modell (Logistische Regression)
- [ ] Hauptmodell (XGBoost)
- [ ] Walk-forward Backtesting (1947–heute)
- [ ] Evaluation (Accuracy, Log-Loss, Brier Score, Kalibrierung)

## Stack

Python · pandas · scikit-learn · XGBoost · matplotlib · Jupyter
