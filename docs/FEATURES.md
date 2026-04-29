# Daten & Features — Glossar

Erklärt jede Rohdatentabelle (von Kaggle) und jedes berechnete Feature im Modell. Geschrieben so, dass es auch ohne tiefes ML-Wissen verständlich ist.

---

## Teil 1 — Die Rohdaten von Kaggle

### `Games.csv` (~73 000 Zeilen, 1 pro Spiel)
Das Haupt-Dataset. Eine Zeile = ein Spiel.

| Spalte | Bedeutung |
|---|---|
| `gameId` | Eindeutige Spiel-ID |
| `gameDate` | Datum + Uhrzeit |
| `hometeamId`, `awayteamId` | Numerische IDs der Teams |
| `hometeamName`, `awayteamName` | Team-Namen (können sich historisch ändern: SuperSonics → Thunder) |
| `homeScore`, `awayScore` | Endpunktzahl |
| `winner` | ID des Siegers |
| `gameType` | `Regular Season`, `Playoffs`, `Pre-Season`, etc. |
| `attendance`, `arenaName`, `officials` | Kontext-Infos (im Modell nicht verwendet) |

### `PlayerStatistics.csv` (~1.6 Mio Zeilen, 1 pro Spieler pro Spiel)
Box-Score auf Spieler-Ebene. Sehr granular.

| Spalte | Bedeutung |
|---|---|
| `personId`, `firstName`, `lastName` | Spieler-Identifikation |
| `gameId` | Welches Spiel |
| `playerteamId` | Für welches Team gespielt |
| `numMinutes` | Gespielte Minuten (NaN = nicht eingesetzt) |
| `points`, `assists`, `reboundsTotal` | Box-Score-Basics |
| `fieldGoalsAttempted`, `fieldGoalsMade` | Wurfquoten-Rohdaten |
| `threePointersAttempted/Made` | Dreier-Statistiken |
| `freeThrowsAttempted/Made` | Freiwürfe |
| `steals`, `blocks`, `turnovers` | Defensive / Ballverlust |
| `plusMinusPoints` | +/- des Spielers während er auf dem Court war (wichtigster "all-in-one" Indikator) |
| `comment` | DNP-Grund: "DNP - Coach's Decision", "DND - Injury", "NWT - Suspension" — wichtig für Verfügbarkeits-Features |

### `TeamStatistics.csv`, `TeamStatisticsExtended.csv`
Team-aggregierte Box-Scores pro Spiel. Wir nutzen sie indirekt — die Aggregation passiert bei uns aus PlayerStatistics.

### `Players.csv`, `TeamHistories.csv`
Stammdaten (Geburtsdatum, Größe, Team-Historie). Im aktuellen Modell nicht verwendet.

### `LeagueSchedule24_25.csv`, `LeagueSchedule25_26.csv`
Spielplan der jeweiligen Saison (zukünftige Spiele).

### `PlayByPlay.parquet`
Sekundengenaue Spielereignisse. Sehr groß, im aktuellen Modell nicht verwendet — wäre eine Erweiterung.

---

## Teil 2 — Berechnete Features

Alle Features werden so gebaut, dass sie **kein Wissen aus dem aktuellen Spiel** enthalten — nur Daten *vor* dem Spiel.

### 2.1 ELO-Rating

ELO ist ein klassisches Schach-Bewertungssystem, hier auf Basketball übertragen.

| Feature | Bedeutung |
|---|---|
| `home_elo_pre` | ELO des Heimteams **vor** dem Spiel |
| `away_elo_pre` | ELO des Auswärtsteams vor dem Spiel |
| `elo_diff` | `home_elo_pre - away_elo_pre`. Positiv = Heim ist stärker |

Mechanik: Jedes Team startet bei 1500. Nach jedem Spiel bekommt der Sieger Punkte vom Verlierer — wenn ein Underdog gewinnt, gibt's mehr Punkte. K-Faktor 20, Heimvorteils-Bonus 100. ELO ~1700 = stark, ~1300 = schwach.

### 2.2 Rolling Form (letzte N Spiele)

Form-Indikatoren über verschiedene Zeitfenster. Reset nach jeder Saison.

| Feature | Bedeutung |
|---|---|
| `home_win_rate_last_5` | Anteil Siege des Heimteams in den letzten 5 Spielen (0 bis 1) |
| `home_win_rate_last_10` | dito, letzte 10 Spiele |
| `home_win_rate_last_20` | dito, letzte 20 Spiele |
| `away_win_rate_last_X` | dasselbe für Auswärtsteam |
| `win_rate_diff_X` | `home - away`. Wer ist gerade in besserer Form? |
| `home_avg_margin_last_X` | Durchschnittliche Punktedifferenz (z.B. +5.3 = im Schnitt 5 Punkte gewonnen) |
| `margin_diff_X` | `home_avg_margin - away_avg_margin` |

Warum mehrere Fenster? 5-Spiele-Fenster fängt sehr aktuelle Form (ist ein Star verletzt?), 20-Spiele-Fenster die fundamentale Stärke.

### 2.3 Pause / Müdigkeit

| Feature | Bedeutung |
|---|---|
| `home_days_since_last_game` | Tage seit letztem Spiel des Heimteams |
| `away_days_since_last_game` | dito Auswärts |
| `home_is_back_to_back` | 1, wenn das Heimteam gestern auch gespielt hat (sehr ermüdend) |
| `away_is_back_to_back` | dito Auswärts |
| `rest_diff` | `home_days - away_days`. Wer ist ausgeruhter? |

Back-to-Back-Spiele führen statistisch zu ~5% niedrigerer Win-Rate.

### 2.4 Head-to-Head

| Feature | Bedeutung |
|---|---|
| `h2h_home_winrate_last5` | Wie oft hat das Heimteam in den letzten 5 direkten Begegnungen mit *diesem spezifischen Gegner* gewonnen? |

Beispiel: Boston spielt Atlanta. Wenn Boston die letzten 5 H2H-Spiele alle gewonnen hat → 1.0. Hat Boston nur eines gewonnen → 0.2.

### 2.5 Player Box-Score (rolling, Team-Aggregat)

Aus `PlayerStatistics.csv` aggregiert: pro Team pro Spiel die Summen, dann rollender Schnitt über die letzten 10 Spiele.

| Feature | Bedeutung |
|---|---|
| `home_pts_roll10` | Durchschnitt erzielter Punkte des Heimteams über letzte 10 Spiele |
| `home_fg_pct_roll10` | Field-Goal-Prozentsatz (Treffer / Versuche, 0-1) |
| `home_tp_pct_roll10` | Drei-Punkte-Prozentsatz |
| `home_ft_pct_roll10` | Freiwurf-Prozentsatz |
| `home_ast_roll10` | Durchschnittliche Assists |
| `home_reb_roll10` | Durchschnittliche Rebounds |
| `home_tov_roll10` | Durchschnittliche Turnovers (Ballverluste) |
| `home_top3_pm_roll10` | Plus/Minus der drei Spieler mit den meisten Minuten — Proxy für Star-Power |

Plus jeweils `away_*` und `*_diff` Varianten.

### 2.6 Strength of Schedule (SoS)

**Was ist das?** Ein 7-Sieg-Lauf gegen Tank-Teams ist nicht so wertvoll wie ein 7-Sieg-Lauf gegen Top-Teams. SoS quantifiziert das.

| Feature | Bedeutung |
|---|---|
| `home_sos_last10` | Durchschnittliches ELO der **Gegner** des Heimteams in den letzten 10 Spielen. Höher = härtere Gegner |
| `away_sos_last10` | dito Auswärts |
| `sos_last10_diff` | Wer hatte den schwereren Spielplan? |
| `home_quality_win_rate_last10` | Anteil der letzten 10 Siege, die gegen Gegner mit ELO > 1500 ("über-Durchschnitt") errungen wurden |
| `home_sos_adj_margin_last10` | Punktedifferenz, korrigiert um Gegnerstärke. Formel: `margin - (1500 - opp_elo) / 25`. Ein +5 gegen einen 1700er-Gegner ist mehr wert als +5 gegen einen 1300er-Gegner |

### 2.7 Star Availability

Aus den Spielerdaten: hat das Team seine besten Spieler im Lineup?

| Feature | Bedeutung |
|---|---|
| `home_top5_avail_last3` | Durchschnitt: wie viele der saison-Top-5 Spieler (nach Gesamtminuten) haben in den letzten 3 Spielen gespielt? Wert zwischen 0 und 5 |
| `away_top5_avail_last3` | dito Auswärts |
| `top5_avail_diff` | Wer hat aktuell mehr Stars verfügbar? |

Identifiziert Verletzungswellen ohne explizite Injury-Reports — wenn die Stars plötzlich nicht mehr in den letzten Spielen waren, fällt der Wert.

---

## Teil 3 — Target & Output-Variablen

| Feature | Bedeutung |
|---|---|
| `home_win` | Zielvariable (0/1): hat das Heimteam gewonnen? |
| `point_diff` | Punktedifferenz (für Regression-Erweiterung nicht nötig) |
| `season` | NBA-Saison als Startjahr (Saison 2023-24 → `season=2023`) |

---

## Teil 4 — Wie das Modell sie nutzt

XGBoost lernt eigenständig **Wechselwirkungen** zwischen Features. Beispiel:
- Ein Team mit hoher `top5_avail_diff` *und* hoher `home_avg_margin_last_10` *und* `home_is_back_to_back=0` ist extrem favorisiert.
- Aber `home_avg_margin_last_10` allein heißt wenig, wenn `top5_avail_last3` zusammengebrochen ist (Stars verletzt).

Im Backtest waren die wichtigsten Features:
1. `elo_diff` (klar — relative Teamstärke)
2. `margin_diff_20` (mittelfristige Form-Differenz)
3. `home_top5_avail_last3` (Verletzungslage)
4. `sos_adj_margin_last10_diff` (qualitäts-bereinigte Form)
5. `home_top3_pm_roll10` (Star-Plus/Minus)

ELO trägt absolut den meisten Erklärungswert — alles andere sind Verfeinerungen.
