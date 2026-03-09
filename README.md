# AFL Model

Predicts AFL game margins and winners using Elo ratings, opponent-adjusted offensive/defensive ratings, and per-team home advantage.

## How it works

### Elo ratings (`src/elo.py`)
Each team has a single rating (starting at 1500) updated after every game. Upsets shift ratings more than expected results. A margin-of-victory multiplier (`log(margin+1)`) means bigger wins shift ratings more, with diminishing returns. Ratings regress 25% toward 1500 at the start of each new season to account for roster changes.

### Offensive / Defensive ratings (`src/ratings.py`)
Each team tracks how many points they typically score (`off`) and concede (`def`), updated via an opponent-adjusted exponentially weighted moving average (EWMA):

```
target_off = league_avg + (points_scored - opponent_def_rating)
new_off    = alpha * target_off + (1 - alpha) * old_off
```

Scoring 135 against a weak defence (def=111) is worth less than scoring 135 against a strong defence (def=82). Alpha = 0.2 — roughly the last 8–10 games dominate.

### Per-team home advantage (`src/ratings.py`)
Each team's historical average margin at home minus average margin away. Captures ground-specific effects (e.g. Brisbane at the Gabba) without splitting the limited EWMA sample.

### Linear model (`src/model.py`)
Ridge regression trained on all past games with exponential time-decay weights — 2026 games weighted ~4x more than 2024 games. Features:

| Feature | Description |
|---|---|
| `elo_diff` | Home Elo minus away Elo (dominant signal) |
| `off_diff` | Home adjusted offence minus away adjusted offence |
| `def_diff` | Away adjusted defence minus home adjusted defence |
| `home_team_advantage` | Per-team historical home ground boost |

Target: `margin = home_score - away_score`. Predicted winner is whichever side has positive predicted margin.

## Data

CSV files in `data/` sourced manually, one per season (`afl-2024-UTC.csv`, `afl-2025-UTC.csv`, `afl-2026-UTC.csv`). Format:

```
Match Number, Round Number, Date, Location, Home Team, Away Team, Result
```

Result is `"home_score - away_score"` for completed games, empty for future games.

## Usage

```bash
python main.py
```

Prints:
1. Current Elo + off/def ratings for all 18 teams
2. Model coefficients (raw and standardised)
3. Predicted winner and margin for all remaining 2026 games

## Project structure

```
src/
  data_prep.py   — load CSVs, parse scores, normalise team names
  elo.py         — sequential Elo rating system
  ratings.py     — opponent-adjusted EWMA off/def ratings + home advantage
  model.py       — feature matrix, Ridge regression, predictions
main.py          — entry point
data/            — season CSVs
```

## Planned improvements
- Head-to-head matchup adjustment (EWMA of margin in specific matchups)
- Backtest accuracy on 2024/2025 seasons
- Pull live data from Squiggle API instead of CSV
