"""
One-time walk-forward reconstruction of predictions for rounds OR through 8.

Simulates what the model would have predicted before each round was played,
using only results available at that point in the season. Writes results to
output/locked_predictions.json without overwriting any entries already there.

Usage:
    python scripts/recover_predictions.py
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import load_season, parse_scores
from src.elo import compute_elo_ratings
from src.ratings import compute_off_def_ratings, compute_home_advantages
from src.model import build_feature_matrix, train_model, predict_games, compute_h2h_residuals

LOCKED_PATH = Path("output/locked_predictions.json")
ROUND_ORDER = ["OR", "1", "2", "3", "4", "5", "6", "7", "8"]


def _game_key(home, away, date_val):
    return f"{home}|{away}|{str(date_val)[:10]}"


def run_pipeline(played_round_strs: list[str]):
    """
    Run the full model pipeline with 2026 results only for the given rounds.
    Returns a predictions DataFrame for all other 2026 games.
    """
    df22 = load_season("data/afl-2022-UTC.csv", 2022)
    df23 = load_season("data/afl-2023-UTC.csv", 2023)
    df24 = load_season("data/afl-2024-UTC.csv", 2024)
    df25 = load_season("data/afl-2025-UTC.csv", 2025)
    df26 = load_season("data/afl-2026-UTC.csv", 2026)

    played_2026 = df26[
        df26["Round Number"].isin(played_round_strs) & df26["Result"].notna()
    ].copy()
    future_games = df26[
        ~(df26["Round Number"].isin(played_round_strs) & df26["Result"].notna())
    ].copy()

    base = [df22, df23, df24, df25]
    if not played_2026.empty:
        base.append(played_2026)

    all_games = pd.concat(base, ignore_index=True)
    all_games = all_games[all_games["Result"].notna()].copy()
    all_games = parse_scores(all_games)
    all_games = all_games.sort_values("Date").reset_index(drop=True)
    ratings_games = all_games[all_games["season"] >= 2024].reset_index(drop=True)

    current_elo, elo_history       = compute_elo_ratings(ratings_games)
    current_ratings, rat_history   = compute_off_def_ratings(ratings_games)
    home_advantages                = compute_home_advantages(ratings_games)

    feature_df = build_feature_matrix(elo_history, rat_history, home_advantages)
    model, features, margin_std    = train_model(feature_df)
    h2h_residuals                  = compute_h2h_residuals(feature_df, model, features)

    return predict_games(
        model, features, current_elo, current_ratings, home_advantages,
        future_games, h2h_residuals, margin_std=margin_std,
    )


def main():
    locked = {}
    if LOCKED_PATH.exists():
        with open(LOCKED_PATH) as f:
            locked = json.load(f)
        print(f"Existing locked entries: {len(locked)}")

    played = []
    for target_round in ROUND_ORDER:
        print(f"\nSimulating predictions before Round {target_round} (played so far: {played or 'none'})...")
        preds = run_pipeline(played)

        target_preds = preds[preds["round"].astype(str) == str(target_round)]
        added = 0
        for _, row in target_preds.iterrows():
            key = _game_key(row["home_team"], row["away_team"], row["date"])
            if key not in locked:
                locked[key] = {
                    "predicted_margin": round(float(row["predicted_margin"]), 1),
                    "win_prob_home":    round(float(row["win_prob_home"]), 3),
                    "predicted_winner": row["predicted_winner"],
                    "h2h_bias":         round(float(row["h2h_bias"]), 1),
                    "locked_at":        f"recovered: pre-round-{target_round}",
                }
                added += 1

        print(f"  Round {target_round}: {len(target_preds)} games found, {added} locked")
        played.append(target_round)

    LOCKED_PATH.parent.mkdir(exist_ok=True)
    with open(LOCKED_PATH, "w") as f:
        json.dump(locked, f, indent=2)
    print(f"\nDone. {len(locked)} total locked predictions saved to {LOCKED_PATH}")


if __name__ == "__main__":
    main()
