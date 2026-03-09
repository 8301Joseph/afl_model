import json
import os
from datetime import datetime, date

from src.data_prep import importData
from src.elo import compute_elo_ratings
from src.ratings import compute_off_def_ratings, compute_home_advantages
from src.model import build_feature_matrix, train_model, predict_games, compute_h2h_residuals, win_prob_from_margin
from src.backtest import run_backtest, print_backtest
from src.ladder import build_probabilistic_ladder, print_ladder, save_ladder


def compute_past_results(feature_df, model, features, margin_std, h2h_residuals, ratings_games):
    """Predict already-played 2026 games and compare to actual results."""
    import numpy as np

    df_2026 = feature_df[feature_df["season"] == 2026].copy()
    if df_2026.empty:
        return []

    rg_rounds = (
        ratings_games[ratings_games["season"] == 2026][["Home Team", "Away Team", "Date", "Round Number"]]
        .rename(columns={"Home Team": "home_team", "Away Team": "away_team", "Date": "date", "Round Number": "round"})
    )
    df_2026 = df_2026.merge(rg_rounds, on=["home_team", "away_team", "date"], how="left")
    df_2026["round"] = df_2026["round"].fillna("")

    h2h_lookup = {
        (r["home_team"], r["away_team"]): r["adj_residual"]
        for _, r in h2h_residuals.iterrows()
    } if h2h_residuals is not None else {}

    predicted_margins = model.predict(df_2026[features].values)

    rows = []
    for i, (_, row) in enumerate(df_2026.iterrows()):
        adjusted = predicted_margins[i] + h2h_lookup.get((row["home_team"], row["away_team"]), 0.0)
        wp = win_prob_from_margin(adjusted, margin_std)
        predicted_winner = row["home_team"] if adjusted > 0 else row["away_team"]
        actual_winner = row["home_team"] if row["margin"] > 0 else row["away_team"]
        rows.append({
            "date":             row["date"],
            "round":            row["round"],
            "home_team":        row["home_team"],
            "away_team":        row["away_team"],
            "home_score":       int(row["home_score"]),
            "away_score":       int(row["away_score"]),
            "actual_margin":    int(row["margin"]),
            "predicted_margin": round(float(adjusted), 1),
            "win_prob_home":    round(float(wp), 3),
            "predicted_winner": predicted_winner,
            "actual_winner":    actual_winner,
            "correct":          predicted_winner == actual_winner,
        })
    return rows


def save_output(predictions, ladder, past_results, path="output/predictions.json"):
    """Save predictions and ladder to JSON for the API to serve."""
    def _serialize(val):
        if isinstance(val, (date, datetime)):
            return val.isoformat()
        return val

    predictions_list = [
        {k: _serialize(v) for k, v in row.items()}
        for row in predictions.to_dict(orient="records")
    ]

    ladder_list = [
        {"position": pos, **{k: _serialize(v) for k, v in row.items()}}
        for pos, row in ladder.iterrows()
    ]

    past_results_list = [
        {k: _serialize(v) for k, v in row.items()}
        for row in past_results
    ]

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "predictions": predictions_list,
        "ladder": ladder_list,
        "results": past_results_list,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Output saved to {path}")


def main():
    # --- 1. Load data ---
    all_games, ratings_games, future_games = importData()
    print(f"All completed games:  {len(all_games)}  (2022–2026, used for backtest warm-up)")
    print(f"Ratings games:       {len(ratings_games)}  (2024–2026, used for current ratings & model)")
    print(f"Future games:        {len(future_games)}\n")

    # --- 2. Compute current ratings from 2024+ only ---
    # 2022/2023 are excluded so current team strengths reflect only recent seasons.
    current_elo, elo_history         = compute_elo_ratings(ratings_games)
    current_ratings, ratings_history = compute_off_def_ratings(ratings_games)
    home_advantages                  = compute_home_advantages(ratings_games)

    # --- 3. Print current standings ---
    print("=== Current Elo Ratings ===")
    for team, rating in sorted(current_elo.items(), key=lambda x: -x[1]):
        off = current_ratings[team]["off"]
        def_ = current_ratings[team]["def"]
        print(f"  {team:<25} Elo: {rating:6.1f}   Off: {off:5.1f}   Def: {def_:5.1f}")

    # --- 4. Build features and train model on 2024+ games ---
    feature_df = build_feature_matrix(elo_history, ratings_history, home_advantages)
    model, features, margin_std = train_model(feature_df)

    import numpy as np
    feature_stds    = feature_df[features].std()
    standardised    = model.coef_ * feature_stds.values

    print(f"\n=== Model Coefficients ===")
    print(f"  {'feature':<22} {'raw coef':>10}  {'std coef':>10}  (std coef = impact of 1 typical shift)")
    for name, coef, std_coef in zip(features, model.coef_, standardised):
        print(f"  {name:<22} {coef:>+10.3f}  {std_coef:>+10.2f}")
    print(f"  {'intercept':<22} {model.intercept_:>+10.3f}")

    # --- 5. Head-to-head matchup biases (2024+ only) ---
    h2h_residuals = compute_h2h_residuals(feature_df, model, features)
    notable = h2h_residuals[h2h_residuals["n_games"] >= 2].head(15)
    print(f"\n=== Top H2H Matchup Biases (vs model expectation, min 2 games) ===")
    print(f"  {'Home':<25} {'Away':<25} {'Raw Avg':>8}  {'Adjusted':>8}  {'Games':>5}")
    for _, r in notable.iterrows():
        direction = "home runs hot" if r["adj_residual"] > 0 else "home runs cold"
        print(f"  {r['home_team']:<25} {r['away_team']:<25} {r['avg_residual']:>+8.1f}  {r['adj_residual']:>+8.1f}  {int(r['n_games']):>5}   ({direction})")

    # --- 6. Backtest using all games (2022/2023 warm up ratings, evaluate 2024/2025) ---
    # Separate rating computation so the backtest walk-forward starts with warm Elos at 2024 R1.
    _, elo_hist_all  = compute_elo_ratings(all_games)
    _, rat_hist_all  = compute_off_def_ratings(all_games)
    ha_all           = compute_home_advantages(all_games)
    feature_df_all   = build_feature_matrix(elo_hist_all, rat_hist_all, ha_all)
    backtest_results = run_backtest(feature_df_all, features, eval_seasons=[2024, 2025])
    print_backtest(backtest_results)

    # --- 7. Predict remaining 2026 games ---
    predictions = predict_games(model, features, current_elo, current_ratings, home_advantages, future_games, h2h_residuals, margin_std=margin_std)

    print(f"\n=== 2026 Predictions ({len(predictions)} games) ===")
    for _, row in predictions.iterrows():
        date_str = row["date"].strftime("%d %b") if hasattr(row["date"], "strftime") else str(row["date"])
        margin   = row["predicted_margin"]
        prob     = row["win_prob_home"] * 100
        bias_str = f"  H2H: {row['h2h_bias']:+.1f}" if row["h2h_bias"] != 0.0 else ""
        print(
            f"  Rd {str(row['round']):<4} {date_str}  "
            f"{row['home_team']:<25} vs {row['away_team']:<25}  "
            f"Winner: {row['predicted_winner']:<25}  "
            f"Margin: {margin:+.1f}  HomeProbElo: {prob:.1f}%{bias_str}"
        )

    # --- 8. Projected final ladder ---
    ladder = build_probabilistic_ladder(ratings_games, predictions)
    print_ladder(ladder)
    save_ladder(ladder)
    past_results = compute_past_results(feature_df, model, features, margin_std, h2h_residuals, ratings_games)
    save_output(predictions, ladder, past_results)


if __name__ == "__main__":
    main()