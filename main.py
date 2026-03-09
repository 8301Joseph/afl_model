from src.data_prep import importData
from src.elo import compute_elo_ratings
from src.ratings import compute_off_def_ratings, compute_home_advantages
from src.model import build_feature_matrix, train_model, predict_games, compute_h2h_residuals


def main():
    # --- 1. Load data ---
    past_games, future_games = importData()
    print(f"Past games loaded:   {len(past_games)}")
    print(f"Future games loaded: {len(future_games)}\n")

    # --- 2. Compute ratings from all past games ---
    current_elo, elo_history         = compute_elo_ratings(past_games)
    current_ratings, ratings_history = compute_off_def_ratings(past_games)
    home_advantages                  = compute_home_advantages(past_games)

    # --- 3. Print current standings ---
    print("=== Current Elo Ratings ===")
    for team, rating in sorted(current_elo.items(), key=lambda x: -x[1]):
        off = current_ratings[team]["off"]
        def_ = current_ratings[team]["def"]
        print(f"  {team:<25} Elo: {rating:6.1f}   Off: {off:5.1f}   Def: {def_:5.1f}")

    # --- 4. Build features and train model on past games ---
    feature_df = build_feature_matrix(elo_history, ratings_history, home_advantages)
    model, features = train_model(feature_df)

    import numpy as np
    feature_stds    = feature_df[features].std()
    standardised    = model.coef_ * feature_stds.values

    print(f"\n=== Model Coefficients ===")
    print(f"  {'feature':<22} {'raw coef':>10}  {'std coef':>10}  (std coef = impact of 1 typical shift)")
    for name, coef, std_coef in zip(features, model.coef_, standardised):
        print(f"  {name:<22} {coef:>+10.3f}  {std_coef:>+10.2f}")
    print(f"  {'intercept':<22} {model.intercept_:>+10.3f}")

    # --- 5. Head-to-head matchup biases ---
    h2h_residuals = compute_h2h_residuals(feature_df, model, features)
    notable = h2h_residuals[h2h_residuals["n_games"] >= 2].head(15)
    print(f"\n=== Top H2H Matchup Biases (vs model expectation, min 2 games) ===")
    print(f"  {'Home':<25} {'Away':<25} {'Raw Avg':>8}  {'Adjusted':>8}  {'Games':>5}")
    for _, r in notable.iterrows():
        direction = "home runs hot" if r["adj_residual"] > 0 else "home runs cold"
        print(f"  {r['home_team']:<25} {r['away_team']:<25} {r['avg_residual']:>+8.1f}  {r['adj_residual']:>+8.1f}  {int(r['n_games']):>5}   ({direction})")

    # --- 6. Predict remaining 2026 games ---
    predictions = predict_games(model, features, current_elo, current_ratings, home_advantages, future_games, h2h_residuals)

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


if __name__ == "__main__":
    main()