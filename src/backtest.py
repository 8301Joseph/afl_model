import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.model import time_decay_weights, compute_h2h_residuals, win_prob_from_margin
from src.metrics import brier_score, log_loss


def run_backtest(feature_df, features, eval_seasons=None, return_eval_df=False):
    """
    Walk-forward backtest: for each round in the eval seasons, train the Ridge
    model on all games played before that round, then predict that round's games.

    Ratings in feature_df are already sequential (pre-game state), so there is
    no lookahead bias in the features or the model coefficients.

    H2H residuals are also computed walk-forward — only from games before each
    round — so they are fully out-of-sample.

    Returns a dict of summary metrics. If return_eval_df=True, returns
    (results, eval_df) where eval_df has one row per game with base_pred/adj_pred
    (margin) and base_prob/adj_prob (win probability) columns — useful for
    comparing against market odds game-by-game.
    """
    df = feature_df.copy().reset_index(drop=True)

    # Sort by date to ensure chronological order
    df = df.sort_values("date").reset_index(drop=True)

    eval_mask = df["season"].isin(eval_seasons) if eval_seasons else pd.Series(True, index=df.index)
    eval_df   = df[eval_mask].copy()

    base_preds = np.full(len(eval_df), np.nan)
    adj_preds  = np.full(len(eval_df), np.nan)
    base_probs = np.full(len(eval_df), np.nan)
    adj_probs  = np.full(len(eval_df), np.nan)

    # Walk forward date by date — games on the same date share a training cutoff
    game_dates = sorted(eval_df["date"].unique())
    eval_df = eval_df.reset_index(drop=True)

    for cutoff in game_dates:
        train = df[df["date"] < cutoff]

        if len(train) < 20:
            continue

        X_train = train[features].values
        y_train = train["margin"].values
        w_train = time_decay_weights(train["date"]).values

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train, sample_weight=w_train)

        # H2H residuals from training data only
        train_preds = model.predict(X_train)

        # Walk-forward margin std (for win-prob calibration) — training residuals only,
        # so probabilities are fully out-of-sample just like the margin predictions.
        margin_std = float(np.std(y_train - train_preds))

        h2h_df = train[["home_team", "away_team", "margin"]].copy()
        h2h_df["residual"] = (h2h_df["margin"] - train_preds).clip(-40, 40)
        K = 5
        h2h = (
            h2h_df.groupby(["home_team", "away_team"])["residual"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_residual", "count": "n_games"})
        )
        h2h["adj_residual"] = h2h["avg_residual"] * h2h["n_games"] / (h2h["n_games"] + K)
        h2h_lookup = dict(zip(zip(h2h["home_team"], h2h["away_team"]), h2h["adj_residual"]))

        # Predict all eval games on this date
        day_idx = eval_df.index[eval_df["date"] == cutoff]
        preds   = model.predict(eval_df.loc[day_idx, features].values)
        for i, idx in enumerate(day_idx):
            home = eval_df.loc[idx, "home_team"]
            away = eval_df.loc[idx, "away_team"]
            base_preds[idx] = preds[i]
            adj_preds[idx]  = preds[i] + h2h_lookup.get((home, away), 0.0)
            base_probs[idx] = win_prob_from_margin(base_preds[idx], margin_std)
            adj_probs[idx]  = win_prob_from_margin(adj_preds[idx], margin_std)

    eval_df["base_pred"] = base_preds
    eval_df["adj_pred"]  = adj_preds
    eval_df["base_prob"] = base_probs
    eval_df["adj_prob"]  = adj_probs

    # Drop any rounds we couldn't predict (shouldn't happen with warm-up data)
    eval_df = eval_df.dropna(subset=["base_pred"])

    eval_df["base_correct"]       = ((eval_df["base_pred"] > 0) == (eval_df["margin"] > 0)).astype(int)
    eval_df["adj_correct"]        = ((eval_df["adj_pred"]  > 0) == (eval_df["margin"] > 0)).astype(int)
    eval_df["home_always_correct"] = eval_df["home_win"]

    def season_stats(subset, pred_col, correct_col, prob_col=None):
        stats = {
            "n":        len(subset),
            "accuracy": subset[correct_col].mean() * 100,
            "mae":      (subset["margin"] - subset[pred_col]).abs().mean(),
            "rmse":     np.sqrt(((subset["margin"] - subset[pred_col]) ** 2).mean()),
        }
        if prob_col is not None:
            stats["brier"]    = brier_score(subset, prob_col)
            stats["log_loss"] = log_loss(subset, prob_col)
        return stats

    results = {
        "overall_base":     season_stats(eval_df, "base_pred", "base_correct", "base_prob"),
        "overall_adj":      season_stats(eval_df, "adj_pred",  "adj_correct",  "adj_prob"),
        "overall_baseline": {
            "n":        len(eval_df),
            "accuracy": eval_df["home_always_correct"].mean() * 100,
            "mae":      eval_df["margin"].abs().mean(),
            "rmse":     np.sqrt((eval_df["margin"] ** 2).mean()),
            # Baseline "probability" = home-ground win rate in the eval set, held constant.
            "brier":    brier_score(eval_df.assign(_p=eval_df["home_win"].mean()), "_p"),
        },
        "by_season": {},
    }

    for season, grp in eval_df.groupby("season"):
        results["by_season"][season] = {
            "base":     season_stats(grp, "base_pred", "base_correct", "base_prob"),
            "adj":      season_stats(grp, "adj_pred",  "adj_correct",  "adj_prob"),
            "baseline": {
                "accuracy": grp["home_always_correct"].mean() * 100,
                "mae":      grp["margin"].abs().mean(),
            },
        }

    if return_eval_df:
        return results, eval_df
    return results


def print_backtest(results):
    has_h2h = results["overall_adj"]["accuracy"] != results["overall_base"]["accuracy"]

    print("\n=== Backtest Results (walk-forward, fully out-of-sample) ===")
    print(f"  {'':30} {'Accuracy':>10}  {'MAE':>8}  {'RMSE':>8}  {'Brier':>8}  {'LogLoss':>8}")
    print(f"  {'-'*80}")

    def fmt_row(label, stats):
        brier    = f"{stats['brier']:>8.4f}"    if "brier"    in stats else f"{'-':>8}"
        loglossv = f"{stats['log_loss']:>8.4f}" if "log_loss" in stats else f"{'-':>8}"
        print(f"  {label:<30} {stats['accuracy']:>9.1f}%  {stats['mae']:>8.1f}  {stats['rmse']:>8.1f}  {brier}  {loglossv}")

    fmt_row("Home-always baseline",   results["overall_baseline"])
    fmt_row("Model (base)",           results["overall_base"])
    if has_h2h:
        fmt_row("Model (base + H2H)", results["overall_adj"])
    print("  (Brier score: mean squared error of win probability, 0=perfect, 0.25=coin-flip. Lower is better.)")

    print(f"\n  {'Season':<10} {'N':>5}  {'Baseline':>10}  {'Model':>10}", end="")
    if has_h2h:
        print(f"  {'+ H2H':>10}", end="")
    print()
    print(f"  {'-'*60}")

    for season, stats in sorted(results["by_season"].items()):
        b = stats["baseline"]
        m = stats["base"]
        a = stats["adj"]
        print(f"  {season:<10} {m['n']:>5}  {b['accuracy']:>9.1f}%  {m['accuracy']:>9.1f}%", end="")
        if has_h2h:
            print(f"  {a['accuracy']:>9.1f}%", end="")
        print()
