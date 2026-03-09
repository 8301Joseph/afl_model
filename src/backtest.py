import numpy as np
import pandas as pd


def run_backtest(feature_df, model, features, h2h_residuals=None, eval_seasons=None):
    """
    Evaluate model accuracy on all past games using pre-game ratings.

    Ratings in feature_df are already computed sequentially (pre-game state),
    so predictions are free of lookahead bias in the features. The Ridge model
    coefficients are trained on all games (minor lookahead), which is acceptable
    for a simplified backtest.

    H2H residuals, if provided, are computed from all games — so their contribution
    is evaluated separately to avoid double-counting. Pass h2h_residuals=None to
    assess the base model only.

    Returns a dict of summary metrics.
    """
    df = feature_df[["date", "season", "home_team", "away_team", "margin", "home_win"]].copy()
    df["base_pred_all"] = model.predict(feature_df[features].values)

    # Filter to evaluation seasons only (e.g. skip warm-up seasons)
    if eval_seasons is not None:
        df = df[df["season"].isin(eval_seasons)].copy()
    df = df.reset_index(drop=True)
    df["base_pred"] = df["base_pred_all"]
    df = df.drop(columns=["base_pred_all"])

    # H2H-adjusted predictions
    if h2h_residuals is not None:
        h2h_lookup = {
            (r["home_team"], r["away_team"]): r["adj_residual"]
            for _, r in h2h_residuals.iterrows()
        }
        df["h2h_bias"] = df.apply(
            lambda r: h2h_lookup.get((r["home_team"], r["away_team"]), 0.0), axis=1
        )
        df["adj_pred"] = df["base_pred"] + df["h2h_bias"]
    else:
        df["adj_pred"] = df["base_pred"]

    df["base_correct"]    = ((df["base_pred"] > 0) == (df["margin"] > 0)).astype(int)
    df["adj_correct"]     = ((df["adj_pred"]  > 0) == (df["margin"] > 0)).astype(int)
    df["home_always_correct"] = df["home_win"]   # baseline: always pick home

    def season_stats(subset, pred_col, correct_col):
        n      = len(subset)
        acc    = subset[correct_col].mean() * 100
        mae    = (subset["margin"] - subset[pred_col]).abs().mean()
        rmse   = np.sqrt(((subset["margin"] - subset[pred_col]) ** 2).mean())
        return {"n": n, "accuracy": acc, "mae": mae, "rmse": rmse}

    results = {}

    # Overall
    results["overall_base"] = season_stats(df, "base_pred", "base_correct")
    results["overall_adj"]  = season_stats(df, "adj_pred",  "adj_correct")
    results["overall_baseline"] = {
        "n":        len(df),
        "accuracy": df["home_always_correct"].mean() * 100,
        "mae":      df["margin"].abs().mean(),   # baseline "predicts" margin=0
        "rmse":     np.sqrt((df["margin"] ** 2).mean()),
    }

    # Per season
    results["by_season"] = {}
    for season, grp in df.groupby("season"):
        results["by_season"][season] = {
            "base":     season_stats(grp, "base_pred", "base_correct"),
            "adj":      season_stats(grp, "adj_pred",  "adj_correct"),
            "baseline": {
                "accuracy": grp["home_always_correct"].mean() * 100,
                "mae":      grp["margin"].abs().mean(),
            },
        }

    return results


def print_backtest(results):
    has_h2h = results["overall_adj"]["accuracy"] != results["overall_base"]["accuracy"]

    print("\n=== Backtest Results ===")
    print(f"  {'':30} {'Accuracy':>10}  {'MAE':>8}  {'RMSE':>8}")
    print(f"  {'-'*58}")

    def fmt_row(label, stats):
        print(f"  {label:<30} {stats['accuracy']:>9.1f}%  {stats['mae']:>8.1f}  {stats['rmse']:>8.1f}")

    fmt_row("Home-always baseline",    results["overall_baseline"])
    fmt_row("Model (base)",            results["overall_base"])
    if has_h2h:
        fmt_row("Model (base + H2H)",  results["overall_adj"])

    print(f"\n  {'Season':<10} {'N':>5}  {'Baseline':>10}  {'Model':>10}", end="")
    if has_h2h:
        print(f"  {'+ H2H':>10}", end="")
    print()
    print(f"  {'-'*58}")

    for season, stats in sorted(results["by_season"].items()):
        b  = stats["baseline"]
        m  = stats["base"]
        a  = stats["adj"]
        print(
            f"  {season:<10} {m['n']:>5}  {b['accuracy']:>9.1f}%  {m['accuracy']:>9.1f}%",
            end="",
        )
        if has_h2h:
            print(f"  {a['accuracy']:>9.1f}%", end="")
        print()
