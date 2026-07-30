"""
scripts/backtest_vs_market.py

Compares the model's walk-forward backtest probabilities against bookmaker
closing odds (aussportsbetting.com), and backtests flat-stake / fractional-
Kelly staking strategies on the model's edge vs those closing prices.

Usage:
    /Users/josephglasson/miniconda3/bin/python3 scripts/backtest_vs_market.py [--refresh-odds]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import importData
from src.elo import compute_elo_ratings
from src.ratings import compute_off_def_ratings, compute_home_advantages
from src.model import build_feature_matrix, train_model
from src.backtest import run_backtest
from src.metrics import brier_score, log_loss
from src.market import load_odds, attach_market_odds
from src.staking import simulate_staking

EVAL_SEASONS = [2024, 2025, 2026]


def build_eval_df():
    all_games, ratings_games, _ = importData()

    # Same feature set the live model uses (see main.py step 4) — only need the
    # `features` list here, so ratings/model themselves are thrown away.
    _, elo_hist_live = compute_elo_ratings(ratings_games)
    _, rat_hist_live = compute_off_def_ratings(ratings_games)
    home_advantages  = compute_home_advantages(ratings_games)
    feature_df_live  = build_feature_matrix(elo_hist_live, rat_hist_live, home_advantages)
    _, features, _   = train_model(feature_df_live)

    # Full history (2022+) for warm-start ratings in the walk-forward backtest.
    _, elo_hist_all = compute_elo_ratings(all_games)
    _, rat_hist_all = compute_off_def_ratings(all_games)
    ha_all          = compute_home_advantages(all_games)
    feature_df_all  = build_feature_matrix(elo_hist_all, rat_hist_all, ha_all)

    _, eval_df = run_backtest(feature_df_all, features, eval_seasons=EVAL_SEASONS, return_eval_df=True)
    return eval_df


def print_market_comparison(matched):
    def stats(prob_col, predicted_home_win):
        return {
            "accuracy": (predicted_home_win == matched["home_win"]).mean() * 100,
            "brier":    brier_score(matched, prob_col),
            "log_loss": log_loss(matched, prob_col),
        }

    baseline_p = matched.assign(_p=matched["home_win"].mean())
    rows = {
        "Home-always baseline": {
            "accuracy": matched["home_win"].mean() * 100,
            "brier":    brier_score(baseline_p, "_p"),
            "log_loss": log_loss(baseline_p, "_p"),
        },
        "Bookmaker (market)": stats("home_prob_market", matched["home_prob_market"] > 0.5),
        "Model (base)":       stats("base_prob", matched["base_prob"] > 0.5),
        "Model (base + H2H)": stats("adj_prob",  matched["adj_prob"]  > 0.5),
    }

    print(f"\n=== Model vs Bookmaker (n={len(matched)} games, {EVAL_SEASONS}, matched to closing odds) ===")
    print(f"  {'':24} {'Accuracy':>10}  {'Brier':>8}  {'LogLoss':>8}")
    print(f"  {'-'*56}")
    for label, s in rows.items():
        print(f"  {label:<24} {s['accuracy']:>9.1f}%  {s['brier']:>8.4f}  {s['log_loss']:>8.4f}")
    print("  Lower Brier/LogLoss is better. Beating the bookmaker on log loss is the real test —")
    print("  accuracy alone rewards confidently backing favourites, which the market already does.")


def print_calibration(matched, prob_col="adj_prob", n_bins=10):
    """
    Bucket games by predicted probability and compare predicted vs. actual
    home-win rate in each bucket, alongside the market's average probability
    for the same games — shows whether the model is over/under-confident,
    and where.
    """
    df = matched.dropna(subset=[prob_col, "home_prob_market"]).copy()
    df["bucket"] = pd.qcut(df[prob_col], q=n_bins, duplicates="drop")

    grouped = df.groupby("bucket", observed=True).agg(
        n=("home_win", "size"),
        avg_model_prob=(prob_col, "mean"),
        actual_win_rate=("home_win", "mean"),
        avg_market_prob=("home_prob_market", "mean"),
    )

    # Expected Calibration Error: n-weighted mean |predicted - actual| across buckets.
    ece = (grouped["n"] * (grouped["avg_model_prob"] - grouped["actual_win_rate"]).abs()).sum() / grouped["n"].sum()

    print(f"\n=== Calibration: {prob_col} vs. actual outcome ({len(df)} games, {n_bins} buckets) ===")
    print(f"  {'Bucket (model prob)':<22} {'N':>4}  {'Model':>7}  {'Actual':>7}  {'Market':>7}  {'Gap':>7}")
    print(f"  {'-'*62}")
    for bucket, row in grouped.iterrows():
        gap = row["avg_model_prob"] - row["actual_win_rate"]
        label = f"{max(bucket.left, 0):.2f}-{min(bucket.right, 1):.2f}"
        print(
            f"  {label:<22} {int(row['n']):>4}  {row['avg_model_prob']:>6.1%}  "
            f"{row['actual_win_rate']:>6.1%}  {row['avg_market_prob']:>6.1%}  {gap:>+6.1%}"
        )
    print(f"  {'-'*62}")
    print(f"  Expected Calibration Error (n-weighted): {ece:.1%}")
    print("  'Gap' > 0 means the model is overconfident in that bucket (predicts higher than it delivers).")
    print("  Compare 'Market' to 'Actual' too — if the market's own gap is smaller, it's better calibrated there.")


def print_staking(matched):
    print(f"\n=== Staking Backtest (model edge vs closing odds, {len(matched)} games w/ odds) ===")
    print(f"  {'Strategy':<28} {'Bets':>6}  {'Win%':>6}  {'Staked':>10}  {'Profit':>10}  {'ROI':>7}  {'MaxDD':>7}")
    print(f"  {'-'*80}")

    configs = [
        ("Flat stake (1% unit)",         "flat",  dict(flat_stake_frac=0.01)),
        ("Quarter-Kelly",                "kelly", dict(kelly_multiplier=0.25)),
        ("Half-Kelly",                   "kelly", dict(kelly_multiplier=0.50)),
        ("Market-following (sanity chk)", "flat",  dict(flat_stake_frac=0.01)),
    ]

    for label, mode, kwargs in configs:
        prob_col = "home_prob_market" if "sanity" in label else "adj_prob"
        summary, _ = simulate_staking(matched, prob_col=prob_col, mode=mode, **kwargs)
        print(
            f"  {label:<28} {summary['n_bets']:>6}  {summary['win_rate']:>5.1f}%  "
            f"${summary['total_staked']:>9,.0f}  ${summary['total_profit']:>9,.0f}  "
            f"{summary['roi_pct']:>+6.1f}%  {summary['max_drawdown_pct']:>6.1f}%"
        )
    print("  Sanity check uses the market's own de-vigged probability as the 'model' — it should")
    print("  land close to -vig% ROI with ~0 bets (the market never disagrees with itself).")


def main():
    refresh_odds = "--refresh-odds" in sys.argv

    print("Loading odds data...")
    odds_df = load_odds(refresh=refresh_odds)
    print(f"  {len(odds_df)} games with closing odds ({odds_df['date'].min().date()} to {odds_df['date'].max().date()})")

    print("Running walk-forward backtest...")
    eval_df = build_eval_df()
    print(f"  {len(eval_df)} backtested games across seasons {EVAL_SEASONS}")

    matched = attach_market_odds(eval_df, odds_df)
    matched = matched.dropna(subset=["home_odds_close", "away_odds_close"]).reset_index(drop=True)
    unmatched = len(eval_df) - len(matched)
    if unmatched:
        print(f"  {unmatched} backtested game(s) had no matching odds row (skipped)")

    print_market_comparison(matched)
    print_calibration(matched)
    print_staking(matched)


if __name__ == "__main__":
    main()
