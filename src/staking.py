"""
src/staking.py — simulate betting the model's edge against bookmaker closing odds.

Bets are placed on whichever side has positive expected value versus the
closing price (model_prob * decimal_odds > 1), sized either as a flat unit
stake or a fractional-Kelly stake. This is a backtest over historical closing
odds, not a live staking plan — it assumes bets could be placed at the closing
price, which in reality moves as market money comes in.
"""
import pandas as pd


def simulate_staking(
    df,
    prob_col,
    mode="kelly",
    kelly_multiplier=0.25,
    flat_stake_frac=0.01,
    starting_bankroll=1000.0,
    min_edge=0.00,
    max_bet_frac=0.2,
):
    """
    df must have: date, home_team, away_team, home_odds_close, away_odds_close,
    home_win, and prob_col (model's P(home wins)).

    mode="flat"  — stake a fixed unit (flat_stake_frac * starting_bankroll) on every +EV bet.
    mode="kelly" — stake kelly_multiplier * Kelly-optimal fraction of the *current* bankroll,
                   capped at max_bet_frac to limit single-bet blowup risk.

    Returns (summary dict, bet log DataFrame).
    """
    df = df.dropna(subset=["home_odds_close", "away_odds_close", prob_col])
    df = df.sort_values("date").reset_index(drop=True)

    bankroll = starting_bankroll
    peak = starting_bankroll
    max_drawdown = 0.0
    records = []

    for _, row in df.iterrows():
        p_home, p_away = row[prob_col], 1 - row[prob_col]
        home_odds, away_odds = row["home_odds_close"], row["away_odds_close"]

        ev_home = p_home * home_odds - 1
        ev_away = p_away * away_odds - 1

        if ev_home < min_edge and ev_away < min_edge:
            continue

        if ev_home >= ev_away:
            side, p, odds, edge, won = "home", p_home, home_odds, ev_home, row["home_win"] == 1
        else:
            side, p, odds, edge, won = "away", p_away, away_odds, ev_away, row["home_win"] == 0

        if mode == "flat":
            stake = starting_bankroll * flat_stake_frac
        elif mode == "kelly":
            b = odds - 1
            f_star = max(0.0, p - (1 - p) / b) * kelly_multiplier
            stake = bankroll * min(f_star, max_bet_frac)
        else:
            raise ValueError(f"unknown staking mode: {mode}")

        if stake <= 0:
            continue

        profit = stake * (odds - 1) if won else -stake
        bankroll += profit
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak > 0 else 0.0)

        records.append({
            "date": row["date"], "home_team": row["home_team"], "away_team": row["away_team"],
            "side": side, "model_prob": p, "odds": odds, "edge": edge,
            "stake": stake, "won": bool(won), "profit": profit, "bankroll": bankroll,
        })

    log = pd.DataFrame(records)
    total_staked = log["stake"].sum() if len(log) else 0.0
    total_profit = log["profit"].sum() if len(log) else 0.0

    summary = {
        "mode":              mode,
        "n_bets":            len(log),
        "win_rate":          log["won"].mean() * 100 if len(log) else 0.0,
        "total_staked":      total_staked,
        "total_profit":      total_profit,
        "roi_pct":           (total_profit / total_staked * 100) if total_staked else 0.0,
        "starting_bankroll": starting_bankroll,
        "final_bankroll":    bankroll,
        "max_drawdown_pct":  max_drawdown * 100,
    }
    return summary, log
