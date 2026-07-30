"""
src/market.py — historical bookmaker odds for AFL games.

Source: aussportsbetting.com's free historical odds spreadsheet. It has
opening/closing home & away decimal odds (averaged across bookmakers) for
every AFL game back to 2009, and is updated as the current season progresses.

Squiggle's public API was checked first (it's already used in sync.py) but
its "tips" are tipster/model predictions, not bookmaker prices — there's no
odds field on games or a bookmaker source in q=sources. This spreadsheet is
the actual bookmaker data.
"""
import numpy as np
import pandas as pd
import requests

ODDS_URL        = "https://www.aussportsbetting.com/historical_data/afl.xlsx"
ODDS_CACHE_PATH = "data/afl_odds.xlsx"

# aussportsbetting uses short team names — map them to data_prep.py's normalised CSV names.
ODDS_TO_CSV = {
    "Adelaide":         "Adelaide Crows",
    "Brisbane":         "Brisbane Lions",
    "GWS Giants":       "GWS Giants",
    "Geelong":          "Geelong Cats",
    "Gold Coast":       "Gold Coast Suns",
    "Sydney":           "Sydney Swans",
    "West Coast":       "West Coast Eagles",
}


def fetch_odds(cache_path=ODDS_CACHE_PATH):
    """Download the latest odds spreadsheet and cache it locally."""
    resp = requests.get(ODDS_URL, headers={"User-Agent": "afl-model/1.0"}, timeout=60)
    resp.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(resp.content)
    return cache_path


def load_odds(cache_path=ODDS_CACHE_PATH, refresh=False):
    """
    Load and normalise the odds spreadsheet.

    Returns a DataFrame with one row per game:
        date, home_team, away_team,
        home_odds_open,  away_odds_open,   (opening line)
        home_odds_close, away_odds_close,  (closing line)
        home_prob_market, away_prob_market (de-vigged implied probabilities, from the close)
    """
    import os
    if refresh or not os.path.exists(cache_path):
        fetch_odds(cache_path)

    df = pd.read_excel(cache_path, header=1)
    df = df.rename(columns={
        "Home Team":        "home_team",
        "Away Team":        "away_team",
        "Home Odds Open":   "home_odds_open",
        "Away Odds Open":   "away_odds_open",
        "Home Odds Close":  "home_odds_close",
        "Away Odds Close":  "away_odds_close",
    })[["Date", "home_team", "away_team", "home_odds_open", "away_odds_open",
        "home_odds_close", "away_odds_close"]]
    df["date"] = pd.to_datetime(df["Date"])
    df = df.drop(columns=["Date"])

    df["home_team"] = df["home_team"].replace(ODDS_TO_CSV)
    df["away_team"] = df["away_team"].replace(ODDS_TO_CSV)

    df = df.dropna(subset=["home_odds_close", "away_odds_close"]).reset_index(drop=True)

    # De-vig: raw 1/odds implied probs sum to > 1 (the bookmaker's overround/vig).
    # Normalising by that sum gives the market's true probability estimate.
    raw_home = 1 / df["home_odds_close"]
    raw_away = 1 / df["away_odds_close"]
    overround = raw_home + raw_away
    df["home_prob_market"] = raw_home / overround
    df["away_prob_market"] = raw_away / overround

    return df


def attach_market_odds(games_df, odds_df, date_col="date", tolerance_days=2):
    """
    Merge odds onto games_df by (home_team, away_team) and nearest date within
    tolerance_days — kickoff timestamps in the two sources use different
    timezone/local-time conventions, so an exact date match is not reliable.

    games_df must have columns: home_team, away_team, <date_col>.
    Returns games_df with home_odds_close, away_odds_close, home_prob_market,
    away_prob_market columns added (NaN where no match was found).
    """
    games = games_df.copy().reset_index(drop=True)
    games["_date"] = pd.to_datetime(games[date_col]).dt.tz_localize(None)

    odds = odds_df.copy()
    odds["_date"] = pd.to_datetime(odds["date"]).dt.tz_localize(None)

    matched_cols = ["home_odds_close", "away_odds_close", "home_prob_market", "away_prob_market"]
    for col in matched_cols:
        games[col] = np.nan

    for (home, away), grp in odds.groupby(["home_team", "away_team"]):
        game_idx = games.index[(games["home_team"] == home) & (games["away_team"] == away)]
        for idx in game_idx:
            game_date = games.loc[idx, "_date"]
            deltas = (grp["_date"] - game_date).abs()
            best = deltas.idxmin()
            if deltas.loc[best] <= pd.Timedelta(days=tolerance_days):
                for col in matched_cols:
                    games.loc[idx, col] = grp.loc[best, col]

    return games.drop(columns=["_date"])
