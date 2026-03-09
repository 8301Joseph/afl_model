import pandas as pd

DATA_2022 = "data/afl-2022-UTC.csv"
DATA_2023 = "data/afl-2023-UTC.csv"
DATA_2024 = "data/afl-2024-UTC.csv"
DATA_2025 = "data/afl-2025-UTC.csv"
DATA_2026 = "data/afl-2026-UTC.csv"

LEAGUE_AVG_SCORE = 86  # 2025 AFL average, used to initialise ratings

# Normalise inconsistent team names across seasons
TEAM_NAME_MAP = {
    "GWS GIANTS":    "GWS Giants",
    "Gold Coast SUNS": "Gold Coast Suns",
}


def normalise_team_names(df):
    df = df.copy()
    df["Home Team"] = df["Home Team"].replace(TEAM_NAME_MAP)
    df["Away Team"] = df["Away Team"].replace(TEAM_NAME_MAP)
    return df


def parse_scores(df):
    """Split 'Result' column (e.g. '132 - 69') into home_score and away_score."""
    scores = df["Result"].str.split(" - ", expand=True).astype(int)
    df = df.copy()
    df["home_score"] = scores[0]
    df["away_score"] = scores[1]
    df["margin"] = df["home_score"] - df["away_score"]   # positive = home win (like basketball spread)
    df["home_win"] = (df["margin"] > 0).astype(int)
    return df


def load_season(path, season_year):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["season"] = season_year
    return normalise_team_names(df)


def importData():
    """
    Returns:
        all_games     — all completed games (2022–2026), used as backtest warm-up + evaluation
        ratings_games — completed games from 2024 onward, used for current ratings and model training
        future_games  — 2026 games not yet played (Result is NaN)

    2022 and 2023 are included in all_games so the backtest walk-forward starts with
    warm ratings at 2024 Round 1 rather than cold 1500s. They are excluded from
    ratings_games so current team strengths reflect only recent seasons.
    """
    df22 = load_season(DATA_2022, 2022)
    df23 = load_season(DATA_2023, 2023)
    df24 = load_season(DATA_2024, 2024)
    df25 = load_season(DATA_2025, 2025)
    df26 = load_season(DATA_2026, 2026)

    # Split 2026 into played vs upcoming based on whether Result exists
    played_2026  = df26.dropna(subset=["Result"]).copy()
    future_games = df26[df26["Result"].isna()].copy()

    all_games = pd.concat([df22, df23, df24, df25, played_2026], ignore_index=True)
    all_games = parse_scores(all_games)
    all_games = all_games.sort_values("Date").reset_index(drop=True)

    ratings_games = all_games[all_games["season"] >= 2024].reset_index(drop=True)

    return all_games, ratings_games, future_games
