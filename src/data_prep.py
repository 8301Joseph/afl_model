import pandas as pd

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
        past_games   — all completed games (2024, 2025, 2026 so far), sorted by date
        future_games — 2026 games not yet played (Result is NaN)
    """
    df24 = load_season(DATA_2024, 2024)
    df25 = load_season(DATA_2025, 2025)
    df26 = load_season(DATA_2026, 2026)

    # Split 2026 into played vs upcoming based on whether Result exists
    played_2026  = df26.dropna(subset=["Result"]).copy()
    future_games = df26[df26["Result"].isna()].copy()

    # Combine all completed games and parse scores
    past_games = pd.concat([df24, df25, played_2026], ignore_index=True)
    past_games = parse_scores(past_games)
    past_games = past_games.sort_values("Date").reset_index(drop=True)

    return past_games, future_games
