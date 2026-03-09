import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DECAY_LAMBDA = 0.002   # time-decay rate; half-life ≈ 350 days (~one season)


def build_feature_matrix(elo_history, ratings_history, home_advantages):
    """
    Merge Elo and off/def rating histories into a single feature dataframe,
    one row per past game.
    """
    elo_df     = pd.DataFrame(elo_history)
    ratings_df = pd.DataFrame(ratings_history)

    df = elo_df.merge(
        ratings_df.drop(columns=["date", "home_team", "away_team"]),
        on="match_number"
    )

    # Add per-team home advantage as a feature (points boost for playing at home)
    df["home_team_advantage"] = df["home_team"].map(home_advantages).fillna(0)

    return df


def time_decay_weights(dates, reference_date=None):
    """
    Exponential decay: recent games matter more.
    weight = exp(-lambda * days_since_game)
    2026 games get ~1.0, 2024 games get ~0.25.
    """
    if reference_date is None:
        reference_date = dates.max()
    days_ago = (reference_date - dates).dt.days.clip(lower=0)
    return np.exp(-DECAY_LAMBDA * days_ago)


def train_model(feature_df):
    """
    Train a Ridge regression to predict margin (home_score - away_score).

    Features used:
        elo_diff          — Elo rating difference (home minus away)
        off_diff          — offensive rating difference
        def_diff          — defensive rating difference (away_def - home_def)
        net_rating_diff   — overall net rating difference
        e_home            — Elo-implied win probability for home team

    Returns:
        model    — fitted Ridge model
        features — list of feature column names
    """
    features = ["elo_diff", "off_diff", "def_diff", "home_team_advantage"]
    target   = "margin"

    X = feature_df[features].values
    y = feature_df[target].values
    w = time_decay_weights(feature_df["date"]).values

    model = Ridge(alpha=1.0)
    model.fit(X, y, sample_weight=w)

    return model, features


def predict_games(model, features, current_elo, current_ratings, home_advantages, future_games):
    """
    Predict margin and winner for each upcoming game.

    Args:
        model           — fitted Ridge model
        features        — list of feature names the model expects
        current_elo     — dict of {team: elo_rating} after all past games
        current_ratings — dict of {team: {"off": float, "def": float}}
        future_games    — dataframe of upcoming games

    Returns:
        predictions dataframe
    """
    from src.elo import DEFAULT_RATING, HOME_ADVANTAGE, expected_win_prob
    from src.data_prep import LEAGUE_AVG_SCORE

    rows = []
    for _, game in future_games.iterrows():
        home = game["Home Team"]
        away = game["Away Team"]

        r_home = current_elo.get(home, DEFAULT_RATING)
        r_away = current_elo.get(away, DEFAULT_RATING)
        e_home = expected_win_prob(r_home + HOME_ADVANTAGE, r_away)
        elo_diff = r_home - r_away

        home_off = current_ratings.get(home, {}).get("off", LEAGUE_AVG_SCORE)
        home_def = current_ratings.get(home, {}).get("def", LEAGUE_AVG_SCORE)
        away_off = current_ratings.get(away, {}).get("off", LEAGUE_AVG_SCORE)
        away_def = current_ratings.get(away, {}).get("def", LEAGUE_AVG_SCORE)

        off_diff        = home_off - away_off
        def_diff        = away_def - home_def
        net_rating_diff = (home_off - home_def) - (away_off - away_def)

        feature_values = {
            "elo_diff":           elo_diff,
            "off_diff":           off_diff,
            "def_diff":           def_diff,
            "net_rating_diff":    net_rating_diff,
            "e_home":             e_home,
            "home_team_advantage": home_advantages.get(home, 0),
        }
        X = np.array([[feature_values[f] for f in features]])
        predicted_margin = model.predict(X)[0]

        rows.append({
            "date":             game["Date"],
            "round":            game["Round Number"],
            "home_team":        home,
            "away_team":        away,
            "location":         game["Location"],
            "home_elo":         round(r_home, 1),
            "away_elo":         round(r_away, 1),
            "win_prob_home":    round(e_home, 3),
            "predicted_margin": round(predicted_margin, 1),
            "predicted_winner": home if predicted_margin > 0 else away,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)