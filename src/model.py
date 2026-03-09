import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

DECAY_LAMBDA = 0.002   # time-decay rate; half-life ≈ 350 days (~one season)


def build_feature_matrix(elo_history, ratings_history, home_advantages):
    """
    Merge Elo and off/def rating histories into a single feature dataframe,
    one row per past game.
    """
    elo_df     = pd.DataFrame(elo_history)
    ratings_df = pd.DataFrame(ratings_history)

    # Both histories are built from the same loop over past_games in identical order,
    # so a positional join is correct. match_number resets each season and cannot be
    # used as a unique key across seasons.
    df = pd.concat(
        [elo_df.reset_index(drop=True),
         ratings_df.drop(columns=["match_number", "date", "home_team", "away_team"]).reset_index(drop=True)],
        axis=1
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

    # Empirical std of residuals — used to convert predicted margin → win probability
    residuals = y - model.predict(X)
    margin_std = float(np.std(residuals))

    return model, features, margin_std


def win_prob_from_margin(predicted_margin, margin_std):
    """
    P(home wins) = P(actual margin > 0) where actual ~ N(predicted_margin, margin_std^2).
    Equivalent to the normal CDF evaluated at predicted_margin / margin_std.
    """
    return float(norm.cdf(predicted_margin / margin_std))


def compute_h2h_residuals(feature_df, model, features):
    """
    For each (home_team, away_team) pair, compute the average residual:
        residual = actual_margin - model_predicted_margin

    Positive residual means the home team consistently outperforms what the model
    expects in this specific matchup (and vice versa).

    Returns a DataFrame with columns:
        home_team, away_team, avg_residual, n_games
    """
    X = feature_df[features].values
    predicted = model.predict(X)
    df = feature_df[["home_team", "away_team", "margin"]].copy()
    df["residual"] = (df["margin"] - predicted).clip(-40, 40)

    h2h = (
        df.groupby(["home_team", "away_team"])["residual"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_residual", "count": "n_games"})
    )

    # Shrink toward zero based on sample size: with few games, trust it less.
    # shrinkage = n / (n + k) where k controls how many games for "full trust".
    # k=5 means 2 games → 29% weight, 5 games → 50%, 10 games → 67%.
    K = 5
    h2h["shrinkage"] = h2h["n_games"] / (h2h["n_games"] + K)
    h2h["adj_residual"] = h2h["avg_residual"] * h2h["shrinkage"]

    h2h = h2h.sort_values("adj_residual", key=abs, ascending=False).reset_index(drop=True)
    return h2h


def predict_games(model, features, current_elo, current_ratings, home_advantages, future_games, h2h_residuals=None, margin_std=None):
    """
    Predict margin and winner for each upcoming game.

    Args:
        model           — fitted Ridge model
        features        — list of feature names the model expects
        current_elo     — dict of {team: elo_rating} after all past games
        current_ratings — dict of {team: {"off": float, "def": float}}
        future_games    — dataframe of upcoming games
        h2h_residuals   — optional DataFrame from compute_h2h_residuals; adds matchup bias
        margin_std      — residual std from training; if provided, win prob is margin-based
                          (more accurate than Elo-only); falls back to Elo if None

    Returns:
        predictions dataframe
    """
    # Build a lookup {(home, away): avg_residual} for quick access
    h2h_lookup = {}
    if h2h_residuals is not None:
        for _, r in h2h_residuals.iterrows():
            h2h_lookup[(r["home_team"], r["away_team"])] = r["adj_residual"]
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

        h2h_bias = h2h_lookup.get((home, away), 0.0)
        adjusted_margin = predicted_margin + h2h_bias

        # Win probability: prefer margin-based (calibrated to model residuals) over Elo-only
        if margin_std is not None:
            win_prob = win_prob_from_margin(adjusted_margin, margin_std)
        else:
            win_prob = e_home

        rows.append({
            "date":             game["Date"],
            "round":            game["Round Number"],
            "home_team":        home,
            "away_team":        away,
            "location":         game["Location"],
            "home_elo":         round(r_home, 1),
            "away_elo":         round(r_away, 1),
            "win_prob_home":    round(win_prob, 3),
            "predicted_margin": round(adjusted_margin, 1),
            "h2h_bias":         round(h2h_bias, 1),
            "predicted_winner": home if adjusted_margin > 0 else away,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)