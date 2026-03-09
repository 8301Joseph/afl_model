import numpy as np

DEFAULT_RATING = 1500   # Starting Elo
HOME_ADVANTAGE  = 60    # Elo points added to home team's effective rating
K               = 35    # how fast ratings shift after each game
SEASON_REGRESS  = 0.25  # pull ratings 25% back toward 1500 each new season
MARGIN_CAP      = 60    # blowouts beyond this (pts) get no extra Elo punishment


def expected_win_prob(rating_a, rating_b):
    """Probability that team A beats team B given their Elo ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def margin_multiplier(margin):
    """
    Scale the Elo update by how big the win was.
    log(margin+1) gives diminishing returns — a 50pt win counts more than 10pt.
    Capped at MARGIN_CAP so blowouts (90+pt losses) don't disproportionately
    crater ratings; beyond that threshold the extra margin is mostly garbage time.
    """
    return np.log(min(abs(margin), MARGIN_CAP) + 1)


def update_elo(r_home, r_away, home_score, away_score):
    """
    Update and return new Elo ratings for both teams after one game.

    Args:
        r_home, r_away  — ratings before the game
        home_score, away_score — actual scores

    Returns:
        new_r_home, new_r_away
    """
    margin = home_score - away_score

    # Home team gets a rating boost for the expected-prob calculation only
    effective_home = r_home + HOME_ADVANTAGE
    e_home = expected_win_prob(effective_home, r_away)

    # Actual outcome from home team's perspective: 1=win, 0.5=draw, 0=loss
    if margin > 0:
        outcome = 1.0
    elif margin == 0:
        outcome = 0.5
    else:
        outcome = 0.0

    mov = margin_multiplier(margin)
    delta = K * mov * (outcome - e_home)

    new_r_home = r_home + delta
    new_r_away = r_away - delta   # zero-sum

    return new_r_home, new_r_away


def season_reset(ratings: dict) -> dict:
    """Regress all ratings toward DEFAULT_RATING at the start of a new season."""
    return {
        team: rating + SEASON_REGRESS * (DEFAULT_RATING - rating)
        for team, rating in ratings.items()
    }


def compute_elo_ratings(past_games):
    """
    Walk through all past games chronologically, updating Elo ratings.

    Returns:
        ratings      — dict of current Elo rating per team (after all past games)
        history      — list of dicts, one per game, with pre-game ratings recorded
    """
    ratings = {}  # team -> current Elo rating
    history = []
    current_season = None

    for _, row in past_games.iterrows():
        home = row["Home Team"]
        away = row["Away Team"]
        season = row["season"]

        # Initialise unseen teams at the default rating
        if home not in ratings:
            ratings[home] = DEFAULT_RATING
        if away not in ratings:
            ratings[away] = DEFAULT_RATING

        # Apply season reset when a new season starts
        if current_season is not None and season != current_season:
            ratings = season_reset(ratings)
        current_season = season

        r_home = ratings[home]
        r_away = ratings[away]
        e_home = expected_win_prob(r_home + HOME_ADVANTAGE, r_away)

        # Record pre-game state (used as features for the model)
        history.append({
            "match_number": row["Match Number"],
            "date":         row["Date"],
            "season":       season,
            "home_team":    home,
            "away_team":    away,
            "home_elo":     r_home,
            "away_elo":     r_away,
            "elo_diff":     r_home - r_away,
            "e_home":       e_home,
            "home_score":   row["home_score"],
            "away_score":   row["away_score"],
            "margin":       row["margin"],
            "home_win":     row["home_win"],
        })

        ratings[home], ratings[away] = update_elo(r_home, r_away, row["home_score"], row["away_score"])

    return ratings, history
