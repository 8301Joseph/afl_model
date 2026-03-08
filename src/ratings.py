from src.data_prep import LEAGUE_AVG_SCORE

ALPHA = 0.3   # EWMA smoothing — higher = more weight on recent games


def compute_off_def_ratings(past_games):
    """
    Walk through past games chronologically, maintaining an exponentially
    weighted moving average (EWMA) of points scored and conceded per team.

    off_rating: how many points this team typically scores
    def_rating: how many points this team typically concedes (lower = better defence)

    Returns:
        ratings  — dict of {team: {"off": float, "def": float}} (current state)
        history  — list of dicts with pre-game ratings recorded per game
    """
    ratings = {}   # team -> {"off": float, "def": float}
    history = []

    for _, row in past_games.iterrows():
        home = row["Home Team"]
        away = row["Away Team"]

        # Initialise unseen teams at league average
        if home not in ratings:
            ratings[home] = {"off": LEAGUE_AVG_SCORE, "def": LEAGUE_AVG_SCORE}
        if away not in ratings:
            ratings[away] = {"off": LEAGUE_AVG_SCORE, "def": LEAGUE_AVG_SCORE}

        home_off = ratings[home]["off"]
        home_def = ratings[home]["def"]
        away_off = ratings[away]["off"]
        away_def = ratings[away]["def"]

        # predicted margin based on current ratings (before update)
        # home scores relative to away's defence, away scores relative to home's defence
        predicted_margin = (home_off - away_def) - (away_off - home_def)

        history.append({
            "match_number":      row["Match Number"],
            "date":              row["Date"],
            "home_team":         home,
            "away_team":         away,
            "home_off":          home_off,
            "home_def":          home_def,
            "away_off":          away_off,
            "away_def":          away_def,
            "off_diff":          home_off - away_off,   # home attacks harder?
            "def_diff":          away_def - home_def,   # away concedes more?
            "net_rating_diff":   (home_off - home_def) - (away_off - away_def),
            "predicted_margin_ratings": predicted_margin,
        })

        # Update ratings after the game using EWMA
        home_scored   = row["home_score"]
        home_conceded = row["away_score"]
        away_scored   = row["away_score"]
        away_conceded = row["home_score"]

        ratings[home]["off"] = ALPHA * home_scored   + (1 - ALPHA) * home_off
        ratings[home]["def"] = ALPHA * home_conceded + (1 - ALPHA) * home_def
        ratings[away]["off"] = ALPHA * away_scored   + (1 - ALPHA) * away_off
        ratings[away]["def"] = ALPHA * away_conceded + (1 - ALPHA) * away_def

    return ratings, history
