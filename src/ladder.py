import pandas as pd
from src.data_prep import LEAGUE_AVG_SCORE


def build_ladder(past_games, predictions):
    """
    Build a projected end-of-season ladder combining:
      - Actual results from completed 2026 games (binary: 4 pts / 0 pts)
      - Probabilistic results for remaining games: each team earns
        expected points = 4 * win_probability, reflecting uncertainty

    Percentage uses actual scores for played games and margin-estimated
    scores (league_avg ± margin/2) for future games.

    Returns a DataFrame sorted by expected ladder position.
    """
    standings = {}  # team -> {exp_pts, exp_wins, pf, pa, played}

    def get(team):
        if team not in standings:
            standings[team] = {"exp_pts": 0.0, "exp_wins": 0.0, "pf": 0.0, "pa": 0.0, "played": 0}
        return standings[team]

    # --- Actual 2026 results (settled, binary) ---
    games_2026 = past_games[past_games["season"] == 2026]
    for _, row in games_2026.iterrows():
        home = row["Home Team"]
        away = row["Away Team"]
        hs   = row["home_score"]
        as_  = row["away_score"]
        m    = row["margin"]

        get(home)["pf"] += hs
        get(home)["pa"] += as_
        get(home)["played"] += 1
        get(away)["pf"] += as_
        get(away)["pa"] += hs
        get(away)["played"] += 1

        if m > 0:
            get(home)["exp_pts"]  += 4
            get(home)["exp_wins"] += 1
        elif m < 0:
            get(away)["exp_pts"]  += 4
            get(away)["exp_wins"] += 1
        else:
            get(home)["exp_pts"]  += 2
            get(away)["exp_pts"]  += 2
            get(home)["exp_wins"] += 0.5
            get(away)["exp_wins"] += 0.5

    # --- Probabilistic future games ---
    for _, row in predictions.iterrows():
        home     = row["home_team"]
        away     = row["away_team"]
        p_home   = row["win_prob_home"]       # model win probability for home team
        p_away   = 1.0 - p_home
        margin   = row["predicted_margin"]

        # Estimated scores from predicted margin
        est_home = LEAGUE_AVG_SCORE + margin / 2
        est_away = LEAGUE_AVG_SCORE - margin / 2

        get(home)["exp_pts"]  += 4 * p_home
        get(home)["exp_wins"] += p_home
        get(home)["pf"]       += est_home
        get(home)["pa"]       += est_away
        get(home)["played"]   += 1

        get(away)["exp_pts"]  += 4 * p_away
        get(away)["exp_wins"] += p_away
        get(away)["pf"]       += est_away
        get(away)["pa"]       += est_home
        get(away)["played"]   += 1

    # --- Build and sort ---
    rows = []
    for team, s in standings.items():
        pct = (s["pf"] / s["pa"] * 100) if s["pa"] > 0 else 0
        rows.append({
            "team":      team,
            "played":    s["played"],
            "exp_wins":  round(s["exp_wins"], 1),
            "exp_pts":   round(s["exp_pts"], 1),
            "pf":        round(s["pf"]),
            "pa":        round(s["pa"]),
            "pct":       round(pct, 1),
        })

    ladder = (
        pd.DataFrame(rows)
        .sort_values(["exp_pts", "pct"], ascending=[False, False])
        .reset_index(drop=True)
    )
    ladder.index += 1
    return ladder


def _format_ladder(ladder):
    lines = []
    lines.append("=== Projected 2026 Final Ladder ===")
    lines.append(f"  {'#':>2}  {'Team':<25} {'P':>3}  {'xW':>5}  {'xPts':>6}  {'%':>7}")
    lines.append(f"  {'-'*55}")
    for pos, row in ladder.iterrows():
        lines.append(
            f"  {pos:>2}  {row['team']:<25} {row['played']:>3}  "
            f"{row['exp_wins']:>5.1f}  {row['exp_pts']:>6.1f}  {row['pct']:>7.1f}%"
        )
        if pos == 8:
            lines.append(f"  {'·' * 55}")
    return "\n".join(lines)


def print_ladder(ladder):
    print("\n" + _format_ladder(ladder))


def save_ladder(ladder, path="ladder.txt"):
    from datetime import date
    content = f"Generated: {date.today()}\n\n" + _format_ladder(ladder) + "\n"
    with open(path, "w") as f:
        f.write(content)
    print(f"  Ladder saved to {path}")
