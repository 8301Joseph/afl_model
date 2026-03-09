import pandas as pd
from src.data_prep import LEAGUE_AVG_SCORE


def _init_standings():
    return {}


def _get(standings, team):
    if team not in standings:
        standings[team] = {"pts": 0.0, "wins": 0.0, "pf": 0.0, "pa": 0.0, "played": 0}
    return standings[team]


def _add_completed_games(standings, past_games):
    """Populate standings with actual 2026 results (binary, settled)."""
    games_2026 = past_games[past_games["season"] == 2026]
    for _, row in games_2026.iterrows():
        home = row["Home Team"]
        away = row["Away Team"]
        hs   = row["home_score"]
        as_  = row["away_score"]
        m    = row["margin"]

        _get(standings, home)["pf"] += hs
        _get(standings, home)["pa"] += as_
        _get(standings, home)["played"] += 1
        _get(standings, away)["pf"] += as_
        _get(standings, away)["pa"] += hs
        _get(standings, away)["played"] += 1

        if m > 0:
            _get(standings, home)["pts"]  += 4
            _get(standings, home)["wins"] += 1
        elif m < 0:
            _get(standings, away)["pts"]  += 4
            _get(standings, away)["wins"] += 1
        else:
            _get(standings, home)["pts"]  += 2
            _get(standings, away)["pts"]  += 2
            _get(standings, home)["wins"] += 0.5
            _get(standings, away)["wins"] += 0.5


def _to_dataframe(standings):
    rows = []
    for team, s in standings.items():
        pct = (s["pf"] / s["pa"] * 100) if s["pa"] > 0 else 0
        rows.append({
            "team":    team,
            "played":  s["played"],
            "wins":    round(s["wins"], 1),
            "pts":     round(s["pts"], 1),
            "pf":      round(s["pf"]),
            "pa":      round(s["pa"]),
            "pct":     round(pct, 1),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["pts", "pct"], ascending=[False, False])
        .reset_index(drop=True)
        .pipe(lambda df: df.assign(**{"#": range(1, len(df) + 1)}).set_index("#"))
    )


def build_probabilistic_ladder(past_games, predictions):
    """
    Projected ladder where future games contribute fractional points:
      home earns 4 * win_prob_home, away earns 4 * (1 - win_prob_home).
    Reflects uncertainty — a 60/40 game gives each team partial credit.
    """
    standings = _init_standings()
    _add_completed_games(standings, past_games)

    for _, row in predictions.iterrows():
        home   = row["home_team"]
        away   = row["away_team"]
        p_home = row["win_prob_home"]
        p_away = 1.0 - p_home
        margin = row["predicted_margin"]

        est_home = LEAGUE_AVG_SCORE + margin / 2
        est_away = LEAGUE_AVG_SCORE - margin / 2

        _get(standings, home)["pts"]    += 4 * p_home
        _get(standings, home)["wins"]   += p_home
        _get(standings, home)["pf"]     += est_home
        _get(standings, home)["pa"]     += est_away
        _get(standings, home)["played"] += 1

        _get(standings, away)["pts"]    += 4 * p_away
        _get(standings, away)["wins"]   += p_away
        _get(standings, away)["pf"]     += est_away
        _get(standings, away)["pa"]     += est_home
        _get(standings, away)["played"] += 1

    return _to_dataframe(standings)


def _format_ladder(ladder):
    lines = []
    lines.append("=== Projected 2026 Final Ladder ===")
    lines.append(f"  {'#':>2}  {'Team':<25} {'P':>3}  {'xW':>5}  {'xPts':>6}  {'%':>7}")
    lines.append(f"  {'-'*55}")
    for pos, row in ladder.iterrows():
        lines.append(
            f"  {pos:>2}  {row['team']:<25} {row['played']:>3}  "
            f"{row['wins']:>5.1f}  {row['pts']:>6.1f}  {row['pct']:>7.1f}%"
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
