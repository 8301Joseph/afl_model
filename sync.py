"""
sync.py — fetch latest AFL results from squiggle and regenerate predictions.

Can be run standalone:   python sync.py
Also called by api.py every 30 minutes and via the /sync endpoint.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CSV_PATH = Path("data/afl-2026-UTC.csv")
LOCKED_PATH = Path("output/locked_predictions.json")
SQUIGGLE_URL = "https://api.squiggle.com.au/"
YEAR = 2026

# Squiggle uses shortened/different team names — map them to the CSV names.
SQUIGGLE_TO_CSV = {
    "Adelaide":              "Adelaide Crows",
    "Sydney":                "Sydney Swans",
    "Geelong":               "Geelong Cats",
    "Gold Coast":            "Gold Coast SUNS",
    "Greater Western Sydney":"GWS GIANTS",
    "West Coast":            "West Coast Eagles",
}


# Squiggle numbers every round (0 = Opening Round, 25+ = finals); the CSV uses
# labels for those. Map Squiggle roundname -> CSV "Round Number" label.
SQUIGGLE_ROUND_LABELS = {
    "Opening Round":      "OR",
    "Wildcard Finals":    "Wildcard",
    "Finals Week 1":      "Finals W1",
    "Semi-Finals":        "Semi Finals",
    "Preliminary Finals": "Prelim Finals",
    "Grand Final":        "Grand Final",
}


def _to_csv_name(squiggle_name: str) -> str:
    return SQUIGGLE_TO_CSV.get(squiggle_name, squiggle_name)


def _round_label(g: dict) -> str:
    """CSV-style round label for a squiggle game (e.g. '12', 'OR', 'Wildcard')."""
    label = SQUIGGLE_ROUND_LABELS.get(g.get("roundname"))
    if label:
        return label
    try:
        return str(int(g["round"]))
    except (ValueError, TypeError):
        return str(g["round"])


def _csv_round(val) -> str:
    """Normalise a CSV 'Round Number' cell to a string label."""
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val).strip()


def _game_key(g: dict) -> tuple:
    return (_to_csv_name(g["hteam"]), _to_csv_name(g["ateam"]), _round_label(g))


def _row_key(row) -> tuple:
    return (row["Home Team"], row["Away Team"], _csv_round(row["Round Number"]))


def _utc_str(unixtime: int) -> str:
    return datetime.fromtimestamp(unixtime, tz=timezone.utc).strftime("%d/%m/%Y %H:%M")


def _has_result(row) -> bool:
    return pd.notna(row.get("Result")) and bool(str(row["Result"]).strip())


def _run_main() -> None:
    subprocess.run([sys.executable, "main.py"], check=True)


def fetch_all_games(year: int = YEAR) -> list:
    """Return all squiggle games for the year."""
    resp = requests.get(
        SQUIGGLE_URL,
        params={"q": "games", "year": year},
        headers={"User-Agent": "afl-model-sync/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("games", [])


def fetch_completed(year: int = YEAR) -> dict:
    """Return {(csv_home, csv_away, round_label): game_dict} for all completed squiggle games."""
    return {
        _game_key(g): g
        for g in fetch_all_games(year)
        if g.get("complete") == 100
        and g.get("hscore") is not None
        and g.get("ascore") is not None
    }


def _update_fixture_dates(df: pd.DataFrame, games: list) -> tuple[pd.DataFrame, int]:
    """
    Update Date column for unplayed games using Squiggle's unixtime.
    Returns the updated dataframe and count of changes.
    """
    fixture = {_game_key(g): _utc_str(g["unixtime"]) for g in games if g.get("unixtime")}

    updated = 0
    for i, row in df.iterrows():
        if _has_result(row):
            continue  # already played — don't touch the date
        new_date = fixture.get(_row_key(row))
        if new_date and new_date != str(row["Date"]).strip():
            df.at[i, "Date"] = new_date
            updated += 1
    return df, updated


def _add_new_fixtures(df: pd.DataFrame, games: list) -> tuple[pd.DataFrame, int]:
    """
    Append squiggle games missing from the CSV (e.g. finals, which are only
    scheduled once the teams are known). Rows are added without a result so
    predictions are generated (and later locked) before the result is written.
    """
    existing = {_row_key(row) for _, row in df.iterrows()}

    # Learn squiggle venue -> CSV location from games already in the CSV
    csv_locations = {_row_key(row): row["Location"] for _, row in df.iterrows()}
    venue_map = {}
    for g in games:
        loc = csv_locations.get(_game_key(g))
        if loc and g.get("venue"):
            venue_map.setdefault(g["venue"], loc)

    new_rows = []
    next_match = int(df["Match Number"].max()) + 1
    for g in sorted(games, key=lambda g: g.get("unixtime") or 0):
        if not g.get("hteam") or not g.get("ateam") or not g.get("unixtime"):
            continue  # teams not yet decided
        if _game_key(g) in existing:
            continue
        home, away, rnd = _game_key(g)
        new_rows.append({
            "Match Number": next_match,
            "Round Number": rnd,
            "Date":         _utc_str(g["unixtime"]),
            "Location":     venue_map.get(g.get("venue"), g.get("venue", "")),
            "Home Team":    home,
            "Away Team":    away,
            "Result":       None,
        })
        next_match += 1
        print(f"[sync]   New fixture: {rnd}: {home} vs {away}")

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df, len(new_rows)


def _lock_pre_result_predictions(game_keys: list[tuple]) -> None:
    """
    Before results are written to the CSV, snapshot the current predictions
    for those games from predictions.json into locked_predictions.json.
    This preserves the pre-result prediction for accuracy evaluation.
    """
    predictions_path = Path("output/predictions.json")
    if not predictions_path.exists() or not game_keys:
        return

    with open(predictions_path) as f:
        current = json.load(f)

    lookup = {
        f"{p['home_team']}|{p['away_team']}|{p['date'][:10]}": p
        for p in current.get("predictions", [])
    }

    # Normalise CSV team names to match predictions.json names
    NORM = {"Gold Coast SUNS": "Gold Coast Suns", "GWS GIANTS": "GWS Giants"}

    locked = json.load(open(LOCKED_PATH)) if LOCKED_PATH.exists() else {}
    added = 0
    for home, away, date_str in game_keys:
        key = f"{NORM.get(home, home)}|{NORM.get(away, away)}|{date_str}"
        if key in locked:
            continue
        if key not in lookup:
            continue
        p = lookup[key]
        locked[key] = {
            "predicted_margin": p["predicted_margin"],
            "win_prob_home":    p["win_prob_home"],
            "predicted_winner": p["predicted_winner"],
            "h2h_bias":         p.get("h2h_bias", 0.0),
            "locked_at":        datetime.now(timezone.utc).isoformat(),
        }
        added += 1

    if added:
        LOCKED_PATH.parent.mkdir(exist_ok=True)
        with open(LOCKED_PATH, "w") as f:
            json.dump(locked, f, indent=2)
        print(f"[sync] Locked {added} pre-result prediction(s).")


def sync(year: int = YEAR) -> bool:
    """
    Fetch latest AFL results/fixtures and update the CSV.

    Results are applied one round at a time in chronological order, regenerating
    predictions after each round, so that when several rounds are caught up at
    once every game's locked prediction only reflects results before it.

    Returns True if any changes were made and predictions were regenerated.
    """
    print(f"[sync] Fetching {year} games from squiggle...")
    try:
        all_games = fetch_all_games(year)
    except Exception as e:
        print(f"[sync] Fetch failed: {e}")
        return False

    completed = {
        _game_key(g): g
        for g in all_games
        if g.get("complete") == 100
        and g.get("hscore") is not None
        and g.get("ascore") is not None
    }
    print(f"[sync] {len(completed)} completed games on squiggle.")

    df = pd.read_csv(CSV_PATH, dtype={"Round Number": str})
    changed = False

    # 1. New fixtures + fixture date changes, then predict them
    df, fixtures_added = _add_new_fixtures(df, all_games)
    df, dates_updated = _update_fixture_dates(df, all_games)
    if dates_updated:
        print(f"[sync] Updated {dates_updated} fixture date(s).")
    if fixtures_added or dates_updated:
        df.to_csv(CSV_PATH, index=False)
        print("[sync] Fixtures changed. Regenerating predictions...")
        _run_main()
        changed = True

    # 2. Find unplayed games that now have a result, grouped by round
    now = datetime.now(timezone.utc)
    pending = {}  # round label -> [(row index, squiggle game)]
    for i, row in df.iterrows():
        if _has_result(row):
            continue
        # Never write a result for a game whose scheduled date is in the future
        game_date = pd.to_datetime(row["Date"], dayfirst=True).tz_localize("UTC")
        if game_date > now:
            continue
        g = completed.get(_row_key(row))
        if g:
            pending.setdefault(_csv_round(row["Round Number"]), []).append((i, g))

    # 3. Apply results round-by-round in date order
    def first_date(rnd):
        return min(pd.to_datetime(df.at[i, "Date"], dayfirst=True) for i, _ in pending[rnd])

    for rnd in sorted(pending, key=first_date):
        games_about_to_finish = []
        for i, g in pending[rnd]:
            row = df.loc[i]
            date_str = pd.to_datetime(row["Date"], dayfirst=True).strftime("%Y-%m-%d")
            games_about_to_finish.append((row["Home Team"], row["Away Team"], date_str))
            df.at[i, "Result"] = f"{int(g['hscore'])} - {int(g['ascore'])}"
            print(
                f"[sync]   Rd {rnd}: "
                f"{row['Home Team']} {int(g['hscore'])} – {int(g['ascore'])} {row['Away Team']}"
            )
        _lock_pre_result_predictions(games_about_to_finish)
        df.to_csv(CSV_PATH, index=False)
        print(f"[sync] {len(pending[rnd])} new result(s) for Rd {rnd}. Regenerating predictions...")
        _run_main()
        changed = True

    print("[sync] Done." if changed else "[sync] No new results or fixture changes.")
    return changed


if __name__ == "__main__":
    sync()
