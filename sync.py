"""
sync.py — fetch latest AFL results from squiggle and regenerate predictions.

Can be run standalone:   python sync.py
Also called by api.py on a daily schedule and via the /sync endpoint.
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


def _to_csv_name(squiggle_name: str) -> str:
    return SQUIGGLE_TO_CSV.get(squiggle_name, squiggle_name)


def fetch_completed(year: int = YEAR) -> dict:
    """Return {(csv_home, csv_away): game_dict} for all completed squiggle games."""
    resp = requests.get(
        SQUIGGLE_URL,
        params={"q": "games", "year": year},
        headers={"User-Agent": "afl-model-sync/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    games = resp.json().get("games", [])
    result = {}
    for g in games:
        if g.get("complete") != 100:
            continue
        if g.get("hscore") is None or g.get("ascore") is None:
            continue
        home = _to_csv_name(g["hteam"])
        away = _to_csv_name(g["ateam"])
        try:
            rnd = int(g["round"])
        except (ValueError, TypeError):
            rnd = g["round"]
        result[(home, away, rnd)] = g
    return result


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

    locked = json.load(open(LOCKED_PATH)) if LOCKED_PATH.exists() else {}
    added = 0
    for home, away, date_str in game_keys:
        key = f"{home}|{away}|{date_str}"
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
    Fetch latest AFL results and update the CSV.
    Returns True if new results were found and predictions were regenerated.
    """
    print(f"[sync] Fetching {year} results from squiggle...")
    try:
        completed = fetch_completed(year)
    except Exception as e:
        print(f"[sync] Fetch failed: {e}")
        return False

    print(f"[sync] {len(completed)} completed games on squiggle.")

    df = pd.read_csv(CSV_PATH)
    now = datetime.now(timezone.utc)
    updated = 0
    games_about_to_finish = []

    for i, row in df.iterrows():
        # Skip rows that already have a result
        if pd.notna(row.get("Result")) and str(row["Result"]).strip():
            continue

        # Never write a result for a game whose scheduled date is in the future
        game_date = pd.to_datetime(row["Date"], dayfirst=True)
        if game_date.tzinfo is None:
            game_date = game_date.tz_localize("UTC")
        if game_date > now:
            continue

        try:
            rnd = int(row["Round Number"])
        except (ValueError, TypeError):
            rnd = row["Round Number"]
        key = (row["Home Team"], row["Away Team"], rnd)
        if key not in completed:
            continue

        g = completed[key]
        date_str = pd.to_datetime(row["Date"], dayfirst=True).strftime("%Y-%m-%d")
        games_about_to_finish.append((row["Home Team"], row["Away Team"], date_str))
        df.at[i, "Result"] = f"{int(g['hscore'])} - {int(g['ascore'])}"
        updated += 1
        print(
            f"[sync]   Rd {row['Round Number']}: "
            f"{row['Home Team']} {int(g['hscore'])} – {int(g['ascore'])} {row['Away Team']}"
        )

    if updated:
        _lock_pre_result_predictions(games_about_to_finish)
        df.to_csv(CSV_PATH, index=False)
        print(f"[sync] {updated} new result(s) written. Regenerating predictions...")
        subprocess.run([sys.executable, "main.py"], check=True)
        print("[sync] Done.")
        return True
    else:
        print("[sync] No new results.")
        return False


if __name__ == "__main__":
    sync()
