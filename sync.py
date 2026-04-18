"""
sync.py — fetch latest AFL results from squiggle and regenerate predictions.

Can be run standalone:   python sync.py
Also called by api.py on a daily schedule and via the /sync endpoint.
"""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CSV_PATH = Path("data/afl-2026-UTC.csv")
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
        df.at[i, "Result"] = f"{int(g['hscore'])} - {int(g['ascore'])}"
        updated += 1
        print(
            f"[sync]   Rd {row['Round Number']}: "
            f"{row['Home Team']} {int(g['hscore'])} – {int(g['ascore'])} {row['Away Team']}"
        )

    if updated:
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
