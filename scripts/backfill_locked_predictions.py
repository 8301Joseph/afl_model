"""
scripts/backfill_locked_predictions.py

Rebuilds locked_predictions.json from scratch by replaying the season
round-by-round. For each completed round, it:

  1. Runs main.py with only prior rounds' results in the CSV
     → generates fresh predictions for all unplayed games
  2. Locks predictions for that round's games before results are seen
  3. Writes the round's actual results into the CSV
  4. Repeats for the next round

After all rounds are processed a final main.py run restores predictions.json
to the correct current state.

Usage:
    /Users/josephglasson/miniconda3/bin/python3 scripts/backfill_locked_predictions.py

"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

CSV_PATH       = Path("data/afl-2026-UTC.csv")
PREDICTIONS_PATH = Path("output/predictions.json")
LOCKED_PATH    = Path("output/locked_predictions.json")

# Match data_prep.py normalisation
TEAM_NAME_MAP = {
    "GWS GIANTS":     "GWS Giants",
    "Gold Coast SUNS": "Gold Coast Suns",
}


def normalise(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def run_main():
    result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError("main.py exited with non-zero status")


def get_rounds_chronological(df: pd.DataFrame) -> list[str]:
    """Return round labels sorted by first game date, ascending."""
    def parse_date(s):
        try:
            return datetime.strptime(str(s).strip(), "%d/%m/%Y %H:%M")
        except ValueError:
            return datetime.max

    first_dates = (
        df.groupby("Round Number")["Date"]
        .apply(lambda col: min(parse_date(d) for d in col))
    )
    return first_dates.sort_values().index.tolist()


def main():
    print("=== AFL Backfill: Rebuilding locked_predictions.json ===\n")

    # ── 0. Validate we're in the project root ────────────────────────────────
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run from the project root.")
        sys.exit(1)

    # ── 1. Load the reference CSV (all current results) ──────────────────────
    ref_df = pd.read_csv(CSV_PATH)          # Date stays as raw string
    all_rounds = get_rounds_chronological(ref_df)

    # Only process rounds that have at least one result
    def result_count(rnd):
        grp = ref_df[ref_df["Round Number"] == rnd]
        return (grp["Result"].notna() & (grp["Result"].str.strip() != "")).sum()

    rounds_to_process = [r for r in all_rounds if result_count(r) > 0]
    print(f"Rounds with results: {rounds_to_process}")
    print(f"Total rounds to backfill: {len(rounds_to_process)}\n")

    # ── 2. Back up originals ─────────────────────────────────────────────────
    shutil.copy(CSV_PATH,    CSV_PATH.with_suffix(".csv.bak"))
    if LOCKED_PATH.exists():
        shutil.copy(LOCKED_PATH, LOCKED_PATH.with_suffix(".json.bak"))
    print(f"Backups written: {CSV_PATH.with_suffix('.csv.bak')}")
    print(f"                 {LOCKED_PATH.with_suffix('.json.bak')}\n")

    # ── 3. Strip ALL 2026 results to start from a blank slate ────────────────
    working_df = ref_df.copy()
    working_df["Result"] = None
    working_df.to_csv(CSV_PATH, index=False)
    print("Cleared all 2026 results. Starting round-by-round replay...\n")

    # ── 4. Round-by-round simulation ─────────────────────────────────────────
    locked: dict = {}

    for rnd in rounds_to_process:
        print(f"─── Round {rnd} ─────────────────────────────")

        # 4a. Generate predictions (no results for this round yet)
        print("  Running main.py...")
        run_main()

        with open(PREDICTIONS_PATH) as f:
            preds = json.load(f)

        # Build lookup: "{norm_home}|{norm_away}|{YYYY-MM-DD}" → prediction
        pred_lookup: dict[str, dict] = {}
        for p in preds["predictions"]:
            date_key = p["date"][:10]           # ISO → YYYY-MM-DD
            key = f"{p['home_team']}|{p['away_team']}|{date_key}"
            pred_lookup[key] = p

        # 4b. Lock predictions for every game in this round
        round_rows = ref_df[ref_df["Round Number"] == rnd]
        locked_this_round = 0
        for _, row in round_rows.iterrows():
            home = normalise(row["Home Team"])
            away = normalise(row["Away Team"])
            # Parse the DD/MM/YYYY HH:MM date string
            try:
                game_date = datetime.strptime(str(row["Date"]).strip(), "%d/%m/%Y %H:%M")
                date_key  = game_date.strftime("%Y-%m-%d")
            except ValueError:
                date_key = str(row["Date"])[:10]

            key = f"{home}|{away}|{date_key}"

            if key in pred_lookup:
                p = pred_lookup[key]
                locked[key] = {
                    "predicted_margin":  p["predicted_margin"],
                    "win_prob_home":     p["win_prob_home"],
                    "predicted_winner":  p["predicted_winner"],
                    "h2h_bias":          p.get("h2h_bias", 0.0),
                    "locked_at":         f"backfill: pre-round-{rnd}",
                }
                locked_this_round += 1
                print(f"  Locked {home} vs {away}: {p['predicted_winner']} ({p['predicted_margin']:+.1f})")
            else:
                print(f"  WARNING: no prediction found for key={key}")

        print(f"  → Locked {locked_this_round}/{len(round_rows)} game(s)")

        # Persist locked predictions after each round
        LOCKED_PATH.parent.mkdir(exist_ok=True)
        with open(LOCKED_PATH, "w") as f:
            json.dump(locked, f, indent=2)

        # 4c. Write this round's actual results into the working CSV
        results_added = 0
        for i, row in working_df.iterrows():
            if str(row["Round Number"]) != str(rnd):
                continue
            ref_match = ref_df[
                (ref_df["Home Team"] == row["Home Team"]) &
                (ref_df["Away Team"] == row["Away Team"]) &
                (ref_df["Round Number"] == rnd)
            ]
            if len(ref_match) == 1:
                result_val = ref_match.iloc[0]["Result"]
                if pd.notna(result_val) and str(result_val).strip():
                    working_df.at[i, "Result"] = result_val
                    results_added += 1

        working_df.to_csv(CSV_PATH, index=False)
        print(f"  → Added {results_added} result(s) to CSV\n")

    # ── 5. Final run to restore predictions.json correctly ───────────────────
    print("=== Final main.py run to restore predictions.json ===")
    run_main()

    print(f"\nBackfill complete.")
    print(f"  Locked predictions total: {len(locked)}")
    print(f"  locked_predictions.json:  {LOCKED_PATH}")
    print(f"  CSV restored to:          {CSV_PATH}")


if __name__ == "__main__":
    main()
