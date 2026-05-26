import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

DATA_FILE = Path("output/predictions.json")
FRONTEND = Path("frontend/index.html")

# ---------------------------------------------------------------------------
# Background sync schedule
# Weekdays : start 13:30 UTC (11:30pm AEST), retry every 30 min → 8am AEST (22:00 UTC)
# Weekends : start 02:00 UTC (1pm AEDT),   retry every 30 min → 4am AEDT (17:00 UTC)
# ---------------------------------------------------------------------------

RETRY_INTERVAL_MIN = 30

# (start_hour_utc, start_minute_utc, retry_hours)
#   Weekdays: 11:30pm AEST (13:30 UTC) → 8am AEST (22:00 UTC) = 8.5 h
#   Weekends: 1pm AEDT (02:00 UTC) → 4am AEDT (17:00 UTC) = 15 h
_WEEKDAY_WINDOW = (13, 30, 8.5)
_WEEKEND_WINDOW = (2,  0,  15.0)

AEST_OFFSET = timedelta(hours=10)  # UTC+10 — close enough for scheduling


def _next_sync_window(now: datetime):
    """
    Return (next_run_utc, remaining_hours) for the next sync opportunity.
    If we're already inside a window (e.g. after a server restart), returns
    now so the loop resumes immediately with the remaining window time.
    """
    for days_ahead in range(3):
        candidate_utc_day = (now + timedelta(days=days_ahead)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        aest_weekday = (candidate_utc_day + AEST_OFFSET).weekday()
        start_hour, start_min, retry_hours = (
            _WEEKEND_WINDOW if aest_weekday >= 5 else _WEEKDAY_WINDOW
        )
        window_start = candidate_utc_day.replace(hour=start_hour, minute=start_min)
        window_end   = window_start + timedelta(hours=retry_hours)

        if days_ahead == 0 and window_start <= now < window_end:
            # Already inside a window — resume immediately with remaining time
            remaining = (window_end - now).total_seconds() / 3600
            return now, remaining

        if window_start > now:
            return window_start, retry_hours

    # fallback (should never hit)
    return now + timedelta(hours=1), 4.0


async def _run_sync() -> bool:
    from sync import sync
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sync)


async def _daily_sync_loop():
    while True:
        now = datetime.now(timezone.utc)
        next_run, retry_hours = _next_sync_window(now)
        wait = (next_run - now).total_seconds()
        print(f"[sync] Next sync in {wait/3600:.1f}h ({next_run.strftime('%Y-%m-%d %H:%M UTC')}, window={retry_hours:.1f}h)")
        await asyncio.sleep(wait)

        deadline = datetime.now(timezone.utc) + timedelta(hours=retry_hours)
        attempt = 0
        while datetime.now(timezone.utc) < deadline:
            attempt += 1
            try:
                await _run_sync()
            except Exception as e:
                print(f"[sync] Sync error (attempt {attempt}): {e}")
            print(f"[sync] Attempt {attempt} done, next in {RETRY_INTERVAL_MIN}m...")
            await asyncio.sleep(RETRY_INTERVAL_MIN * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sync on startup so a fresh deploy immediately picks up latest results
    asyncio.create_task(_run_sync())
    asyncio.create_task(_daily_sync_loop())
    yield


app = FastAPI(title="AFL Model API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _load():
    if not DATA_FILE.exists():
        raise HTTPException(status_code=503, detail="Predictions not yet generated. Run main.py first.")
    with open(DATA_FILE) as f:
        return json.load(f)


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions")
def get_predictions():
    data = _load()
    return {
        "generated_at": data["generated_at"],
        "predictions": data["predictions"],
    }


@app.get("/ladder")
def get_ladder():
    data = _load()
    return {
        "generated_at": data["generated_at"],
        "ladder": data["ladder"],
    }


@app.get("/results")
def get_results():
    data = _load()
    return {
        "generated_at": data["generated_at"],
        "results": data.get("results", []),
    }


@app.get("/ratings")
def get_ratings():
    data = _load()
    return {
        "generated_at": data["generated_at"],
        "ratings": data.get("ratings", []),
    }


@app.post("/sync")
async def trigger_sync():
    """Manually trigger a result sync and prediction regeneration."""
    try:
        await _run_sync()
        return {"status": "ok", "message": "Sync complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/round/{round_number}")
def get_round(round_number: int):
    data = _load()
    games = [g for g in data["predictions"] if g["round"] == round_number]
    if not games:
        raise HTTPException(status_code=404, detail=f"No games found for round {round_number}")
    return {
        "generated_at": data["generated_at"],
        "round": round_number,
        "games": games,
    }
