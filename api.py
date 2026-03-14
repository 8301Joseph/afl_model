import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

DATA_FILE = Path("output/predictions.json")
FRONTEND = Path("frontend/index.html")

# ---------------------------------------------------------------------------
# Background sync schedule
# Weekdays : start 20:00 UTC (06:00 AEST), retry for 4 h → covers night games
# Weekends : start 01:00 UTC (noon AEDT), retry for 14 h → covers all-day games
# ---------------------------------------------------------------------------

RETRY_INTERVAL_MIN = 30

# (start_hour_utc, retry_hours)  — indexed by weekday() of the AEST date
#   0=Mon … 4=Fri → night-game window; 5=Sat, 6=Sun → all-day window
_WEEKDAY_WINDOW = (20, 4)   # 06:00 AEST start, 4 h retry
_WEEKEND_WINDOW = (2, 15)   # 1pm AEDT (02:00 UTC) start, 15 h retry (1pm→4am AEDT)

AEST_OFFSET = timedelta(hours=10)  # UTC+10 — close enough for scheduling


def _next_sync_window(now: datetime):
    """Return (next_run_utc, retry_hours) for the next sync window."""
    for days_ahead in range(3):
        candidate_utc_day = (now + timedelta(days=days_ahead)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        aest_weekday = (candidate_utc_day + AEST_OFFSET).weekday()
        start_hour, retry_hours = (
            _WEEKEND_WINDOW if aest_weekday >= 5 else _WEEKDAY_WINDOW
        )
        candidate = candidate_utc_day.replace(hour=start_hour)
        if candidate > now:
            return candidate, retry_hours
    # fallback (should never hit)
    return now + timedelta(hours=1), 4


async def _run_sync() -> bool:
    from sync import sync
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sync)


async def _daily_sync_loop():
    while True:
        now = datetime.now(timezone.utc)
        next_run, retry_hours = _next_sync_window(now)
        wait = (next_run - now).total_seconds()
        print(f"[sync] Next sync in {wait/3600:.1f}h ({next_run.strftime('%Y-%m-%d %H:%M UTC')}, window={retry_hours}h)")
        await asyncio.sleep(wait)

        deadline = datetime.now(timezone.utc) + timedelta(hours=retry_hours)
        attempt = 0
        while datetime.now(timezone.utc) < deadline:
            attempt += 1
            try:
                found = await _run_sync()
            except Exception as e:
                print(f"[sync] Sync error (attempt {attempt}): {e}")
                found = False
            if found:
                print(f"[sync] Results found on attempt {attempt}.")
                break
            print(f"[sync] No new results (attempt {attempt}), retrying in {RETRY_INTERVAL_MIN}m...")
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


@app.get("/", response_class=FileResponse)
def serve_frontend():
    return FileResponse(FRONTEND)


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
