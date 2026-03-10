import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

DATA_FILE = Path("output/predictions.json")
FRONTEND = Path("frontend/index.html")

# ---------------------------------------------------------------------------
# Background sync — runs daily at 06:00 AEST (20:00 UTC)
# ---------------------------------------------------------------------------

SYNC_HOUR_UTC = 20  # 06:00 AEST = 20:00 UTC


async def _run_sync():
    from sync import sync
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sync)


async def _daily_sync_loop():
    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=SYNC_HOUR_UTC, minute=0, second=0, microsecond=0)
        if now >= next_run:
            # Already past today's window — schedule for tomorrow
            next_run = next_run.replace(day=next_run.day + 1)
        wait = (next_run - now).total_seconds()
        print(f"[sync] Next scheduled sync in {wait/3600:.1f}h ({next_run.strftime('%Y-%m-%d %H:%M UTC')})")
        await asyncio.sleep(wait)
        try:
            await _run_sync()
        except Exception as e:
            print(f"[sync] Scheduled sync error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
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
