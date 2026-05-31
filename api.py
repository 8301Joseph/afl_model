import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

DATA_FILE = Path("output/predictions.json")
FRONTEND = Path("frontend/index.html")

SYNC_INTERVAL_MIN = 30


async def _run_sync() -> bool:
    from sync import sync
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, sync)


async def _sync_loop():
    """Sync every 30 minutes. Simple and reliable on Railway restarts/redeploys."""
    while True:
        await asyncio.sleep(SYNC_INTERVAL_MIN * 60)
        try:
            await _run_sync()
        except Exception as e:
            print(f"[sync] Sync error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sync on startup so a fresh deploy immediately picks up latest results
    asyncio.create_task(_run_sync())
    asyncio.create_task(_sync_loop())
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
