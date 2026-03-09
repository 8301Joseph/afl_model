import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

DATA_FILE = Path("output/predictions.json")
FRONTEND = Path("frontend/index.html")

app = FastAPI(title="AFL Model API")

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
