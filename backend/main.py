"""HTTP API for the screening web app.

Run from the repo root:
    uvicorn backend.main:app --reload --port 8000
Then open http://localhost:8000/docs to try every endpoint. If frontend/dist
exists (after `npm run build`), the same server also serves the website at /.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import service
from .schemas import Health, ModelInfo, PredictResponse, ResultsSummary, VoiceSample

app = FastAPI(title="Parkinson's screening API", version="1.0.0",
              description="Research prototype. Not a medical device and not a diagnosis.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("QPD_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

predictor = service.Predictor()


@app.get("/api/health", response_model=Health)
def health():
    return {"status": "ok", "demo": predictor.demo, "reason": predictor.reason,
            "models": predictor.model_ids(), "voice_columns": predictor.voice_columns}


@app.get("/api/models", response_model=list[ModelInfo])
def models():
    return predictor.models()


@app.get("/api/samples", response_model=list[VoiceSample])
def samples():
    return service.voice_samples()


@app.get("/api/results", response_model=ResultsSummary)
def results():
    return service.results_summary()


@app.post("/api/reload", response_model=Health)
def reload():
    """Pick up a new results/model_bundle.joblib without restarting the server."""
    predictor.reload()
    return health()


@app.post("/api/predict", response_model=PredictResponse)
def predict(
    spiral: UploadFile = File(..., description="PNG or JPG of a hand-drawn spiral"),
    model: str = Form("vqc"),
    voice_csv: UploadFile | None = File(None, description="CSV with a header and one row of the 22 features"),
    voice_json: str | None = Form(None, description='JSON object, e.g. {"PPE": 0.28, ...}'),
):
    try:
        if voice_csv is not None:
            voice = service.voice_from_csv(voice_csv.file.read(), predictor.voice_columns)
        elif voice_json:
            try:
                values = json.loads(voice_json)
            except json.JSONDecodeError:
                raise service.InputError("The voice values are not valid JSON.")
            if not isinstance(values, dict):
                raise service.InputError("The voice values must be a JSON object of feature: value.")
            voice = service.voice_from_mapping(values, predictor.voice_columns)
        else:
            raise service.InputError("Add the voice features: upload a CSV or choose a sample recording.")
        image = service.load_image(spiral.file.read(service.MAX_IMAGE_BYTES + 1))
        return predictor.predict(voice, image, model)
    except service.InputError as e:
        raise HTTPException(status_code=422, detail=str(e))


_site = service.ROOT / "frontend" / "dist"
if _site.is_dir():
    app.mount("/", StaticFiles(directory=_site, html=True), name="site")
