"""
Marvin Voice Command Backend (SDK version)
"""

import os
import tempfile
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from edge_impulse_linux.runner import ImpulseRunner

# ─── CONFIG ──────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.eim"

SAMPLE_RATE = 16000
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))

LABELS = ["Clock", "Forecast", "Greet", "Hello", "Noise", "Off", "Stop", "Unknown", "Weather"]

# ─── APP ─────────────────────────────────────────

app = FastAPI(
    title="Marvin Voice Command API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODEL (loaded at startup) ───────────────────

runner = None

@app.on_event("startup")
async def load_model():
    global runner
    print("📂 Files in /app:", os.listdir(BASE_DIR))
    print("📌 Using model path:", MODEL_PATH)

    print("[EIM] Loading model...")
    runner = ImpulseRunner(str(MODEL_PATH))
    model_info = runner.init()
    print("[EIM] Model loaded:", model_info)

# ─── HELPERS ─────────────────────────────────────

def wav_bytes_to_samples(wav_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name

    try:
        with wave.open(tmp_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            dtype = np.int16
            max_val = 32768.0
        else:
            raise ValueError("Only 16-bit WAV supported")

        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / max_val

        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples
    finally:
        os.unlink(tmp_path)


def raw_pcm_to_samples(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def build_response(eim_result: dict, audio_duration_s: float) -> dict:
    classification = eim_result["result"]["classification"]

    best_label = max(classification, key=classification.get)
    best_score = classification[best_label]

    is_valid = (
        best_label not in ("Noise", "Unknown")
        and best_score >= CONFIDENCE_THRESHOLD
    )

    return {
        "command": best_label if is_valid else None,
        "confidence": round(best_score, 4),
        "all_scores": {k: round(v, 4) for k, v in classification.items()},
        "valid": is_valid,
        "audio_duration_s": round(audio_duration_s, 3),
        "threshold": CONFIDENCE_THRESHOLD,
    }

# ─── ROUTES ──────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "model": str(MODEL_PATH),
        "labels": LABELS,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": runner is not None}


@app.post("/classify/raw")
async def classify_raw(file: UploadFile = File(...)):
    contents = await file.read()

    if len(contents) < 100:
        raise HTTPException(400, "Audio too short")

    samples = raw_pcm_to_samples(contents)
    duration = len(samples) / SAMPLE_RATE

    result = runner.classify(samples.tolist())
    return JSONResponse(build_response(result, duration))


@app.post("/classify/wav")
async def classify_wav(file: UploadFile = File(...)):
    contents = await file.read()

    if len(contents) < 44:
        raise HTTPException(400, "Invalid WAV")

    samples = wav_bytes_to_samples(contents)
    duration = len(samples) / SAMPLE_RATE

    result = runner.classify(samples.tolist())
    return JSONResponse(build_response(result, duration))
