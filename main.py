"""
Marvin Voice Command Backend (SDK version)
Fix: runner.init() moved into lifespan so uvicorn binds the port FIRST,
     then loads the model. This is required for Render to detect the open port.
"""

import os
import tempfile
import threading
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from edge_impulse_linux.runner import ImpulseRunner

# ─── CONFIG ──────────────────────────────────────
_default_model = Path(__file__).parent / "model.eim"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(_default_model)))
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000   # 1 second at 16 kHz
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))

LABELS = ["Clock", "Forecast", "Greet", "Hello",
          "Noise", "Off", "Stop", "Unknown", "Weather"]

# ─── RUNNER WRAPPER ──────────────────────────────
# runner is populated inside lifespan (after uvicorn has bound the port)
runner: ImpulseRunner | None = None
runner_lock = threading.Lock()


# ─── APP LIFECYCLE ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    # Ensure model is executable
    MODEL_PATH.chmod(0o755)
    print(f"[EIM] Loading model: {MODEL_PATH}")
    runner = ImpulseRunner(str(MODEL_PATH))
    runner.init()
    print("[EIM] Model ready ✓")
    yield
    print("[EIM] Shutting down model...")
    runner.stop()
    runner = None


# ─── APP ─────────────────────────────────────────
app = FastAPI(
    title="Marvin Voice Command API",
    description="Edge Impulse voice model inference for ESP32 Marvin device",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── HELPERS ─────────────────────────────────────
def _to_window(samples: np.ndarray) -> list:
    """Pad or trim to exactly WINDOW_SAMPLES floats in [-1, 1]."""
    samples = samples.astype(np.float32)
    if len(samples) >= WINDOW_SAMPLES:
        samples = samples[:WINDOW_SAMPLES]
    else:
        samples = np.pad(samples, (0, WINDOW_SAMPLES - len(samples)))
    return samples.tolist()


def wav_bytes_to_samples(wav_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    try:
        with wave.open(tmp_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        return samples
    finally:
        os.unlink(tmp_path)


def raw_pcm_to_samples(data: bytes, bits: int = 16) -> np.ndarray:
    if bits == 16:
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits == 32:
        return np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported bit depth: {bits}")


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


def _run_classify(samples: np.ndarray) -> dict:
    """Thread-safe classify call."""
    if runner is None:
        raise RuntimeError("Model not loaded")
    with runner_lock:
        return runner.classify(_to_window(samples))


# ─── ROUTES ──────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Marvin Voice Command API",
        "status": "running",
        "model": str(MODEL_PATH),
        "labels": LABELS,
        "sample_rate": SAMPLE_RATE,
        "endpoints": ["/classify/wav", "/classify/raw", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": runner is not None}


@app.post("/classify/raw")
async def classify_raw(file: UploadFile = File(...), bits: int = 16):
    """
    Preferred endpoint for ESP32.
    Send raw signed int16 PCM, 16 kHz mono.
    1 second = 32 000 bytes.
    """
    contents = await file.read()
    if len(contents) < 100:
        raise HTTPException(400, "Audio data too short")
    try:
        samples = raw_pcm_to_samples(contents, bits=bits)
    except Exception as e:
        raise HTTPException(400, f"Failed to decode PCM: {e}")

    duration = len(samples) / SAMPLE_RATE
    try:
        result = _run_classify(samples)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    return JSONResponse(build_response(result, duration))


@app.post("/classify/wav")
async def classify_wav(file: UploadFile = File(...)):
    """Send a WAV file (16 kHz, mono, 16-bit recommended)."""
    contents = await file.read()
    if len(contents) < 44:
        raise HTTPException(400, "Invalid WAV file")
    try:
        samples = wav_bytes_to_samples(contents)
    except Exception as e:
        raise HTTPException(400, f"Failed to decode WAV: {e}")

    duration = len(samples) / SAMPLE_RATE
    try:
        result = _run_classify(samples)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    return JSONResponse(build_response(result, duration))
