"""
Marvin Voice Command Backend
Talks directly to the .eim model via UNIX socket — no edge-impulse-linux SDK needed.
This avoids the pyaudio / six dependency chain entirely.
"""
 
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
 
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
 
# ─── CONFIG ──────────────────────────────────────
_default_model = Path(__file__).parent / "model.eim"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(_default_model)))
SOCKET_PATH = "/tmp/marvin_runner.sock"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000          # 1 second at 16 kHz
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
LABELS = ["Clock", "Forecast", "Greet", "Hello",
          "Noise", "Off", "Stop", "Unknown", "Weather"]
 
 
# ─── EIM RUNNER ──────────────────────────────────
class EIMRunner:
    """
    Manages the .eim subprocess and communicates via UNIX socket.
 
    Protocol (one connection per inference):
      1. Client sends  {"id":1, "hello":1}  → server replies with model info
      2. Client sends  {"id":2, "classify":[...16000 floats...]}
      3. Server replies with classification result
    """
 
    def __init__(self, model_path: Path, socket_path: str):
        self.model_path = model_path
        self.socket_path = socket_path
        self.process = None
        self.lock = threading.Lock()
        self.ready = False
 
    def start(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model_path.chmod(0o755)
 
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
 
        print(f"[EIM] Starting: {self.model_path}")
        self.process = subprocess.Popen(
            [str(self.model_path), self.socket_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
 
        # Wait up to 15 s for the socket to appear
        for _ in range(150):
            if os.path.exists(self.socket_path):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("EIM model did not create socket in time")
 
        print("[EIM] Ready ✓")
        self.ready = True
 
    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.ready = False
 
    def _recv_json(self, sock: socket.socket) -> dict:
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n")[0])
 
    def classify(self, samples: np.ndarray) -> dict:
        if not self.ready:
            raise RuntimeError("EIM runner not started")
 
        # Pad / trim to exactly WINDOW_SAMPLES float32 values
        samples = samples.astype(np.float32)
        if len(samples) >= WINDOW_SAMPLES:
            samples = samples[:WINDOW_SAMPLES]
        else:
            samples = np.pad(samples, (0, WINDOW_SAMPLES - len(samples)))
 
        with self.lock:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(15)
            sock.connect(self.socket_path)
            try:
                # Handshake
                sock.sendall(json.dumps({"id": 1, "hello": 1}).encode() + b"\n")
                hello = self._recv_json(sock)
                if not hello.get("success"):
                    raise RuntimeError(f"EIM hello failed: {hello}")
 
                # Classify
                sock.sendall(
                    json.dumps({"id": 2, "classify": samples.tolist()}).encode() + b"\n"
                )
                return self._recv_json(sock)
            finally:
                sock.close()
 
 
# ─── APP LIFECYCLE ────────────────────────────────
eim = EIMRunner(MODEL_PATH, SOCKET_PATH)
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    eim.start()
    yield
    eim.stop()
 
 
app = FastAPI(
    title="Marvin Voice Command API",
    description="Edge Impulse voice model inference for ESP32 Marvin device",
    version="3.0.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# ─── HELPERS ─────────────────────────────────────
def wav_bytes_to_samples(wav_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        with wave.open(tmp, "rb") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            s = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sw}")
        if n_ch > 1:
            s = s.reshape(-1, n_ch).mean(axis=1)
        return s
    finally:
        os.unlink(tmp)
 
 
def raw_pcm_to_samples(data: bytes, bits: int = 16) -> np.ndarray:
    if bits == 16:
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if bits == 32:
        return np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported bit depth: {bits}")
 
 
def build_response(eim_result: dict, duration: float) -> dict:
    clf = eim_result["result"]["classification"]
    best = max(clf, key=clf.get)
    score = clf[best]
    valid = best not in ("Noise", "Unknown") and score >= CONFIDENCE_THRESHOLD
    return {
        "command": best if valid else None,
        "confidence": round(score, 4),
        "all_scores": {k: round(v, 4) for k, v in clf.items()},
        "valid": valid,
        "audio_duration_s": round(duration, 3),
        "threshold": CONFIDENCE_THRESHOLD,
    }
 
 
# ─── ROUTES ──────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Marvin Voice Command API",
        "status": "running",
        "labels": LABELS,
        "sample_rate": SAMPLE_RATE,
        "endpoints": ["/classify/raw", "/classify/wav", "/health"],
    }
 
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": eim.ready}
 
@app.post("/classify/raw")
async def classify_raw(file: UploadFile = File(...), bits: int = 16):
    """Preferred ESP32 endpoint. Send raw signed int16 PCM, 16 kHz mono. 1 sec = 32000 bytes."""
    data = await file.read()
    if len(data) < 100:
        raise HTTPException(400, "Audio too short")
    try:
        samples = raw_pcm_to_samples(data, bits)
    except Exception as e:
        raise HTTPException(400, f"Bad PCM: {e}")
    try:
        result = eim.classify(samples)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    return JSONResponse(build_response(result, len(samples) / SAMPLE_RATE))
 
@app.post("/classify/wav")
async def classify_wav(file: UploadFile = File(...)):
    """Send a WAV file (16 kHz mono 16-bit recommended)."""
    data = await file.read()
    if len(data) < 44:
        raise HTTPException(400, "Invalid WAV")
    try:
        samples = wav_bytes_to_samples(data)
    except Exception as e:
        raise HTTPException(400, f"Bad WAV: {e}")
    try:
        result = eim.classify(samples)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    return JSONResponse(build_response(result, len(samples) / SAMPLE_RATE))
