"""
main.py — Member A (Backend & Integration Lead)

This is the single entrypoint for the whole project:
- Serves the frontend (Member D's index.html) from one address, so the
  user only ever sees "upload/record -> progress -> report" (no visible
  backend, no separate services — matches the "efficient, frontend-only"
  requirement).
- Owns video/audio extraction (ffmpeg + OpenCV frame sampling), since
  that's the shared step every other module depends on.
- Calls into each teammate's module through a fixed function signature
  (see modules/*.py). As B, C and D replace their stubs with real code,
  this file does not need to change — only the insides of modules/*.py do.

Run (from the repo root):
    pip install -r requirements.txt
    uvicorn main:app --reload
    open http://127.0.0.1:8000
(add --host 0.0.0.0 only if teammates need to reach it over your network)
"""

import math
import shutil
import subprocess
import time
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from modules import audio, content, language, vision
import db

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = BASE_DIR / "static"

# Config every teammate can tune without touching this file's logic
MAX_FRAMES = 60                  # max frames handed to the vision models (raise if machines cope)
MAX_UPLOAD_MB = 300              # adjust if test videos are bigger
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}

app = FastAPI(title="Speaking Evaluation & Coaching Platform")

# Member D owns db.py's real implementation (schema, history queries).
db.init()


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """Safety net: any error we didn't anticipate still comes back as JSON
    (with the real message) instead of the plain-text 'Internal Server
    Error' that breaks the frontend's res.json()."""
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {type(exc).__name__}: {exc}"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_native(obj):
    """Recursively convert numpy types (np.float32, np.int64, arrays...)
    into plain Python types so FastAPI/JSON can serialize the report.
    Modules that use numpy (like vision.py) often return np.float32
    without meaning to."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return to_native(obj.item())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
    

# ---------------------------------------------------------------------------
# Extraction helpers (Member A owns this: it's the shared input every
# other member's module consumes, so it lives in the integration layer)
# ---------------------------------------------------------------------------

def has_audio_stream(video_path: Path) -> bool:
    """True if the file contains at least one audio track."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=20,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True  # if ffprobe itself fails, let ffmpeg try and report


def extract_audio(video_path: Path, out_path: Path) -> Path:
    """Pull a mono 16kHz WAV out of the video with ffmpeg (what
    faster-whisper and librosa both expect)."""
    if not has_audio_stream(video_path):
        raise RuntimeError("No audio detected in this video.")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
    return out_path


def _shrink(f, w=960):
    """Downscale a frame to at most w pixels wide (keeps aspect ratio)
    so memory stays small and the vision models run faster."""
    h0, w0 = f.shape[:2]
    return f if w0 <= w else cv2.resize(f, (w, int(h0 * w / w0)))


def sample_frames(video_path: Path, max_frames: int = MAX_FRAMES):
    """Return at most max_frames evenly spaced frames (BGR numpy arrays),
    downscaled. We deliberately do NOT run vision models on every frame —
    memory and CPU time would explode on longer videos."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video for frame sampling")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) if total > 0 else 15

    frames, idx = [], 0
    while cap.grab():
        if idx % step == 0:
            ok, f = cap.retrieve()
            if ok:
                frames.append(_shrink(f))
                # unknown-length videos (e.g. browser webm): keep memory bounded
                if len(frames) >= max_frames * 2:
                    frames = frames[::2]
                    step *= 2
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted — video may be corrupt or empty")

    if len(frames) > max_frames:
        picks = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in picks]
    return frames


def get_video_duration_seconds(video_path: Path) -> float:
    """Ask ffprobe for the duration (works on browser-recorded webm,
    where OpenCV's frame count is often 0)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, timeout=20,
        )
        return round(float(out.stdout.strip()), 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    """Serve Member D's dashboard as the one page the user ever sees."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            {"error": "static/index.html not found — Member D's dashboard goes here"},
            status_code=404,
        )
    return FileResponse(index_path)


# NOTE: plain def (not async def) on purpose. The pipeline is blocking
# CPU work; FastAPI runs plain-def routes in a thread pool, so the server
# (and /api/health) stays responsive while a video is being analyzed.
@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)):
    """
    The one endpoint that runs the whole pipeline end-to-end:

      upload -> extract audio + sample frames
             -> vision (posture, expression)      [Member B]
             -> speech-to-text + grammar/vocab     [Member C]
             -> content/rubric scoring via Ollama  [Member C]
             -> combine into one multi-parameter report
             -> save session, return to frontend

    Returns a JSON report broken out by parameter — never a single
    blended score, per the requirement.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    session_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{session_id}{ext}"
    audio_path = UPLOAD_DIR / f"{session_id}.wav"

    try:
        # Stream the upload to disk instead of reading it all into memory
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        size_mb = video_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(413, f"File too large ({size_mb:.0f}MB > {MAX_UPLOAD_MB}MB limit)")

        started = time.time()

        duration_s = get_video_duration_seconds(video_path)
        extract_audio(video_path, audio_path)
        frames = sample_frames(video_path)

        # --- Member B: computer vision -------------------------------
        posture_result = vision.analyze_posture(frames)
        expression_result = vision.analyze_expression(frames)

        # --- Member C: speech + language ------------------------------
        transcript = audio.transcribe(audio_path)
        delivery_result = audio.compute_delivery_metrics(transcript, duration_s)
        language_result = language.analyze_grammar_and_vocab(transcript)

        # --- Member C: content scoring via local LLM (Ollama) ---------
        content_result = content.score_content(transcript)

        report = {
            "session_id": session_id,
            "duration_seconds": duration_s,
            "processing_time_seconds": round(time.time() - started, 2),
            "parameters": {
                "content": content_result,
                "delivery": delivery_result,
                "posture": posture_result,
                "expression": expression_result,
                "language": language_result,
            },
            "transcript": transcript["text"],
        }

        # Convert numpy types -> plain Python so JSON serialization can't fail
        report = to_native(report)
        
        db.save_session(session_id, report)
        return report

    except HTTPException:
        raise  # already a clean, intentional error (e.g. 413 too large)

    except Exception as e:
        # Print the full traceback in the terminal for debugging, and send
        # a clear message to the frontend (silent audio, no face, etc).
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {e}")

    finally:
        # Clean up temp files regardless of success/failure
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


@app.get("/api/history")
def history(limit: int = 20):
    """Backs Member D's progress-over-sessions view."""
    return to_native(db.get_recent_sessions(limit=limit))


@app.get("/api/health")
def health():
    """Quick check that every dependency (Ollama, ffmpeg, etc.) is
    reachable — useful when everyone's confirming their laptop can run
    the full stack."""
    checks = {}

    checks["ffmpeg"] = shutil.which("ffmpeg") is not None
    checks["ffprobe"] = shutil.which("ffprobe") is not None

    try:
        checks["ollama"] = content.is_ollama_reachable()
    except Exception:
        checks["ollama"] = False

    checks["db"] = db.is_ready()

    all_ok = all(checks.values())
    return JSONResponse({"ok": all_ok, "checks": checks}, status_code=200 if all_ok else 503)


# Static assets (css/js/images) alongside index.html, if Member D adds any
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")