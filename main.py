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
  (see modules/*.py). Right now those modules return realistic-looking
  STUB data so the whole pipeline runs end-to-end today. As B and C
  finish their real implementations, this file does not need to change —
  only the insides of modules/*.py do. That's the whole point of the
  skeleton-first approach in the workplan.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload
    open http://127.0.0.1:8000
"""

import shutil
import subprocess
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
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
FRAME_SAMPLE_EVERY_N = 10       # Member B: raise this if posture/expression is too slow
MAX_UPLOAD_MB = 300              # adjust if test videos are bigger
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}

app = FastAPI(title="Speaking Evaluation & Coaching Platform")

# db.init() should create the SQLite file/tables if they don't exist yet.
# Member D owns db.py's real implementation (schema, history queries).
db.init()


# ---------------------------------------------------------------------------
# Extraction helpers (Member A owns this: it's the shared input every
# other member's module consumes, so it lives in the integration layer,
# not inside any one member's module)
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, out_path: Path) -> Path:
    """Pull a mono 16kHz WAV out of the video with ffmpeg (what
    faster-whisper and librosa both expect)."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
    return out_path


def sample_frames(video_path: Path, every_n: int = FRAME_SAMPLE_EVERY_N):
    """Yield every Nth frame as a numpy array (BGR), for efficiency —
    we deliberately do NOT run vision models on every single frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video for frame sampling")

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted — video may be corrupt or empty")
    return frames


def get_video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return round(frame_count / fps, 2) if fps else 0.0


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


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
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

    # Stream the upload to disk instead of reading it all into memory
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        video_path.unlink(missing_ok=True)
        raise HTTPException(413, f"File too large ({size_mb:.0f}MB > {MAX_UPLOAD_MB}MB limit)")

    started = time.time()
    try:
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

        db.save_session(session_id, report)
        return report

    except Exception as e:
        # Surface a clear error to the frontend instead of a raw 500 —
        # matches the "clear error states" requirement for edge cases
        # (silent audio, no face in frame, corrupt upload, etc).
        raise HTTPException(500, f"Analysis failed: {e}")

    finally:
        # Clean up temp files regardless of success/failure
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


@app.get("/api/history")
def history(limit: int = 20):
    """Backs Member D's progress-over-sessions view."""
    return db.get_recent_sessions(limit=limit)


@app.get("/api/health")
def health():
    """Quick check that every dependency (Ollama, ffmpeg, etc.) is
    reachable — useful on Day 1 when everyone's confirming their laptop
    can run the full stack."""
    checks = {}

    checks["ffmpeg"] = shutil.which("ffmpeg") is not None

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
