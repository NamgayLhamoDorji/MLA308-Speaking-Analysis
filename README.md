# Speaking Platform — Backend Skeleton (Member A)

## What this is
The Day 1–2 shared skeleton from the workplan: a working FastAPI app that
takes a video upload, extracts audio + sampled frames, and calls into
each teammate's module. Every ML-heavy call is currently a **stub**
returning realistic fake data, so the full pipeline runs end-to-end
*today* — nobody is blocked waiting on anyone else's model.

## Folder layout
```
main.py              <- Member A: routes, extraction, wiring (this is the file to run)
db.py                <- Member D: session storage (currently in-memory stub)
modules/
  vision.py           <- Member B: posture + expression (stub)
  audio.py             <- Member C: transcription + delivery metrics (stub)
  language.py           <- Member C: grammar + vocabulary (stub)
  content.py             <- Member C: Ollama rubric scoring (stub)
static/index.html    <- Placeholder page; Member D replaces this with the real dashboard
requirements.txt     <- pip installs (grows as stubs go real)
```

## Day 1 setup (everyone, on their own laptop — Dell Ryzen 7, i5, and Mac all fine)
This project has no heavy training step, so no GPU is required — everything
below runs on CPU. On the Mac, Ollama and faster-whisper both run natively
on Apple Silicon; on the Windows/Intel laptops they run fine on CPU too,
just slightly slower. Use `phi3:mini` (not `llama3.1:8b`) as your default
Ollama model — it's ~2.3GB, runs acceptably on all three machine types, and
is what `modules/content.py` is already set to.

1. **Python 3.10+** — check with `python3 --version`.
2. **ffmpeg** — needed for audio extraction.
   - Mac: `brew install ffmpeg`
   - Windows: `winget install ffmpeg` (or download from ffmpeg.org and add to PATH)
   - Confirm: `ffmpeg -version`
3. **git** — clone/pull the shared repo.
4. **Ollama** (Member C's content scoring, and everyone should confirm it
   works today since it's the one component that behaves differently per OS):
   - Install from https://ollama.com
   - `ollama pull phi3:mini`
   - `ollama run phi3:mini` — type something, confirm you get a reply, then exit.
5. **Project setup:**
   ```bash
   cd member_a_backend
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
6. **Run it:**
   ```bash
   uvicorn main:app --reload
   ```
   Open http://127.0.0.1:8000 — upload any short video file and click
   Analyze. You should get back a JSON report with five stub-scored
   parameters (content, delivery, posture, expression, language) in a
   few seconds. If that works, your machine can run the full stack.
7. **Sanity check endpoint:** http://127.0.0.1:8000/api/health — confirms
   ffmpeg and Ollama are both reachable from Python.

## How the team plugs in (no changes to main.py needed)
Each stub function has a `# TODO(Member X)` comment showing exactly what
to replace and the exact input/output shape main.py expects:

- **Member B** — fill in `modules/vision.py`'s two functions with real
  YOLO-Pose and expression-model inference. `frames` arrives already
  sampled (every 10th frame by default — see `FRAME_SAMPLE_EVERY_N` in
  `main.py` if that needs tuning for speed).
- **Member C** — fill in `modules/audio.py` (faster-whisper),
  `modules/language.py` (LanguageTool), and `modules/content.py`
  (Ollama rubric prompt from your professional-speaking research).
- **Member D** — replace `static/index.html` with the real dashboard,
  and replace `db.py`'s in-memory list with real SQLite (`init()`,
  `save_session()`, `get_recent_sessions()` — same three functions,
  real implementation).

Every module returns the same shape: `{"score", "label", "details",
"feedback"}`. Keep that stable even as the internals change — it's what
the frontend renders as a meter strip per parameter.

## What's real right now vs. stubbed
| Piece | Status |
|---|---|
| Video upload, size/type validation | Real |
| Audio extraction (ffmpeg) | Real |
| Frame sampling (OpenCV) | Real |
| Posture / expression scores | **Stub** — random numbers |
| Transcript | **Stub** — fixed fake sentence |
| WPM / filler count | Real calculation, but on stub transcript |
| Grammar / vocabulary | **Stub** — random numbers |
| Content score (Ollama) | **Stub** — not yet calling Ollama |
| Session storage | **Stub** — in-memory, resets on restart |

Swapping each "Stub" row to real only requires editing that one module —
this is deliberate, so B, C, and D can all work in parallel starting today.
