# Speaking Evaluation & Coaching Platform (MLA308)

A website where a user uploads a video of themselves speaking. The system scores the
speech on six separate parameters, gives feedback for each one, saves every session,
and shows progress over time so the speaker can see whether they are improving.

Everything uses free tools. No paid APIs are used.

## The six scores

| Score | What it looks at | How |
|---|---|---|
| Content | Structure, clarity, evidence, relevance | Local Ollama model (phi3:mini) rates the transcript against our rubric; our code computes the score |
| Delivery | Pace, filler words, pauses | From the Whisper transcript and word timings |
| Tone | Pitch range, pace variation, emphasis, how statements end | Pitch tracked from the audio with numpy, measured relative to the speaker |
| Posture | Shoulder tilt, off-centre, sway, in frame | MediaPipe Pose on about 60 sampled frames |
| Expression | Expressiveness, eye contact, smiles, eyes open, face visible | MediaPipe face landmarks |
| Language | Grammar and vocabulary variety | LanguageTool and our own vocabulary measure |

Labels everywhere: Strong is 75 or more, Developing is 55 to 74, Needs work is under 55.
The dashboard also shows an overall score out of 100 (content 25%, delivery 20%, tone 15%,
expression 15%, language 15%, posture 10%). The six scores stay separate as the main result.

## Tools used

FastAPI and uvicorn (server), ffmpeg and OpenCV (audio and frames), faster-whisper
(speech to text, base.en), Ollama with phi3:mini (content), LanguageTool (grammar),
MediaPipe (posture and expression), numpy (tone), SQLite (sessions), one HTML file with
plain JavaScript (dashboard), Cloudflare quick tunnel (public link).

## Setup (Windows, macOS or Linux, CPU only)

1. **Python 3.10 or newer.** Check with `python --version`.
2. **ffmpeg** (includes ffprobe). Windows: `winget install ffmpeg`. Mac: `brew install ffmpeg`.
   Check with `ffmpeg -version`.
3. **Ollama.** Install from https://ollama.com, then run `ollama pull phi3:mini`.
4. **Project:**
   ```
   python -m venv venv
   venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```
5. **First run downloads models.** faster-whisper downloads base.en the first time it is used,
   and MediaPipe may download the pose model if it is not bundled. Allow internet for the first run.

## Run

```
uvicorn main:app
```

Open http://127.0.0.1:8000, upload a video and wait for the report (about 1 to 2 minutes for a
3-minute video on a laptop CPU).

Health check: http://127.0.0.1:8000/api/health should show `ffmpeg`, `ffprobe`, `ollama` and `db` all `true`.

Optional settings (set before starting the server):

- `WHISPER_MODEL=tiny.en` faster, less accurate (or `small.en` slower, more accurate)
- `OLLAMA_MODEL=...` use a different Ollama model
- `MAX_FRAMES=30` fewer video frames for the vision step

## Public link (no localhost)

The free Cloudflare quick tunnel gives a public address that reaches the laptop running the app.
Keep both windows open.

```
# window 1
uvicorn main:app
# window 2
cloudflared tunnel --url http://127.0.0.1:8000
```

Open the `https://....trycloudflare.com` address that cloudflared prints. The address changes each
time the tunnel restarts, and it stops working when the laptop or server is off.

## Testing

- `python run_tests.py test_videos --manifest test_videos.csv` runs every test video through the
  full pipeline and marks PASS or FAIL against what each video should show. Results are saved to
  `test_results.csv` and `test_reports/`.
- `python calibrate_vision.py <folder>` shows the raw posture and expression measurements.
- `python calibrate_tone.py <folder>` shows the raw tone measurements and writes `tone_results.csv`.

Latest run: 10 videos, 6 PASS and 4 FAIL. The silent video is rejected with "No audio speech
detected". The failures are explained under Limitations.

## Folder layout

```
main.py                   server, extraction (ffmpeg, OpenCV), wiring of all modules
db.py                     SQLite session storage and history
modules/
  vision.py               posture and expression (MediaPipe)
  audio.py                speech to text, delivery metrics
  language.py             grammar and vocabulary
  prosody.py              tone from the audio
  content.py              content scoring with Ollama
static/index.html         the dashboard (one file, no external libraries)
run_tests.py              end-to-end test runner
calibrate_vision.py       vision measurement viewer
calibrate_tone.py         tone measurement viewer
test_videos.csv           what each test video should show
```

Every module returns `{"score", "label", "details", "feedback"}`.

## Limitations

- **Content scores vary** by about 20 points between runs, because phi3:mini is a small model.
- **Whisper tends to drop "um" and "uh"**, so filler words are undercounted (the filler-heavy test video scored delivery 64, expected under 55).
- **Posture averages over the whole video**, so swaying or slouching for part of it is diluted (the swaying test video scored posture 83, expected under 55).
- **Wide shots and edited clips** make face detection unreliable, so expression can be 0 on stage footage. Webcam recordings are fairer.
- **Eye contact is an estimate** from head direction and iris position, not a measurement.
- **Privacy:** videos and audio are processed on the local machine, and the uploaded video and audio are deleted after processing. The one exception is grammar checking: if no LanguageTool server runs at `localhost:8081`, the transcript text (not the video) is sent to the free public LanguageTool server. If it cannot be reached, the language score is capped at 75 and marked unchecked.
- **Tone thresholds** were set from 12 clips of 4 people and are a first calibration. Deliberate uptalk was not detected reliably. Very noisy audio, or fewer than 8 recognised words, is left out of tone instead of scored.
- **Thresholds have not been compared with human raters.**
- **English only.** Dzongkha was scoped out because free Dzongkha speech and language tools are limited.

## Team

- Member A: backend, integration, testing
- Member B: vision (posture, expression)
- Member C: speech, language, tone, content
- Member D: database, dashboard, documentation