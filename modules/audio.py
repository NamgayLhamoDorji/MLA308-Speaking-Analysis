"""
modules/audio.py — Member C (Speech & Language) owns this file.

Contract with main.py:

    transcribe(audio_path: Path) -> dict
        Must return: {"text": str, "words": [{"word": str, "start": float, "end": float}, ...]}
        Word-level timestamps are required for WPM/filler/pause detection.

    compute_delivery_metrics(transcript: dict, duration_seconds: float) -> dict
        Same output shape as vision.py's functions:
        {"score": float, "label": str, "details": {...}, "feedback": str}

TODO(Member C): replace transcribe() with real faster-whisper inference,
and compute_delivery_metrics() with real WPM / filler-word-rate / pause
detection from the word-level timestamps.
"""

import random


def transcribe(audio_path) -> dict:
    # TODO(Member C): load faster-whisper once at module import time
    # (not per-request — it's slow to load), then run it here on
    # `audio_path` and return real text + word timestamps.
    fake_text = (
        "Um, so today I want to talk about, uh, the importance of "
        "clear communication in the workplace."
    )
    words = []
    t = 0.0
    for w in fake_text.split():
        words.append({"word": w, "start": round(t, 2), "end": round(t + 0.3, 2)})
        t += 0.4
    return {"text": fake_text, "words": words}


def compute_delivery_metrics(transcript: dict, duration_seconds: float) -> dict:
    # TODO(Member C): compute real WPM from transcript["words"] and
    # duration_seconds, detect filler words ("um", "uh", "like", "you
    # know"...), and detect pauses (gaps between word end/start times).
    word_count = len(transcript["words"])
    wpm = round(word_count / (duration_seconds / 60), 1) if duration_seconds else 0
    filler_count = sum(1 for w in transcript["words"] if w["word"].lower().strip(",.") in {"um", "uh"})

    score = round(random.uniform(55, 90), 1)
    return {
        "score": score,
        "label": "Good pace" if 110 <= wpm <= 160 else "Check pace",
        "details": {
            "words_per_minute": wpm,
            "filler_word_count": filler_count,
            "note": "STUB SCORE — pace/filler counts above are real, but the 0-100 score formula is a placeholder",
        },
        "feedback": "Placeholder feedback: aim for 110-160 WPM and cut down on filler words.",
    }
