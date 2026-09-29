"""
modules/audio.py — Member C (Speech & Language) owns this file.

Contract with main.py (unchanged):

    transcribe(audio_path) -> {"text": str, "words": [{"word", "start", "end"}, ...]}
    compute_delivery_metrics(transcript, duration_seconds)
        -> {"score", "label", "details", "feedback"}

How it works
------------
transcribe()  : faster-whisper (local, free). Model is loaded once per
                process. Word-level timestamps drive everything below.
delivery      : three explainable sub-scores, combined 40/35/25:
                  pace    - words per minute over the *speaking span*
                            (first word -> last word), target 120-160 WPM
                  fillers - um/uh/er..., "you know", "I mean", and "like"
                            only when set off by commas (rate per 100 words)
                  pauses  - gaps between words; long pauses (>= 2 s) and
                            the share of time spent silent are penalised

The numeric thresholds are named constants below so the team can tune
them during calibration (Days 8-11) and cite the research behind them.
"""

import os
import re

# ---------------------------------------------------------------------------
# Config (tune during calibration)
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base.en")   # "tiny.en" = faster, "small.en" = more accurate
KEEP_FILLERS = os.getenv("WHISPER_KEEP_FILLERS", "1") == "1"

MIN_WORDS = 8                     # below this we can't judge delivery
PACE_TARGET = (120, 160)          # words per minute
PAUSE_MIN_S = 0.8                 # a gap this long counts as a pause
LONG_PAUSE_S = 2.0                # ...and this long counts as an awkward pause

HARD_FILLERS = {"um", "umm", "uh", "uhm", "er", "erm", "ah", "hmm", "mm"}
FILLER_PHRASES = [("you", "know"), ("i", "mean")]

# Whisper tends to "clean up" ums and uhs. Priming it with a prompt that
# contains them makes it much more likely to transcribe them, which we
# need in order to count them.
_FILLER_PRIMING = "Um, so, uh, you know, I mean, like, well, um. "

_whisper = None


def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper


def _norm(token: str) -> str:
    return re.sub(r"[^\w']", "", token.lower())


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(audio_path) -> dict:
    model = _get_whisper()
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        vad_filter=True,                 # skips silence, avoids hallucinated text
        beam_size=1,                     # fast on CPU
        condition_on_previous_text=False,
        initial_prompt=_FILLER_PRIMING if KEEP_FILLERS else None,
    )

    words, parts = [], []
    for seg in segments:                 # generator: transcription runs here
        parts.append(seg.text.strip())
        for w in seg.words or []:
            token = w.word.strip()
            if token:
                words.append({
                    "word": token,
                    "start": round(float(w.start), 2),
                    "end": round(float(w.end), 2),
                })

    return {"text": " ".join(p for p in parts if p).strip(), "words": words}


# ---------------------------------------------------------------------------
# Delivery metrics
# ---------------------------------------------------------------------------

def _count_fillers(words):
    """Return (total, {filler: count}). 'like' counts only when set off
    by commas, e.g. 'it was, like, huge' — not 'I like coffee'."""
    counts = {}

    def add(name):
        counts[name] = counts.get(name, 0) + 1

    toks = [w["word"] for w in words]
    norm = [_norm(t) for t in toks]
    i = 0
    while i < len(norm):
        n = norm[i]
        if n in HARD_FILLERS:
            add(n)
        elif i + 1 < len(norm) and (n, norm[i + 1]) in FILLER_PHRASES:
            add(f"{n} {norm[i + 1]}")
            i += 1
        elif n == "like":
            after_comma = i > 0 and toks[i - 1].endswith(",")
            before_comma = toks[i].endswith(",")
            if after_comma or before_comma:
                add("like")
        i += 1
    return sum(counts.values()), counts


def _pause_stats(words):
    gaps = [
        words[i + 1]["start"] - words[i]["end"]
        for i in range(len(words) - 1)
    ]
    pauses = [g for g in gaps if g >= PAUSE_MIN_S]
    long_pauses = [g for g in pauses if g >= LONG_PAUSE_S]
    return pauses, long_pauses


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def compute_delivery_metrics(transcript: dict, duration_seconds: float) -> dict:
    words = transcript.get("words") or []
    n = len(words)

    if n < MIN_WORDS:
        return {
            "score": 0.0,
            "label": "Not enough speech",
            "details": {
                "word_count": n,
                "note": f"Fewer than {MIN_WORDS} words were detected, so delivery can't be judged.",
            },
            "feedback": "We could hardly hear any speech. Check your microphone, "
                        "speak closer to it, and try a longer recording.",
        }

    span = max(words[-1]["end"] - words[0]["start"], 1.0)   # seconds of actual speaking
    wpm = round(n / (span / 60), 1)
    wpm_overall = round(n / (duration_seconds / 60), 1) if duration_seconds else None

    filler_total, filler_counts = _count_fillers(words)
    filler_rate = round(filler_total / n * 100, 1)

    pauses, long_pauses = _pause_stats(words)
    pause_total = sum(pauses)
    pause_ratio = pause_total / span
    long_per_min = len(long_pauses) / (span / 60)

    # --- sub-scores (each 0-100) ---
    lo, hi = PACE_TARGET
    if wpm < lo:
        pace_score = _clamp(100 - (lo - wpm) * 2.0)      # 70 wpm -> 0
    elif wpm > hi:
        pace_score = _clamp(100 - (wpm - hi) * 2.5)      # 200 wpm -> 0
    else:
        pace_score = 100.0

    filler_score = _clamp(100 - max(0.0, filler_rate - 1.0) * 12.5)   # 9 per 100 words -> 0
    pause_score = _clamp(100 - long_per_min * 15 - max(0.0, pause_ratio - 0.30) * 150)

    score = round(0.40 * pace_score + 0.35 * filler_score + 0.25 * pause_score, 1)

    # --- feedback ---
    bits = []
    if wpm < lo:
        bits.append(f"speed up a little (you spoke at {wpm:.0f} WPM; aim for {lo}-{hi})")
    elif wpm > hi:
        bits.append(f"slow down (you spoke at {wpm:.0f} WPM; aim for {lo}-{hi})")
    if filler_rate > 3:
        top = max(filler_counts, key=filler_counts.get)
        bits.append(f"cut back on filler words, especially \"{top}\"")
    if long_pauses:
        bits.append("avoid long silences — plan your next point before you pause")
    elif pause_ratio > 0.35:
        bits.append("reduce the amount of silence between sentences")

    feedback = (
        "Good pace, few fillers, and comfortable pauses — a confident delivery."
        if not bits else "Try to " + "; ".join(bits) + "."
    )

    return {
        "score": score,
        "label": "Strong delivery" if score >= 80 else "Developing" if score >= 60 else "Needs work",
        "details": {
            "word_count": n,
            "words_per_minute": wpm,
            "wpm_target": f"{lo}-{hi}",
            "wpm_including_silence": wpm_overall,
            "filler_word_count": filler_total,
            "filler_rate_per_100_words": filler_rate,
            "filler_breakdown": filler_counts,
            "pause_count": len(pauses),
            "long_pause_count": len(long_pauses),
            "longest_pause_seconds": round(max(pauses), 2) if pauses else 0.0,
            "silence_share_pct": round(pause_ratio * 100, 1),
            "sub_scores": {
                "pace": round(pace_score, 1),
                "fillers": round(filler_score, 1),
                "pauses": round(pause_score, 1),
            },
        },
        "feedback": feedback,
    }