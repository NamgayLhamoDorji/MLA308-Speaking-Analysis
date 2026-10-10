"""
modules/prosody.py — tone / intonation scoring (Member C).

Contract:

    analyze_prosody(wav_path, transcript) -> {"score", "label", "details", "feedback"}

`wav_path` is the mono 16 kHz WAV that main.py already extracts with ffmpeg.
`transcript` is the dict from audio.transcribe(): {"text": str, "words": [{"word","start","end"}]}.

What it measures (no new packages: only numpy and the standard library):

    pitch_range_semitones    spread of pitch (90th - 10th percentile). Flat, monotone
                             speech has a small range; lively speech has a wide one.
    emphasis_peaks_per_min   how often pitch jumps clearly above the speaker's recent
                             baseline (stressed words).
    loudness_range_db        spread of loudness (90th - 10th percentile) on voiced frames.
    pace_variation_cv        how much speaking speed changes between 5-second windows
                             (coefficient of variation of words per second).
    statements_ending_in_fall_pct
                             % of statements whose last ~0.5 s of pitch falls (sounds sure).
    uptalk_pct               % of statements that end with rising pitch (sounds unsure).
    median_pitch_hz_not_scored
                             the speaker's typical pitch. NOT scored: voices differ,
                             and a low or high voice is not "better".

Every score is relative to the speaker's OWN pitch (semitones around their median),
so a deep voice and a high voice are treated the same way.

The numbers in TONE_BANDS are starting values. Tune them after comparing the
scores with human ratings (calibrate_tone.py prints each raw metric).
"""

import wave

import numpy as np

# ---------------------------------------------------------------------------
# Bands: 100 points at "full", 0 at "zero", straight line between.
# Works for higher-is-better (full > zero) and lower-is-better (full < zero).
# ---------------------------------------------------------------------------

TONE_BANDS = {
    "pitch_range":    {"full": 9.0,  "zero": 3.0,  "weight": 0.45},   # semitones
    "emphasis":       {"full": 40.0, "zero": 10.0, "weight": 0.10},   # peaks per minute
    "loudness":       {"full": 24.0, "zero": 12.0, "weight": 0.05},   # dB
    "pace_variation": {"full": 0.25, "zero": 0.10, "weight": 0.25},   # coefficient of variation
    "fall":           {"full": 60.0, "zero": 15.0, "weight": 0.075},  # % statements ending in a fall
    "uptalk":         {"full": 10.0, "zero": 50.0, "weight": 0.075},  # % statements ending in a rise (lower is better)
}

LABEL_STRONG, LABEL_DEVELOPING = 75, 55     # same cut-offs as the rest of the project

# Signal settings
SR_EXPECTED = 16000
FRAME_S, HOP_S = 0.040, 0.010
F0_MIN, F0_MAX = 70.0, 400.0
VOICING_MIN = 0.35               # normalised autocorrelation needed to call a frame "voiced"
ENERGY_BELOW_PEAK_DB = 35.0      # frames quieter than (loud level - this) are treated as silence
OUTLIER_SEMITONES = 12.0         # drop pitch estimates this far from the speaker's median (octave errors)
BASELINE_S = 1.5                 # emphasis is measured against a rolling baseline this long
EMPHASIS_MIN_ST = 2.0            # a peak must rise this many semitones above the baseline
EMPHASIS_MIN_FRAMES = 5
ENDING_WINDOW_S = 0.5
ENDING_MIN_FRAMES = 8
FALL_ST, RISE_ST = -0.5, 1.5     # end-of-sentence change that counts as a fall / a rise
MIN_VOICED_S = 5.0
MIN_STATEMENTS = 3
PACE_WINDOW_S = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def band_score(value, full, zero):
    if value is None:
        return None
    t = (value - zero) / (full - zero)
    return round(max(0.0, min(1.0, t)) * 100, 1)


def _label(score):
    return "Strong" if score >= LABEL_STRONG else "Developing" if score >= LABEL_DEVELOPING else "Needs work"


def _read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, n_ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("expected 16-bit WAV")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, sr


def _frames(x, size, hop):
    n = 1 + (len(x) - size) // hop if len(x) >= size else 0
    if n <= 0:
        return np.zeros((0, size), dtype=np.float32)
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _median_filter(v, k=5):
    pad = k // 2
    p = np.pad(v, pad, mode="edge")
    return np.median(np.stack([p[i:i + len(v)] for i in range(k)]), axis=0)


def pitch_track(x, sr):
    """Per 10 ms frame: (f0_hz or nan, loudness_db). Pitch is nan on unvoiced/silent frames."""
    size, hop = int(FRAME_S * sr), int(HOP_S * sr)
    fr = _frames(x, size, hop)
    if len(fr) == 0:
        return np.array([]), np.array([])
    win = np.hanning(size).astype(np.float32)
    fr = fr * win
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)

    nfft = 2048
    spec = np.fft.rfft(fr, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)[:, :size]
    ac = ac / (ac[:, :1] + 1e-12)

    lag_lo, lag_hi = int(sr / F0_MAX), int(sr / F0_MIN)
    seg = ac[:, lag_lo:lag_hi]
    best = np.argmax(seg, axis=1)
    peak = seg[np.arange(len(seg)), best]
    lag = (best + lag_lo).astype(np.float64)

    # parabolic interpolation for a finer lag
    for i in range(len(lag)):
        b = best[i]
        if 0 < b < seg.shape[1] - 1:
            a, c = seg[i, b - 1], seg[i, b + 1]
            d = a - 2 * peak[i] + c
            if abs(d) > 1e-9:
                lag[i] += 0.5 * (a - c) / d

    loud_level = np.percentile(db, 95)
    voiced = (peak >= VOICING_MIN) & (db >= loud_level - ENERGY_BELOW_PEAK_DB)
    f0 = np.where(voiced, sr / lag, np.nan)

    # Remove octave errors: stay within OUTLIER_SEMITONES of the speaker's median pitch.
    if np.sum(~np.isnan(f0)) >= 5:
        med = np.nanmedian(f0)
        st = 12 * np.log2(f0 / med)
        f0 = np.where(np.abs(st) <= OUTLIER_SEMITONES, f0, np.nan)
    return f0, db


def _smooth_semitones(f0):
    """Semitones around the median pitch, median-filtered over voiced runs."""
    ok = ~np.isnan(f0)
    if ok.sum() == 0:
        return np.full_like(f0, np.nan), float("nan")
    med = float(np.nanmedian(f0))
    st = np.full_like(f0, np.nan, dtype=np.float64)
    st[ok] = 12 * np.log2(f0[ok] / med)
    out = st.copy()
    # median-filter inside each continuous voiced run only
    i, n = 0, len(st)
    while i < n:
        if np.isnan(st[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(st[j]):
            j += 1
        if j - i >= 5:
            out[i:j] = _median_filter(st[i:j], 5)
        i = j
    return out, med


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _emphasis_peaks(st):
    """Number of runs where pitch sits >= EMPHASIS_MIN_ST above a rolling baseline."""
    n = len(st)
    if n == 0:
        return 0
    k = int(BASELINE_S / HOP_S)
    excess = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(st[i]):
            continue
        lo, hi = max(0, i - k // 2), min(n, i + k // 2)
        w = st[lo:hi]
        w = w[~np.isnan(w)]
        if len(w) >= 10:
            excess[i] = st[i] - np.median(w)
    above = np.nan_to_num(excess, nan=-99) >= EMPHASIS_MIN_ST
    peaks, run = 0, 0
    for a in above:
        if a:
            run += 1
        else:
            if run >= EMPHASIS_MIN_FRAMES:
                peaks += 1
            run = 0
    if run >= EMPHASIS_MIN_FRAMES:
        peaks += 1
    return peaks


def _pace_variation(words):
    if len(words) < 10:
        return None
    t0, t1 = words[0]["start"], words[-1]["end"]
    if t1 - t0 < 2 * PACE_WINDOW_S:
        return None
    edges = np.arange(t0, t1, PACE_WINDOW_S)
    rates = []
    for a in edges:
        b = a + PACE_WINDOW_S
        if b > t1 + 1e-6 and (t1 - a) < PACE_WINDOW_S * 0.6:
            continue
        cnt = sum(1 for w in words if a <= w["start"] < b)
        rates.append(cnt / PACE_WINDOW_S)
    rates = np.array(rates)
    if len(rates) < 2 or rates.mean() < 1e-6:
        return None
    return float(rates.std() / rates.mean())


def _sentence_endings(words):
    """End times of statements (not questions), from punctuation Whisper puts on words."""
    ends = []
    for w in words:
        tok = w["word"].strip()
        if tok.endswith(".") or tok.endswith("!"):
            ends.append(w["end"])
    return ends


def _ending_shapes(st, ends):
    fall = rise = counted = 0
    for t_end in ends:
        a = int((t_end - ENDING_WINDOW_S) / HOP_S)
        b = int(t_end / HOP_S)
        a = max(a, 0)
        seg = st[a:b]
        pts = np.where(~np.isnan(seg))[0]
        if len(pts) < ENDING_MIN_FRAMES:
            continue
        vals = seg[pts]
        third = max(len(vals) // 3, 2)
        delta = float(np.mean(vals[-third:]) - np.mean(vals[:third]))
        counted += 1
        if delta <= FALL_ST:
            fall += 1
        elif delta >= RISE_ST:
            rise += 1
    if counted < MIN_STATEMENTS:
        return None, None, counted
    return fall / counted * 100, rise / counted * 100, counted


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_prosody(wav_path, transcript) -> dict:
    words = (transcript or {}).get("words") or []
    try:
        x, sr = _read_wav(wav_path)
    except Exception as e:
        return _too_little(f"Could not read the audio ({type(e).__name__}).", 0.0)
    if sr != SR_EXPECTED:
        # keep it simple: resample by linear interpolation to 16 kHz
        n_new = int(len(x) * SR_EXPECTED / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n_new), np.arange(len(x)), x).astype(np.float32)
        sr = SR_EXPECTED

    if len(words) < 8:
        return _too_little(f"Only {len(words)} words were recognised, so tone cannot be judged.", 0.0)

    f0, db = pitch_track(x, sr)
    if len(f0) == 0:
        return _too_little("The audio was too short.", 0.0)
    voiced_s = float(np.sum(~np.isnan(f0)) * HOP_S)
    if voiced_s < MIN_VOICED_S:
        return _too_little(f"Only {voiced_s:.1f} s of voiced speech was found (need {MIN_VOICED_S:.0f} s).", voiced_s)

    st, med_hz = _smooth_semitones(f0)
    ok = ~np.isnan(st)
    p10, p90 = np.percentile(st[ok], [10, 90])
    pitch_range = float(p90 - p10)

    span_s = (words[-1]["end"] - words[0]["start"]) if len(words) >= 2 else len(x) / sr
    span_s = max(span_s, voiced_s, 1.0)
    peaks = _emphasis_peaks(st)
    peaks_per_min = peaks / (span_s / 60)

    l10, l90 = np.percentile(db[ok], [10, 90])
    loud_range = float(l90 - l10)

    cv = _pace_variation(words)
    fall_pct, rise_pct, n_statements = _ending_shapes(st, _sentence_endings(words))

    raw = {"pitch_range": pitch_range, "emphasis": peaks_per_min, "loudness": loud_range,
           "pace_variation": cv, "fall": fall_pct, "uptalk": rise_pct}
    subs = {k: band_score(raw[k], b["full"], b["zero"]) for k, b in TONE_BANDS.items()}
    have = {k: v for k, v in subs.items() if v is not None}
    wsum = sum(TONE_BANDS[k]["weight"] for k in have)
    score = round(sum(v * TONE_BANDS[k]["weight"] for k, v in have.items()) / wsum, 1)

    # Noise guard: wildly erratic pitch/pace usually means background noise, not expression.
    if (cv is not None and cv > 0.6) or peaks_per_min > 60:
        return {
            "score": None,
            "label": "Audio too noisy",
            "details": {"voiced_speech_seconds": round(voiced_s, 1),
                        "pace_variation_cv": None if cv is None else round(cv, 3),
                        "emphasis_peaks_per_min": round(peaks_per_min, 1),
                        "note": "Measurements look erratic, which usually means background noise."},
            "feedback": "We could not judge your tone reliably. Record in a quiet room, close to the microphone.",
        }

    tips = []
    if subs["pitch_range"] is not None and subs["pitch_range"] < 60:
        tips.append("vary your pitch more: let your voice rise and fall instead of staying on one note")
    if subs["emphasis"] is not None and subs["emphasis"] < 60:
        tips.append("stress your key words, for example by saying them a little higher and louder")
    if subs["loudness"] is not None and subs["loudness"] < 60:
        tips.append("use more volume contrast between important and less important phrases")
    if subs["pace_variation"] is not None and subs["pace_variation"] < 60:
        tips.append("change your speed: slow down for key points and speed up for easy ones")
    if subs["uptalk"] is not None and subs["uptalk"] < 60:
        tips.append("end statements with a falling voice; rising endings can make you sound unsure")
    feedback = ("Lively, expressive tone with good variety." if not tips else "Try to " + "; ".join(tips) + ".")

    return {
        "score": score,
        "label": _label(score),
        "details": {
            "voiced_speech_seconds": round(voiced_s, 1),
            "pitch_range_semitones": round(pitch_range, 2),
            "emphasis_peaks_per_min": round(peaks_per_min, 1),
            "loudness_range_db": round(loud_range, 1),
            "pace_variation_cv": None if cv is None else round(cv, 3),
            "statements_ending_in_fall_pct": None if fall_pct is None else round(fall_pct, 1),
            "uptalk_pct": None if rise_pct is None else round(rise_pct, 1),
            "statements_checked": n_statements,
            "median_pitch_hz_not_scored": round(med_hz, 1),
            "sub_scores": subs,
            "note": "Pitch is measured relative to the speaker's own voice. The thresholds are starting values to be tuned.",
        },
        "feedback": feedback,
    }


def _too_little(reason, voiced_s):
    return {
        "score": None,
        "label": "Not enough speech",
        "details": {"voiced_speech_seconds": round(voiced_s, 1), "note": reason},
        "feedback": "We could not measure your tone. Speak for at least 10 to 20 seconds, close to the microphone.",
    }