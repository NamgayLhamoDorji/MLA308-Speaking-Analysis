"""
modules/content.py — Member C (Speech & Language) owns this file.

Contract with main.py (unchanged):

    score_content(transcript) -> {"score", "label", "details", "feedback"}
    is_ollama_reachable() -> bool

The LLM (Ollama, free, local) rates four rubric criteria 1-10. The final
0-100 score is computed HERE from those ratings with fixed weights —
small models are unreliable at arithmetic and drift when asked for one
overall number, so we only ask for the ratings.

If Ollama is down or returns unusable output, we fall back to a simple
rule-based estimate (clearly labelled in details["scored_by"]) so the
whole pipeline still returns a report.

No paid API is called anywhere in this file.
"""

import json
import os
import re

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi3:mini")   # llama3.1:8b on stronger laptops
OLLAMA_TIMEOUT_S = 180        # first call loads the model into RAM — can be slow
MAX_TRANSCRIPT_CHARS = 6000   # phi3:mini has a small context window
MIN_WORDS = 20

# Weights come from the rubric research (structure matters most in the
# public-speaking literature; adjust once the team finalises the rubric).
WEIGHTS = {"structure": 0.30, "clarity": 0.25, "evidence": 0.20, "relevance": 0.25}

RUBRIC_PROMPT_TEMPLATE = """You are an experienced public-speaking coach evaluating the CONTENT of a spoken presentation.

Rate each criterion as an integer from 1 to 10:
- structure: clear opening that states the topic, logically ordered body, and a closing/takeaway
- clarity: ideas are easy to follow, precise, and free of rambling
- evidence: uses concrete examples, facts, numbers, stories or reasoning to support points
- relevance: stays focused on one main point without off-topic tangents

Calibration: typical student speeches score 4-7. Use 9-10 only for exceptional work and 1-3 for very weak work. Do not give every criterion the same number unless it truly fits.

The transcript below was produced by speech-to-text. IGNORE punctuation, capitalisation, spelling and filler words (um, uh). Judge the ideas only.

Transcript:
\"\"\"
{transcript}
\"\"\"

Respond ONLY with JSON in exactly this shape:
{{"structure": <int>, "clarity": <int>, "evidence": <int>, "relevance": <int>,
  "main_point": "<the speaker's main point in one sentence>",
  "strengths": "<one sentence>",
  "improvements": "<one or two sentences>",
  "feedback": "<2-3 sentences of constructive coaching addressed to the speaker as 'you'>"}}
"""


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def is_ollama_reachable() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# LLM call + defensive parsing
# ---------------------------------------------------------------------------

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _parse_llm_json(raw: str):
    """Models sometimes wrap JSON in prose or code fences — pull out the
    first {...} block and validate the four ratings."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    ratings = {}
    for k in WEIGHTS:
        try:
            ratings[k] = int(_clamp(round(float(data[k])), 1, 10))
        except (KeyError, TypeError, ValueError):
            return None
    data["_ratings"] = ratings
    return data


def _ask_ollama(prompt: str):
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",                       # forces valid JSON output
            "options": {"temperature": 0.2, "num_predict": 500},
        },
        timeout=OLLAMA_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json().get("response", "")


# ---------------------------------------------------------------------------
# Offline fallback (only used if Ollama is unavailable/unusable)
# ---------------------------------------------------------------------------

_OPEN = ("today", "talk about", "going to", "let me", "my topic", "i want to", "welcome", "introduce")
_CLOSE = ("in conclusion", "to sum up", "in summary", "to conclude", "finally", "takeaway", "thank you", "remember")
_EXAMPLE = ("for example", "for instance", "such as", "imagine", "when i", "story", "one time")
_EVIDENCE = ("research", "study", "studies", "percent", "%", "according to", "data", "statistics")


def _heuristic_score(text: str):
    low = text.lower()
    words = low.split()
    head, tail = " ".join(words[:60]), " ".join(words[-60:])
    has_open = any(k in head for k in _OPEN)
    has_close = any(k in tail for k in _CLOSE)
    examples = sum(low.count(k) for k in _EXAMPLE)
    has_evidence = any(k in low for k in _EVIDENCE) or bool(re.search(r"\d", low))

    score = (
        35
        + (15 if has_open else 0)
        + (15 if has_close else 0)
        + min(examples, 3) * 7
        + (8 if has_evidence else 0)
        + min(len(words) / 150, 1) * 10
    )
    return round(_clamp(score, 0, 100), 1), {
        "has_opening": has_open, "has_closing": has_close,
        "example_markers": examples, "has_evidence_markers": has_evidence,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _label(score):
    return "Strong" if score >= 75 else "Developing" if score >= 55 else "Needs work"


def score_content(transcript: dict) -> dict:
    text = (transcript.get("text") or "").strip()
    n_words = len(text.split())

    if n_words < MIN_WORDS:
        return {
            "score": 0.0,
            "label": "Not enough speech",
            "details": {"word_count": n_words,
                        "note": f"Fewer than {MIN_WORDS} words — too short to judge content."},
            "feedback": "Your recording was too short to evaluate. Aim for at least "
                        "30-60 seconds with a clear opening, a few points, and a closing.",
        }

    prompt = RUBRIC_PROMPT_TEMPLATE.format(transcript=text[:MAX_TRANSCRIPT_CHARS])

    parsed, error = None, None
    for _ in range(2):                              # one retry if the JSON is unusable
        try:
            parsed = _parse_llm_json(_ask_ollama(prompt))
            if parsed:
                break
            error = "model returned unusable JSON"
        except requests.RequestException as e:
            error = f"Ollama unreachable: {type(e).__name__}"
            break                                   # no point retrying a dead server

    if parsed:
        ratings = parsed["_ratings"]
        score = round(sum(ratings[k] * w for k, w in WEIGHTS.items()) * 10, 1)
        return {
            "score": score,
            "label": _label(score),
            "details": {
                "scored_by": f"ollama:{MODEL_NAME}",
                "criteria_ratings_out_of_10": ratings,
                "weights": WEIGHTS,
                "main_point": str(parsed.get("main_point", ""))[:300],
                "strengths": str(parsed.get("strengths", ""))[:300],
                "improvements": str(parsed.get("improvements", ""))[:400],
            },
            "feedback": str(parsed.get("feedback") or parsed.get("improvements") or "")[:600]
                        or "Work on a clearer opening, supporting examples, and a strong closing.",
        }

    score, signals = _heuristic_score(text)
    return {
        "score": score,
        "label": _label(score),
        "details": {
            "scored_by": "heuristic-fallback",
            "note": f"LLM scoring unavailable ({error}); this is a rough rule-based estimate.",
            "signals": signals,
        },
        "feedback": "Make sure you have a clear opening that names your topic, concrete "
                    "examples or facts in the middle, and a closing takeaway.",
    }