"""
modules/content.py — Member C (Speech & Language) 

Contract with main.py (unchanged):

    score_content(transcript) -> {"score", "label", "details", "feedback"}
    is_ollama_reachable() -> bool

Follows Professional_Speaking_Rubric.docx, section 4:

  1. The local LLM (Ollama, free) first answers three true/false checklist
     questions (opening? closing? example?) and then rates four criteria
     1-10 against written anchors.
  2. The 0-100 score is computed HERE from the ratings with fixed weights —
     small models are unreliable at arithmetic.
  3. Consistency caps in code stop a generous model from contradicting its
     own checklist:
        no opening AND no closing -> structure can't exceed 5
        no example/fact/number/story -> evidence can't exceed 4
  4. One retry on bad JSON, then a clearly labelled rule-based estimate if
     Ollama is down, so the report always completes.

No paid API is called anywhere in this file.
"""

import json
import os
import re

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi3:mini")   # llama3.1:8b on stronger laptops
OLLAMA_TIMEOUT_S = 180        # first call loads the model into RAM and can be slow
MAX_TRANSCRIPT_CHARS = 6000   # phi3:mini has a small context window
MIN_WORDS = 20

# Rubric section 4 weights
WEIGHTS = {"structure": 0.30, "clarity": 0.25, "evidence": 0.20, "relevance": 0.25}
STRUCTURE_CAP_NO_OPEN_CLOSE = 5
EVIDENCE_CAP_NO_EXAMPLE = 4

LABEL_STRONG, LABEL_DEVELOPING = 75, 55   # same cut-offs as the dashboard

RUBRIC_PROMPT = """You are a strict but fair public-speaking coach. Score the CONTENT of a spoken presentation (what is said, not how it sounds).

STEP 1. Answer these checklist questions with true or false:
- has_opening: does the speaker say what the talk is about near the start?
- has_closing: does the speaker finish with a conclusion, summary or takeaway?
- has_example: is there at least one concrete example, fact, number or story?

STEP 2. Rate each criterion from 1 to 10 using these anchors (use in-between numbers when it fits):
structure
  2 = no clear start or end, ideas in random order
  5 = either an opening or a closing is missing, order mostly logical
  8 = topic stated up front, two or more clear points in a logical order, ends with a takeaway
clarity
  2 = hard to follow, rambling or vague
  5 = understandable but wordy, repeats itself, or has unclear parts
  8 = every point is precise and easy to follow
evidence
  2 = only opinions or claims, no support
  5 = one example or fact, thinly explained
  8 = several concrete examples, facts, numbers or stories that really support the points
relevance
  2 = wanders between unrelated topics
  5 = one main point but with noticeable tangents
  8 = everything serves one main point

Calibration: typical student speeches score 4 to 7. Use 9 or 10 only for exceptional work and 1 to 3 for very weak work. Do not give every criterion the same number unless it truly fits.

The transcript was produced by speech-to-text. IGNORE punctuation, capitalisation, spelling, and filler words (um, uh). Judge the ideas only.

Transcript:
\"\"\"
<<TRANSCRIPT>>
\"\"\"

Respond ONLY with JSON in exactly this shape:
{"has_opening": <true|false>, "has_closing": <true|false>, "has_example": <true|false>,
 "structure": <int>, "clarity": <int>, "evidence": <int>, "relevance": <int>,
 "main_point": "<the speaker's main point in one sentence>",
 "strengths": "<one sentence>",
 "improvements": "<one or two sentences>",
 "feedback": "<2-3 sentences of constructive coaching addressed to the speaker as 'you'>"}
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
# Rule-based signals (used for the fallback score, and to fill in a checklist
# answer the model forgot)
# ---------------------------------------------------------------------------

_OPEN = ("today", "talk about", "going to", "let me", "my topic", "i want to", "welcome", "introduce")
_CLOSE = ("in conclusion", "to sum up", "in summary", "to conclude", "finally", "takeaway", "thank you", "remember")
_EXAMPLE = ("for example", "for instance", "such as", "imagine", "when i", "story", "one time")
_EVIDENCE = ("research", "study", "studies", "percent", "%", "according to", "data", "statistics")


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _signals(text: str) -> dict:
    low = text.lower()
    words = low.split()
    head, tail = " ".join(words[:60]), " ".join(words[-60:])
    examples = sum(low.count(k) for k in _EXAMPLE)
    has_evidence = any(k in low for k in _EVIDENCE) or bool(re.search(r"\d", low))
    return {
        "has_opening": any(k in head for k in _OPEN),
        "has_closing": any(k in tail for k in _CLOSE),
        "example_markers": examples,
        "has_evidence_markers": has_evidence,
        "has_example": examples > 0 or has_evidence,
        "word_count": len(words),
    }


def _heuristic_score(text: str):
    s = _signals(text)
    score = (
        35
        + (15 if s["has_opening"] else 0)
        + (15 if s["has_closing"] else 0)
        + min(s["example_markers"], 3) * 7
        + (8 if s["has_evidence_markers"] else 0)
        + min(s["word_count"] / 150, 1) * 10
    )
    return round(_clamp(score, 0, 100), 1), {
        k: s[k] for k in ("has_opening", "has_closing", "example_markers", "has_evidence_markers")
    }


# ---------------------------------------------------------------------------
# LLM call + defensive parsing + consistency caps
# ---------------------------------------------------------------------------

def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.strip().lower() in ("true", "yes"):
            return True
        if v.strip().lower() in ("false", "no"):
            return False
    return None


def _parse_llm_json(raw: str):
    """Pull the JSON object out of the model's reply (models sometimes wrap
    it in prose or code fences) and validate the four ratings."""
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
    if not isinstance(data, dict):
        return None

    ratings = {}
    for k in WEIGHTS:
        try:
            ratings[k] = int(_clamp(round(float(data[k])), 1, 10))
        except (KeyError, TypeError, ValueError):
            return None
    data["_ratings"] = ratings
    data["_checklist"] = {k: _as_bool(data.get(k)) for k in ("has_opening", "has_closing", "has_example")}
    return data


def apply_consistency_caps(ratings: dict, checklist: dict):
    """Return (capped_ratings, list_of_caps_applied). `checklist` values
    must already be real booleans."""
    r, caps = dict(ratings), []
    if not checklist["has_opening"] and not checklist["has_closing"] and r["structure"] > STRUCTURE_CAP_NO_OPEN_CLOSE:
        r["structure"] = STRUCTURE_CAP_NO_OPEN_CLOSE
        caps.append(f"structure capped at {STRUCTURE_CAP_NO_OPEN_CLOSE} (no opening and no closing)")
    if not checklist["has_example"] and r["evidence"] > EVIDENCE_CAP_NO_EXAMPLE:
        r["evidence"] = EVIDENCE_CAP_NO_EXAMPLE
        caps.append(f"evidence capped at {EVIDENCE_CAP_NO_EXAMPLE} (no example, fact, number or story)")
    return r, caps


def _ask_ollama(prompt: str):
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",                       # forces valid JSON output
            "options": {"temperature": 0.2, "num_predict": 600},
        },
        timeout=OLLAMA_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json().get("response", "")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _label(score):
    return "Strong" if score >= LABEL_STRONG else "Developing" if score >= LABEL_DEVELOPING else "Needs work"


def score_content(transcript: dict) -> dict:
    text = (transcript.get("text") or "").strip()
    n_words = len(text.split())

    if n_words < MIN_WORDS:
        return {
            "score": 0.0,
            "label": "Not enough speech",
            "details": {"word_count": n_words,
                        "note": f"Fewer than {MIN_WORDS} words, which is too short to judge content."},
            "feedback": "Your recording was too short to evaluate. Aim for at least "
                        "30-60 seconds with a clear opening, a few points, and a closing.",
        }

    prompt = RUBRIC_PROMPT.replace("<<TRANSCRIPT>>", text[:MAX_TRANSCRIPT_CHARS])

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
        # Fill any checklist answer the model skipped from the rule-based signals.
        sig = _signals(text)
        checklist, filled = {}, []
        for k, v in parsed["_checklist"].items():
            if v is None:
                v = bool(sig[k]); filled.append(k)
            checklist[k] = v

        ratings, caps = apply_consistency_caps(parsed["_ratings"], checklist)
        score = round(sum(ratings[k] * w for k, w in WEIGHTS.items()) * 10, 1)

        details = {
            "scored_by": f"ollama:{MODEL_NAME}",
            "checklist": checklist,
            "criteria_ratings_out_of_10": ratings,
            "weights": WEIGHTS,
            "main_point": str(parsed.get("main_point", ""))[:300],
            "strengths": str(parsed.get("strengths", ""))[:300],
            "improvements": str(parsed.get("improvements", ""))[:400],
        }
        if caps:
            details["consistency_caps_applied"] = caps
        if filled:
            details["checklist_filled_by_rules"] = filled

        return {
            "score": score,
            "label": _label(score),
            "details": details,
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