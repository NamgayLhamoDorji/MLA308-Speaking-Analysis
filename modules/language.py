"""
modules/language.py — Member C (Speech & Language) 

Contract with main.py (unchanged):

    analyze_grammar_and_vocab(transcript) -> {"score", "label", "details", "feedback"}

Grammar    : LanguageTool (rule-based, free, no API key). It tries, in order:
               1. $LANGUAGETOOL_URL if set
               2. a self-hosted server at http://localhost:8081
               3. the free public endpoint https://api.languagetool.org
             If none respond, the module still returns a result (vocabulary
             only) instead of crashing the pipeline.
             We disable spelling / punctuation / casing rules on purpose:
             the transcript is auto-generated, so those "errors" belong to
             Whisper, not the speaker. Only real grammar/word-choice issues
             are counted.
Vocabulary : MATTR (moving-average type-token ratio, window 50) — unlike a
             plain unique/total ratio it isn't unfairly lower for longer
             speeches — plus a list of over-used content words.
"""

import os
import re
import time

import requests

LT_URLS = [u for u in (
    os.getenv("LANGUAGETOOL_URL"),
    "http://localhost:8081/v2/check",
    "https://api.languagetool.org/v2/check",
) if u]
LT_TIMEOUT_S = 15
LT_MAX_CHARS = 15000     # public endpoint limit is 20k chars per request
LT_DISABLED_CATEGORIES = "TYPOS,PUNCTUATION,CASING,TYPOGRAPHY,WHITESPACE"
STYLE_WEIGHT = 0.4       # style suggestions count less than real grammar errors

MATTR_WINDOW = 50

_STOPWORDS = set("""
a about after all also an and any are as at be because been but by can could did do does
for from get had has have he her him his how i if in into is it its just like me more my
no not of on one or our out over so some than that the their them then there these they
this to up us was we were what when which who will with would you your very really
""".split())


def _tokens(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def _mattr(tokens, window=MATTR_WINDOW):
    if len(tokens) <= window:
        return len(set(tokens)) / max(len(tokens), 1)
    ratios = [
        len(set(tokens[i:i + window])) / window
        for i in range(len(tokens) - window + 1)
    ]
    return sum(ratios) / len(ratios)


def _vocab_stats(tokens):
    mattr = _mattr(tokens)
    counts = {}
    for t in tokens:
        if t not in _STOPWORDS and len(t) > 3:
            counts[t] = counts.get(t, 0) + 1
    overused = sorted(
        ((w, c) for w, c in counts.items() if c >= 4),
        key=lambda x: -x[1],
    )[:5]
    # ~0.50 is repetitive speech, ~0.75+ is varied speech
    score = _clamp((mattr - 0.50) / (0.75 - 0.50) * 100)
    return {
        "mattr": round(mattr, 2),
        "unique_words": len(set(tokens)),
        "overused_words": {w: c for w, c in overused},
    }, score


# ---------------------------------------------------------------------------
# Grammar (LanguageTool)
# ---------------------------------------------------------------------------

LT_RETRIES = 2           # attempts per server before moving on to the next one


def _call_languagetool(text):
    """Return (matches, url_used) or (None, None) if no server responded.

    Each server gets LT_RETRIES attempts (the free public endpoint sometimes
    answers 429/5xx or times out once and works a second later), so one
    hiccup no longer forces the 75-point cap.
    """
    for url in LT_URLS:
        for attempt in range(LT_RETRIES):
            try:
                r = requests.post(
                    url,
                    data={
                        "text": text[:LT_MAX_CHARS],
                        "language": "en-US",
                        "disabledCategories": LT_DISABLED_CATEGORIES,
                    },
                    timeout=LT_TIMEOUT_S,
                )
                if r.status_code == 200:
                    return r.json().get("matches", []), url
                if r.status_code not in (429, 500, 502, 503, 504):
                    break                      # a real client error: retrying will not help
            except (requests.RequestException, ValueError):
                pass
            if attempt + 1 < LT_RETRIES:
                time.sleep(1.5)
    return None, None


def _fallback_grammar(text):
    """Very small offline check used only when LanguageTool is unreachable:
    accidental immediate repeats like 'the the'."""
    return re.findall(r"\b(\w+)\s+\1\b", text.lower())


def analyze_grammar_and_vocab(transcript: dict) -> dict:
    text = (transcript.get("text") or "").strip()
    tokens = _tokens(text)
    n = len(tokens)

    if n < 8:
        return {
            "score": 0.0,
            "label": "Not enough speech",
            "details": {"word_count": n,
                        "note": "Too little text to assess grammar or vocabulary."},
            "feedback": "We need a longer recording to assess your language use.",
        }

    vocab, vocab_score = _vocab_stats(tokens)

    matches, lt_url = _call_languagetool(text)
    grammar_checked = matches is not None

    details = {"word_count": n, "vocabulary_variety_mattr": vocab["mattr"],
               "unique_words": vocab["unique_words"],
               "overused_words": vocab["overused_words"],
               "grammar_checked": grammar_checked}
    feedback_bits = []

    if grammar_checked:
        grammar, style, examples = 0, 0, []
        for m in matches:
            cat = (m.get("rule", {}).get("category", {}).get("id") or "").upper()
            if cat == "STYLE":
                style += 1
            else:
                grammar += 1
            if len(examples) < 5:
                ctx = m.get("context", {})
                snippet = ctx.get("text", "")[ctx.get("offset", 0):][: ctx.get("length", 0) + 20]
                examples.append({"issue": m.get("message", ""), "text": snippet.strip()})

        weighted = grammar + STYLE_WEIGHT * style
        per100 = weighted / n * 100
        grammar_score = _clamp(100 - per100 * 15)      # ~6.7 weighted errors/100 words -> 0
        score = round(0.65 * grammar_score + 0.35 * vocab_score, 1)

        details.update({
            "grammar_error_count": grammar,
            "style_suggestion_count": style,
            "errors_per_100_words": round(per100, 1),
            "examples": examples,
            "languagetool_source": "self-hosted" if "localhost" in (lt_url or "") else "public/free endpoint",
            "sub_scores": {"grammar": round(grammar_score, 1), "vocabulary": round(vocab_score, 1)},
        })
        if per100 > 2:
            feedback_bits.append("review the grammar issues flagged in the details (e.g. agreement and word choice)")
    else:
        repeats = _fallback_grammar(text)
        # Grammar was never checked, so never award a top score.
        score = round(min(75.0, _clamp(vocab_score - 10 * len(repeats))), 1)
        details.update({
            "accidental_repeats": len(repeats),
            "languagetool_source": "unreachable (score capped at 75)",
            "note": "LanguageTool was unreachable, so only vocabulary and word-repeat checks were run.",
            "sub_scores": {"vocabulary": round(vocab_score, 1)},
        })
        if repeats:
            feedback_bits.append("avoid accidentally repeating words")

    if vocab_score < 50:
        feedback_bits.append("vary your vocabulary more")
    if vocab["overused_words"]:
        top = next(iter(vocab["overused_words"]))
        feedback_bits.append(f"use synonyms for \"{top}\" (you repeated it often)")

    feedback = (
        "Clear, accurate language with good variety."
        if not feedback_bits else "Try to " + "; ".join(feedback_bits) + "."
    )

    return {
        "score": score,
        "label": ("Clean" if score >= 75 else "Some issues" if score >= 50 else "Needs work")
                 if grammar_checked else "Vocabulary only (grammar not checked)",
        "details": details,
        "feedback": feedback,
    }