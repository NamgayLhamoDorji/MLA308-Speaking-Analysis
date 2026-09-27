"""
modules/language.py — Member C (Speech & Language) owns this file.

Contract with main.py:

    analyze_grammar_and_vocab(transcript: dict) -> dict
        Same {"score", "label", "details", "feedback"} shape as the
        other modules.

TODO(Member C): replace the STUB body with a real call to a self-hosted
LanguageTool server (grammar/spelling errors) plus vocabulary-variety
stats (type-token ratio, repeated-word rate) computed from transcript["text"].
"""

import random


def analyze_grammar_and_vocab(transcript: dict) -> dict:
    # TODO(Member C): POST transcript["text"] to your local LanguageTool
    # server, count real errors by category, and compute vocabulary
    # variety (unique words / total words) from the same text.
    text = transcript["text"]
    words = text.split()
    unique_ratio = round(len(set(w.lower().strip(",.") for w in words)) / max(len(words), 1), 2)

    score = round(random.uniform(55, 90), 1)
    return {
        "score": score,
        "label": "Clean" if score >= 75 else "Some issues",
        "details": {
            "grammar_error_count": random.randint(0, 3),
            "vocabulary_variety_ratio": unique_ratio,
            "note": "STUB SCORE — replace grammar_error_count with real LanguageTool output",
        },
        "feedback": "Placeholder feedback: watch for repeated words and check subject-verb agreement.",
    }
