"""
modules/content.py — Member C (Speech & Language) owns this file.

Contract with main.py:

    score_content(transcript: dict) -> dict
        Same {"score", "label", "details", "feedback"} shape.

    is_ollama_reachable() -> bool
        Used by main.py's /api/health check.

TODO(Member C):
  1. Design the rubric prompt from your "how professionals speak"
     research (structure, clarity, evidence, relevance).
  2. Send transcript["text"] + rubric to Ollama (e.g. llama3.1:8b, or
     phi3:mini on weaker laptops) via a local HTTP call to
     http://localhost:11434/api/generate — no API key, fully local.
  3. Parse the model's response into a score + written feedback.

No paid API is called anywhere in this file — Ollama runs on localhost.
"""

import random

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "phi3:mini"  # swap to llama3.1:8b on stronger laptops

RUBRIC_PROMPT_TEMPLATE = """You are evaluating a spoken presentation transcript for a public-speaking coaching tool.
Score it 0-100 on: structure, clarity, evidence/examples, and relevance to a clear main point.
Give a one-sentence justification and 2-3 sentences of constructive coaching feedback.

Transcript:
{transcript}

Respond ONLY as JSON: {{"score": <int>, "justification": "...", "feedback": "..."}}
"""


def is_ollama_reachable() -> bool:
    # TODO(Member C): real check, e.g.:
    #   import requests
    #   return requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).status_code == 200
    return True  # STUB — always reports reachable until real check is wired in


def score_content(transcript: dict) -> dict:
    # TODO(Member C): build RUBRIC_PROMPT_TEMPLATE.format(transcript=...),
    # POST to f"{OLLAMA_URL}/api/generate" with {"model": MODEL_NAME,
    # "prompt": prompt, "stream": False}, then json.loads() the model's
    # "response" field (models sometimes wrap JSON in text — strip/parse
    # defensively).
    score = round(random.uniform(55, 90), 1)
    return {
        "score": score,
        "label": "Strong" if score >= 75 else "Developing",
        "details": {
            "note": "STUB DATA — replace with real Ollama rubric-based scoring",
        },
        "feedback": "Placeholder feedback: strengthen your opening with a clearer main point.",
    }
