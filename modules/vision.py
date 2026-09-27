"""
modules/vision.py — Member B (Computer Vision) owns this file.

Contract with main.py (do not change these function names/signatures —
Member A's pipeline calls exactly these two functions):

    analyze_posture(frames: list[np.ndarray]) -> dict
    analyze_expression(frames: list[np.ndarray]) -> dict

`frames` is a list of sampled BGR frames (numpy arrays) — already sampled
every Nth frame by main.py for efficiency, so don't re-sample here.

Each function must return a dict shaped like:
    {
        "score": <float 0-100>,
        "label": <short string, e.g. "Good", "Needs work">,
        "details": <dict of whatever sub-metrics you want to show/debug>,
        "feedback": <1-3 sentences of plain-language coaching text>
    }
That exact shape is what the frontend dashboard (Member D) renders as a
meter strip, so keep it stable even while the internals change.

TODO(Member B): replace the STUB bodies below with real YOLO-Pose /
expression-model inference. Everything else (main.py, the report shape,
the frontend) should keep working unmodified once you do.
"""

import random


def analyze_posture(frames) -> dict:
    # TODO(Member B): run YOLOv8-Pose (or YOLO11-Pose) on `frames`,
    # extract keypoints, and compute alignment / shoulder-tilt /
    # stillness / staying-in-frame metrics -> turn into a 0-100 score.
    score = round(random.uniform(55, 90), 1)
    return {
        "score": score,
        "label": "Good" if score >= 70 else "Needs work",
        "details": {
            "frames_analyzed": len(frames),
            "shoulder_tilt_avg_deg": round(random.uniform(0, 12), 1),
            "in_frame_pct": round(random.uniform(80, 100), 1),
            "note": "STUB DATA — replace with real YOLO-Pose inference",
        },
        "feedback": "Placeholder feedback: keep shoulders level and stay centered in frame.",
    }


def analyze_expression(frames) -> dict:
    # TODO(Member B): run a face detector + expression classifier on
    # `frames`, aggregate into an engagement/expressiveness score.
    score = round(random.uniform(55, 90), 1)
    return {
        "score": score,
        "label": "Engaged" if score >= 70 else "Flat",
        "details": {
            "frames_with_face_detected": max(0, len(frames) - random.randint(0, 2)),
            "dominant_expression": random.choice(["neutral", "engaged", "smiling"]),
            "note": "STUB DATA — replace with real expression-model inference",
        },
        "feedback": "Placeholder feedback: vary your expression to match the energy of your content.",
    }
