"""
modules/vision.py 

Contract with main.py (do not change these function names/signatures —
Member A's pipeline calls exactly these two functions):

    analyze_posture(frames: list[np.ndarray]) -> dict
    analyze_expression(frames: list[np.ndarray]) -> dict

`frames` is a list of sampled BGR frames (numpy arrays) — already sampled
every Nth frame by main.py for efficiency, so we don't re-sample here.

Each function returns:
    {
        "score": <float 0-100>,
        "label": <short string>,
        "details": <dict of sub-metrics>,
        "feedback": <1-3 sentences of plain-language coaching text>
    }

--------------------------------------------------------------------------
How this works
--------------------------------------------------------------------------
Posture   -> YOLOv8-Pose (ultralytics), using the yolov8n-pose.pt weights
             already sitting in the repo root. We pull out the shoulder
             keypoints per frame and turn them into:
               - shoulder tilt (level shoulders vs. leaning)
               - how centered/in-frame the speaker stayed
               - how much the shoulder midpoint drifted (swaying/pacing)

Expression -> MediaPipe Face Mesh (468 landmarks). Rather than a trained
             emotion classifier (heavy, hard to explain, needs its own
             training data), we use landmark-geometry heuristics: mouth
             width/openness, eyebrow raise, eye openness, and — most
             importantly — how much the face *varies* across frames,
             since a frozen face reads as "flat" even if it's not
             frowning. This is explainable in the viva and needs no
             extra model download beyond `pip install mediapipe`.

Install (add to requirements.txt):
    ultralytics==8.3.0
    mediapipe==0.10.14
"""

import math
from pathlib import Path
from statistics import mean, pstdev

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Lazy-loaded models — loaded once per process (not once per request)
# ---------------------------------------------------------------------------

_pose_model = None
_face_mesh = None

# yolov8n-pose.pt lives at the repo root, one level up from modules/
POSE_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "yolov8n-pose.pt"

# COCO keypoint indices used by YOLOv8/11-Pose
L_SHOULDER, R_SHOULDER = 5, 6
KEYPOINT_CONF_THRESHOLD = 0.4


def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO
        _pose_model = YOLO(str(POSE_WEIGHTS_PATH))
    return _pose_model


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _face_mesh


# ---------------------------------------------------------------------------
# Posture
# ---------------------------------------------------------------------------

def _extract_person_keypoints(result):
    """Return the (17, 3) [x, y, conf] keypoint array for the most
    prominent person in a single YOLO-Pose result, or None if nobody
    was detected. If several people are in frame, we assume the one
    with the largest bounding box is the speaker (closest to camera)."""
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    boxes = result.boxes
    if boxes is not None and len(boxes) > 1:
        areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).cpu().numpy()
        best_idx = int(np.argmax(areas))
    else:
        best_idx = 0

    return result.keypoints.data[best_idx].cpu().numpy()  # (17, 3)


def analyze_posture(frames) -> dict:
    """Score shoulder level, how centered/in-frame the speaker stayed,
    and how still they held their upper body."""
    model = _get_pose_model()

    frame_h, frame_w = frames[0].shape[:2]
    tilt_angles = []
    centered_offsets = []
    shoulder_midpoints = []
    frames_with_person = 0

    results = model(frames, verbose=False)

    for result, frame in zip(results, frames):
        kp = _extract_person_keypoints(result)
        if kp is None:
            continue

        l_sh, r_sh = kp[L_SHOULDER], kp[R_SHOULDER]
        if l_sh[2] < KEYPOINT_CONF_THRESHOLD or r_sh[2] < KEYPOINT_CONF_THRESHOLD:
            continue

        frames_with_person += 1

        dx = r_sh[0] - l_sh[0]
        dy = r_sh[1] - l_sh[1]
        tilt_angles.append(math.degrees(math.atan2(abs(dy), abs(dx))))

        mid_x = (l_sh[0] + r_sh[0]) / 2
        mid_y = (l_sh[1] + r_sh[1]) / 2
        centered_offsets.append(abs(mid_x - frame_w / 2) / (frame_w / 2) * 100)
        shoulder_midpoints.append((mid_x / frame_w, mid_y / frame_h))

    total_frames = len(frames)
    in_frame_pct = round(frames_with_person / total_frames * 100, 1) if total_frames else 0.0

    if not tilt_angles:
        return {
            "score": 0.0,
            "label": "No person detected",
            "details": {
                "frames_analyzed": total_frames,
                "frames_with_person": 0,
                "in_frame_pct": 0.0,
                "note": "Could not detect a person in any sampled frame.",
            },
            "feedback": "We couldn't detect you clearly in the video. "
                        "Make sure you're well-lit and fully visible in frame.",
        }

    avg_tilt = round(mean(tilt_angles), 1)
    avg_offset_pct = round(mean(centered_offsets), 1)

    if len(shoulder_midpoints) > 1:
        xs = [p[0] for p in shoulder_midpoints]
        ys = [p[1] for p in shoulder_midpoints]
        movement_pct = round((pstdev(xs) + pstdev(ys)) / 2 * 100, 1)
    else:
        movement_pct = 0.0

    tilt_score = max(0.0, 100 - avg_tilt * 6)
    centered_score = max(0.0, 100 - avg_offset_pct * 1.5)
    stillness_score = max(0.0, 100 - movement_pct * 8)

    score = round(
        0.35 * tilt_score
        + 0.25 * centered_score
        + 0.20 * stillness_score
        + 0.20 * in_frame_pct,
        1,
    )

    feedback_bits = []
    if avg_tilt > 8:
        feedback_bits.append("keep your shoulders level")
    if avg_offset_pct > 20:
        feedback_bits.append("stay more centered in the frame")
    if movement_pct > 6:
        feedback_bits.append("reduce swaying or pacing")
    if in_frame_pct < 90:
        feedback_bits.append("stay fully in frame throughout")

    feedback = (
        "Nice, steady, well-framed posture throughout."
        if not feedback_bits
        else "Try to " + "; ".join(feedback_bits) + "."
    )

    return {
        "score": score,
        "label": "Good" if score >= 70 else "Needs work",
        "details": {
            "frames_analyzed": total_frames,
            "frames_with_person": frames_with_person,
            "in_frame_pct": in_frame_pct,
            "shoulder_tilt_avg_deg": avg_tilt,
            "centered_offset_avg_pct": avg_offset_pct,
            "movement_pct": movement_pct,
        },
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------

# MediaPipe Face Mesh landmark indices used below (468-point mesh)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
LEFT_BROW, RIGHT_BROW = 105, 334


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _face_metrics(landmarks, w, h):
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]

    interocular = _dist(pts[LEFT_EYE_OUTER], pts[RIGHT_EYE_OUTER])
    if interocular < 1e-3:
        return None

    mouth_width = _dist(pts[MOUTH_LEFT], pts[MOUTH_RIGHT])
    mouth_height = _dist(pts[MOUTH_TOP], pts[MOUTH_BOTTOM])

    eye_open = (
        _dist(pts[LEFT_EYE_TOP], pts[LEFT_EYE_BOTTOM])
        + _dist(pts[RIGHT_EYE_TOP], pts[RIGHT_EYE_BOTTOM])
    ) / (2 * interocular)

    brow_raise = (
        _dist(pts[LEFT_BROW], pts[LEFT_EYE_TOP])
        + _dist(pts[RIGHT_BROW], pts[RIGHT_EYE_TOP])
    ) / (2 * interocular)

    return {
        "smile_ratio": mouth_width / interocular,
        "mouth_open_ratio": mouth_height / interocular,
        "eye_open": eye_open,
        "brow_raise": brow_raise,
    }


def analyze_expression(frames) -> dict:
    """Score facial engagement using landmark geometry — smile width,
    mouth movement, eyebrow raise, eye openness, and how much the face
    varies over time — rather than a trained emotion classifier."""
    face_mesh = _get_face_mesh()

    per_frame_metrics = []
    frames_with_face = 0

    for frame in frames:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            continue
        frames_with_face += 1
        metrics = _face_metrics(result.multi_face_landmarks[0].landmark, w, h)
        if metrics:
            per_frame_metrics.append(metrics)

    total_frames = len(frames)

    if not per_frame_metrics:
        return {
            "score": 0.0,
            "label": "No face detected",
            "details": {
                "frames_analyzed": total_frames,
                "frames_with_face_detected": 0,
                "note": "Could not detect a face in any sampled frame.",
            },
            "feedback": "We couldn't detect your face clearly. Face the "
                        "camera directly with good lighting.",
        }

    smile_ratios = [m["smile_ratio"] for m in per_frame_metrics]
    mouth_open_ratios = [m["mouth_open_ratio"] for m in per_frame_metrics]
    eye_open_vals = [m["eye_open"] for m in per_frame_metrics]
    brow_vals = [m["brow_raise"] for m in per_frame_metrics]

    avg_smile = mean(smile_ratios)
    avg_eye_open = mean(eye_open_vals)

    smile_variation = pstdev(smile_ratios) if len(smile_ratios) > 1 else 0.0
    mouth_variation = pstdev(mouth_open_ratios) if len(mouth_open_ratios) > 1 else 0.0
    brow_variation = pstdev(brow_vals) if len(brow_vals) > 1 else 0.0
    expressiveness = smile_variation + mouth_variation + brow_variation

    face_detected_pct = round(frames_with_face / total_frames * 100, 1) if total_frames else 0.0

    # Rough generic thresholds — tune against your team's own pilot
    # recordings once you have a few real videos to test against.
    expressiveness_score = min(100.0, expressiveness * 4000)
    eye_openness_score = min(100.0, max(0.0, (avg_eye_open - 0.15) * 800))
    smile_presence_score = min(100.0, max(0.0, (avg_smile - 1.0) * 250))

    score = round(
        0.45 * expressiveness_score
        + 0.30 * eye_openness_score
        + 0.15 * smile_presence_score
        + 0.10 * face_detected_pct,
        1,
    )

    if avg_smile > 1.25:
        dominant_expression = "smiling"
    elif mean(mouth_open_ratios) > 0.25:
        dominant_expression = "speaking / animated"
    else:
        dominant_expression = "neutral"

    feedback_bits = []
    if expressiveness_score < 40:
        feedback_bits.append("let your expression vary more with your content instead of staying flat")
    if eye_openness_score < 40:
        feedback_bits.append("keep your eyes open and engaged with the camera")
    if face_detected_pct < 80:
        feedback_bits.append("face the camera more directly so your expression is visible")

    feedback = (
        "Good, expressive delivery — your face matched the energy of your content."
        if not feedback_bits
        else "Try to " + "; ".join(feedback_bits) + "."
    )

    return {
        "score": score,
        "label": "Engaged" if score >= 70 else "Flat",
        "details": {
            "frames_analyzed": total_frames,
            "frames_with_face_detected": frames_with_face,
            "dominant_expression": dominant_expression,
            "expressiveness_index": round(expressiveness, 4),
            "avg_eye_openness": round(avg_eye_open, 3),
        },
        "feedback": feedback,
    }
