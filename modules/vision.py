"""
modules/vision.py — Member B 

Contract with main.py (do not change these names/signatures):

    analyze_posture(frames: list[np.ndarray]) -> dict
    analyze_expression(frames: list[np.ndarray]) -> dict

`frames` is a list of sampled BGR frames, already sampled and downscaled
by main.py. Each function returns:
    {"score": 0-100, "label": str, "details": dict, "feedback": str}

Every number below comes from Professional_Speaking_Rubric.docx
(sections 6 and 7). To change a threshold, edit POSTURE_BANDS or
EXPRESSION_BANDS — nothing else needs to change.

How a band works: each metric earns 100 at its "full" value, 0 at its
"zero" value, and slopes in a straight line between them. The five/four
metric scores are then combined with the "weight" values (they add up to 1).

Also exposed for calibrate_vision.py:
    measure_posture(frames)    -> raw metrics (no scoring)
    measure_expression(frames) -> raw metrics (no scoring)
    score_posture(raw)         -> (score, sub_scores)
    score_expression(raw)      -> (score, sub_scores)

Face landmarks: uses the MediaPipe Face Mesh "solutions" API when it is
installed (mediapipe 0.10.14), otherwise the newer Face Landmarker "tasks"
API with face_landmarker.task from the repo root. Both give 478 landmarks
(including irises), so the metrics are identical.

Install: ultralytics==8.3.0  mediapipe==0.10.14
"""

import atexit
import math
import threading
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Rubric bands — edit here after calibration (rubric section 9)
# ---------------------------------------------------------------------------

POSTURE_BANDS = {
    #  metric        unit                              full   zero   weight
    "tilt":       {"full": 3.0,  "zero": 15.0, "weight": 0.25},   # degrees from horizontal
    "off_centre": {"full": 10.0, "zero": 60.0, "weight": 0.20},   # % of half the frame width
    "sway":       {"full": 2.0,  "zero": 9.0,  "weight": 0.30},   # % of frame (spread of shoulder midpoint)
    "in_frame":   {"full": 95.0, "zero": 50.0, "weight": 0.25},   # % of frames with both shoulders found
}

EXPRESSION_BANDS = {
    "expressiveness": {"full": 0.12, "zero": 0.03, "weight": 0.35},  # (p90-p10)/median of smile width and brow height
    "eye_contact":    {"full": 70.0, "zero": 20.0, "weight": 0.30},  # % of face frames facing camera, irises centred
    "smile_moments":  {"full": 15.0, "zero": 0.0,  "weight": 0.15},  # % of frames mouth >= 10% wider than relaxed
    "eyes_open":      {"full": 92.0, "zero": 65.0, "weight": 0.10},  # % of frames eye height >= 60% of own wide-open
    "face_visible":   {"full": 90.0, "zero": 50.0, "weight": 0.10},  # % of frames with a face found
}

# Other measurement settings (also tunable)
KEYPOINT_CONF_THRESHOLD = 0.4
SHOULDERS_TOO_SMALL_PCT = 15.0       # shoulders narrower than this % of frame width -> "move closer"
SHOULDERS_TOO_BIG_PCT = 75.0         # wider than this -> "move back"

SMILE_WIDER_BY = 1.10                # smile moment = mouth this much wider than the speaker's relaxed mouth
RELAXED_PERCENTILE = 25              # "relaxed mouth" = this percentile of the speaker's mouth width
EYE_OPEN_FRACTION = 0.60             # "eyes open" = at least this fraction of the speaker's own wide-open height
WIDE_OPEN_PERCENTILE = 90            # "wide open" = this percentile of the speaker's eye height
YAW_MAX = 0.20                       # head counts as facing the camera below this left/right asymmetry
IRIS_H_BAND = (0.36, 0.64)           # iris horizontal position across the eye (0.5 = centred)
IRIS_V_BAND = (0.25, 0.75)           # iris vertical position between lids

LABEL_STRONG, LABEL_DEVELOPING = 75, 55   # same cut-offs as the dashboard (rubric section 3)

ROOT = Path(__file__).resolve().parent.parent
POSE_WEIGHTS_PATH = ROOT / "yolov8n-pose.pt"
FACE_TASK_PATH = ROOT / "face_landmarker.task"

# COCO keypoint indices used by YOLOv8-Pose
L_SHOULDER, R_SHOULDER = 5, 6

# MediaPipe face landmark indices (478-point mesh with irises)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
R_EYE_OUTER, R_EYE_INNER, R_EYE_TOP, R_EYE_BOTTOM = 33, 133, 159, 145
L_EYE_OUTER, L_EYE_INNER, L_EYE_TOP, L_EYE_BOTTOM = 263, 362, 386, 374
R_IRIS, L_IRIS = 468, 473
L_BROW, R_BROW = 105, 334
NOSE_TIP, CHEEK_R, CHEEK_L = 1, 234, 454


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def band_score(value, full, zero):
    """100 at `full`, 0 at `zero`, straight line in between. Works whether
    'better' is higher (full > zero) or lower (full < zero)."""
    if value is None:
        return 0.0
    if full == zero:
        return 100.0 if value >= full else 0.0
    t = (value - zero) / (full - zero)
    return round(max(0.0, min(1.0, t)) * 100, 1)


def _combine(raw_by_metric, bands):
    subs = {k: band_score(raw_by_metric[k], b["full"], b["zero"]) for k, b in bands.items()}
    score = round(sum(subs[k] * bands[k]["weight"] for k in bands), 1)
    return score, subs


def _label(score):
    return "Strong" if score >= LABEL_STRONG else "Developing" if score >= LABEL_DEVELOPING else "Needs work"


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _spread(values):
    """(90th - 10th percentile) / median — how much a signal varies,
    relative to the speaker's own typical value."""
    v = np.asarray(values, dtype=float)
    med = float(np.median(v))
    if len(v) < 2 or med < 1e-6:
        return 0.0
    p10, p90 = np.percentile(v, [10, 90])
    return float((p90 - p10) / med)


# ---------------------------------------------------------------------------
# Lazy-loaded models (once per process)
# ---------------------------------------------------------------------------

_pose_model = None
_face_backend = None
_face_lock = threading.Lock()      # FastAPI runs requests in threads; MediaPipe objects aren't thread-safe


def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO
        _pose_model = YOLO(str(POSE_WEIGHTS_PATH))
    return _pose_model


def _get_face_backend():
    global _face_backend
    if _face_backend is None:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5)
            _face_backend = ("solutions", fm)
        else:
            if not FACE_TASK_PATH.exists():
                raise RuntimeError(
                    f"face_landmarker.task not found at {FACE_TASK_PATH}. "
                    "Keep it in the project root next to main.py.")
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(FACE_TASK_PATH)),
                running_mode=mp_vision.RunningMode.IMAGE, num_faces=1)
            _face_backend = ("tasks", mp_vision.FaceLandmarker.create_from_options(options))
            def _close(lm=_face_backend[1]):
                try:
                    lm.close()
                except Exception:
                    pass                                    # harmless at interpreter shutdown
            atexit.register(_close)
    return _face_backend


def _detect_landmarks(rgb):
    """Landmarks (objects with .x/.y in 0-1) for the first face, or None."""
    kind, detector = _get_face_backend()
    with _face_lock:
        if kind == "solutions":
            res = detector.process(rgb)
            return res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
        import mediapipe as mp
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return res.face_landmarks[0] if res.face_landmarks else None


# ---------------------------------------------------------------------------
# Posture — rubric section 6
# ---------------------------------------------------------------------------

def _extract_person_keypoints(result):
    """(17, 3) [x, y, conf] for the largest person in frame (assumed to be
    the speaker), or None if nobody was detected."""
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None
    boxes = result.boxes
    if boxes is not None and len(boxes) > 1:
        areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).cpu().numpy()
        best = int(np.argmax(areas))
    else:
        best = 0
    return result.keypoints.data[best].cpu().numpy()


def measure_posture(frames) -> dict:
    """Raw posture measurements (no scoring)."""
    total = len(frames)
    raw = {"frames_analyzed": total, "frames_with_person": 0, "in_frame_pct": 0.0,
           "tilt_deg": None, "off_centre_pct": None, "sway_pct": None, "shoulder_width_pct": None}
    if total == 0:
        return raw

    results = _get_pose_model()(frames, verbose=False)

    tilts, offsets, mids, widths = [], [], [], []
    for result, frame in zip(results, frames):
        kp = _extract_person_keypoints(result)
        if kp is None:
            continue
        l_sh, r_sh = kp[L_SHOULDER], kp[R_SHOULDER]
        if l_sh[2] < KEYPOINT_CONF_THRESHOLD or r_sh[2] < KEYPOINT_CONF_THRESHOLD:
            continue
        fh, fw = frame.shape[:2]
        dx, dy = float(r_sh[0] - l_sh[0]), float(r_sh[1] - l_sh[1])

        # Angle of the shoulder line from horizontal, always 0-90 degrees.
        # (YOLO's "left" shoulder is the speaker's own left, which sits on the
        # RIGHT of the image, so dx is usually negative. Using atan2(dy, dx)
        # directly gives ~180 deg for a perfectly level speaker — that was the
        # old tilt bug. abs(dx) removes it.)
        tilts.append(abs(math.degrees(math.atan2(dy, abs(dx)))))

        mid_x, mid_y = (l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2
        offsets.append(abs(mid_x - fw / 2) / (fw / 2) * 100)
        mids.append((mid_x / fw, mid_y / fh))
        widths.append(abs(dx) / fw * 100)

    n = len(tilts)
    raw["frames_with_person"] = n
    raw["in_frame_pct"] = round(n / total * 100, 1)
    if n == 0:
        return raw

    xs, ys = [m[0] for m in mids], [m[1] for m in mids]
    sway = (float(np.std(xs)) + float(np.std(ys))) / 2 * 100 if n > 1 else 0.0

    raw.update(tilt_deg=float(np.mean(tilts)), off_centre_pct=float(np.mean(offsets)),
               sway_pct=sway, shoulder_width_pct=float(np.median(widths)))
    return raw


def score_posture(raw: dict):
    metrics = {
        "tilt": raw["tilt_deg"], "off_centre": raw["off_centre_pct"],
        "sway": raw["sway_pct"], "in_frame": raw["in_frame_pct"],
    }
    return _combine(metrics, POSTURE_BANDS)


def analyze_posture(frames) -> dict:
    raw = measure_posture(frames)

    if raw["tilt_deg"] is None:
        return {
            "score": 0.0, "label": "No person detected",
            "details": {"frames_analyzed": raw["frames_analyzed"], "frames_with_person": 0,
                        "in_frame_pct": 0.0,
                        "note": "Could not find both shoulders in any sampled frame."},
            "feedback": "We couldn't see you clearly. Sit or stand so your head and both "
                        "shoulders are in frame, with good lighting.",
        }

    score, subs = score_posture(raw)

    tips = []
    if subs["tilt"] < 70:
        tips.append("keep your shoulders level (a tilted camera can also make them look uneven)")
    if subs["off_centre"] < 70:
        tips.append("stay centred in the frame")
    if subs["sway"] < 70:
        tips.append("reduce swaying or pacing while you speak")
    if subs["in_frame"] < 70:
        tips.append("stay fully in frame throughout")
    if raw["shoulder_width_pct"] < SHOULDERS_TOO_SMALL_PCT:
        tips.append("move closer to the camera so your upper body is easier to see")
    elif raw["shoulder_width_pct"] > SHOULDERS_TOO_BIG_PCT:
        tips.append("move back from the camera so more of your upper body is in frame")

    return {
        "score": score,
        "label": _label(score),
        "details": {
            "frames_analyzed": raw["frames_analyzed"],
            "frames_with_person": raw["frames_with_person"],
            "in_frame_pct": raw["in_frame_pct"],
            "shoulder_tilt_avg_deg": round(raw["tilt_deg"], 1),
            "off_centre_avg_pct": round(raw["off_centre_pct"], 1),
            "sway_pct": round(raw["sway_pct"], 1),
            "shoulder_width_pct_of_frame": round(raw["shoulder_width_pct"], 1),
            "sub_scores": subs,
        },
        "feedback": "Steady, level and well framed — nice posture throughout." if not tips
                    else "Try to " + "; ".join(tips) + ".",
    }


# ---------------------------------------------------------------------------
# Expression — rubric section 7
# ---------------------------------------------------------------------------

def _face_frame(landmarks, w, h):
    """Per-frame measurements, all relative to the distance between the
    eyes so they don't depend on how close the speaker is to the camera."""
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]
    inter = _dist(pts[R_EYE_OUTER], pts[L_EYE_OUTER])
    if inter < 1e-3:
        return None

    out = {
        "mouth_width": _dist(pts[MOUTH_LEFT], pts[MOUTH_RIGHT]) / inter,
        "brow_height": (_dist(pts[L_BROW], pts[L_EYE_TOP]) + _dist(pts[R_BROW], pts[R_EYE_TOP])) / (2 * inter),
        "eye_height": (_dist(pts[R_EYE_TOP], pts[R_EYE_BOTTOM]) + _dist(pts[L_EYE_TOP], pts[L_EYE_BOTTOM])) / (2 * inter),
    }

    # Head facing the camera? Compare nose-to-cheek distance on each side.
    d_r, d_l = _dist(pts[NOSE_TIP], pts[CHEEK_R]), _dist(pts[NOSE_TIP], pts[CHEEK_L])
    yaw = abs(d_r - d_l) / (d_r + d_l) if (d_r + d_l) > 1e-6 else 1.0
    facing = yaw <= YAW_MAX

    # Irises near the centre of each eye? (needs the 478-point mesh)
    centred = False
    if len(pts) > L_IRIS:
        def iris_ok(iris, outer, inner, top, bottom):
            span_x = pts[inner][0] - pts[outer][0]
            span_y = pts[bottom][1] - pts[top][1]
            if abs(span_x) < 1e-3 or abs(span_y) < 1e-3:
                return False
            hx = (pts[iris][0] - pts[outer][0]) / span_x
            vy = (pts[iris][1] - pts[top][1]) / span_y
            return IRIS_H_BAND[0] <= hx <= IRIS_H_BAND[1] and IRIS_V_BAND[0] <= vy <= IRIS_V_BAND[1]
        centred = (iris_ok(R_IRIS, R_EYE_OUTER, R_EYE_INNER, R_EYE_TOP, R_EYE_BOTTOM)
                   and iris_ok(L_IRIS, L_EYE_OUTER, L_EYE_INNER, L_EYE_TOP, L_EYE_BOTTOM))

    out["looking_at_camera"] = bool(facing and centred)
    out["yaw"] = yaw
    return out


def measure_expression(frames) -> dict:
    """Raw expression measurements (no scoring)."""
    total = len(frames)
    raw = {"frames_analyzed": total, "frames_with_face": 0, "face_visible_pct": 0.0,
           "expressiveness": None, "eye_contact_pct": None, "smile_moments_pct": None,
           "eyes_open_pct": None, "avg_yaw": None}
    per_frame, with_face = [], 0

    for frame in frames:
        h, w = frame.shape[:2]
        landmarks = _detect_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if landmarks is None:
            continue
        with_face += 1
        m = _face_frame(landmarks, w, h)
        if m:
            per_frame.append(m)

    raw["frames_with_face"] = with_face
    raw["face_visible_pct"] = round(with_face / total * 100, 1) if total else 0.0
    n = len(per_frame)
    if n == 0:
        return raw

    mouth = [m["mouth_width"] for m in per_frame]
    brow = [m["brow_height"] for m in per_frame]
    eyes = [m["eye_height"] for m in per_frame]

    relaxed_mouth = float(np.percentile(mouth, RELAXED_PERCENTILE))
    wide_open = float(np.percentile(eyes, WIDE_OPEN_PERCENTILE))

    raw.update(
        expressiveness=(_spread(mouth) + _spread(brow)) / 2,
        eye_contact_pct=100 * sum(m["looking_at_camera"] for m in per_frame) / n,
        smile_moments_pct=100 * sum(x >= relaxed_mouth * SMILE_WIDER_BY for x in mouth) / n,
        eyes_open_pct=100 * sum(x >= EYE_OPEN_FRACTION * wide_open for x in eyes) / n,
        avg_yaw=float(np.mean([m["yaw"] for m in per_frame])),
    )
    return raw


def score_expression(raw: dict):
    metrics = {
        "expressiveness": raw["expressiveness"], "eye_contact": raw["eye_contact_pct"],
        "smile_moments": raw["smile_moments_pct"], "eyes_open": raw["eyes_open_pct"],
        "face_visible": raw["face_visible_pct"],
    }
    return _combine(metrics, EXPRESSION_BANDS)


def analyze_expression(frames) -> dict:
    raw = measure_expression(frames)

    if raw["expressiveness"] is None:
        return {
            "score": 0.0, "label": "No face detected",
            "details": {"frames_analyzed": raw["frames_analyzed"], "frames_with_face_detected": 0,
                        "note": "Could not find a face in any sampled frame."},
            "feedback": "We couldn't see your face clearly. Face the camera with good, "
                        "even lighting and keep your whole face in frame.",
        }

    score, subs = score_expression(raw)

    tips = []
    if subs["expressiveness"] < 50:
        tips.append("let your expression change with your message instead of staying flat")
    if subs["eye_contact"] < 60:
        tips.append("look towards the camera lens more often (glancing at notes now and then is fine)")
    if subs["eyes_open"] < 70:
        tips.append("keep your eyes open and engaged")
    if subs["face_visible"] < 70:
        tips.append("face the camera more directly so your face stays visible")
    if not tips and subs["smile_moments"] == 0:
        tips.append("add a natural smile where it fits your message")

    return {
        "score": score,
        "label": _label(score),
        "details": {
            "frames_analyzed": raw["frames_analyzed"],
            "frames_with_face_detected": raw["frames_with_face"],
            "face_visible_pct": raw["face_visible_pct"],
            "expressiveness_index": round(raw["expressiveness"], 3),
            "eye_contact_pct": round(raw["eye_contact_pct"], 1),
            "smile_moments_pct": round(raw["smile_moments_pct"], 1),
            "eyes_open_pct": round(raw["eyes_open_pct"], 1),
            "sub_scores": subs,
            "note": "Eye contact is an estimate: it cannot tell the camera lens from the screen below it.",
        },
        "feedback": "Engaged and expressive — your face supports your message." if not tips
                    else "Try to " + "; ".join(tips) + ".",
    }