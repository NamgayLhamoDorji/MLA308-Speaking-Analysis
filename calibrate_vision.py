"""
calibrate_vision.py — Member B's calibration tool (rubric section 9).

Runs ONLY the vision module (no ffmpeg, Ollama or server needed) on several
test videos and shows how each raw metric and sub-score behaves, so you can
decide whether the bands in modules/vision.py need moving.

Usage (project root, venv active):
    python calibrate_vision.py test_videos/                 # every video in a folder
    python calibrate_vision.py a.mp4 b.mp4 c.mp4            # or list files
    python calibrate_vision.py test_videos/ --max-frames 40

Output:
    - per-video raw metrics, sub-scores and final scores
    - a min / median / max table across all videos
    - a warning for any metric that scores 0 (or 100) on EVERY video —
      that means its band is in the wrong place (move it in POSTURE_BANDS /
      EXPRESSION_BANDS at the top of modules/vision.py, then run again)
    - calibration_results.csv (one row per video) for your report
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import cv2

from modules import vision

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}

POSTURE_RAW = ["frames_with_person", "in_frame_pct", "tilt_deg", "off_centre_pct", "sway_pct", "shoulder_width_pct"]
EXPRESSION_RAW = ["frames_with_face", "face_visible_pct", "expressiveness", "eye_contact_pct",
                  "smile_moments_pct", "eyes_open_pct", "avg_yaw"]


def sample_frames(path, max_frames, width=960):
    """Evenly spaced frames, downscaled — same idea as main.py's sampler."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("could not open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) if total > 0 else 15

    frames, i = [], 0
    while cap.grab():
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                h0, w0 = f.shape[:2]
                frames.append(f if w0 <= width else cv2.resize(f, (width, int(h0 * width / w0))))
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError("no frames could be read")
    if len(frames) > max_frames:
        idx = [round(k * (len(frames) - 1) / (max_frames - 1)) for k in range(max_frames)]
        frames = [frames[k] for k in idx]
    return frames


def find_videos(inputs):
    out = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.exists():
            out.append(p)
        else:
            print(f"  (skipping {item}: not found)")
    return out


def fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) and abs(v) < 1 else f"{v:.1f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="video files and/or folders")
    ap.add_argument("--max-frames", type=int, default=60, help="same as MAX_FRAMES in main.py")
    ap.add_argument("--out", default="calibration_results.csv")
    args = ap.parse_args()

    videos = find_videos(args.inputs)
    if not videos:
        sys.exit("No videos found.")

    rows = []
    for v in videos:
        print(f"\n=== {v.name} ===")
        try:
            frames = sample_frames(v, args.max_frames)
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            continue

        p_raw = vision.measure_posture(frames)
        e_raw = vision.measure_expression(frames)
        p_score, p_sub = (vision.score_posture(p_raw) if p_raw["tilt_deg"] is not None else (0.0, {}))
        e_score, e_sub = (vision.score_expression(e_raw) if e_raw["expressiveness"] is not None else (0.0, {}))

        row = {"video": v.name, "frames": len(frames), "posture_score": p_score, "expression_score": e_score}
        for k in POSTURE_RAW:
            row[f"posture_{k}"] = p_raw.get(k)
        for k, s in p_sub.items():
            row[f"posture_sub_{k}"] = s
        for k in EXPRESSION_RAW:
            row[f"expression_{k}"] = e_raw.get(k)
        for k, s in e_sub.items():
            row[f"expression_sub_{k}"] = s
        rows.append(row)

        print(f"  POSTURE    score {p_score:5.1f}   " + ("  ".join(f"{k}={fmt(p_raw.get(k))}" for k in POSTURE_RAW)))
        if p_sub:
            print("             sub-scores: " + "  ".join(f"{k}={s}" for k, s in p_sub.items()))
        print(f"  EXPRESSION score {e_score:5.1f}   " + ("  ".join(f"{k}={fmt(e_raw.get(k))}" for k in EXPRESSION_RAW)))
        if e_sub:
            print("             sub-scores: " + "  ".join(f"{k}={s}" for k, s in e_sub.items()))

    if not rows:
        sys.exit("\nNothing could be analysed.")

    # ---- min / median / max across videos ----
    print("\n" + "=" * 78)
    print(f"SUMMARY across {len(rows)} video(s)")
    print("=" * 78)
    print(f"{'metric':<34}{'min':>10}{'median':>10}{'max':>10}")
    stuck = []
    cols = [c for c in rows[0] if c not in ("video",)]
    for c in cols:
        vals = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
        if not vals:
            continue
        print(f"{c:<34}{fmt(min(vals)):>10}{fmt(statistics.median(vals)):>10}{fmt(max(vals)):>10}")
        if "_sub_" in c and len(vals) >= 3:
            if all(v == 0 for v in vals):
                stuck.append((c, 0))
            elif all(v == 100 for v in vals):
                stuck.append((c, 100))

    if stuck:
        print("\nCHECK THESE BANDS — the sub-score is the same on every video:")
        for c, v in stuck:
            kind, name = c.split("_sub_")
            bands = "POSTURE_BANDS" if kind == "posture" else "EXPRESSION_BANDS"
            side = "zero" if v == 0 else "full"
            print(f"  - {c} is {v} everywhere -> move the '{side}' value of '{name}' in {bands}")
    else:
        print("\nNo metric is stuck at 0 or 100 on every video.")

    # ---- CSV ----
    all_cols = []
    for r in rows:
        for k in r:
            if k not in all_cols:
                all_cols.append(k)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {args.out}  ({len(rows)} row(s))")
    print("Next: ask 2-3 classmates to rate each video's posture and expression 1-10 (without seeing "
          "the scores), then compare the rankings.")


if __name__ == "__main__":
    main()