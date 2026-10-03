"""
calibrate_vision.py - run the posture + expression scorers on your own test
videos and print the raw numbers, so thresholds can be set from real data
(workplan days 8-11: "calibrate thresholds against 6-8 real test videos").

Usage (from the repo root, with the venv active):
    python calibrate_vision.py videos/*.mp4
    python calibrate_vision.py clip1.mov clip2.mov --every 10 --max-frames 120

For each video it prints the scores and the raw metrics, then a summary of
the minimum / median / maximum of every metric across all videos, and writes
calibration_results.csv. Compare that table with POSTURE_BANDS and
EXPRESSION_BANDS at the top of modules/vision.py:
  * a good video should score 75+ and a weak one under 55
  * if every video lands at 0 or 100 for a metric, move its band
"""

import argparse
import csv
import statistics
import sys

import cv2

from modules import vision

POSTURE_KEYS = ["shoulder_tilt_avg_deg", "centered_offset_avg_pct", "movement_pct",
                "in_frame_pct", "shoulder_width_pct"]
EXPRESSION_KEYS = ["expressiveness_index", "smile_moments_pct", "eye_contact_pct",
                   "eyes_open_pct", "avg_smile_ratio", "avg_eye_openness"]


def sample_frames(path, every, max_frames):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--every", type=int, default=10, help="keep every Nth frame (main.py default is 10)")
    ap.add_argument("--max-frames", type=int, default=120)
    args = ap.parse_args()

    rows = []
    for path in args.videos:
        frames = sample_frames(path, args.every, args.max_frames)
        if not frames:
            print(f"!! could not read frames from {path}", file=sys.stderr)
            continue
        posture = vision.analyze_posture(frames)
        expression = vision.analyze_expression(frames)
        row = {"video": path, "frames": len(frames),
               "posture_score": posture["score"], "expression_score": expression["score"]}
        row.update({k: posture["details"].get(k) for k in POSTURE_KEYS})
        row.update({k: expression["details"].get(k) for k in EXPRESSION_KEYS})
        rows.append(row)
        print(f"\n{path}  ({len(frames)} frames)")
        print(f"  posture    {posture['score']:5.1f}  {posture['label']:<12} {posture['feedback']}")
        print(f"  expression {expression['score']:5.1f}  {expression['label']:<12} {expression['feedback']}")
        for k in POSTURE_KEYS + EXPRESSION_KEYS:
            print(f"    {k:<26} {row[k]}")

    if not rows:
        return

    print("\n=== SUMMARY across videos (min / median / max) ===")
    for k in ["posture_score", "expression_score"] + POSTURE_KEYS + EXPRESSION_KEYS:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        if vals:
            print(f"  {k:<26} {min(vals):8.3f} {statistics.median(vals):8.3f} {max(vals):8.3f}")

    with open("calibration_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nSaved calibration_results.csv")


if __name__ == "__main__":
    main()
