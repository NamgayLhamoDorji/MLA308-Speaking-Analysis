"""
test_vision.py — Member B's standalone test.

Runs ONLY your vision module on a video, so you don't need ffmpeg, Ollama or
the other members' code.

Usage (from the project root, venv active):
    python test_vision.py path/to/video.mp4
    python test_vision.py path/to/video.mp4 --every 5
"""
import argparse
import json
import time

import cv2

from modules import vision


def load_frames(path, every_n):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=10, help="same as FRAME_SAMPLE_EVERY_N in main.py")
    args = ap.parse_args()

    frames = load_frames(args.video, args.every)
    print(f"Loaded {len(frames)} sampled frames")

    t = time.time()
    posture = vision.analyze_posture(frames)
    print(f"\n=== POSTURE ({time.time() - t:.1f}s) ===")
    print(json.dumps(posture, indent=2))

    t = time.time()
    expression = vision.analyze_expression(frames)
    print(f"\n=== EXPRESSION ({time.time() - t:.1f}s) ===")
    print(json.dumps(expression, indent=2))
