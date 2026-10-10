"""
run_tests.py — Member A's full-pipeline test runner (workplan days 8-11 and 14-16).

Runs every video in a folder through the SAME steps as POST /api/analyze
(ffmpeg audio, frame sampling, vision, Whisper, language, Ollama content),
then checks each result against what that video is supposed to show.
It does not touch the database, so test runs don't pollute "My progress".

Usage (project root, venv active, Ollama running):
    python run_tests.py test_videos/
    python run_tests.py test_videos/ --manifest test_videos.csv --out test_results.csv

The manifest (test_videos_template.csv is a starting point) has one row per video:

    video, tests, expect_strong, expect_weak, expect_error

    expect_strong : parameters that should score 75+   e.g.  delivery;posture
    expect_weak   : parameters that should score under 55   e.g.  posture
    expect_error  : "yes" (or part of the message, e.g. "No audio") when the
                    video SHOULD be rejected with an error — for edge cases

Outputs:
    test_results.csv          one row per video (scores, overall, PASS/FAIL, notes)
    test_reports/<video>.json the full report for each video, for debugging
    a console summary with the best and worst video for every parameter

A FAIL is not a bug in this script: it means the scores don't "feel right"
for that video, so the owner of that parameter should look at their
thresholds (Member B: modules/vision.py bands; Member C: modules/audio.py
and content.py).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PARAMS = ["content", "delivery", "tone", "posture", "expression", "language"]
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
STRONG, WEAK = 75, 55                       # same cut-offs as the dashboard and rubric

# Keep in sync with WEIGHTS in static/index.html (the overall result).
OVERALL_WEIGHTS = {"content": 0.25, "delivery": 0.20, "tone": 0.15, "posture": 0.10, "expression": 0.15, "language": 0.15}


def overall_score(scores):
    got = {k: v for k, v in scores.items() if isinstance(v, (int, float)) and k in OVERALL_WEIGHTS}
    if not got:
        return None
    w = sum(OVERALL_WEIGHTS[k] for k in got)
    return round(sum(v * OVERALL_WEIGHTS[k] for k, v in got.items()) / w, 1)


def split_list(cell):
    return [x.strip().lower() for x in (cell or "").replace(",", ";").split(";") if x.strip()]


def check_expectations(scores, error, expect_strong, expect_weak, expect_error):
    """Return (status, notes). status is PASS, FAIL or n/a."""
    expect_error = (expect_error or "").strip()
    if expect_error:
        if not error:
            return "FAIL", "expected an error but the video was analysed"
        wanted = expect_error.lower()
        if wanted not in ("yes", "true", "y", "1") and wanted not in error.lower():
            return "FAIL", f"got a different error than expected: {error[:80]}"
        return "PASS", "rejected with an error, as expected"

    if error:
        return "FAIL", f"unexpected error: {error[:120]}"

    strong, weak = split_list(expect_strong), split_list(expect_weak)
    if not strong and not weak:
        return "n/a", "no expectations in the manifest"

    problems = []
    for p in strong:
        s = scores.get(p)
        if not isinstance(s, (int, float)) or s < STRONG:
            problems.append(f"{p}={s} (expected {STRONG}+)")
    for p in weak:
        s = scores.get(p)
        if not isinstance(s, (int, float)) or s >= WEAK:
            problems.append(f"{p}={s} (expected under {WEAK})")
    return ("FAIL", "; ".join(problems)) if problems else ("PASS", "all expectations met")


def load_manifest(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"(manifest {path} not found — running without expectations)")
        return {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        return {(r.get("video") or "").strip().lower(): r for r in csv.DictReader(f) if (r.get("video") or "").strip()}


def analyze_video(path, main):
    """Same steps as main.analyze(), minus saving to the database."""
    from modules import audio, content, language, prosody, vision

    wav = main.UPLOAD_DIR / f"test_{path.stem}.wav"
    try:
        duration = main.get_video_duration_seconds(path)
        main.extract_audio(path, wav)                    # raises "No audio detected..." for silent videos
        frames = main.sample_frames(path)

        posture = vision.analyze_posture(frames)
        expression = vision.analyze_expression(frames)
        transcript = audio.transcribe(wav)
        if len(transcript["text"].split()) < 3:
            raise RuntimeError("No audio speech detected in this video.")
        delivery = audio.compute_delivery_metrics(transcript, duration)
        lang = language.analyze_grammar_and_vocab(transcript)
        tone = prosody.analyze_prosody(wav, transcript)
        cont = content.score_content(transcript)
    finally:
        wav.unlink(missing_ok=True)

    return {
        "duration_seconds": duration,
        "parameters": {"content": cont, "delivery": delivery, "tone": tone, "posture": posture,
                       "expression": expression, "language": lang},
        "transcript": transcript["text"],
    }


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder containing the test videos")
    ap.add_argument("--manifest", default="test_videos.csv")
    ap.add_argument("--out", default="test_results.csv")
    ap.add_argument("--reports", default="test_reports")
    args = ap.parse_args()

    folder = Path(args.folder)
    videos = sorted(f for f in folder.iterdir() if f.suffix.lower() in VIDEO_EXTS) if folder.is_dir() else []
    if not videos:
        sys.exit(f"No videos found in {folder}")

    manifest = load_manifest(args.manifest)
    print("Loading the app's models (first run is slow)...")
    import main as app_main                              # imports the same helpers the website uses
    Path(args.reports).mkdir(exist_ok=True)

    rows = []
    for v in videos:
        m = manifest.get(v.name.lower(), {})
        print(f"\n--- {v.name}  ({m.get('tests', 'not in manifest')}) ---")
        t0 = time.time()
        row = {"video": v.name, "tests": m.get("tests", ""), "duration_s": None, "run_s": None,
               **{p: None for p in PARAMS}, "overall": None, "result": "", "notes": "", "error": ""}
        report, error = None, None
        try:
            report = analyze_video(v, app_main)
        except Exception as e:                            # a crash is a finding, not a reason to stop
            error = f"{type(e).__name__}: {e}"
        row["run_s"] = round(time.time() - t0, 1)

        if report:
            report = app_main.to_native(report)
            row["duration_s"] = report["duration_seconds"]
            for p in PARAMS:
                row[p] = report["parameters"][p].get("score")
            row["overall"] = overall_score({p: row[p] for p in PARAMS})
            d = report["parameters"]["delivery"].get("details", {})
            row["wpm"], row["filler_count"] = d.get("words_per_minute"), d.get("filler_word_count")
            (Path(args.reports) / f"{v.stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            row["wpm"] = row["filler_count"] = None
        row["error"] = error or ""

        row["result"], row["notes"] = check_expectations(
            {p: row[p] for p in PARAMS}, error,
            m.get("expect_strong"), m.get("expect_weak"), m.get("expect_error"))
        rows.append(row)

        scores = "  ".join(f"{p[:4]}={row[p]}" for p in PARAMS)
        print(f"  {scores}  overall={row['overall']}   [{row['result']}] {row['notes']}")

    for name in manifest:                                   # videos promised in the manifest but not recorded yet
        if name not in {v.name.lower() for v in videos}:
            print(f"  (manifest lists {name} but it is not in {folder})")

    # ---- summary ----
    ok = [r for r in rows if not r["error"]]
    print("\n" + "=" * 70)
    passed = sum(r["result"] == "PASS" for r in rows)
    failed = sum(r["result"] == "FAIL" for r in rows)
    print(f"{len(rows)} video(s): {passed} PASS, {failed} FAIL, {len(rows) - passed - failed} without expectations")
    if len(ok) >= 2:
        print("\nBest and worst video for each parameter (the good speaker should be near the top):")
        for p in PARAMS + ["overall"]:
            have = [r for r in ok if isinstance(r[p], (int, float))]
            if len(have) >= 2:
                hi, lo = max(have, key=lambda r: r[p]), min(have, key=lambda r: r[p])
                print(f"  {p:<11} best: {hi['video']} ({hi[p]})   worst: {lo['video']} ({lo[p]})")
    fails = [r for r in rows if r["result"] == "FAIL"]
    if fails:
        print("\nTo fix (tell the owner of each parameter):")
        for r in fails:
            print(f"  {r['video']}: {r['notes']}")

    cols = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {args.out} and the full reports in {args.reports}/")


if __name__ == "__main__":
    main_cli()