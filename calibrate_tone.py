import csv, statistics, subprocess, sys, tempfile
from pathlib import Path

from modules import audio, prosody

EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".wav", ".mp3", ".m4a"}
KEYS = ["voiced_speech_seconds", "pitch_range_semitones", "emphasis_peaks_per_min",
        "loudness_range_db", "pace_variation_cv", "statements_ending_in_fall_pct",
        "uptalk_pct", "median_pitch_hz_not_scored"]


def find(inputs):
    out = []
    for i in inputs:
        p = Path(i)
        out += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    return out


def main():
    files = find(sys.argv[1:])
    if not files:
        sys.exit("Usage: python calibrate_tone.py <videos or folder>")
    rows = []
    for f in files:
        print(f"\n=== {f.name} ===")
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            r = subprocess.run(["ffmpeg", "-y", "-i", str(f), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("  ffmpeg failed:", r.stderr[-200:]); continue
            res = prosody.analyze_prosody(wav, audio.transcribe(wav))
        d = res["details"]
        print(f"  TONE score={res['score']}  label={res['label']}")
        for k in KEYS:
            if k in d: print(f"    {k}: {d[k]}")
        print("    sub_scores:", d.get("sub_scores"))
        print("    feedback:", res["feedback"])
        row = {"file": f.name, "tone_score": res["score"], **{k: d.get(k) for k in KEYS}}
        row.update({f"sub_{k}": v for k, v in (d.get("sub_scores") or {}).items()})
        rows.append(row)
    if not rows:
        sys.exit("Nothing analysed.")
    scores = [r["tone_score"] for r in rows if isinstance(r["tone_score"], (int, float))]
    if scores:
        print(f"\nScores: min {min(scores):.1f}  median {statistics.median(scores):.1f}  max {max(scores):.1f}")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open("tone_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("Saved tone_results.csv")


if __name__ == "__main__":
    main()