"""
test_speech.py — Member C's standalone test.

Runs ONLY the speech/language modules, so you don't need the server, YOLO
or MediaPipe.

Usage (project root, venv active):
    python test_speech.py path/to/video_or_audio.mp4
    python test_speech.py path/to/video.mp4 --skip-llm      # skip Ollama
    python test_speech.py --text "Some transcript text here..." # no audio at all

faster-whisper decodes mp4/webm/wav directly, so no ffmpeg step is needed.
"""
import argparse
import json
import time

from modules import audio, content, language


def show(title, result, t0):
    print(f"\n=== {title} ({time.time() - t0:.1f}s) ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("media", nargs="?", help="video or audio file")
    ap.add_argument("--text", help="skip transcription and use this text")
    ap.add_argument("--skip-llm", action="store_true", help="skip Ollama content scoring")
    args = ap.parse_args()

    if not args.media and not args.text:
        raise SystemExit("Give a media file or --text")

    if args.text:
        words, t = [], 0.0
        for w in args.text.split():
            words.append({"word": w, "start": round(t, 2), "end": round(t + 0.3, 2)})
            t += 0.4
        transcript = {"text": args.text, "words": words}
        duration = t
    else:
        t0 = time.time()
        transcript = audio.transcribe(args.media)
        print(f"\n=== TRANSCRIPT ({time.time() - t0:.1f}s, {len(transcript['words'])} words) ===")
        print(transcript["text"])
        duration = transcript["words"][-1]["end"] if transcript["words"] else 0.0

    t0 = time.time()
    show("DELIVERY", audio.compute_delivery_metrics(transcript, duration), t0)

    t0 = time.time()
    show("LANGUAGE", language.analyze_grammar_and_vocab(transcript), t0)

    if not args.skip_llm:
        print(f"\nOllama reachable: {content.is_ollama_reachable()}")
        t0 = time.time()
        show("CONTENT", content.score_content(transcript), t0)