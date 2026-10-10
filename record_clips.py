"""
record_clips.py — Member B: record the two webcam test clips (with microphone audio,
because run_tests.py rejects silent videos).

Usage (project root):
    python record_clips.py upright         # saves test_videos/01d_upright_webcam.mp4 (good posture reference)
    python record_clips.py slouched        # saves test_videos/02b_slouched_webcam.mp4
    python record_clips.py swaying         # saves test_videos/03b_swaying_webcam.mp4

Find your device names first:
    Windows : ffmpeg -list_devices true -f dshow -i dummy
    macOS   : ffmpeg -f avfoundation -list_devices true -i ""
    Linux   : the defaults (/dev/video0, default ALSA mic) usually work

Then pass them if the defaults fail, e.g. on Windows:
    python record_clips.py swaying --video "Integrated Camera" --audio "Microphone Array (Realtek)"

What to do while it records (about 25 s, it stops by itself):
    upright  : sit tall, centred, shoulders level, stay still, talk normally
    slouched : sit hunched and leaning to one side, talk normally
    swaying  : stand or sit and rock / sway from side to side the whole time, talk normally
Talk about anything (e.g. introduce yourself). Whisper needs real speech.
"""
import argparse
import platform
import subprocess
import sys
from pathlib import Path

NAMES = {"upright": "01d_upright_webcam.mp4", "slouched": "02b_slouched_webcam.mp4", "swaying": "03b_swaying_webcam.mp4"}


def build_cmd(out, seconds, video, audio):
    system = platform.system()
    if system == "Windows":
        if not video or not audio:
            sys.exit("On Windows, pass --video and --audio (see ffmpeg -list_devices true -f dshow -i dummy).")
        src = ["-f", "dshow", "-i", f"video={video}:audio={audio}"]
    elif system == "Darwin":
        src = ["-f", "avfoundation", "-framerate", "30", "-i", f"{video or '0'}:{audio or '0'}"]
    else:
        src = ["-f", "v4l2", "-i", video or "/dev/video0", "-f", "alsa", "-i", audio or "default"]
    return ["ffmpeg", "-y", *src, "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "16000", "-ac", "1", str(out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=NAMES)
    ap.add_argument("--seconds", type=int, default=25)
    ap.add_argument("--video", help="camera device name or index")
    ap.add_argument("--audio", help="microphone device name or index")
    ap.add_argument("--folder", default="test_videos")
    a = ap.parse_args()

    Path(a.folder).mkdir(exist_ok=True)
    out = Path(a.folder) / NAMES[a.kind]
    input(f"Get ready to {a.kind}. Press Enter, then start speaking. Recording {a.seconds}s... ")
    r = subprocess.run(build_cmd(out, a.seconds, a.video, a.audio))
    print(f"\nSaved {out}" if r.returncode == 0 else "\nffmpeg failed: check the device names above.")


if __name__ == "__main__":
    main()