#!/usr/bin/env python3
"""TempCompass multiple-choice questions, 5 s windows and the reader frames for the caption-index kill test.

Data: the lmms-eval/TempCompass dataset on Hugging Face (ungated, CC BY-NC 4.0), downloaded to local/tempcompass/
(multi-choice/test-*.parquet and videos/<video_id>.mp4; 410 videos, median 10 s). A video is cut into windows of
5 s, [5p, 5p + 5), the last one shorter when the video does not end on the grid. Window p has frames at
t = 5p + k s (k = 0..4, t < duration), the 1 fps instants the captioner samples, saved with the long edge at
224 px (16:9 sources -> 224x126, 40 Claude image tokens).

    .venv/bin/python scripts/index_qa/tempcompass.py sample   # writes results/index_qa/kill_test_items.json
    .venv/bin/python scripts/index_qa/tempcompass.py frames   # frames for the sampled videos
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
TC = ROOT / "local/tempcompass"
VIDEOS = TC / "videos"
FRAMES = TC / "frames"
OUT = ROOT / "results/index_qa"
ITEMS = OUT / "kill_test_items.json"
WINDOW_S = 5
LONG_EDGE = 224
DIMS = ["action", "speed", "direction", "order", "attribute_change"]
OPTION = re.compile(r"^\s*([A-F])\.\s*(.*?)\s*$")


def parse_question(question: str, answer: str) -> dict:
    """Split a TempCompass question into its stem and lettered options, and find the answer letter.
    Tolerates the dataset's few malformed options ("C.Both", " D. away")."""
    stem, options = [], {}
    for line in question.split("\n"):
        m = OPTION.match(line)
        if m:
            options[m.group(1)] = m.group(2)
        elif line.strip():
            stem.append(line.strip())
    m = OPTION.match(answer)
    if len(stem) != 1 or not m or m.group(1) not in options:
        raise ValueError(f"cannot parse question {question!r} / answer {answer!r}")
    return dict(stem=stem[0], options=options, answer=m.group(1))


def load_questions() -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(TC / "multi-choice/test-00000-of-00001.parquet")
    return [dict(video_id=r.video_id, dim=r.dim, **parse_question(r.question, r.answer))
            for r in df.itertuples()]


def sample(per_dim: int = 20, seed: int = 0) -> list[dict]:
    """per_dim questions per dimension, each on a different video (TempCompass asks several near-paraphrases per
    video, so one question per video keeps the items independent). Seeded; dimensions in a fixed order."""
    rng = random.Random(seed)
    qs = load_questions()
    used, out = set(), []
    for dim in DIMS:
        pool = [q for q in qs if q["dim"] == dim]
        rng.shuffle(pool)
        for q in pool:
            if len([o for o in out if o["dim"] == dim]) == per_dim:
                break
            if q["video_id"] not in used:
                used.add(q["video_id"])
                out.append(q)
    for i, q in enumerate(out):
        q["item"] = f"tc{i:03d}|{q['video_id']}|{q['dim']}"
    return out


def items() -> list[dict]:
    if not ITEMS.exists():
        sys.exit(f"{ITEMS} is missing: run scripts/index_qa/tempcompass.py sample first")
    return json.loads(ITEMS.read_text())


def first_per_dim(qs: list[dict], n: int | None) -> list[dict]:
    """The first n questions of each aspect, in their order in qs (all of them if n is None)."""
    if n is None:
        return qs
    seen: dict[str, int] = {}
    out = []
    for q in qs:
        if seen.get(q["dim"], 0) < n:
            seen[q["dim"]] = seen.get(q["dim"], 0) + 1
            out.append(q)
    return out


def video_path(video_id: str) -> Path:
    return VIDEOS / f"{video_id}.mp4"


def duration_s(video_id: str) -> float:
    cap = cv2.VideoCapture(str(video_path(video_id)))
    try:
        return cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1)
    finally:
        cap.release()


def windows(duration: float) -> list[tuple[float, float]]:
    """[start, end) of each 5 s window; a last window under 0.5 s is dropped (no frame of its own)."""
    n = max(1, math.ceil(duration / WINDOW_S - 0.1))
    return [(p * WINDOW_S, min((p + 1) * WINDOW_S, duration)) for p in range(n)]


def frame_times(start: float, end: float) -> list[int]:
    return [t for t in range(int(start), int(start) + WINDOW_S) if t < end - 0.05]


def frame_path(video_id: str, p: int, k: int) -> Path:
    return FRAMES / video_id / f"{p:02d}_{k}.jpg"


def window_frames(video_id: str) -> list[list[Path]]:
    """Per window, the paths of its frames in time order."""
    return [[frame_path(video_id, p, t - int(s)) for t in frame_times(s, e)]
            for p, (s, e) in enumerate(windows(duration_s(video_id)))]


def middle(frames: list[Path]) -> Path:
    """The single frame of the one-frame arm: k = 2 in a full window, the middle one of a shorter window."""
    return frames[len(frames) // 2]


def extract(video_id: str) -> int:
    scale = f"scale='if(gte(iw,ih),{LONG_EDGE},-2)':'if(gte(iw,ih),-2,{LONG_EDGE})'"
    made = 0
    for p, (s, e) in enumerate(windows(duration_s(video_id))):
        for t in frame_times(s, e):
            out = frame_path(video_id, p, t - int(s))
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.jpg")
            # Input seek, first frame at or after t (as in scripts/unshuffle/frames.py).
            cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", str(t), "-i", str(video_path(video_id)),
                   "-an", "-frames:v", "1", "-vf", scale, "-q:v", "2", str(tmp)]
            subprocess.run(cmd, check=True)
            if not tmp.exists():
                sys.exit(f"{video_id} t={t}: no frame extracted")
            tmp.replace(out)
            made += 1
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "frames"])
    args = ap.parse_args()
    if args.cmd == "sample":
        if ITEMS.exists():
            sys.exit(f"{ITEMS} exists; the kill-test sample is fixed once drawn")
        OUT.mkdir(parents=True, exist_ok=True)
        qs = sample()
        ITEMS.write_text(json.dumps(qs, indent=1))
        n_win = sum(len(windows(duration_s(q["video_id"]))) for q in qs)
        print(f"{len(qs)} items on {len({q['video_id'] for q in qs})} videos, {n_win} windows -> {ITEMS.relative_to(ROOT)}")
    else:
        for q in items():
            print(f"{q['video_id']}: {extract(q['video_id'])} new frames")


if __name__ == "__main__":
    main()
