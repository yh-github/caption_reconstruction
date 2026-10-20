#!/usr/bin/env python3
"""TempCompass multiple-choice questions, 5 s windows and the reader frames for the caption-index kill test.

Data: the lmms-eval/TempCompass dataset on Hugging Face (ungated, CC BY-NC 4.0), downloaded to local/tempcompass/
(multi-choice/test-*.parquet and videos/<video_id>.mp4; 410 videos, median 10 s). A video is cut into windows of
5 s, [5p, 5p + 5), the last one shorter when the video does not end on the grid. Window p has frames at
t = 5p + k s (k = 0..4, t < duration), the 1 fps instants the captioner samples, saved with the long edge at
224 px (16:9 sources -> 224x126, 40 Claude image tokens).

Question sets (docs/new_direction/research_log.md): "dev", the 100 development questions (K0-K2); "confirm", 60 per
aspect on the 310 videos the dev set does not use (C1); and "pairs", C4's minimal pairs: each confirmatory question
asked, with its own wording and option order, on a sibling video (one built from the same filmed clip) whose correct
option differs (sample_pairs).

    .venv/bin/python scripts/index_qa/tempcompass.py sample [--set confirm|pairs]   # writes the set's items file
    .venv/bin/python scripts/index_qa/tempcompass.py frames [--set confirm] [--long-edge 448]   # the set's frames
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
SETS = {"dev": OUT / "kill_test_items.json", "confirm": OUT / "confirm_items.json", "pairs": OUT / "pairs_items.json"}
ITEMS = SETS["dev"]
# C1 amendment 2 (research log): questions dropped from every arm, before any reader call on the Gemini captions,
# because the captioner returned no caption for one of their windows
EXCLUDED = {
    "tcc132|1058907236_1021488277_1|order": "frame captions: window 1 answered ERROR (spliced clips)",
    "tcc153|1022115337|order": "video captions: windows 0-1 blocked (PROHIBITED_CONTENT)",
    "tcc169|1096238125_1098970151_1|order": "frame captions: window 1 answered ERROR (spliced clips)",
}
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


def sample(per_dim: int = 20, seed: int = 0, exclude: frozenset[str] = frozenset(), order: list[str] = DIMS,
           prefix: str = "tc", single_first: bool = False) -> list[dict]:
    """per_dim questions per dimension, each on a different video (TempCompass asks several near-paraphrases per
    video, so one question per video keeps the items independent), none on an excluded video. Seeded; dimensions
    are filled in the given order. single_first takes, within a dimension, videos that only that dimension asks
    about before videos that several dimensions ask about."""
    rng = random.Random(seed)
    qs = [q for q in load_questions() if q["video_id"] not in exclude]
    n_dims = {v: len({q["dim"] for q in qs if q["video_id"] == v}) for v in {q["video_id"] for q in qs}}
    used, out = set(), []
    for dim in order:
        pool = [q for q in qs if q["dim"] == dim]
        rng.shuffle(pool)
        if single_first:
            pool.sort(key=lambda q: n_dims[q["video_id"]] > 1)
        for q in pool:
            if len([o for o in out if o["dim"] == dim]) == per_dim:
                break
            if q["video_id"] not in used:
                used.add(q["video_id"])
                out.append(q)
    for i, q in enumerate(out):
        q["item"] = f"{prefix}{i:03d}|{q['video_id']}|{q['dim']}"
    return out


def sample_confirm() -> list[dict]:
    """C1: 60 questions per aspect on the videos the dev set does not use. Aspects are filled rarest first (fewest
    unused videos) and from single-aspect videos first, so a video with questions in several aspects cannot starve
    a rare one."""
    dev = frozenset(q["video_id"] for q in items("dev"))
    qs = [q for q in load_questions() if q["video_id"] not in dev]
    order = sorted(DIMS, key=lambda d: (len({q["video_id"] for q in qs if q["dim"] == d}), DIMS.index(d)))
    out = sample(per_dim=60, seed=1, exclude=dev, order=order, prefix="tcc", single_first=True)
    short = {d: n for d in DIMS if (n := sum(q["dim"] == d for q in out)) < 60}
    if short:
        sys.exit(f"too few unused videos for 60 questions per aspect: {short}")
    return out


def siblings(video_id: str, all_ids: list[str]) -> list[str]:
    """The other videos TempCompass built from the same filmed clip(s), as the same kind of video: a plain video and
    its reversed version; the stacked versions of one clip (different speed pairs); the two-clip videos of the same
    two clips (other order or layout)."""
    t = video_type(video_id)
    if t in ("plain", "reversed"):
        base = video_id.split("_")[0]
        out = [base, f"{base}_reverse"]
    elif t == "stacked":
        out = [v for v in all_ids if video_type(v) == "stacked" and v.rsplit("_", 1)[0] == video_id.rsplit("_", 1)[0]]
    else:
        out = [v for v in all_ids if video_type(v) == "two_clip" and set(v.split("_")[:2]) == set(source_clips(video_id))]
    return sorted(v for v in set(out) if v != video_id and v in all_ids)


def sample_pairs(has_captions) -> list[dict]:
    """C4's minimal pairs (research log): for each confirmatory question, the first sibling video (sorted by ID) with a
    TempCompass question of the same aspect, stem and option set whose correct option differs, and whose windows are
    already captioned by Gemini (has_captions(video_id)). The pair item keeps the confirmatory question's stem and
    option order, so the two reads differ only in the video; its answer is the letter of the sibling's correct option
    in that order."""
    qs = load_questions()
    by_video: dict[str, list[dict]] = {}
    for q in qs:
        by_video.setdefault(q["video_id"], []).append(q)
    ids = sorted(by_video)
    out = []
    for q in items("confirm"):
        right = q["options"][q["answer"]]
        for s in siblings(q["video_id"], ids):
            m = [x for x in by_video[s] if x["dim"] == q["dim"] and x["stem"] == q["stem"]
                 and sorted(x["options"].values()) == sorted(q["options"].values())
                 and x["options"][x["answer"]] != right]
            if m and has_captions(s):
                text = m[0]["options"][m[0]["answer"]]
                letter = next(L for L, o in q["options"].items() if o == text)
                out.append(dict(video_id=s, dim=q["dim"], stem=q["stem"], options=q["options"], answer=letter,
                                pair_of=q["item"], pair_video=q["video_id"]))
                break
    for i, p in enumerate(out):
        p["item"] = f"tcp{i:03d}|{p['video_id']}|{p['dim']}"
    return out


def items(set_: str = "dev") -> list[dict]:
    """The set's questions, without EXCLUDED ones (the items file itself stays as sampled)."""
    if not SETS[set_].exists():
        sys.exit(f"{SETS[set_]} is missing: run scripts/index_qa/tempcompass.py sample --set {set_} first")
    return [q for q in json.loads(SETS[set_].read_text()) if q["item"] not in EXCLUDED]


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


VIDEO_TYPES = ["plain", "two_clip", "reversed", "stacked"]


def video_type(video_id: str) -> str:
    """How TempCompass built the video, from its ID (A3, research log): "reversed" (<id>_reverse, one clip played
    backwards; direction and attribute change), "stacked" (<id>_concat_<k>, the same clip twice, one above the other,
    at equal or different speeds; speed), "two_clip" (<a>_<b>_<k>, two different clips, one after the other or one
    above the other; order), or "plain" (<id>, one clip as filmed)."""
    if re.fullmatch(r"\d+_reverse", video_id):
        return "reversed"
    if re.fullmatch(r"\d+_concat_\d+", video_id):
        return "stacked"
    if re.fullmatch(r"\d+_\d+_\d+", video_id):
        return "two_clip"
    if re.fullmatch(r"\d+", video_id):
        return "plain"
    raise ValueError(f"unknown TempCompass video ID {video_id!r}")


def source_clips(video_id: str) -> list[str]:
    """The IDs of the filmed clips a video is built from (a reversed or stacked video shares its clip with the plain
    one; a two-clip video has two)."""
    t = video_type(video_id)
    if t == "two_clip":
        return video_id.split("_")[:2]
    return [video_id.split("_")[0]]


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


def frame_path(video_id: str, p: int, k: int, long_edge: int = LONG_EDGE) -> Path:
    """The reader frames; frames at another size (C6: 448 px) go to frames_<long_edge>/."""
    root = FRAMES if long_edge == LONG_EDGE else TC / f"frames_{long_edge}"
    return root / video_id / f"{p:02d}_{k}.jpg"


def window_frames(video_id: str, long_edge: int = LONG_EDGE) -> list[list[Path]]:
    """Per window, the paths of its frames in time order."""
    return [[frame_path(video_id, p, t - int(s), long_edge) for t in frame_times(s, e)]
            for p, (s, e) in enumerate(windows(duration_s(video_id)))]


def middle(frames: list[Path]) -> Path:
    """The single frame of the one-frame arm: k = 2 in a full window, the middle one of a shorter window."""
    return frames[len(frames) // 2]


def extract(video_id: str, long_edge: int = LONG_EDGE) -> int:
    scale = f"scale='if(gte(iw,ih),{long_edge},-2)':'if(gte(iw,ih),-2,{long_edge})'"
    made = 0
    for p, (s, e) in enumerate(windows(duration_s(video_id))):
        for t in frame_times(s, e):
            out = frame_path(video_id, p, t - int(s), long_edge)
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
    ap.add_argument("--set", dest="set_", choices=list(SETS), default="dev")
    ap.add_argument("--long-edge", type=int, default=LONG_EDGE)
    args = ap.parse_args()
    if args.cmd == "sample":
        path = SETS[args.set_]
        if path.exists():
            sys.exit(f"{path} exists; a sample is fixed once drawn")
        OUT.mkdir(parents=True, exist_ok=True)
        if args.set_ == "pairs":
            sys.path.insert(0, str(ROOT / "scripts"))
            from index_qa.caption import captions
            qs = sample_pairs(lambda v: captions(v) is not None)
        else:
            qs = sample() if args.set_ == "dev" else sample_confirm()
        path.write_text(json.dumps(qs, indent=1))
        n_win = sum(len(windows(duration_s(q["video_id"]))) for q in qs)
        print(f"{len(qs)} items on {len({q['video_id'] for q in qs})} videos, {n_win} windows -> {path.relative_to(ROOT)}")
    else:
        for q in items(args.set_):
            print(f"{q['video_id']}: {extract(q['video_id'], args.long_edge)} new frames")


if __name__ == "__main__":
    main()
