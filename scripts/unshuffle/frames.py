#!/usr/bin/env python3
"""Save the frames the captioner saw, as JPEGs for the frame readers. No API calls.

Piece p covers [5p, 5p + 5), the span scripts/caption_pieces.py cut for Gemini, which sampled it at 1 fps. The
readers get the 5 frames at exactly t = 5p + k s (k = 0..4), the same moments as SigLIP rows 5p..5p+4, so the
frame readers and the frame-embedding arms see the same frames. (Which frame inside each second Gemini took
is not documented; it is within the same second.) The long edge is scaled to 448 px (640x360 sources ->
448x252, about 150 Claude image tokens per frame; the captioner used about 66 Gemini tokens per frame).

    .venv/bin/python scripts/unshuffle/frames.py [--videos V ...]
Writes local/unshuffle_frames/<video file id>/<pp>_<k>.jpg (gitignored).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio_ffmpeg  # noqa: E402

from unshuffle import data  # noqa: E402
from unshuffle.core import N_PIECES, PIECE_S  # noqa: E402

FRAMES = data.ROOT / "local/unshuffle_frames"
LONG_EDGE = 448
VIDEO_DIRS = [data.ROOT / "local/wild_videos_raw/Videos1", data.ROOT / "local/wild_videos_raw/Videos2"]


def frame_paths(video: str, piece: int) -> list[Path]:
    return [FRAMES / data.file_id(video) / f"{piece:02d}_{k}.jpg" for k in range(PIECE_S)]


def source(video: str) -> Path:
    """The raw file the captioner used (caption_pieces.py looks videos up the same way)."""
    from experiment_executor.yt_video_processing import _build_local_cache, _normalize_name
    global _CACHE
    if _CACHE is None:
        _CACHE = _build_local_cache(VIDEO_DIRS)
    if _normalize_name(video) not in _CACHE:
        sys.exit(f"no raw video for {video}")
    return _CACHE[_normalize_name(video)]


_CACHE: dict | None = None


def extract(video: str) -> int:
    src, made = source(video), 0
    scale = (f"scale='if(gte(iw,ih),{LONG_EDGE},-2)':'if(gte(iw,ih),-2,{LONG_EDGE})'")
    for p in range(N_PIECES):
        for k, out in enumerate(frame_paths(video, p)):
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            # Input seek, first frame at or after t: the method scripts/check_frame_timestamps.py verified against
            # the SigLIP rows (offset 0). The fps=1 filter instead picks a frame about 0.5 s into each second.
            tmp = out.with_suffix(".tmp.jpg")
            cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", str(p * PIECE_S + k), "-i", str(src),
                   "-an", "-frames:v", "1", "-vf", scale, "-q:v", "2", str(tmp)]
            subprocess.run(cmd, check=True)
            if not tmp.exists():
                sys.exit(f"{video} piece {p} frame {k}: no frame extracted")
            tmp.replace(out)
            made += 1
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", help="default: the 30 pilot videos")
    args = ap.parse_args()
    videos = args.videos or data.pilot_videos()
    for v in videos:
        print(f"{v}: {extract(v)} new frames")


if __name__ == "__main__":
    main()
