"""Caption 60 s clips as 12 pieces of 5 s, with and without context (Gemini, paid key only).

Two modes, same output format (one 1-2 sentence description per 5 s piece):
  in_context  one call per video: the model sees the first 60 s and describes each of the 12 pieces
              (prompts/yt_video_processing/piece_in_context.txt)
  isolated    one call per piece: the model sees only that 5 s piece and is not told where it comes from
              (prompts/yt_video_processing/piece_isolated.txt)

Generation settings match the original captions (config/gen_captions/wild4.yaml): gemini-3-flash-preview,
temperature 0.8, dynamic thinking, seed 0x5EED, 1 fps, local mp4s sent as inline bytes. Pieces are
re-encoded so they start exactly on the second (stream copy would snap to keyframes).

Auth: reads GEMINI_PAID_API_KEY only, never GEMINI_API_KEY. Spending is capped by --max-usd, summed over
every call logged in results/piece_captions/calls.jsonl (all modes), at the --usd-in/--usd-out prices.

  source ~/.secrets && .venv/bin/python scripts/caption_pieces.py isolated --videos Joe-Robinet_11-clip-0 --limit-calls 1
  source ~/.secrets && .venv/bin/python scripts/caption_pieces.py in_context --pilot 30 --max-usd 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import diskcache
import imageio_ffmpeg
import pandas as pd
from google import genai
from google.genai import types

from data_models.captions_only import CaptionedInterval
from experiment_executor.yt_video_processing import _build_local_cache, _normalize_name
from llm.llm_interaction import LLM_Manager_Builder, LLM_ResponseBlocked, LLM_ResponseError
from llm.parsers import parse_llm_response_list

N_PIECES, PIECE_S = 12, 5
VIDEO_DIRS = [ROOT / "local/wild_videos_raw/Videos1", ROOT / "local/wild_videos_raw/Videos2"]
OUT = ROOT / "results/piece_captions"
CALLS_LOG = OUT / "calls.jsonl"
PROMPTS = {
    "in_context": ROOT / "prompts/yt_video_processing/piece_in_context.txt",
    "isolated": ROOT / "prompts/yt_video_processing/piece_isolated.txt",
}
SCHEMAS = {"in_context": "list[CaptionedInterval]", "isolated": "text/plain"}
LLM_CONFIG = dict(model_name="gemini-3-flash-preview", system_instructions=None, temperature=0.8,
                  thought_budget=-1, seed=0x5EED, fps=1)


def channel(video_id: str) -> str:
    return re.sub(r"_\d+(-clip-\d+|-manual)?$", "", video_id)


def benchmark_videos() -> list[str]:
    df = pd.read_csv(ROOT / "results/unified_benchmark_master.csv", usecols=["video_id"])
    return sorted(df.video_id.unique())


def select_pilot(n: int, seed: int) -> list[str]:
    """n videos spread over channels: shuffle channels and videos, then take round-robin."""
    rng = random.Random(seed)
    by_ch = defaultdict(list)
    for v in benchmark_videos():
        by_ch[channel(v)].append(v)
    chans = sorted(by_ch)
    rng.shuffle(chans)
    for c in chans:
        rng.shuffle(by_ch[c])
    out = []
    while len(out) < n and any(by_ch.values()):
        for c in chans:
            if by_ch[c] and len(out) < n:
                out.append(by_ch[c].pop())
    return out


def cut(src: Path, start: float, dur: float) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", str(start), "-i", str(src),
               "-t", str(dur), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac",
               "-movflags", "+faststart", tmp.name]
        subprocess.run(cmd, check=True)
        return Path(tmp.name).read_bytes()


def duration_s(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        return cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1)
    finally:
        cap.release()


def content(video: bytes, prompt: str) -> types.Content:
    return types.Content(parts=[
        types.Part(inline_data=types.Blob(data=video, mime_type="video/mp4"),
                   video_metadata=types.VideoMetadata(fps=LLM_CONFIG["fps"])),
        types.Part(text=prompt),
    ])


def spent_usd() -> float:
    if not CALLS_LOG.exists():
        return 0.0
    return sum(json.loads(l).get("usd") or 0 for l in CALLS_LOG.open())


def log_call(rec: dict) -> None:
    CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CALLS_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def ts_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["in_context", "isolated"])
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--videos", nargs="+", help="video ids")
    sel.add_argument("--pilot", type=int, help="n benchmark videos spread over channels")
    ap.add_argument("--pilot-seed", type=int, default=0)
    ap.add_argument("--pieces", type=int, nargs="+", help="isolated mode: only these piece indices (0-11)")
    ap.add_argument("--max-usd", type=float, default=1.0, help="stop when logged spend reaches this")
    ap.add_argument("--usd-in", type=float, default=0.50, help="$ per 1M input tokens")
    ap.add_argument("--usd-out", type=float, default=3.00, help="$ per 1M output + thinking tokens")
    ap.add_argument("--sleep", type=float, default=15.0, help="seconds between calls")
    ap.add_argument("--limit-calls", type=int, help="stop after this many API calls")
    ap.add_argument("--dry-run", action="store_true", help="list work, no API calls")
    args = ap.parse_args()

    videos = args.videos or select_pilot(args.pilot, args.pilot_seed)
    paths = _build_local_cache(VIDEO_DIRS)
    jobs = []  # (video_id, piece or None, out_path)
    for v in videos:
        if _normalize_name(v) not in paths:
            sys.exit(f"no local video for {v}")
        if args.mode == "in_context":
            jobs.append((v, None, OUT / "in_context" / f"{v}.json"))
        else:
            for p in args.pieces or range(N_PIECES):
                jobs.append((v, p, OUT / "isolated" / v / f"{p:02d}.json"))
    todo = [j for j in jobs if not j[2].exists()]
    print(f"{args.mode}: {len(videos)} videos, {len(jobs)} calls, {len(todo)} to do, "
          f"spent so far ${spent_usd():.4f} of ${args.max_usd}")
    if args.dry_run:
        for v, p, out in todo:
            print(f"  {v} piece={p} -> {out.relative_to(ROOT)}")
        return

    key = os.environ.get("GEMINI_PAID_API_KEY")
    if not key:
        sys.exit("GEMINI_PAID_API_KEY is not set (run: source ~/.secrets)")
    client = genai.Client(api_key=key)
    cache = diskcache.Cache(str(ROOT / "disk_cache/piece_captions"))
    llm = LLM_Manager_Builder(client, cache).from_config({**LLM_CONFIG, "response_schema": SCHEMAS[args.mode]})
    prompt = PROMPTS[args.mode].read_text()

    n_calls = 0
    for v, p, out in todo:
        if args.limit_calls is not None and n_calls >= args.limit_calls:
            print(f"stopping: --limit-calls {args.limit_calls}")
            break
        spent = spent_usd()
        if spent >= args.max_usd:
            print(f"stopping: spent ${spent:.4f} >= --max-usd {args.max_usd}")
            break
        src = paths[_normalize_name(v)]
        if duration_s(src) < N_PIECES * PIECE_S:
            print(f"skip {v}: shorter than {N_PIECES * PIECE_S} s")
            continue
        start, dur = (0.0, N_PIECES * PIECE_S) if p is None else (p * PIECE_S, PIECE_S)
        t0 = time.time()
        res = llm.call(content(cut(src, start, dur), prompt))
        n_calls += 1

        raw = res.raw_response
        usage = raw.usage_metadata.model_dump(exclude_none=True) if raw and raw.usage_metadata else {}
        n_in = usage.get("prompt_token_count", 0)
        n_out = usage.get("candidates_token_count", 0) + usage.get("thoughts_token_count", 0)
        usd = (n_in * args.usd_in + n_out * args.usd_out) / 1e6
        rec = dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"), mode=args.mode, video_id=v,
                   piece=p, model=LLM_CONFIG["model_name"], model_version=raw.model_version if raw else None,
                   cached=raw is None and not isinstance(res, LLM_ResponseError), usage=usage,
                   usd=round(usd, 6), secs=round(time.time() - t0, 1), error=None)

        if isinstance(res, LLM_ResponseError):
            rec["error"] = f"{type(res).__name__}: {res.exception}"
            log_call(rec)
            if isinstance(res, LLM_ResponseBlocked):
                print(f"blocked: {v} piece={p}, continuing")
                continue
            sys.exit(f"API error, stopping: {rec['error']}")

        if args.mode == "isolated":
            pieces = [dict(piece=p, start_s=start, end_s=start + PIECE_S, caption=(res.text or "").strip())]
            ok = bool(pieces[0]["caption"])
        else:
            items = parse_llm_response_list(list[CaptionedInterval], res.text or "")
            pieces = [dict(piece=i, start_s=i * PIECE_S, end_s=(i + 1) * PIECE_S, caption=c.caption.strip(),
                           model_start=c.start, model_end=c.end) for i, c in enumerate(items)]
            ok = len(items) == N_PIECES and all(
                abs(ts_seconds(c.start) - ts_seconds(items[0].start) - i * PIECE_S) < 0.5 for i, c in enumerate(items))
        if not ok:
            rec["error"] = f"invalid output ({len(pieces)} pieces)"
        log_call(rec)
        print(f"{v} piece={p} ok={ok} in={n_in} out={n_out} ${usd:.4f} {rec['model_version']} {rec['secs']}s")

        out.parent.mkdir(parents=True, exist_ok=True)
        dest = out if ok else out.with_suffix(".invalid.json")
        dest.write_text(json.dumps(dict(video_id=v, mode=args.mode, prompt=str(PROMPTS[args.mode].relative_to(ROOT)),
                                        model_version=rec["model_version"], pieces=pieces,
                                        thoughts=res.thoughts, raw_text=res.text), indent=1))
        time.sleep(args.sleep)
    print(f"done: {n_calls} calls, total logged spend ${spent_usd():.4f}")


if __name__ == "__main__":
    main()
