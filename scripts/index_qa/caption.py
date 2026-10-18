#!/usr/bin/env python3
"""Caption every 5 s window of a question set's videos in isolation (Gemini, paid key only).

The same captioner, settings and prompt as the WildQA isolated piece captions (scripts/caption_pieces.py:
gemini-3.8-flash, temperature 0.8, dynamic thinking, 1 fps, audio removed, prompts/yt_video_processing/
piece_isolated.txt). A window shorter than 5 s (the end of a video) gets the prompt with its own length in
place of "5-second"; the prompt hash is saved per window. Calls are logged to results/piece_captions/calls.jsonl
(mode "tc_isolated"), so --max-usd caps the spend summed over all captioning runs, as in caption_pieces.py.

--input frames (C1's coverage control) gives Gemini, instead of the video, exactly the window's JPEG frames that
the frames5 reader arm shows, with the prompt the self-captions use (frames_prompt). Those captions go to the
captioner directory <model>-frames (mode "tc_isolated_frames").

    source ~/.secrets && .venv/bin/python scripts/index_qa/caption.py --dry-run
    source ~/.secrets && .venv/bin/python scripts/index_qa/caption.py --limit-calls 3 --max-usd 2.5
    source ~/.secrets && .venv/bin/python scripts/index_qa/caption.py --set confirm --input frames --max-usd 7.5
Writes results/index_qa/captions/<captioner>/<video_id>/<pp>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import caption_pieces as cp  # noqa: E402
from index_qa import tempcompass as tc  # noqa: E402

CAPTIONS = tc.OUT / "captions"
MODE = "tc_isolated"
GEMINI_LEAD = "Describe what happens in the attached 5-second video clip,"


def prompt_for(base: str, dur: float) -> str:
    return base if dur >= tc.WINDOW_S - 0.5 else base.replace("5-second", f"{max(1, round(dur))}-second")


def frames_prompt(n_frames: int, dur: float) -> str:
    """The video caption prompt with its first clause describing frames instead of a video clip; everything else
    is identical. Used for captions of the frames5 JPEGs, by Gemini (--input frames) and by Haiku (self_caption.py)."""
    base = cp.PROMPTS["isolated"].read_text()
    if GEMINI_LEAD not in base:
        sys.exit("the Gemini caption prompt changed; update GEMINI_LEAD")
    lead = (f"The attached images are {n_frames} frames from a {max(1, round(dur))}-second video clip, taken one "
            f"second apart and shown in time order. Describe what happens in the clip,")
    return base.replace(GEMINI_LEAD, lead)


def frames_content(frames: list[Path], prompt: str):
    return cp.types.Content(parts=[*(cp.types.Part(inline_data=cp.types.Blob(data=f.read_bytes(), mime_type="image/jpeg"))
                                     for f in frames), cp.types.Part(text=prompt)])


def caption_path(model: str, video_id: str, p: int) -> Path:
    return CAPTIONS / model / video_id / f"{p:02d}.json"


def captions(video_id: str, model: str = cp.LLM_CONFIG["model_name"]) -> list[str] | None:
    """The window captions of a video in time order, or None if any window is missing."""
    out = []
    for p in range(len(tc.windows(tc.duration_s(video_id)))):
        f = caption_path(model, video_id, p)
        if not f.exists():
            return None
        out.append(json.loads(f.read_text())["caption"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=cp.LLM_CONFIG["model_name"])
    ap.add_argument("--set", dest="set_", choices=list(tc.SETS), default="dev")
    ap.add_argument("--input", choices=["video", "frames"], default="video")
    ap.add_argument("--max-usd", type=float, default=2.5, help="stop when the spend logged in calls.jsonl reaches this")
    ap.add_argument("--usd-in", type=float, default=0.50, help="$ per 1M input tokens")
    ap.add_argument("--usd-out", type=float, default=3.00, help="$ per 1M output + thinking tokens")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between calls")
    ap.add_argument("--limit-calls", type=int, help="stop after this many API calls")
    ap.add_argument("--dry-run", action="store_true", help="list work, no API calls")
    args = ap.parse_args()

    captioner = args.model if args.input == "video" else f"{args.model}-frames"
    mode = MODE if args.input == "video" else f"{MODE}_frames"
    jobs = [(q["video_id"], p, s, e, frames) for q in tc.items(args.set_)
            for p, ((s, e), frames) in enumerate(zip(tc.windows(tc.duration_s(q["video_id"])),
                                                     tc.window_frames(q["video_id"])))]
    if args.input == "frames" and (missing := [f for j in jobs for f in j[4] if not f.exists()]):
        sys.exit(f"{len(missing)} frames missing, e.g. {missing[0]}: run tempcompass.py frames --set {args.set_}")
    todo = [j for j in jobs if not caption_path(captioner, j[0], j[1]).exists()]
    print(f"{len(jobs)} windows, {len(todo)} to do, spent so far ${cp.spent_usd():.4f} of ${args.max_usd}")
    if args.dry_run:
        return

    key = os.environ.get("GEMINI_PAID_API_KEY")
    if not key:
        sys.exit("GEMINI_PAID_API_KEY is not set (run: source ~/.secrets)")
    client = cp.genai.Client(api_key=key)
    cache = cp.diskcache.Cache(str(cp.ROOT / "disk_cache/piece_captions"))
    llm = cp.LLM_Manager_Builder(client, cache).from_config(
        {**cp.LLM_CONFIG, "model_name": args.model, "response_schema": cp.SCHEMAS["isolated"]})
    base = cp.PROMPTS["isolated"].read_text()

    n_calls = 0
    for v, p, s, e, frames in todo:
        if args.limit_calls is not None and n_calls >= args.limit_calls:
            print(f"stopping: --limit-calls {args.limit_calls}")
            break
        spent = cp.spent_usd()
        if spent >= args.max_usd:
            print(f"stopping: spent ${spent:.4f} >= --max-usd {args.max_usd}")
            break
        if args.input == "video":
            prompt = prompt_for(base, e - s)
            t0 = time.time()
            request = cp.content(cp.cut(tc.video_path(v), s, e - s), prompt)
        else:
            prompt = frames_prompt(len(frames), e - s)
            t0 = time.time()
            request = frames_content(frames, prompt)
        for wait in [*cp.RETRY_WAITS_S, None]:
            res = llm.call(request)
            if not (isinstance(res, cp.LLM_ResponseError) and cp.is_transient(str(res.exception))) or wait is None:
                break
            print(f"server busy, retrying in {wait} s")
            time.sleep(wait)
        n_calls += 1

        raw = res.raw_response
        usage = raw.usage_metadata.model_dump(exclude_none=True) if raw and raw.usage_metadata else {}
        n_in = usage.get("prompt_token_count", 0)
        n_out = usage.get("candidates_token_count", 0) + usage.get("thoughts_token_count", 0)
        usd = (n_in * args.usd_in + n_out * args.usd_out) / 1e6
        rec = dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"), mode=mode, video_id=v, piece=p,
                   model=args.model, model_version=raw.model_version if raw else None,
                   cached=raw is None and not isinstance(res, cp.LLM_ResponseError), usage=usage,
                   usd=round(usd, 6), secs=round(time.time() - t0, 1), error=None)
        if isinstance(res, cp.LLM_ResponseError):
            rec["error"] = f"{type(res).__name__}: {res.exception}"
            cp.log_call(rec)
            if isinstance(res, cp.LLM_ResponseBlocked):
                print(f"blocked: {v} window {p}, continuing")
                continue
            sys.exit(f"API error, stopping: {rec['error']}")

        caption = (res.text or "").strip()
        ok = bool(caption) and not cp.is_error(caption)
        if not ok:
            rec["error"] = "invalid output"
        cp.log_call(rec)
        print(f"{v} window={p} ok={ok} in={n_in} out={n_out} ${usd:.4f} {rec['secs']}s")
        out = caption_path(captioner, v, p)
        out.parent.mkdir(parents=True, exist_ok=True)
        dest = out if ok else out.with_suffix(".invalid.json")
        dest.write_text(json.dumps(dict(
            video_id=v, window=p, start_s=s, end_s=e, caption=caption, mode=mode, captioner=args.model,
            prompt=str(cp.PROMPTS["isolated"].relative_to(cp.ROOT)),
            frames=[str(f.relative_to(tc.ROOT)) for f in frames] if args.input == "frames" else None,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()[:16], model_version=rec["model_version"],
            audio=False, usage=usage, thoughts=res.thoughts), indent=1))
        time.sleep(args.sleep)
    print(f"done: {n_calls} calls, total logged spend ${cp.spent_usd():.4f}")


if __name__ == "__main__":
    main()
