#!/usr/bin/env python3
"""A Claude model captions each 5 s window from exactly the frames the frames5 arm shows (K2 and C1: Haiku; C3:
Sonnet; research log).

The prompt is the Gemini window-caption prompt (prompts/yt_video_processing/piece_isolated.txt) with its first
clause, "Describe what happens in the attached 5-second video clip,", replaced by a description of the frames;
everything else is identical (caption.frames_prompt, also used for Gemini's captions of the same frames). One isolated `claude -p` call per window (scripts/unshuffle/claude_reader.py's
call_claude), never seeing the question. An empty or "ERROR" caption gets one retry. Every Claude model counts
image tokens by the same rule, so all captioners spend the same visual tokens on a window.

    .venv/bin/python scripts/index_qa/self_caption.py --per-dim 10 --dry-run
    .venv/bin/python scripts/index_qa/self_caption.py --per-dim 10 [--limit 3]
    .venv/bin/python scripts/index_qa/self_caption.py --set confirm
    .venv/bin/python scripts/index_qa/self_caption.py --set confirm --model sonnet --limit 3   # C3's 3-call check
Writes results/index_qa/captions/<captioner>/<video_id>/<pp>.json (captioner(): Haiku's are read by reader.py's
selfcaptions arm, Sonnet's by sonnetcaptions) and one row per attempt to results/index_qa/self_captions.jsonl.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import caption_pieces as cp  # noqa: E402
from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import caption_path, frames_prompt as prompt_for  # noqa: E402
from unshuffle.claude_reader import EFFORT, MODELS, call_claude  # noqa: E402

LOG = tc.OUT / "self_captions.jsonl"


def captioner(model: str) -> str:
    """The captions folder of a Claude captioner. Haiku keeps K2's name ("-self": it captions for itself as the
    reader); other models caption for another reader, named like Gemini's frame captions ("-frames")."""
    return "claude-haiku-5-5-self" if model == "haiku" else f"{MODELS[model]}-frames"


def blocks_for(prompt: str, frames: list[Path]) -> list[dict]:
    imgs = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(f.read_bytes()).decode()}} for f in frames]
    return [*imgs, {"type": "text", "text": prompt}]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="haiku")
    ap.add_argument("--set", dest="set_", choices=list(tc.SETS), default="dev")
    ap.add_argument("--per-dim", type=int, help="only the videos of the first N questions of each aspect")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--effort", default=EFFORT)
    ap.add_argument("--max-5h", type=float, default=0.7)
    ap.add_argument("--max-7d", type=float, default=0.9)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    model_id, name = MODELS[args.model], captioner(args.model)

    jobs = []
    for q in tc.first_per_dim(tc.items(args.set_), args.per_dim):
        v = q["video_id"]
        for p, ((s, e), frames) in enumerate(zip(tc.windows(tc.duration_s(v)), tc.window_frames(v))):
            if not caption_path(name, v, p).exists():
                jobs.append((v, p, s, e, frames))
    jobs = jobs[: args.limit]
    print(f"{len(jobs)} windows to caption with {model_id} (effort {args.effort}) into {name}")
    if args.dry_run:
        if jobs:
            v, p, s, e, frames = jobs[0]
            print(f"--- {v} window {p}: {len(frames)} frames\n{prompt_for(len(frames), e - s)}")
        return
    with LOG.open("a") as log:
        for n, (v, p, s, e, frames) in enumerate(jobs):
            prompt = prompt_for(len(frames), e - s)
            for attempt in (1, 2):
                out = call_claude(blocks_for(prompt, frames), model_id, args.effort)
                init = next((x for x in out["events"] if x.get("type") == "system" and x.get("subtype") == "init"), {})
                res = next((x for x in out["events"] if x.get("type") == "result"), {})
                caption = (res.get("result") or "").strip()
                ok = bool(caption) and not cp.is_error(caption) and not res.get("is_error")
                rl = next((x.get("rate_limit_info", {}) for x in out["events"] if x.get("type") == "rate_limit_event"), {})
                util_5h = rl.get("unifiedWindows", {}).get("five_hour", {}).get("utilization")
                util_7d = rl.get("unifiedWindows", {}).get("seven_day", {}).get("utilization")
                row = dict(video_id=v, window=p, attempt=attempt, ok=ok, caption=caption, captioner=name,
                           model_reported=init.get("model"), tools=init.get("tools"),
                           mcp_servers=init.get("mcp_servers"), num_turns=res.get("num_turns"), usage=res.get("usage"),
                           model_usage=res.get("modelUsage"), util_5h=util_5h, util_7d=util_7d,
                           seconds=out["seconds"], time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                if ok:
                    break
            if row["model_reported"] and row["model_reported"] != model_id:
                sys.exit(f"the CLI reports model {row['model_reported']}, not {model_id}; stopping")
            if row["tools"] or row["mcp_servers"] or (row["num_turns"] or 1) != 1:
                sys.exit(f"isolation not confirmed (tools={row['tools']}, mcp={row['mcp_servers']}); stopping")
            words = len(caption.split())
            in_tok = (res.get("usage") or {}).get("input_tokens")
            print(f"  [{n + 1}/{len(jobs)}] {v} window {p}: ok={ok} {words} words in={in_tok} 5h={util_5h} "
                  f"7d={util_7d} {out['seconds']}s", flush=True)
            if not ok:
                if n == 0:
                    sys.exit(f"first call failed; stopping ({out['stderr'][-300:]})")
                continue
            dest = caption_path(name, v, p)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(dict(
                video_id=v, window=p, start_s=s, end_s=e, caption=caption, captioner=model_id,
                effort=args.effort, frames=[str(f.relative_to(tc.ROOT)) for f in frames],
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()[:16], usage=res.get("usage"),
                model_usage=res.get("modelUsage")), indent=1))
            if util_5h is not None and util_5h >= args.max_5h:
                sys.exit(f"5-hour quota use reached {util_5h:.0%}; stopping")
            if util_7d is not None and util_7d >= args.max_7d:
                sys.exit(f"7-day quota use reached {util_7d:.0%}; stopping")


if __name__ == "__main__":
    main()
