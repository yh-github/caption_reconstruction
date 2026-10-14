#!/usr/bin/env python3
"""Listwise Claude reader for the unshuffle puzzles: the same model orders the pieces from their frames or from
their isolated captions. Runs through `claude -p` on the user's subscription (quota, not money).

Isolation per call, as in scripts/blind_llm_runner.py: a fresh `claude -p` process, no tools, an empty MCP
config, an empty working directory outside the repo, no session persistence, prompt caching off, a short fixed
system prompt, the full model ID and the effort level pinned and recorded. The user message goes in through
stream-json, so frames can be sent as JPEG image blocks (5 per piece, local/unshuffle_frames/, frames.py).

Item sets (pilot pre-registration §5):
    smoke        6 items: video 1 P12 and G6-even, video 2 G6-even; shuffle 1; frames and captions
    calibration  the first 10 pilot videos x {P12, G6 even pieces} x 2 shuffles x {frames, captions}: 80 items
A format failure gets one retry; if that fails too the item is invalid and scores as chance.

    .venv/bin/python scripts/unshuffle/claude_reader.py run --items smoke --model haiku [--dry-run]
    .venv/bin/python scripts/unshuffle/claude_reader.py run --items calibration --model sonnet [--limit 20]
    .venv/bin/python scripts/unshuffle/claude_reader.py check [--items calibration]
Outputs (append-only; the latest row per item counts):
    results/unshuffle/claude/<model>.jsonl          one row per attempt: status, raw answer, order, metrics, usage,
                                                    the account's 5-hour and 7-day quota use reported with the call
    results/unshuffle/claude/<model>__events.jsonl  every stream-json event per attempt (images replaced by paths)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blind_llm_runner import EMPTY_MCP, empty_cwd  # noqa: E402
from unshuffle import core, data  # noqa: E402
from unshuffle.frames import frame_paths  # noqa: E402

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
EFFORT = "medium"
TIMEOUT_S = 900
OUT = data.OUT / "claude"
PROMPTS = data.ROOT / "prompts/unshuffle"
LETTERS = "ABCDEFGHIJKL"
MODALITIES = ["frames", "captions"]


# ----------------------------------------------------------------------------- items
def load_puzzles() -> dict[str, dict]:
    path = data.OUT / "puzzles.json"
    if not path.exists():
        sys.exit(f"{path} is missing: run scripts/unshuffle/run_free.py first")
    return {p["puzzle"]: p for p in json.loads(path.read_text())}


def make_item(pz: dict, shuffle: int, modality: str, caption_set: str = "isolated") -> dict:
    presented = pz["shuffles"][shuffle]
    tag = modality if modality == "frames" else f"captions-{caption_set}"
    return dict(item=f"{pz['puzzle']}|s{shuffle}|{tag}", video=pz["video"], puzzle=pz["puzzle"], set=pz["set"],
                k=pz["k"], shuffle=shuffle, modality=modality, caption_set=None if modality == "frames" else caption_set,
                gap_s=pz["gap_s"], presented=presented)


def item_set(name: str) -> list[dict]:
    puzzles = load_puzzles()
    vids = data.pilot_videos()[:10]
    if name == "smoke":
        keys = [(vids[0], "P12|0"), (vids[0], "G6|0"), (vids[1], "G6|0")]
        return [make_item(puzzles[f"{v}|{p}"], 0, m) for v, p in keys for m in MODALITIES]
    if name == "calibration":  # video-major, so a --limit run covers whole videos with both modalities
        return [make_item(puzzles[f"{v}|{p}"], s, m) for v in vids for p in ("P12|0", "G6|0")
                for s in (0, 1) for m in MODALITIES]
    sys.exit(f"unknown item set {name}")


# ----------------------------------------------------------------------------- prompt
def parts() -> dict:
    return json.loads((PROMPTS / "parts.json").read_text())


def build(item: dict) -> tuple[list[dict], str]:
    """The user message's content blocks, and a loggable copy of the prompt (images as file paths)."""
    P, n = parts(), len(item["presented"])
    letters = LETTERS[:n]
    gap = item["gap_s"]
    geometry = (P["geometry_contiguous"].format(span=n * core.PIECE_S) if gap == 0
                else P["geometry_gapped"].format(gap=gap))
    modality = (P["modality_frames"].format(k=core.PIECE_S) if item["modality"] == "frames"
                else P["modality_captions"])
    tpl = (PROMPTS / "listwise.txt").read_text().strip()
    head, tail = tpl.split("{pieces}")
    fill = dict(n=n, piece_s=core.PIECE_S, geometry=geometry, modality=modality, letters=", ".join(letters))
    head, tail = head.format(**fill).rstrip(), tail.format(**fill).strip()
    blocks, log = [], [head]
    if item["modality"] == "captions":
        caps = data.captions(item["video"])[item["caption_set"]]
        body = "\n\n".join(f"{P['piece_label'].format(letter=L)} {caps[p]}" for L, p in zip(letters, item["presented"]))
        text = f"{head}\n\n{body}\n\n{tail}"
        return [{"type": "text", "text": text}], text
    blocks.append({"type": "text", "text": head})
    for L, p in zip(letters, item["presented"]):
        label = P["piece_label"].format(letter=L)
        blocks.append({"type": "text", "text": label})
        log.append(label)
        for f in frame_paths(item["video"], p):
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                       "data": base64.b64encode(f.read_bytes()).decode()}})
            log.append(f"[image {f.relative_to(data.ROOT)}]")
    blocks.append({"type": "text", "text": tail})
    log.append(tail)
    return blocks, "\n".join(log)


def parse(text: str, n: int) -> list[str] | None:
    """The last JSON object in the text whose "order" is a permutation of the n letters."""
    dec, best, want = json.JSONDecoder(), None, set(LETTERS[:n])
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        order = obj.get("order") if isinstance(obj, dict) else None
        if isinstance(order, list) and all(isinstance(x, str) for x in order):
            order = [x.strip().upper() for x in order]
            if len(order) == n and set(order) == want:
                best = order
    return best


# ----------------------------------------------------------------------------- calls
def call_claude(blocks: list[dict], model_id: str, effort: str) -> dict:
    if "ANTHROPIC_API_KEY" in os.environ:
        sys.exit("ANTHROPIC_API_KEY is set: claude would bill the API, not the subscription. Unset it first.")
    cmd = ["claude", "-p", "--model", model_id, "--effort", effort, "--system-prompt", parts()["system"],
           "--tools", "", "--strict-mcp-config", "--mcp-config", EMPTY_MCP, "--no-session-persistence",
           "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]
    msg = json.dumps({"type": "user", "message": {"role": "user", "content": blocks}}) + "\n"
    env = {**os.environ, "DISABLE_PROMPT_CACHING": "1"}
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=msg, capture_output=True, text=True, cwd=empty_cwd(), env=env, timeout=TIMEOUT_S)
        stdout, stderr, rc, timed_out = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, timed_out = None, True
    events, junk = [], []
    for line in stdout.splitlines():
        if line.strip():
            try:
                events.append(_strip_images(json.loads(line)))
            except json.JSONDecodeError:
                junk.append(line)
    return dict(events=events, stdout_unparsed=junk, stderr=stderr, returncode=rc, timed_out=timed_out,
                seconds=round(time.time() - t0, 1))


def _strip_images(obj):
    """Drop base64 image data from echoed events, so the events file stays small."""
    if isinstance(obj, dict):
        if obj.get("type") == "base64" and "data" in obj:
            return {**obj, "data": f"<{len(obj['data'])} chars>"}
        return {k: _strip_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_images(v) for v in obj]
    return obj


def record(item: dict, model: str, effort: str, attempt: int, out: dict) -> dict:
    init = next((e for e in out["events"] if e.get("type") == "system" and e.get("subtype") == "init"), {})
    res = next((e for e in out["events"] if e.get("type") == "result"), {})
    text = res.get("result", "") or ""
    n = len(item["presented"])
    letters = parse(text, n) if text else None
    if out["timed_out"]:
        status = "timeout"
    elif not res or res.get("is_error") or out["returncode"] not in (0, None):
        status = "error"
    else:
        status = "ok" if letters else "invalid"
    order = [item["presented"][LETTERS.index(L)] for L in letters] if letters else None
    metrics = core.score(order) if order else core.chance(n)
    rl = next((e.get("rate_limit_info", {}) for e in out["events"] if e.get("type") == "rate_limit_event"), {})
    windows = rl.get("unifiedWindows", {})
    thinking = [b.get("thinking", "") for e in out["events"] if e.get("type") == "assistant"
                for b in e.get("message", {}).get("content", []) if b.get("type") == "thinking"]
    return dict(item=item["item"], video=item["video"], puzzle=item["puzzle"], set=item["set"], shuffle=item["shuffle"],
                modality=item["modality"], caption_set=item["caption_set"], model=model, model_id=MODELS[model],
                model_reported=init.get("model"), tools=init.get("tools"), mcp_servers=init.get("mcp_servers"),
                effort=effort, attempt=attempt, status=status, raw=text, letters=letters, order=order,
                presented=item["presented"], position_bias=core.position_bias(order, item["presented"]) if order else None,
                **{f"m_{k}": v for k, v in metrics.items()},
                n_thinking_blocks=len(thinking), num_turns=res.get("num_turns"), usage=res.get("usage"),
                model_usage=res.get("modelUsage"), error=(res.get("subtype") if status == "error" else None)
                or (out["stderr"][-1000:] or None if status != "ok" else None),
                util_5h=windows.get("five_hour", {}).get("utilization"),
                util_7d=windows.get("seven_day", {}).get("utilization"),
                seconds=out["seconds"], time=time.strftime("%Y-%m-%dT%H:%M:%S"))


def rows_path(model: str) -> Path:
    return OUT / f"{model}.jsonl"


def latest(model: str) -> dict[str, dict]:
    rows = {}
    if rows_path(model).exists():
        for line in rows_path(model).read_text().splitlines():
            r = json.loads(line)
            rows[r["item"]] = r
    return rows


def cmd_run(args):
    items = item_set(args.items)
    prev = latest(args.model)
    retry = set(args.retry or [])
    todo = [it for it in items if it["item"] not in prev or prev[it["item"]]["status"] in retry][: args.limit]
    print(f"{args.items}: {len(items)} items, {sum(it['item'] in prev for it in items)} saved, running {len(todo)} "
          f"with {MODELS[args.model]} (effort {args.effort})")
    if args.dry_run:
        for it in todo[:2]:
            print(f"--- {it['item']} presented={it['presented']}\n{build(it)[1]}\n")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    first = last = None
    with open(rows_path(args.model), "a") as f, open(OUT / f"{args.model}__events.jsonl", "a") as fe:
        for n, it in enumerate(todo):
            blocks, log = build(it)
            for attempt in (1, 2):  # one retry on a format failure (paper_plan.md, reader protocol)
                out = call_claude(blocks, MODELS[args.model], args.effort)
                fe.write(json.dumps(dict(item=it["item"], attempt=attempt, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         model=args.model, effort=args.effort, prompt=log, **out),
                                    ensure_ascii=False) + "\n")
                fe.flush()
                r = record(it, args.model, args.effort, attempt, out)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                u = r["usage"] or {}
                if r["util_5h"] is not None:
                    first, last = first or (r["util_5h"], r["util_7d"]), (r["util_5h"], r["util_7d"])
                print(f"  [{n + 1}/{len(todo)}] {it['item']} try {attempt}: {r['status']} tau={r['m_tau']:+.2f} "
                      f"adj={r['m_adj_und']:.2f} bias={r['position_bias']} model={r['model_reported']} "
                      f"turns={r['num_turns']} in={u.get('input_tokens')} out={u.get('output_tokens')} "
                      f"quota 5h={r['util_5h']} 7d={r['util_7d']} {r['seconds']}s", flush=True)
                if r["status"] != "invalid":
                    break
            if r["status"] in ("error", "timeout"):
                print(f"    {r['status']}: {r['error'] or r['raw'][:300]}")
                if n == 0:
                    sys.exit("first call failed; stopping before spending more quota")
            if r["model_reported"] and r["model_reported"] != MODELS[args.model]:
                sys.exit(f"the CLI reports model {r['model_reported']}, not {MODELS[args.model]}; stopping")
            if r["tools"] or r["mcp_servers"] or (r["num_turns"] or 1) != 1:
                sys.exit(f"isolation not confirmed (tools={r['tools']}, mcp={r['mcp_servers']}, "
                         f"turns={r['num_turns']}); stopping")
            if r["util_5h"] is not None and r["util_5h"] >= args.max_5h:
                print(f"5-hour quota use reached {r['util_5h']:.0%} (limit --max-5h {args.max_5h:.0%}); stopping")
                break
    if first:
        print(f"quota (account-wide, all sessions): 5h {first[0]:.0%} -> {last[0]:.0%}, 7d {first[1]:.0%} -> {last[1]:.0%}")


def cmd_check(args):
    import pandas as pd
    items = {it["item"] for it in item_set(args.items)}
    rows = []
    for m in MODELS:
        for r in latest(m).values():
            if r["item"] in items:
                rows.append(r)
    if not rows:
        sys.exit("no saved rows for these items")
    df = pd.DataFrame(rows)
    df["valid"] = df.status == "ok"
    df["in_tok"] = [(u or {}).get("input_tokens") for u in df.usage]
    df["out_tok"] = [(u or {}).get("output_tokens") for u in df.usage]
    g = df.groupby(["model", "modality", "set"])
    print(g.agg(n=("item", "size"), valid=("valid", "mean"), tau=("m_tau", "mean"), adj_und=("m_adj_und", "mean"),
                adj_dir=("m_adj_dir", "mean"), local_dir=("m_local_dir", "mean"), exact=("m_exact", "mean"),
                pos_bias=("position_bias", "mean"), in_tok=("in_tok", "median"), out_tok=("out_tok", "median"),
                secs=("seconds", "median")).round(3).to_string())
    print("\nper model and modality, mean over videos with 95% channel-cluster CI (tau, undirected adjacency):")
    for (m, mod), d in df.groupby(["model", "modality"]):
        for k in ["m_tau", "m_adj_und", "m_local_dir"]:
            est, lo, hi = core.cluster_boot(d.groupby("video")[k].mean().to_dict())
            print(f"  {m:<6} {mod:<8} {k[2:]:<9} {est:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({d.video.nunique()} videos)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--items", choices=["smoke", "calibration"], required=True)
    r.add_argument("--model", choices=list(MODELS), required=True)
    r.add_argument("--effort", default=EFFORT)
    r.add_argument("--limit", type=int)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--max-5h", type=float, default=0.8)
    r.add_argument("--retry", nargs="+", choices=["invalid", "error", "timeout"])
    c = sub.add_parser("check")
    c.add_argument("--items", choices=["smoke", "calibration"], default="calibration")
    args = ap.parse_args()
    {"run": cmd_run, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
