#!/usr/bin/env python3
"""Claude answers the kill-test TempCompass questions from one representation of the video per call.

Arms (per 5 s window; Claude image tokens by the ceil(w/28) x ceil(h/28) rule, 224x126 -> 40):
    blind     the question and options only
    captions  the isolated Gemini caption of each window (scripts/index_qa/caption.py), about 45 tokens
    frame1    one 224 px frame from the middle of each window, 40 tokens
    frames5   the window's frames 1 s apart (up to 5), 224 px, up to 200 tokens
    selfcaptions  Haiku's own isolated caption of each window's frames5 frames (scripts/index_qa/self_caption.py);
              same prompt text as captions, so only the captioner differs
    framecaptions Gemini's isolated caption of the same frames5 frames (caption.py --input frames): the same input
              as selfcaptions, with Gemini as the captioner (C1's coverage control)
    frameclipcaptions  Gemini's isolated caption of the same frames played as a 1 fps clip (caption.py --input
              frameclip): the visual tokens and prompt of the captions arm, the pixels of selfcaptions (C2)
    sonnetcaptions  Sonnet's isolated caption of the same frames5 frames (self_caption.py --model sonnet): the
              pixels, visual tokens and prompt of selfcaptions, with Sonnet as the captioner (C3)
    describe5 the frames5 input, but the reader first writes one sentence per window, then answers, in one call
              (prompts/index_qa/mcq_describe.txt; C5)
    frames5_448  frames5 with the frames at 448 px (448x252 -> 144 tokens each, up to 720 per window; C6)
    sonnetcaptions448  Sonnet's isolated caption of the frames5_448 frames (self_caption.py --model sonnet
              --long-edge 448): sonnetcaptions with 3.6x the captioner's visual tokens (C9)
    sonnetwhole  Sonnet's single caption of all of a video's frames5 frames in one call (self_caption.py --model
              sonnet --whole), with the length limit times the number of windows: sonnetcaptions without the split
              into windows (C10)
    sonnetaspects  sonnetwhole with one prompt bullet that names the kinds of question to come (self_caption.py
              --model sonnet --whole --aspects; caption.ASPECT_BULLET). The reader's prompt is sonnetwhole's (C11)
    qwencaptions  Qwen2.5-VL-7B's own isolated caption of each window's frames5 frames, with the self-captions'
              prompt: selfcaptions for the open-weights reader (C12; written on Kaggle by qwen_kaggle.py)
Open-weights readers (OPEN_MODELS, C12) do not run here: open_reader.py exports these prompts for qwen_kaggle.py and
imports its outputs to results/index_qa/open/<model>.jsonl, which latest() and the analyses read like Claude's rows.
Calls are isolated exactly as in scripts/unshuffle/claude_reader.py (whose call_claude this reuses): a fresh
`claude -p` per item, no tools, empty MCP config, empty working directory, prompt caching off, pinned model ID.
A format failure gets one retry; after that the item scores as wrong.

    .venv/bin/python scripts/index_qa/reader.py run --model haiku --dry-run
    .venv/bin/python scripts/index_qa/reader.py run --model haiku --limit 8
    .venv/bin/python scripts/index_qa/reader.py run --model sonnet --arms captions frames5 --per-dim 10
    .venv/bin/python scripts/index_qa/reader.py check [--models haiku sonnet] [--per-dim 10]
    .venv/bin/python scripts/index_qa/reader.py run --set confirm --model haiku --arms blind captions ...
    .venv/bin/python scripts/index_qa/reader.py confirm      # C1's pre-registered statistics (research log)
    .venv/bin/python scripts/index_qa/reader.py c2           # C2's statistics (research log)
    .venv/bin/python scripts/index_qa/reader.py c3           # C3's statistics (research log)
    .venv/bin/python scripts/index_qa/reader.py thinking [--set dev]   # A1: thinking tokens (research log)
    .venv/bin/python scripts/index_qa/reader.py cost         # G4: cost per question and arm, index cost, Q*
    .venv/bin/python scripts/index_qa/reader.py types [--set dev]      # A3: by video type, clip-clustered CIs
    .venv/bin/python scripts/index_qa/reader.py run --set pairs --model haiku --arms frames5 ...   # C4's sibling reads
    .venv/bin/python scripts/index_qa/reader.py c4           # C4: minimal pairs
    .venv/bin/python scripts/index_qa/reader.py c5           # C5: describe, then answer
    .venv/bin/python scripts/index_qa/reader.py c6           # C6: 448 px frames
    .venv/bin/python scripts/index_qa/reader.py c8           # C8: Sonnet with the question only
    .venv/bin/python scripts/index_qa/reader.py c9           # C9: Sonnet captions of 448 px frames
    .venv/bin/python scripts/index_qa/reader.py c10          # C10: one Sonnet caption per video
    .venv/bin/python scripts/index_qa/reader.py c12          # C12: the open-weights reader (open_reader.py)
    .venv/bin/python scripts/index_qa/reader.py tables       # the paper's generated tables (docs/paper/acl2027)
Outputs (append-only; the latest row per item counts): results/index_qa/claude/<model>.jsonl and
<model>__events.jsonl, as in the unshuffle reader.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import caption_pieces as cp  # noqa: E402
from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import (ASPECTS, CAPTIONS, MODE, WHOLE, caption_path, caption_windows, captions,  # noqa: E402
                              end_line, visual_tokens)
from unshuffle.claude_reader import EFFORT, MODELS, call_claude  # noqa: E402

ARMS = ["blind", "captions", "frame1", "frames5", "selfcaptions", "framecaptions", "frameclipcaptions",
        "sonnetcaptions", "describe5", "frames5_448", "sonnetcaptions448", "sonnetwhole", "sonnetaspects",
        "qwencaptions"]
K0_ARMS = ARMS[:4]
# Open-weights readers (C12): run on Kaggle by qwen_kaggle.py, never by cmd_run
OPEN_MODELS = {"qwen": "Qwen/Qwen2.5-VL-7B-Instruct"}
OPEN_OUT = tc.OUT / "open"
# Caption arms -> the captioner whose window (or, for a WHOLE captioner, video) captions they read (Claude's:
# self_caption.captioner)
CAPTIONERS = {"captions": "gemini-3.8-flash", "selfcaptions": "claude-haiku-5-5-self",
              "framecaptions": "gemini-3.8-flash-frames", "frameclipcaptions": "gemini-3.8-flash-frameclip",
              "sonnetcaptions": "claude-sonnet-5-5-frames", "sonnetcaptions448": "claude-sonnet-5-5-frames448",
              "sonnetwhole": f"claude-sonnet-5-5-frames{WHOLE}",
              "sonnetaspects": f"claude-sonnet-5-5-frames{ASPECTS}{WHOLE}",
              "qwencaptions": "qwen2.5-vl-7b-instruct-self"}
# caption.py --input that writes each Gemini caption arm, and self_caption.py --model (and --long-edge, --whole,
# --aspects) for each Claude one
CAPTION_INPUT = {"captions": "video", "framecaptions": "frames", "frameclipcaptions": "frameclip"}
CLAUDE_CAPTIONER = {"selfcaptions": "haiku", "sonnetcaptions": "sonnet", "sonnetcaptions448": "sonnet",
                    "sonnetwhole": "sonnet", "sonnetaspects": "sonnet"}
CAPTIONER_LONG_EDGE = {"sonnetcaptions448": 448}
OPEN_CAPTIONER = {"qwencaptions": "qwen"}
# Frame arms that show every frame of a window, with the frame size and the answer template
FRAMES_ALL = {"frames5": (tc.LONG_EDGE, "mcq.txt"), "describe5": (tc.LONG_EDGE, "mcq_describe.txt"),
              "frames5_448": (448, "mcq.txt")}
OUT = tc.OUT / "claude"
PROMPTS = tc.ROOT / "prompts/index_qa"


def parts() -> dict:
    return json.loads((PROMPTS / "parts.json").read_text())


def item_set(per_dim: int | None = None, set_: str = "dev") -> list[dict]:
    """Question-major, so a --limit run covers whole questions with every arm. per_dim keeps the first per_dim
    questions of each aspect in sample order (K1/K2 use 10: a balanced half of the K0 sample)."""
    return [dict(q, item=f"{q['item']}|{arm}", question_item=q["item"], arm=arm)
            for q in tc.first_per_dim(tc.items(set_), per_dim) for arm in ARMS]


def _image(path: Path) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(path.read_bytes()).decode()}}


def build(item: dict, image=_image) -> tuple[list[dict], str]:
    """The user message's content blocks, and a loggable copy of the prompt (images as file paths). `image` makes an
    image block from a frame path (open_reader.py exports paths instead of base64)."""
    P, arm, v = parts(), item["arm"], item["video_id"]
    dur = tc.duration_s(v)
    wins = tc.windows(dur)
    size, template = FRAMES_ALL.get(arm, (tc.LONG_EDGE, "mcq.txt"))
    tpl = (PROMPTS / template).read_text().strip()
    head, tail = tpl.split("{evidence}")
    whole = CAPTIONERS.get(arm, "").endswith(WHOLE)
    modality = (("wholecaption" if whole else "captions") if arm in CAPTIONERS
                else "frames5" if arm in FRAMES_ALL else arm)
    fill = dict(duration=f"{dur:.1f}".rstrip("0").rstrip("."), modality=P[f"modality_{modality}"], stem=item["stem"],
                options="\n".join(f"{L}. {o}" for L, o in item["options"].items()),
                letters=", ".join(item["options"]))
    head, tail = head.format(**fill).rstrip(), tail.format(**fill).strip()
    if arm == "blind":
        text = f"{head}\n\n{tail}"
        return [{"type": "text", "text": text}], text
    if arm in CAPTIONERS:
        caps = captions(v, CAPTIONERS[arm])
        if caps is None:
            sys.exit(f"{CAPTIONERS[arm]} captions missing for {v}: run scripts/index_qa/"
                     + (f"self_caption.py --model {CLAUDE_CAPTIONER[arm]} --long-edge "
                        f"{CAPTIONER_LONG_EDGE.get(arm, tc.LONG_EDGE)}{' --whole' if whole else ''}"
                        f"{' --aspects' if ASPECTS in CAPTIONERS[arm] else ''} first"
                        if arm in CLAUDE_CAPTIONER else "qwen_kaggle.py, then open_reader.py import"
                        if arm in OPEN_CAPTIONER else f"caption.py --input {CAPTION_INPUT[arm]} first"))
        body = "\n".join(f"{P['window_label'].format(start=s, end=round(e, 1))} {c}"
                         for (s, e), c in zip(caption_windows(v, CAPTIONERS[arm]), caps))
        text = f"{head}\n\n{body}\n\n{tail}"
        return [{"type": "text", "text": text}], text
    blocks, log = [{"type": "text", "text": head}], [head]
    for (s, e), frames in zip(wins, tc.window_frames(v, size)):
        shown = [tc.middle(frames)] if arm == "frame1" else frames
        if arm in FRAMES_ALL:
            label = P["window_label"].format(start=s, end=round(e, 1))
            blocks.append({"type": "text", "text": label})
            log.append(label)
        for f in shown:
            if arm == "frame1":
                label = P["frame_label"].format(t=s + int(f.stem.split("_")[1]))
                blocks.append({"type": "text", "text": label})
                log.append(label)
            blocks.append(image(f))
            log.append(f"[image {f.relative_to(tc.ROOT)}]")
    blocks.append({"type": "text", "text": tail})
    log.append(tail)
    return blocks, "\n".join(log)


def parse(text: str, letters: list[str]) -> str | None:
    """The answer of the last JSON object in the text whose "answer" is one of the option letters."""
    dec, best = json.JSONDecoder(), None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        a = obj.get("answer") if isinstance(obj, dict) else None
        if isinstance(a, str) and a.strip().upper().rstrip(".") in letters:
            best = a.strip().upper().rstrip(".")
    return best


def record(item: dict, model: str, effort: str, attempt: int, out: dict) -> dict:
    init = next((e for e in out["events"] if e.get("type") == "system" and e.get("subtype") == "init"), {})
    res = next((e for e in out["events"] if e.get("type") == "result"), {})
    text = res.get("result", "") or ""
    letter = parse(text, list(item["options"])) if text else None
    if out["timed_out"]:
        status = "timeout"
    elif not res or res.get("is_error") or out["returncode"] not in (0, None):
        status = "error"
    else:
        status = "ok" if letter else "invalid"
    rl = next((e.get("rate_limit_info", {}) for e in out["events"] if e.get("type") == "rate_limit_event"), {})
    windows = rl.get("unifiedWindows", {})
    thinking = [b.get("thinking", "") for e in out["events"] if e.get("type") == "assistant"
                for b in e.get("message", {}).get("content", []) if b.get("type") == "thinking"]
    return dict(item=item["item"], question_item=item["question_item"], video_id=item["video_id"], dim=item["dim"],
                arm=item["arm"], n_options=len(item["options"]), answer=item["answer"], letter=letter,
                correct=letter == item["answer"], model=model, model_id=MODELS[model],
                model_reported=init.get("model"), tools=init.get("tools"), mcp_servers=init.get("mcp_servers"),
                effort=effort, attempt=attempt, status=status, raw=text, n_thinking_blocks=len(thinking),
                num_turns=res.get("num_turns"), usage=res.get("usage"), model_usage=res.get("modelUsage"),
                error=(res.get("subtype") if status == "error" else None)
                or (out["stderr"][-1000:] or None if status != "ok" else None),
                util_5h=windows.get("five_hour", {}).get("utilization"),
                util_7d=windows.get("seven_day", {}).get("utilization"),
                seconds=out["seconds"], time=time.strftime("%Y-%m-%dT%H:%M:%S"))


def rows_path(model: str) -> Path:
    return (OPEN_OUT if model in OPEN_MODELS else OUT) / f"{model}.jsonl"


def latest(model: str) -> dict[str, dict]:
    """The newest row per item. A partial row (from a stopped run) is skipped, so its item is run again."""
    rows = {}
    if rows_path(model).exists():
        for line in rows_path(model).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["item"]] = r
    return rows


def cmd_run(args):
    items = [it for it in item_set(args.per_dim, args.set_) if it["arm"] in args.arms]
    if args.skip_missing:
        gone = {it["item"] for it in items
                if it["arm"] in CAPTIONERS and captions(it["video_id"], CAPTIONERS[it["arm"]]) is None}
        for i in sorted(gone):
            print(f"skipped, a window has no caption: {i}")
        items = [it for it in items if it["item"] not in gone]
    prev = latest(args.model)
    retry = set(args.retry or [])
    todo = [it for it in items if it["item"] not in prev or prev[it["item"]]["status"] in retry][: args.limit]
    print(f"{len(items)} items, {sum(it['item'] in prev for it in items)} saved, running {len(todo)} "
          f"with {MODELS[args.model]} (effort {args.effort})")
    if args.dry_run:
        for it in todo[: len(ARMS)]:
            print(f"--- {it['item']} (answer {it['answer']})\n{build(it)[1]}\n")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    end_line(rows_path(args.model))
    end_line(OUT / f"{args.model}__events.jsonl")
    first = last = None
    with open(rows_path(args.model), "a") as f, open(OUT / f"{args.model}__events.jsonl", "a") as fe:
        for n, it in enumerate(todo):
            blocks, log = build(it)
            for attempt in (1, 2):
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
                print(f"  [{n + 1}/{len(todo)}] {it['item']} try {attempt}: {r['status']} {r['letter']}/{r['answer']} "
                      f"model={r['model_reported']} turns={r['num_turns']} in={u.get('input_tokens')} "
                      f"out={u.get('output_tokens')} quota 5h={r['util_5h']} 7d={r['util_7d']} {r['seconds']}s",
                      flush=True)
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


def paired_boot(diff: list[float], n_boot: int = 10_000, seed: int = 0, level: float = 0.95
                ) -> tuple[float, float, float]:
    """Mean of per-question differences and a percentile CI (one question per video, so questions are
    independent)."""
    import numpy as np
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [50 * (1 - level), 50 * (1 + level)])
    return float(d.mean()), float(lo), float(hi)


def contrast_boot(d_a: list[float], d_b: list[float], n_boot: int = 10_000, seed: int = 0, level: float = 0.95
                  ) -> tuple[float, float, float]:
    """mean(d_a) - mean(d_b) for per-question differences in two disjoint question groups, with a percentile CI
    from resampling questions within each group."""
    import numpy as np
    a, b = np.asarray(d_a, float), np.asarray(d_b, float)
    rng = np.random.default_rng(seed)
    bs = (a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1)
          - b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(1))
    lo, hi = np.percentile(bs, [50 * (1 - level), 50 * (1 + level)])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def cluster_boot(diff: list[float], clusters: list, n_boot: int = 10_000, seed: int = 0, level: float = 0.95
                 ) -> tuple[float, float, float]:
    """Mean of per-question differences and a percentile CI from resampling clusters of questions (A3: questions
    whose videos share a filmed clip), each resample's mean taken over its questions."""
    import numpy as np
    d = np.asarray(diff, float)
    keys = sorted(set(clusters), key=str)
    at = {k: i for i, k in enumerate(keys)}
    idx = np.array([at[c] for c in clusters])
    s, n = np.bincount(idx, d, len(keys)), np.bincount(idx, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    bs = s[pick].sum(1) / n[pick].sum(1)
    lo, hi = np.percentile(bs, [50 * (1 - level), 50 * (1 + level)])
    return float(d.mean()), float(lo), float(hi)


def clip_clusters(video_ids: list[str]) -> list[int]:
    """Per video, a cluster label: videos that share a filmed clip, directly or through other videos, share a label."""
    parent: dict = {}

    def root(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for v in video_ids:
        for c in tc.source_clips(v):
            parent[root(("video", v))] = root(("clip", c))
    labels: dict = {}
    return [labels.setdefault(root(("video", v)), len(labels)) for v in video_ids]


def retention_boot(cap: list[float], frames: list[float], blind: list[float], n_boot: int = 10_000, seed: int = 0,
                   level: float = 0.95, clusters: list | None = None) -> tuple[float, float, float]:
    """(acc(captions) - acc(blind)) / (acc(frames) - acc(blind)) over the same questions, with a percentile CI from
    resampling questions, or clusters of questions when `clusters` is given (as in cluster_boot)."""
    import numpy as np
    x = np.column_stack([cap, frames, blind]).astype(float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    else:
        keys = sorted(set(clusters), key=str)
        at = {k: i for i, k in enumerate(keys)}
        idx = np.array([at[c] for c in clusters])
        s = np.column_stack([np.bincount(idx, x[:, j], len(keys)) for j in range(3)])
        n = np.bincount(idx, minlength=len(keys)).astype(float)
        pick = rng.integers(0, len(keys), size=(n_boot, len(keys)))
        m = s[pick].sum(1) / n[pick].sum(1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        bs = (m[:, 0] - m[:, 2]) / (m[:, 1] - m[:, 2])
    lo, hi = np.nanpercentile(bs, [50 * (1 - level), 50 * (1 + level)])
    c, f, b = x.mean(0)
    return float((c - b) / (f - b)), float(lo), float(hi)


# C1 (research log): the aspect groups, fixed in D1 from development data before any confirmatory data existed
LOW = {"direction", "speed"}
HIGH = {"action", "order", "attribute_change"}


def cmd_check(args):
    """Arms are named <model>:<arm> (e.g. sonnet:frames5), so differences can cross readers. Only questions in the
    --per-dim subset count, and each difference uses the questions both arms answered."""
    import pandas as pd
    keep = {q["item"] for q in tc.first_per_dim(tc.items(args.set_), args.per_dim)}
    df = pd.DataFrame([r for m in args.models for r in latest(m).values() if r["question_item"] in keep])
    if df.empty:
        sys.exit("no saved rows")
    df["arm"] = df.model + ":" + df.arm
    df["in_tok"] = [(u or {}).get("input_tokens") for u in df.usage]
    df["out_tok"] = [(u or {}).get("output_tokens") for u in df.usage]
    df["chance"] = 1 / df.n_options
    print(df.groupby("arm").agg(n=("item", "size"), valid=("status", lambda s: (s == "ok").mean()),
                                acc=("correct", "mean"), chance=("chance", "mean"), in_tok=("in_tok", "median"),
                                out_tok=("out_tok", "median"), thinking=("n_thinking_blocks", lambda s: (s > 0).mean()),
                                secs=("seconds", "median")).round(3).to_string())
    print("\naccuracy by dimension:")
    print(df.pivot_table(index="dim", columns="arm", values="correct", aggfunc="mean").round(2).to_string())
    wide = df.pivot_table(index="question_item", columns="arm", values="correct", aggfunc="first").astype(float)
    pairs = [p.split("-") for p in args.pairs] if args.pairs else [
        (f"{m}:{a}", f"{m}:{b}") for m in args.models
        for a, b in [("captions", "blind"), ("frame1", "blind"), ("frames5", "blind"), ("captions", "frame1"),
                     ("captions", "frames5"), ("frames5", "frame1"), ("selfcaptions", "frames5"),
                     ("captions", "selfcaptions")]]
    print("\npaired differences in accuracy, 95% bootstrap CI over questions:")
    for a, b in pairs:
        if a in wide and b in wide:
            d = (wide[a] - wide[b]).dropna()
            est, lo, hi = paired_boot(d.tolist())
            print(f"  {a:>21} - {b:<21} {est:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(d)})")


def confirm_table():
    """The confirmatory questions, their aspects, and correctness per question (rows) and model:arm (columns)."""
    import pandas as pd
    qs = tc.items("confirm")
    keep = {q["item"] for q in qs}
    rows = [r for m in ("haiku", "sonnet", *OPEN_MODELS) for r in latest(m).values() if r["question_item"] in keep]
    if not rows:
        sys.exit("no confirmatory rows")
    df = pd.DataFrame(rows)
    df["arm"] = df.model + ":" + df.arm
    wide = df.pivot_table(index="question_item", columns="arm", values="correct", aggfunc="first").astype(float)
    return qs, {q["item"]: q["dim"] for q in qs}, wide


def cmd_confirm(args):
    """C1's pre-registered statistics on the confirmatory set, exactly as listed in the research log. Primary tests
    (P1-P3) use 98.33% CIs (Bonferroni over three); secondary ones 95%. A failed call counts as wrong."""
    qs, dim, wide = confirm_table()
    n_q = len(qs)
    print(f"{n_q} confirmatory questions; answered per arm:")
    print(wide.notna().sum().to_string())
    print("\naccuracy:")
    print(wide.mean().round(3).to_string())
    print("\naccuracy by aspect:")
    print(wide.groupby(wide.index.map(dim)).mean().round(2).T.to_string())

    def pair(a, b):
        if a not in wide or b not in wide:
            return None
        return (wide[a] - wide[b]).dropna()

    P = 1 - 0.05 / 3

    def show(name, a, b, level, pred):
        d = pair(a, b)
        if d is None:
            print(f"  {name}: not run yet")
            return
        est, lo, hi = paired_boot(d.tolist(), level=level)
        print(f"  {name}: {a} - {b} = {est:+.3f} [{lo:+.3f}, {hi:+.3f}] ({level:.2%} CI, n={len(d)}; "
              f"prediction {pred:+.2f})")

    def show_contrast(name, a, b, level, pred):
        d = pair(a, b)
        if d is None:
            print(f"  {name}: not run yet")
            return
        low = d[d.index.map(lambda i: dim[i] in LOW)]
        high = d[d.index.map(lambda i: dim[i] in HIGH)]
        est, lo, hi = contrast_boot(low.tolist(), high.tolist(), level=level)
        print(f"  {name}: [{a} - {b}] on direction+speed ({low.mean():+.3f}, n={len(low)}) minus on "
              f"action+order+attribute change ({high.mean():+.3f}, n={len(high)}) = {est:+.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"({level:.2%} CI; prediction {pred:+.2f})")

    print("\nprimary (pass if the CI excludes 0 in the predicted direction):")
    show_contrast("P1 aspect, Sonnet", "sonnet:captions", "sonnet:frames5", P, -0.20)
    show("P2 format, Haiku", "haiku:selfcaptions", "haiku:frames5", P, -0.08)
    show("P3 captioner, Haiku", "haiku:framecaptions", "haiku:selfcaptions", P, +0.06)
    print("\nsecondary (95% CIs):")
    show("S1 caption loss, Sonnet", "sonnet:captions", "sonnet:frames5", 0.95, -0.07)
    cols = ["sonnet:captions", "sonnet:frames5", "sonnet:blind"]
    if all(c in wide for c in cols):
        x = wide[cols].dropna()
        est, lo, hi = retention_boot(*(x[c].tolist() for c in cols))
        print(f"  S1 retention, Sonnet: {est:.3f} [{lo:.3f}, {hi:.3f}] (n={len(x)}; prediction 0.85)")
    else:
        print("  S1 retention, Sonnet: not run yet")
    show("S2 coverage, Haiku", "haiku:captions", "haiku:framecaptions", 0.95, +0.02)
    show("S3 total, Haiku (= S2 + P3 + P2)", "haiku:captions", "haiku:frames5", 0.95, 0.00)
    show_contrast("S4 aspect, Haiku", "haiku:captions", "haiku:frames5", 0.95, -0.12)
    show("S5 matched cost, Haiku", "haiku:captions", "haiku:frame1", 0.95, +0.05)
    show("S5 matched cost, Sonnet", "sonnet:captions", "sonnet:frame1", 0.95, +0.02)
    show("S6 cheap reader on captions vs. strong reader on frames", "haiku:captions", "sonnet:frames5", 0.95, -0.13)
    print("\nby aspect, paired differences (descriptive):")
    for a, b in [("sonnet:captions", "sonnet:frames5"), ("haiku:selfcaptions", "haiku:frames5"),
                 ("haiku:framecaptions", "haiku:selfcaptions"), ("haiku:captions", "haiku:framecaptions")]:
        d = pair(a, b)
        if d is not None:
            print(f"  {a} - {b}: " + ", ".join(f"{k} {v:+.2f}" for k, v in d.groupby(d.index.map(dim)).mean().items()))


# C2 (research log): Haiku arms, all on the questions that every one of them answered
C2_ARMS = ["haiku:frames5", "haiku:selfcaptions", "haiku:frameclipcaptions", "haiku:framecaptions", "haiku:captions"]


def cmd_c2(args):
    """C2's statistics, as listed in the research log: the captioner effect at matched visual tokens, the visual
    budget and the source terms, with 95% CIs. A failed call counts as wrong."""
    qs, dim, wide = confirm_table()
    if missing := [a for a in C2_ARMS if a not in wide]:
        sys.exit(f"not run yet: {missing}")
    x = wide[C2_ARMS].dropna()
    print(f"{len(qs)} confirmatory questions, {len(x)} answered in every C2 arm; dropped: "
          f"{sorted(set(wide.index) - set(x.index)) or 'none'}")
    print("\nGemini visual tokens per window, median over the logged calls (all sets):")
    tok = {mode: visual_tokens(mode) for mode in (MODE, f"{MODE}_frames", f"{MODE}_frameclip")}
    for mode, t in tok.items():
        v = sorted(t.values())
        print(f"  {mode}: {v[len(v) // 2] if v else None} (n={len(v)})")
    both = tok[MODE].keys() & tok[f"{MODE}_frameclip"].keys()
    same = sum(tok[MODE][k] == tok[f"{MODE}_frameclip"][k] for k in both)
    print(f"  frame clip vs. video, same window: equal tokens in {same} of {len(both)} windows; total "
          f"{sum(tok[f'{MODE}_frameclip'][k] for k in both)} vs. {sum(tok[MODE][k] for k in both)}")
    print("\naccuracy:")
    print(x.mean().round(3).to_string())
    print("\naccuracy by aspect:")
    print(x.groupby(x.index.map(dim)).mean().round(2).T.to_string())
    print("\neffects (95% CIs):")
    terms = [("C2a captioner at matched visual tokens", "haiku:frameclipcaptions", "haiku:selfcaptions", +0.10),
             ("C2b visual budget", "haiku:framecaptions", "haiku:frameclipcaptions", +0.05),
             ("C2c source", "haiku:captions", "haiku:frameclipcaptions", 0.00),
             ("P3 (C1) on these questions", "haiku:framecaptions", "haiku:selfcaptions", +0.06),
             ("P2 (C1) format on these questions", "haiku:selfcaptions", "haiku:frames5", -0.08),
             ("S3 (C1) total = C2c + C2a + P2", "haiku:captions", "haiku:frames5", 0.00)]
    for name, a, b, pred in terms:
        est, lo, hi = paired_boot((x[a] - x[b]).tolist())
        print(f"  {name}: {a} - {b} = {est:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(x)}; prediction {pred:+.2f})")
    print("\nby aspect, paired differences (descriptive):")
    for name, a, b, _ in terms[:3]:
        d = x[a] - x[b]
        print(f"  {name}: " + ", ".join(f"{k} {v:+.2f}" for k, v in d.groupby(d.index.map(dim)).mean().items()))


# C3 (research log): Claude arms, all on the questions that every one of them answered
C3_ARMS = ["haiku:frames5", "haiku:selfcaptions", "haiku:sonnetcaptions", "sonnet:frames5"]


def caption_tokens(name: str) -> dict[tuple[str, int], int]:
    """Input tokens (prompt and frames) of each window's caption call, from a Claude captioner's caption files."""
    out = {}
    for f in (CAPTIONS / name).glob("*/*.json"):
        d = json.loads(f.read_text())
        out[(d["video_id"], d["window"])] = (d.get("usage") or {}).get("input_tokens")
    return out


def cmd_c3(args):
    """C3's statistics, as listed in the research log: the captioner effect with the pixels, prompt and visual tokens
    of the self-captions, and the index comparisons around it, with 95% CIs. A failed call counts as wrong."""
    qs, dim, wide = confirm_table()
    if missing := [a for a in C3_ARMS if a not in wide]:
        sys.exit(f"not run yet: {missing}")
    x = wide[C3_ARMS].dropna()
    print(f"{len(qs)} confirmatory questions, {len(x)} answered in every C3 arm; dropped: "
          f"{sorted(set(wide.index) - set(x.index)) or 'none'}")
    tok = {"haiku": caption_tokens(CAPTIONERS["selfcaptions"]), "sonnet": caption_tokens(CAPTIONERS["sonnetcaptions"])}
    both = sorted(tok["haiku"].keys() & tok["sonnet"].keys())
    d = sorted(tok["sonnet"][k] - tok["haiku"][k] for k in both)
    print(f"\ncaption-call input tokens, Sonnet - Haiku, same window ({len(both)} windows): median "
          f"{d[len(d) // 2] if d else None}, range [{d[0] if d else None}, {d[-1] if d else None}]")
    print("\naccuracy:")
    print(x.mean().round(3).to_string())
    print("\naccuracy by aspect:")
    print(x.groupby(x.index.map(dim)).mean().round(2).T.to_string())
    print("\neffects (95% CIs):")
    terms = [("C3a captioner at identical visual tokens", "haiku:sonnetcaptions", "haiku:selfcaptions", +0.09),
             ("C3b total within Claude = C3a + P2", "haiku:sonnetcaptions", "haiku:frames5", +0.12),
             ("C3c index (Sonnet captions once, Haiku answers) vs. Sonnet on frames", "haiku:sonnetcaptions",
              "sonnet:frames5", -0.07),
             ("P2 (C1) modality on these questions", "haiku:selfcaptions", "haiku:frames5", -0.08)]
    for name, a, b, pred in terms:
        est, lo, hi = paired_boot((x[a] - x[b]).tolist())
        print(f"  {name}: {a} - {b} = {est:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(x)}; prediction {pred:+.2f})")
    a, b = "haiku:sonnetcaptions", "haiku:frameclipcaptions"
    if b in wide:
        dd = (wide[a] - wide[b]).dropna()
        est, lo, hi = paired_boot(dd.tolist())
        print(f"  C3d Sonnet vs. Gemini as captioner (about matched tokens): {a} - {b} = {est:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] (n={len(dd)}; prediction +0.00)")
    else:
        print("  C3d Sonnet vs. Gemini as captioner: C2's frameclipcaptions not run yet")
    print("\nby aspect, paired differences (descriptive):")
    for name, a, b, _ in terms[:3]:
        dd = x[a] - x[b]
        print(f"  {name}: " + ", ".join(f"{k} {v:+.2f}" for k, v in dd.groupby(dd.index.map(dim)).mean().items()))


def thinking_tokens(r: dict) -> int:
    return ((r.get("usage") or {}).get("output_tokens_details") or {}).get("thinking_tokens") or 0


def spearman(a, b) -> float:
    """Spearman's rho of two aligned pandas Series (average ranks for ties)."""
    return float(a.rank().corr(b.rank()))


# A1 (research log): thinking-token pairs to compare, and the escalation illustration (cheap arm, strong arm)
THINK_PAIRS = [("haiku:frames5", "haiku:captions"), ("haiku:frames5", "haiku:selfcaptions"),
               ("haiku:captions", "haiku:framecaptions"), ("sonnet:frames5", "sonnet:captions"),
               ("haiku:frames5", "sonnet:frames5"), ("haiku:captions", "sonnet:captions")]
ESCALATE = ("haiku:captions", "sonnet:frames5")


def cmd_thinking(args):
    """A1 (research log; exploratory): thinking tokens per arm, whether the same questions draw thinking in different
    arms, and accuracy with and without thinking. At effort medium the model decides per call whether to think, so
    thinking is observed, not set. A failed call counts as wrong."""
    import pandas as pd
    keep = {q["item"] for q in tc.items(args.set_)}
    rows = [r for m in ("haiku", "sonnet") for r in latest(m).values() if r["question_item"] in keep]
    if not rows:
        sys.exit("no saved rows")
    df = pd.DataFrame(rows)
    df["arm"] = df.model + ":" + df.arm
    df["think"] = [thinking_tokens(r) for r in rows]
    df["cost"] = [sum(v.get("costUSD", 0) for v in (r.get("model_usage") or {}).values()) for r in rows]
    print(f"{args.set_} set: {df.question_item.nunique()} questions")
    print("\nper arm: share of calls that think, thinking tokens, accuracy without / with thinking, "
          "mean list-price $ per call (the CLI's costUSD):")
    stats = []
    for arm, d in df.groupby("arm"):
        stats.append(dict(arm=arm, n=len(d), thinks=(d.think > 0).mean(), mean=d.think.mean(), median=d.think.median(),
                          p90=d.think.quantile(0.9), acc_no_think=d.correct[d.think == 0].mean(),
                          acc_think=d.correct[d.think > 0].mean(), usd=d.cost.mean()))
    print(pd.DataFrame(stats).set_index("arm").round({"thinks": 2, "mean": 1, "acc_no_think": 3, "acc_think": 3,
                                                      "usd": 5}).to_string())
    print("\nmean thinking tokens by aspect:")
    print(df.pivot_table(index="arm", columns="dim", values="think", aggfunc="mean").round(0).to_string())
    t = df.pivot_table(index="question_item", columns="arm", values="think", aggfunc="first")
    qdim = df.groupby("question_item").dim.first()
    print("\nsame questions, thinking tokens in two arms: Spearman, all questions and the mean within aspects:")
    for a, b in THINK_PAIRS:
        if a in t and b in t:
            x = t[[a, b]].dropna()
            within = [spearman(g[a], g[b]) for _, g in x.groupby(qdim.reindex(x.index)) if g[a].nunique() > 1
                      and g[b].nunique() > 1]
            mean_within = f"{sum(within) / len(within):+.2f}" if within else "n/a"
            print(f"  {a} vs. {b}: {spearman(x[a], x[b]):+.2f} (n={len(x)}); within aspects {mean_within}")
    cheap, strong = ESCALATE
    if cheap in t and strong in t:
        c = df.pivot_table(index="question_item", columns="arm", values="correct", aggfunc="first").astype(float)
        u = df.pivot_table(index="question_item", columns="arm", values="cost", aggfunc="first")
        q = t[[cheap, strong]].dropna().index
        up = t.loc[q, cheap] > 0
        acc = c.loc[q, strong].where(up, c.loc[q, cheap]).mean()
        usd = (u.loc[q, cheap] + u.loc[q, strong].where(up, 0)).mean()
        print(f"\nescalation (illustration, post hoc): {cheap}, and {strong} when the first thought "
              f"({up.mean():.0%} of {len(q)} questions): accuracy {acc:.3f}, ${usd:.5f} per question; always {strong}: "
              f"{c.loc[q, strong].mean():.3f}, ${u.loc[q, strong].mean():.5f}; always {cheap}: "
              f"{c.loc[q, cheap].mean():.3f}, ${u.loc[q, cheap].mean():.5f}")


# A3 (research log): differences reported by video type; the C3 ones once C3 has run (C3 amendment 1)
TYPE_PAIRS = [("P2 modality", "haiku:selfcaptions", "haiku:frames5"),
              ("P3 captioner (Gemini, image path)", "haiku:framecaptions", "haiku:selfcaptions"),
              ("C2a captioner (Gemini, matched tokens)", "haiku:frameclipcaptions", "haiku:selfcaptions"),
              ("S1 Sonnet, captions - frames", "sonnet:captions", "sonnet:frames5"),
              ("S3 Haiku, captions - frames", "haiku:captions", "haiku:frames5"),
              ("C3a captioner (Sonnet)", "haiku:sonnetcaptions", "haiku:selfcaptions"),
              ("C3b total within Claude", "haiku:sonnetcaptions", "haiku:frames5"),
              ("C3c index vs. Sonnet on frames", "haiku:sonnetcaptions", "sonnet:frames5")]
PRIOR_ARM = "haiku:blind"


def cmd_types(args):
    """A3 (research log; exploratory): accuracy and the main differences by how TempCompass built the video
    (tc.video_type), with CIs from resampling clusters of questions whose videos share a filmed clip; and, among wrong
    answers, the share that equal the question-only answer. A failed call counts as wrong."""
    import pandas as pd
    qs = tc.items(args.set_)
    q = {x["item"]: x for x in qs}
    rows = [r for m in ("haiku", "sonnet") for r in latest(m).values() if r["question_item"] in q]
    if not rows:
        sys.exit("no saved rows")
    df = pd.DataFrame(rows)
    df["arm"] = df.model + ":" + df.arm
    wide = df.pivot_table(index="question_item", columns="arm", values="correct", aggfunc="first").astype(float)
    letter = df.pivot_table(index="question_item", columns="arm", values="letter", aggfunc="first")
    vtype = pd.Series({i: tc.video_type(q[i]["video_id"]) for i in wide.index})
    clus = pd.Series(dict(zip(wide.index, clip_clusters([q[i]["video_id"] for i in wide.index]))))
    print(f"{args.set_} set: {len(wide)} questions in {clus.nunique()} clusters of videos that share a filmed clip "
          f"(largest {clus.value_counts().max()})")
    print("\nquestions by video type and aspect:")
    aspect = pd.Series({i: q[i]["dim"] for i in wide.index}, name="aspect")
    print(pd.crosstab(vtype.rename("video type"), aspect).reindex(tc.VIDEO_TYPES).to_string())
    print("\naccuracy by video type:")
    acc = wide.groupby(vtype).mean().reindex(tc.VIDEO_TYPES).T.round(2)
    acc["all"] = wide.mean().round(3)
    print(acc.to_string())
    print("\ndifferences, all questions: 95% CI over questions, then over clip clusters:")
    pairs = [(n, a, b) for n, a, b in TYPE_PAIRS if a in wide and b in wide]
    for name, a, b in pairs:
        d = (wide[a] - wide[b]).dropna()
        est, lo, hi = paired_boot(d.tolist())
        _, clo, chi = cluster_boot(d.tolist(), clus[d.index].tolist())
        print(f"  {name}: {a} - {b} = {est:+.3f} [{lo:+.3f}, {hi:+.3f}], clustered [{clo:+.3f}, {chi:+.3f}] "
              f"(n={len(d)})")
    print("\nby video type, 95% CI over clip clusters:")
    for name, a, b in pairs:
        d = (wide[a] - wide[b]).dropna()
        out = []
        for t in tc.VIDEO_TYPES:
            dt = d[vtype[d.index] == t]
            if len(dt):
                est, lo, hi = cluster_boot(dt.tolist(), clus[dt.index].tolist())
                out.append(f"{t} {est:+.2f} [{lo:+.2f}, {hi:+.2f}] n={len(dt)}")
        print(f"  {name}: " + "; ".join(out))
    if PRIOR_ARM in letter:
        no = pd.Series({i: len(q[i]["options"]) for i in wide.index})
        print(f"\nwrong answers equal to the question-only answer ({PRIOR_ARM}), share and n, by video type; "
              f"{(1 / (no - 1)).mean():.2f} if wrong answers were uniform over the wrong options:")
        for a in wide.columns:
            if a == PRIOR_ARM:
                continue
            wrong = (wide[a] == 0) & letter[PRIOR_ARM].notna() & letter[a].notna()
            out = []
            for t in tc.VIDEO_TYPES + [None]:
                m = wrong & (vtype == t) if t else wrong
                share = (letter.loc[m, a] == letter.loc[m, PRIOR_ARM]).mean() if m.any() else float("nan")
                out.append(f"{t or 'all'} {share:.2f} ({int(m.sum())})")
            print(f"  {a}: " + "; ".join(out))


def type_effects(wide, terms: list[tuple], vtype, clus, label: str = "type",
                 groups: list[str] = tc.VIDEO_TYPES) -> None:
    """For each (name, a, b, prediction) term, a - b over the questions both arms answered: overall with 95% CIs over
    questions and over clip clusters, then by video type (or the groups of another vtype series) over clip
    clusters."""
    for name, a, b, pred in terms:
        if a not in wide or b not in wide:
            print(f"  {name}: {a} or {b} not run yet")
            continue
        d = (wide[a] - wide[b]).dropna()
        est, lo, hi = paired_boot(d.tolist())
        _, clo, chi = cluster_boot(d.tolist(), clus[d.index].tolist())
        by = []
        for t in groups:
            dt = d[vtype[d.index] == t]
            if len(dt):
                e, l, h = cluster_boot(dt.tolist(), clus[dt.index].tolist())
                by.append(f"{t} {e:+.2f} [{l:+.2f}, {h:+.2f}] n={len(dt)}")
        print(f"  {name}: {a} - {b} = {est:+.3f} [{lo:+.3f}, {hi:+.3f}], clustered [{clo:+.3f}, {chi:+.3f}] "
              f"(n={len(d)}; prediction {pred:+.2f})\n      by {label}: " + "; ".join(by))


def confirm_frame():
    """Correctness per confirmatory question and model:arm, with each question's video type and clip cluster."""
    import pandas as pd
    qs, dim, wide = confirm_table()
    q = {x["item"]: x for x in qs}
    vtype = pd.Series({i: tc.video_type(q[i]["video_id"]) for i in wide.index})
    clus = pd.Series(dict(zip(wide.index, clip_clusters([q[i]["video_id"] for i in wide.index]))))
    return wide, vtype, clus


# C5 and C6 (research log): terms, as (name, a, b, prediction)
C5_TERMS = [("C5a describing first (same input, same call)", "haiku:describe5", "haiku:frames5", +0.03),
            ("C5b describe-then-answer vs. separate question-blind self-captions", "haiku:describe5",
             "haiku:selfcaptions", +0.00),
            ("P2 (C1) modality, for reference", "haiku:selfcaptions", "haiku:frames5", -0.08)]
C6_TERMS = [("C6a frame size (448 vs. 224 px)", "haiku:frames5_448", "haiku:frames5", +0.04),
            ("C6b Sonnet captions vs. 448 px frames", "haiku:sonnetcaptions", "haiku:frames5_448", +0.07),
            ("C6c Gemini video captions vs. 448 px frames", "haiku:captions", "haiku:frames5_448", +0.08),
            ("C6d Gemini frame captions vs. 448 px frames", "haiku:framecaptions", "haiku:frames5_448", +0.13)]


def cmd_c5(args):
    """C5's statistics (research log): describe, then answer, against the frames5 and self-caption arms; 95% CIs over
    questions and over clip clusters, and by video type. A failed call counts as wrong."""
    wide, vtype, clus = confirm_frame()
    cols = [c for c in ("haiku:frames5", "haiku:selfcaptions", "haiku:describe5") if c in wide]
    print("accuracy by video type:")
    acc = wide[cols].groupby(vtype).mean().reindex(tc.VIDEO_TYPES).T.round(2)
    acc["all"] = wide[cols].mean().round(3)
    print(acc.to_string())
    print("\neffects:")
    type_effects(wide, C5_TERMS, vtype, clus)


def cmd_c6(args):
    """C6's statistics (research log): Haiku on 448 px frames against 224 px frames and the caption arms; tokens per
    call; 95% CIs over questions and over clip clusters, and by video type. A failed call counts as wrong."""
    wide, vtype, clus = confirm_frame()
    rows = [r for r in latest("haiku").values() if r["arm"] in ("frames5", "frames5_448")
            and r["question_item"] in wide.index]
    for arm in ("frames5", "frames5_448"):
        t = sorted((r["usage"] or {}).get("input_tokens", 0) for r in rows if r["arm"] == arm)
        print(f"haiku:{arm}: median input tokens {t[len(t) // 2] if t else None} (n={len(t)})")
    cols = [c for c in ("haiku:frames5", "haiku:frames5_448", "haiku:sonnetcaptions", "haiku:captions",
                        "haiku:framecaptions", "sonnet:frames5") if c in wide]
    print("\naccuracy by video type:")
    acc = wide[cols].groupby(vtype).mean().reindex(tc.VIDEO_TYPES).T.round(2)
    acc["all"] = wide[cols].mean().round(3)
    print(acc.to_string())
    print("\neffects:")
    type_effects(wide, C6_TERMS, vtype, clus)


# C8 and C9 (research log): terms, as (name, a, b, prediction)
C8_TERMS = [("C8a Sonnet vs. Haiku, question only", "sonnet:blind", "haiku:blind", +0.02),
            ("C8b Sonnet, gain from frames", "sonnet:frames5", "sonnet:blind", +0.37)]
C8_RETENTION = [("sonnet:captions", +0.98), ("sonnet:sonnetcaptions", +0.76)]
C9_TERMS = [("C9a captioner visual budget (Sonnet, 448 vs. 224 px)", "haiku:sonnetcaptions448",
             "haiku:sonnetcaptions", +0.03),
            ("C9b index vs. direct", "haiku:sonnetcaptions448", "sonnet:frames5", -0.06),
            ("C9c Sonnet: own 448 px captions vs. frames", "sonnet:sonnetcaptions448", "sonnet:frames5", -0.07),
            ("C9d vs. Gemini frame captions", "haiku:sonnetcaptions448", "haiku:framecaptions", -0.06),
            ("C9e Sonnet: own captions, 448 vs. 224 px", "sonnet:sonnetcaptions448", "sonnet:sonnetcaptions", +0.02)]


def cmd_c8(args):
    """C8's statistics (research log): Sonnet with the question only; Sonnet's retention (Prop. 1's share of the frame
    gain kept by captions) with 95% CIs over questions; and, among Sonnet's wrong answers, the share equal to its
    question-only answer, by video type. A failed call counts as wrong."""
    import pandas as pd
    wide, vtype, clus = confirm_frame()
    if "sonnet:blind" not in wide:
        sys.exit("sonnet:blind not run yet")
    cols = [c for c in ("haiku:blind", "sonnet:blind", "sonnet:frames5", "sonnet:captions", "sonnet:sonnetcaptions",
                        "sonnet:sonnetcaptions448") if c in wide]
    print("accuracy by video type:")
    acc = wide[cols].groupby(vtype).mean().reindex(tc.VIDEO_TYPES).T.round(2)
    acc["all"] = wide[cols].mean().round(3)
    print(acc.to_string())
    print("\neffects:")
    type_effects(wide, C8_TERMS, vtype, clus)
    print("\nSonnet's retention, (captions - question only) / (frames - question only), 95% CI over questions:")
    for arm, pred in C8_RETENTION:
        x = wide[[arm, "sonnet:frames5", "sonnet:blind"]].dropna()
        est, lo, hi = retention_boot(x[arm].tolist(), x["sonnet:frames5"].tolist(), x["sonnet:blind"].tolist())
        print(f"  {arm}: {est:.2f} [{lo:.2f}, {hi:.2f}] (n={len(x)}; prediction {pred:.2f})")
    q = {x["item"]: x for x in tc.items("confirm")}
    rows = [r for r in latest("sonnet").values() if r["question_item"] in wide.index]
    letter = pd.DataFrame(rows).pivot_table(index="question_item", columns="arm", values="letter", aggfunc="first")
    no = pd.Series({i: len(q[i]["options"]) for i in wide.index})
    print(f"\nSonnet's wrong answers equal to its question-only answer, share (n wrong), by video type; "
          f"{(1 / (no - 1)).mean():.2f} if wrong answers were uniform over the wrong options:")
    for arm in [c.split(":")[1] for c in cols if c.startswith("sonnet:") and c != "sonnet:blind"]:
        wrong = (wide[f"sonnet:{arm}"] == 0) & letter["blind"].notna() & letter[arm].notna()
        out = []
        for t in tc.VIDEO_TYPES + [None]:
            m = wrong & (vtype == t) if t else wrong
            share = (letter.loc[m[m].index, arm] == letter.loc[m[m].index, "blind"]).mean() if m.any() else float("nan")
            out.append(f"{t or 'all'} {share:.2f} ({int(m.sum())})")
        print(f"  sonnet:{arm}: " + "; ".join(out))


# D3 (research log): Table 5 of the draft (§8.1) per reader, with the captions it reads; and the retention of every
# index each reader reads
HEADROOM = [("sonnet", "sonnetcaptions"), ("haiku", "sonnetcaptions")]
RETENTION_ARMS = {"haiku": ["selfcaptions", "sonnetcaptions", "sonnetcaptions448", "sonnetwhole", "frameclipcaptions",
                            "framecaptions", "captions"],
                  "sonnet": ["sonnetcaptions", "sonnetcaptions448", "sonnetwhole", "captions"]}


def cmd_headroom(args):
    """D3's statistics (research log), descriptive: per video type, a reader's question-only, frames and captions
    accuracy, the perception shortfall (1 - frames), the index loss (frames - captions) and the retention, with 95% CIs
    over clip clusters; then each index's retention per reader, overall. A failed call counts as wrong."""
    wide, vtype, clus = confirm_frame()
    for model, arm in HEADROOM:
        b, f, c = (f"{model}:{a}" for a in ("blind", "frames5", arm))
        x = wide[[b, f, c]].dropna()
        print(f"\n{model} reads {arm} (n={len(x)}); 95% CIs over clip clusters:")
        for t in tc.VIDEO_TYPES + [None]:
            xt = x[vtype[x.index] == t] if t else x
            k = clus[xt.index].tolist()
            short, sl, sh = cluster_boot((1 - xt[f]).tolist(), k)
            loss, ll, lh = cluster_boot((xt[f] - xt[c]).tolist(), k)
            r, rl, rh = retention_boot(xt[c].tolist(), xt[f].tolist(), xt[b].tolist(), clusters=k)
            print(f"  {t or 'all'} (n={len(xt)}): question only {xt[b].mean():.2f}, frames {xt[f].mean():.2f}, "
                  f"captions {xt[c].mean():.2f}; shortfall {short:.2f} [{sl:.2f}, {sh:.2f}], "
                  f"index loss {loss:+.2f} [{ll:+.2f}, {lh:+.2f}], retention {r:.2f} [{rl:.2f}, {rh:.2f}]")
    print("\nretention per index, overall: 95% CI over questions; over clip clusters")
    for model, arms in RETENTION_ARMS.items():
        for arm in arms:
            cols = [f"{model}:{a}" for a in (arm, "frames5", "blind")]
            if not all(col in wide for col in cols):
                print(f"  {model}:{arm}: not run yet")
                continue
            x = wide[cols].dropna()
            v = [x[col].tolist() for col in cols]
            est, lo, hi = retention_boot(*v)
            _, clo, chi = retention_boot(*v, clusters=clus[x.index].tolist())
            print(f"  {model}:{arm}: {est:.2f} [{lo:.2f}, {hi:.2f}]; [{clo:.2f}, {chi:.2f}] (n={len(x)})")


def cmd_c9(args):
    """C9's statistics (research log): Sonnet's captions of 448 px frames, read by Haiku and Sonnet, against Sonnet's
    224 px captions, Sonnet on frames and Gemini's frame captions; the captioners' input tokens per window; 95% CIs
    over questions and over clip clusters, and by video type. A failed call counts as wrong."""
    wide, vtype, clus = confirm_frame()
    t224, t448 = caption_tokens(CAPTIONERS["sonnetcaptions"]), caption_tokens(CAPTIONERS["sonnetcaptions448"])
    both = sorted(t224.keys() & t448.keys())
    d = sorted(t448[k] - t224[k] for k in both)
    if d:
        print(f"caption-call input tokens, 448 - 224 px, same window ({len(d)} windows): median {d[len(d) // 2]}, "
              f"range [{d[0]}, {d[-1]}] (5 landscape frames: 5 x (144 - 40) = 520)")
    cols = [c for c in ("haiku:sonnetcaptions", "haiku:sonnetcaptions448", "haiku:framecaptions", "sonnet:frames5",
                        "sonnet:sonnetcaptions", "sonnet:sonnetcaptions448") if c in wide]
    print("\naccuracy by video type:")
    acc = wide[cols].groupby(vtype).mean().reindex(tc.VIDEO_TYPES).T.round(2)
    acc["all"] = wide[cols].mean().round(3)
    print(acc.to_string())
    print("\neffects:")
    type_effects(wide, C9_TERMS, vtype, clus)


# C10 (research log): one caption per video against one per window; terms as (name, a, b, prediction)
C10_TERMS = [("C10a splitting, Haiku reads (whole video vs. windows)", "haiku:sonnetwhole", "haiku:sonnetcaptions",
              +0.00),
             ("C10b splitting, Sonnet reads", "sonnet:sonnetwhole", "sonnet:sonnetcaptions", +0.01),
             ("C10c index vs. direct", "haiku:sonnetwhole", "sonnet:frames5", -0.09),
             ("C10d Sonnet: own whole-video captions vs. frames", "sonnet:sonnetwhole", "sonnet:frames5", -0.08)]
WINDOW_GROUPS = ["1", "2", "3+"]


def cmd_c10(args):
    """C10's statistics (research log): Sonnet's whole-video captions against its window captions, read by Haiku and
    Sonnet; caption words and captioning $ per video, whole vs. the windows summed; 95% CIs over questions and over
    clip clusters, by video type and by the number of windows. A failed call counts as wrong."""
    import pandas as pd
    wide, vtype, clus = confirm_frame()
    vid = {q["item"]: q["video_id"] for q in tc.items("confirm")}
    nw = {v: len(tc.windows(tc.duration_s(v))) for v in set(vid.values())}
    wgroup = pd.Series({i: WINDOW_GROUPS[min(nw[vid[i]], 3) - 1] for i in wide.index})
    whole, win = CAPTIONERS["sonnetwhole"], CAPTIONERS["sonnetcaptions"]
    words, usd = [], []
    for v in sorted(nw):
        cw, cs = captions(v, whole), captions(v, win)
        if cw is None or cs is None:
            continue
        words.append((nw[v], len(cw[0].split()), sum(len(c.split()) for c in cs)))
    iw, iv = index_usd("sonnetwhole"), index_usd("sonnetcaptions")
    for v in sorted(nw):
        w = [iv.get((v, p)) for p in range(nw[v])]
        if (v, 0) in iw and all(c is not None for c in w):
            usd.append((nw[v], iw[v, 0], sum(w)))
    print("whole-video caption vs. the window captions summed, per video; median ratio (n videos) by windows:")
    for name, rows in (("words", words), ("captioning $", usd)):
        out = []
        for g in WINDOW_GROUPS + ["all"]:
            r = sorted(a / b for n, a, b in rows if b and (g == "all" or WINDOW_GROUPS[min(n, 3) - 1] == g))
            out.append(f"{g} {r[len(r) // 2]:.2f} ({len(r)})" if r else f"{g} -")
        print(f"  {name}: " + "; ".join(out))
    if usd:
        print(f"  captioning $ per video, mean: whole {sum(a for _, a, _ in usd) / len(usd):.5f}, windows "
              f"{sum(b for _, _, b in usd) / len(usd):.5f}")
    cols = [c for c in ("haiku:sonnetcaptions", "haiku:sonnetwhole", "sonnet:frames5", "sonnet:sonnetcaptions",
                        "sonnet:sonnetwhole") if c in wide]
    for label, by, groups in (("video type", vtype, tc.VIDEO_TYPES), ("windows", wgroup, WINDOW_GROUPS)):
        print(f"\naccuracy by {label}:")
        acc = wide[cols].groupby(by).mean().reindex(groups).T.round(2)
        acc["all"] = wide[cols].mean().round(3)
        print(acc.to_string())
        print("n: " + ", ".join(f"{g} {int((by == g).sum())}" for g in groups))
    print("\neffects:")
    type_effects(wide, C10_TERMS, vtype, clus)
    print("\neffects by the number of windows:")
    type_effects(wide, C10_TERMS, wgroup, clus, "windows", WINDOW_GROUPS)


# C11 (research log): whole-video captions written knowing the kinds of question (the aspect bullet) against C10's;
# terms as (name, a, b, prediction)
C11_TERMS = [("C11a knowing the question kinds, Haiku reads", "haiku:sonnetaspects", "haiku:sonnetwhole", +0.03),
             ("C11b knowing the question kinds, Sonnet reads", "sonnet:sonnetaspects", "sonnet:sonnetwhole", +0.03),
             ("C11c index vs. direct", "haiku:sonnetaspects", "sonnet:frames5", -0.05),
             ("C11d Sonnet: own aspect captions vs. frames", "sonnet:sonnetaspects", "sonnet:frames5", -0.05)]
REVERSAL_WORDS = re.compile(r"revers|backward|rewind", re.I)


def cmd_c11(args):
    """C11's statistics (research log): Sonnet's whole-video captions with the aspect bullet against C10's without it,
    read by Haiku and Sonnet; the captioner's input tokens and the caption words, per video; the share of captions
    that say the clip runs backwards, on reversed and other videos; 95% CIs over questions and over clip clusters, by
    video type. A failed call counts as wrong."""
    import pandas as pd
    wide, vtype, clus = confirm_frame()
    asp, whole = CAPTIONERS["sonnetaspects"], CAPTIONERS["sonnetwhole"]
    ta, tw = caption_tokens(asp), caption_tokens(whole)
    d = sorted(ta[k] - tw[k] for k in ta.keys() & tw.keys())
    if d:
        print(f"caption-call input tokens, aspects - whole, same video ({len(d)} videos): median {d[len(d) // 2]}, "
              f"range [{d[0]}, {d[-1]}] (the bullet alone: a constant)")
    vt = {q["video_id"]: tc.video_type(q["video_id"]) for q in tc.items("confirm")}
    rows = []
    for v in sorted(vt):
        ca, cw = captions(v, asp), captions(v, whole)
        if ca is not None and cw is not None:
            rows.append(dict(v=v, rev=vt[v] == "reversed", wa=len(ca[0].split()), ww=len(cw[0].split()),
                             ma=bool(REVERSAL_WORDS.search(ca[0])), mw=bool(REVERSAL_WORDS.search(cw[0]))))
    if rows:
        w = pd.DataFrame(rows)
        print(f"words per caption, median ({len(w)} videos): aspects {w.wa.median():.0f}, whole {w.ww.median():.0f}; "
              f"median ratio {(w.wa / w.ww).median():.2f}")
        for rev, g in w.groupby("rev"):
            print(f"  says the clip runs backwards ({'reversed' if rev else 'other'} videos, n={len(g)}): "
                  f"aspects {g.ma.mean():.2f}, whole {g.mw.mean():.2f}")
    cols = [c for c in ("haiku:sonnetwhole", "haiku:sonnetaspects", "sonnet:frames5", "sonnet:sonnetwhole",
                        "sonnet:sonnetaspects") if c in wide]
    print("\naccuracy by video type:")
    acc = wide[cols].groupby(vtype).mean().reindex(tc.VIDEO_TYPES).T.round(2)
    acc["all"] = wide[cols].mean().round(3)
    print(acc.to_string())
    print("\neffects:")
    type_effects(wide, C11_TERMS, vtype, clus)


# C12 (research log): the open-weights reader, Qwen2.5-VL-7B, on the arms of the main decomposition
C12_ARMS = ["qwen:blind", "qwen:frames5", "qwen:qwencaptions", "qwen:selfcaptions", "qwen:sonnetcaptions",
            "qwen:frameclipcaptions", "qwen:framecaptions", "qwen:captions"]
C12_TERMS = [("C12a modality (P2 for Qwen)", "qwen:qwencaptions", "qwen:frames5", -0.01),
             ("C12b captioner at identical input, Sonnet (C3a for Qwen)", "qwen:sonnetcaptions", "qwen:qwencaptions",
              +0.09),
             ("C12c captioner at identical input, Haiku", "qwen:selfcaptions", "qwen:qwencaptions", +0.05),
             ("C12d captioner at about matched tokens, Gemini (C2a for Qwen)", "qwen:frameclipcaptions",
              "qwen:qwencaptions", +0.08),
             ("C12e captioner with Gemini's image budget (P3 for Qwen)", "qwen:framecaptions", "qwen:qwencaptions",
              +0.14),
             ("C12f index vs. frames (S3 for Qwen)", "qwen:captions", "qwen:frames5", +0.12),
             ("C12g the cap: Qwen vs. Haiku on Sonnet's captions", "qwen:sonnetcaptions", "haiku:sonnetcaptions",
              -0.05),
             ("C12h question-only floor, Qwen vs. Haiku", "qwen:blind", "haiku:blind", 0.00)]
C12_RETENTION = [("qwen:qwencaptions", 0.95), ("qwen:sonnetcaptions", 1.40), ("qwen:captions", 1.60)]


CAPTION_WORD_LIMIT = 45  # caption.frames_prompt: "at most 45 words in total" per 5 s window


def caption_lengths(arms=("qwencaptions", "selfcaptions", "sonnetcaptions", "frameclipcaptions", "framecaptions",
                          "captions")):
    """Words per caption on the confirmatory windows, per captioner, and the share over the prompt's limit. Added to
    C12 after its entry, when Qwen's captions ran over the limit (descriptive)."""
    print(f"caption words per window, confirmatory windows (prompt limit {CAPTION_WORD_LIMIT}):")
    vids = sorted({q["video_id"] for q in tc.items("confirm")})
    for arm in arms:
        model = CAPTIONERS[arm]
        words = sorted(len(json.loads(f.read_text())["caption"].split()) for v in vids
                       for p in range(len(caption_windows(v, model))) if (f := caption_path(model, v, p)).exists())
        if not words:
            print(f"  {arm}: none yet")
            continue
        over = sum(w > CAPTION_WORD_LIMIT for w in words)
        print(f"  {arm} ({model}): n={len(words)}, median {words[len(words) // 2]}, mean "
              f"{sum(words) / len(words):.1f}, max {words[-1]}, over the limit {over / len(words):.2f}")


def cmd_c12(args):
    """C12's statistics (research log): Qwen2.5-VL-7B (greedy, on Kaggle) on the main decomposition's arms; its
    caption and reader tokens; the terms with 95% CIs over questions and over clip clusters, by video type; Qwen's
    retention per index. An unparsed answer counts as wrong."""
    wide, vtype, clus = confirm_frame()
    if missing := [a for a in C12_ARMS if a not in wide]:
        sys.exit(f"not run yet: {missing}")
    rows = [r for r in latest("qwen").values() if r["question_item"] in wide.index]
    print(f"qwen rows: {len(rows)}; status: "
          + ", ".join(f"{s} {sum(r['status'] == s for r in rows)}" for s in sorted({r['status'] for r in rows})))
    for arm in [a.split(":")[1] for a in C12_ARMS]:
        u = [r["usage"] for r in rows if r["arm"] == arm]
        t, vis = sorted(x["input_tokens"] for x in u), sorted(x["visual_tokens"] for x in u)
        print(f"  qwen:{arm}: median input tokens {t[len(t) // 2]}, visual {vis[len(vis) // 2]} (n={len(u)})")
    caps = [json.loads(f.read_text()) for f in (CAPTIONS / CAPTIONERS["qwencaptions"]).glob("*/*.json")]
    if caps:
        vis = sorted(c["usage"]["visual_tokens"] for c in caps)
        print(f"Qwen captions: {len(caps)} windows; visual tokens per window median {vis[len(vis) // 2]}; ERROR or "
              f"empty {sum(cp.is_error(c['caption']) or not c['caption'].strip() for c in caps)}")
    caption_lengths()
    cols = C12_ARMS + [c for c in ("haiku:blind", "haiku:frames5", "haiku:sonnetcaptions", "sonnet:frames5") if c in wide]
    print("\naccuracy by video type:")
    acc = wide[cols].groupby(vtype).mean().reindex(tc.VIDEO_TYPES).T.round(2)
    acc["all"] = wide[cols].mean().round(3)
    print(acc.to_string())
    print("\neffects:")
    type_effects(wide, C12_TERMS, vtype, clus)
    print("\nQwen's retention, (captions - question only) / (frames - question only), 95% CI over questions; "
          "clustered:")
    for arm, pred in C12_RETENTION:
        x = wide[[arm, "qwen:frames5", "qwen:blind"]].dropna()
        cols3 = [x[c].tolist() for c in (arm, "qwen:frames5", "qwen:blind")]
        est, lo, hi = retention_boot(*cols3)
        _, clo, chi = retention_boot(*cols3, clusters=clus[x.index].tolist())
        print(f"  {arm}: {est:.2f} [{lo:.2f}, {hi:.2f}]; [{clo:.2f}, {chi:.2f}] (n={len(x)}; prediction {pred:.2f})")


# The paper's appendix table of accuracy by video type and aspect: arm -> row label, in row order
PAPER_ARMS = {"blind": "question only", "frame1": "1 frame", "frames5": "5 frames", "frames5_448": "5 frames, 448\\,px",
              "describe5": "describe, then answer", "selfcaptions": "self-captions",
              "sonnetcaptions": "Sonnet captions", "sonnetcaptions448": "Sonnet captions, 448\\,px",
              "sonnetwhole": "Sonnet whole-video", "sonnetaspects": "Sonnet aspect-aware",
              "frameclipcaptions": "Gemini frame-clip", "framecaptions": "Gemini frame",
              "captions": "Gemini video", "qwencaptions": "Qwen captions"}
PAPER_TABLES = tc.ROOT / "docs/paper/acl2027/generated"


def cmd_tables(args):
    """Writes the paper's generated tables (docs/paper/acl2027/generated/*.tex) from the confirmatory rows: accuracy by
    video type and by aspect, for every arm of every reader."""
    import pandas as pd
    qs, _, _ = confirm_table()
    wide, vtype, _ = confirm_frame()
    aspect = pd.Series({x["item"]: x["dim"] for x in qs})[wide.index]
    aspects = sorted(aspect.unique())
    lines = []
    for model in ("haiku", "sonnet", *OPEN_MODELS):
        cols = [f"{model}:{a}" for a in PAPER_ARMS if f"{model}:{a}" in wide]
        if not cols:
            continue
        by_t = wide[cols].groupby(vtype).mean().reindex(tc.VIDEO_TYPES).T
        by_a = wide[cols].groupby(aspect).mean().reindex(aspects).T
        lines.append("\\midrule")
        for c in cols:
            vals = [*by_t.loc[c], *by_a.loc[c], wide[c].mean()]
            # selfcaptions are Haiku's; only Claude readers read their own under that arm (an open reader's are its own arm)
            label = "Haiku captions" if model in OPEN_MODELS and c.endswith(":selfcaptions") else PAPER_ARMS[c.split(':')[1]]
            lines.append(f"{model.capitalize()}, {label} & "
                         + " & ".join(f"{v:.2f}" for v in vals) + " \\\\")
    n_t = vtype.value_counts().reindex(tc.VIDEO_TYPES)
    n_a = aspect.value_counts().reindex(aspects)
    head = (" & ".join(["Reader, input", *(f"{t.replace('_', '-')} ({n_t[t]})" for t in tc.VIDEO_TYPES),
                        *(f"{a.replace('attribute_change', 'attr.')} ({n_a[a]})" for a in aspects),
                        f"all ({len(wide)})"]) + " \\\\")
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    out = PAPER_TABLES / "acc_by_type_aspect.tex"
    out.write_text("% Generated by scripts/index_qa/reader.py tables; do not edit.\n"
                   f"\\begin{{tabular}}{{@{{}}l{'r' * (len(tc.VIDEO_TYPES) + len(aspects) + 1)}@{{}}}}\n\\toprule\n"
                   + head + "\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"{out}: {sum(l.endswith(chr(92) * 2) for l in lines)} rows")


# C4 (research log): arms read on both videos of a minimal pair, and the terms on pair accuracy
C4_ARMS = ["haiku:frames5", "haiku:selfcaptions", "haiku:sonnetcaptions", "haiku:captions", "sonnet:frames5",
           "sonnet:sonnetcaptions"]
C4_TERMS = [("C4a modality", "haiku:selfcaptions", "haiku:frames5", +0.06),
            ("C4b captioner (Sonnet)", "haiku:sonnetcaptions", "haiku:selfcaptions", +0.09),
            ("C4c index vs. Sonnet on frames", "haiku:sonnetcaptions", "sonnet:frames5", -0.12),
            ("C4d Gemini video captions vs. frames", "haiku:captions", "haiku:frames5", +0.18),
            ("C4e Sonnet: own captions vs. frames", "sonnet:sonnetcaptions", "sonnet:frames5", -0.06)]
PAIR_GROUPS = {"plain": "reversal", "reversed": "reversal", "stacked": "stacked", "two_clip": "two_clip"}


def pair_frame():
    """Per minimal pair (index: the pair item) and model:arm: correct on both videos, and the same answer letter on
    both; with the pair's group (reversal, stacked, two_clip) and the clip cluster of its confirmatory video."""
    import pandas as pd
    pairs = tc.items("pairs")
    keep = {p["pair_of"] for p in pairs} | {p["item"] for p in pairs}
    rows = [r for m in ("haiku", "sonnet") for r in latest(m).values() if r["question_item"] in keep]
    df = pd.DataFrame(rows)
    df["arm"] = df.model + ":" + df.arm
    cor = df.pivot_table(index="question_item", columns="arm", values="correct", aggfunc="first").astype(float)
    let = df.pivot_table(index="question_item", columns="arm", values="letter", aggfunc="first")
    idx = [p["item"] for p in pairs]
    a, b = [p["pair_of"] for p in pairs], idx
    both = pd.DataFrame(cor.reindex(a).values * cor.reindex(b).values, index=idx, columns=cor.columns)
    same = pd.DataFrame((let.reindex(a).values == let.reindex(b).values).astype(float), index=idx, columns=let.columns)
    missing = cor.reindex(a).isna().values | cor.reindex(b).isna().values
    both = both.mask(missing)
    same = same.reindex(columns=cor.columns).mask(missing)
    orig = pd.DataFrame(cor.reindex(a).values, index=idx, columns=cor.columns)
    sib = pd.DataFrame(cor.reindex(b).values, index=idx, columns=cor.columns)
    group = pd.Series({p["item"]: PAIR_GROUPS[tc.video_type(p["pair_video"])] for p in pairs})
    clus = pd.Series(dict(zip(idx, clip_clusters([p["pair_video"] for p in pairs]))))
    chance = pd.Series({p["item"]: 1 / len(p["options"]) ** 2 for p in pairs})
    return both, same, orig, sib, group, clus, chance


def cmd_c4(args):
    """C4's statistics (research log): per arm, accuracy on each video of a minimal pair, pair accuracy (right on both)
    and the same-answer rate (one letter for two videos whose correct options differ); the pair-accuracy terms with 95%
    CIs over clip clusters, overall and by group. A failed call counts as wrong."""
    import pandas as pd
    both, same, orig, sib, group, clus, chance = pair_frame()
    arms = [a for a in C4_ARMS if a in both and both[a].notna().any()]
    if not arms:
        sys.exit("no arm has read both videos of a pair yet")
    print(f"{len(both)} minimal pairs: " + ", ".join(f"{g} {n}" for g, n in group.value_counts().items())
          + f"; chance pair accuracy {chance.mean():.3f}")
    stats = []
    for a in arms:
        m = both[a].notna()
        stats.append(dict(arm=a, n=int(m.sum()), acc_confirm=orig.loc[m, a].mean(), acc_sibling=sib.loc[m, a].mean(),
                          pair_acc=both.loc[m, a].mean(), same_answer=same.loc[m, a].mean()))
    print(pd.DataFrame(stats).set_index("arm").round(3).to_string())
    for name, tab in (("pair accuracy", both), ("same-answer rate", same)):
        print(f"\n{name} by group:")
        print(tab[arms].groupby(group).mean().T.round(2).to_string())
    print("\npair-accuracy differences, 95% CI over clip clusters (overall, then by group):")
    for name, a, b, pred in C4_TERMS:
        if a not in arms or b not in arms:
            print(f"  {name}: {a} or {b} not run yet")
            continue
        d = (both[a] - both[b]).dropna()
        est, lo, hi = cluster_boot(d.tolist(), clus[d.index].tolist())
        by = []
        for g in ("reversal", "stacked", "two_clip"):
            dg = d[group[d.index] == g]
            e, l, h = cluster_boot(dg.tolist(), clus[dg.index].tolist())
            by.append(f"{g} {e:+.2f} [{l:+.2f}, {h:+.2f}] n={len(dg)}")
        print(f"  {name}: {a} - {b} = {est:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(d)}; prediction {pred:+.2f})\n"
              f"      by group: " + "; ".join(by))
    ref = "sonnet:frames5"
    if ref in arms:
        print(f"\nstacked pairs, same-answer rate minus {ref}'s (H1: > 0 for every caption arm):")
        for a in arms:
            if a != ref:
                d = (same[a] - same[ref])[group == "stacked"].dropna()
                e, l, h = cluster_boot(d.tolist(), clus[d.index].tolist())
                print(f"  {a}: {e:+.2f} [{l:+.2f}, {h:+.2f}] (n={len(d)})")


# Claude list prices, $ per million input / output tokens (thinking bills as output), from the claude-api skill's
# model table (cached 2026-10-06); Haiku's hold for prompts up to 100K tokens. The CLI's costUSD should match them.
PRICES = {"haiku": (0.10, 0.50), "sonnet": (2.00, 10.00)}
# Where each caption arm's one-time captioning cost comes from: Gemini's call log (by caption.py mode), or a Claude
# captioner's caption files at the list prices
INDEX_SOURCE = {"captions": ("gemini", MODE), "framecaptions": ("gemini", f"{MODE}_frames"),
                "frameclipcaptions": ("gemini", f"{MODE}_frameclip"), "selfcaptions": ("claude", "haiku"),
                "sonnetcaptions": ("claude", "sonnet"), "sonnetcaptions448": ("claude", "sonnet"),
                "sonnetwhole": ("claude", "sonnet"), "sonnetaspects": ("claude", "sonnet")}
# Break-even pairs (Prop. 2 for one caption arm against one frame arm): (caption arm, frame arm)
QSTAR_PAIRS = [("haiku:captions", "haiku:frames5"), ("sonnet:captions", "sonnet:frames5"),
               ("haiku:captions", "sonnet:frames5"), ("haiku:framecaptions", "sonnet:frames5"),
               ("haiku:frameclipcaptions", "sonnet:frames5"), ("haiku:sonnetcaptions", "sonnet:frames5"),
               ("haiku:sonnetcaptions", "haiku:frames5_448"), ("sonnet:sonnetcaptions", "sonnet:frames5"),
               ("haiku:sonnetcaptions448", "sonnet:frames5"), ("sonnet:sonnetcaptions448", "sonnet:frames5"),
               ("haiku:sonnetwhole", "sonnet:frames5"), ("sonnet:sonnetwhole", "sonnet:frames5"),
               ("haiku:sonnetaspects", "sonnet:frames5"), ("sonnet:sonnetaspects", "sonnet:frames5")]


def list_usd(model: str, usage: dict | None) -> float:
    """A Claude call's list-price cost from its usage (prompt caching is off in every call)."""
    u = usage or {}
    p_in, p_out = PRICES[model]
    return (u.get("input_tokens", 0) * p_in + u.get("output_tokens", 0) * p_out) / 1e6


def index_usd(arm: str) -> dict[tuple[str, int], float]:
    """$ per (video_id, window) to caption a caption arm's window once: the latest successful call per window (window
    0 only, for a whole-video captioner)."""
    kind, key = INDEX_SOURCE[arm]
    out = {}
    if kind == "gemini":
        for line in cp.CALLS_LOG.open():
            r = json.loads(line)
            if r["mode"] == key and not r.get("error") and not r.get("cached"):
                out[r["video_id"], r["piece"]] = r["usd"]
    else:
        for f in (CAPTIONS / CAPTIONERS[arm]).glob("*/*.json"):
            d = json.loads(f.read_text())
            out[d["video_id"], d["window"]] = list_usd(key, d.get("usage"))
    return out


def qstar_boot(index: list[float], c_cap: list[float], c_frames: list[float], n_boot: int = 10_000, seed: int = 0,
               level: float = 0.95) -> tuple[float, float, float, float]:
    """Prop. 2's break-even for one caption arm against one frame arm: Q* = mean index cost per video / (mean cost
    per question on frames - on captions), with a percentile CI from resampling questions (one per video, so
    videos too). Where the saving is <= 0, Q* = inf (never). Also returns the share of resamples that never pay."""
    import numpy as np
    x = np.column_stack([index, c_cap, c_frames]).astype(float)

    def q(m):
        save = m[..., 2] - m[..., 1]
        return np.where(save > 0, m[..., 0] / np.where(save > 0, save, 1.0), np.inf), save

    rng = np.random.default_rng(seed)
    qs, save = q(x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1))
    lo, hi = np.percentile(qs, [50 * (1 - level), 50 * (1 + level)], method="nearest")  # no inf - inf interpolation
    return float(q(x.mean(0))[0]), float(lo), float(hi), float((save <= 0).mean())


def cmd_cost(args):
    """G4 (plan §5, step 6): per arm on the confirmatory questions, tokens, list-price dollars and latency per
    question; per caption arm, the one-time captioning cost per video; and Q* for the pairs in QSTAR_PAIRS."""
    import pandas as pd
    qs = tc.items("confirm")
    keep = {q["item"] for q in qs}
    rows = [r for m in PRICES for r in latest(m).values() if r["question_item"] in keep]
    df = pd.DataFrame(rows)
    df["arm"] = df.model + ":" + df.arm
    df["in_tok"] = [(u or {}).get("input_tokens", 0) for u in df.usage]
    df["out_tok"] = [(u or {}).get("output_tokens", 0) for u in df.usage]
    df["think"] = [thinking_tokens(r) for r in rows]
    df["usd"] = [list_usd(r["model"], r["usage"]) for r in rows]
    df["cli_usd"] = [sum(v.get("costUSD", 0) for v in (r.get("model_usage") or {}).values()) for r in rows]
    rel = ((df.usd - df.cli_usd).abs() / df.cli_usd).max()
    print(f"{len(qs)} confirmatory questions. List prices $/M tokens (input, output incl. thinking): {PRICES}; "
          f"largest relative difference from the CLI's costUSD over {len(df)} calls: {rel:.1e}")
    vid = {q["item"]: q["video_id"] for q in qs}
    idx = {}
    for arm in INDEX_SOURCE:
        per_win = index_usd(arm)
        per_vid = {}
        for v in set(vid.values()):
            w = [per_win.get((v, p)) for p in range(len(caption_windows(v, CAPTIONERS[arm])))]
            per_vid[v] = sum(w) if all(c is not None for c in w) else None
        idx[arm] = per_vid
    stats = []
    for arm, d in df.groupby("arm"):
        cap = arm.split(":")[1]
        iv = [idx[cap][vid[q]] for q in d.question_item] if cap in idx else []
        iv = [c for c in iv if c is not None]
        stats.append(dict(arm=arm, n=len(d), acc=d.correct.mean(), in_tok=d.in_tok.median(), out_tok=d.out_tok.mean(),
                          think=d.think.mean(), usd_per_q=d.usd.mean(), secs=d.seconds.median(),
                          index_usd_per_video=sum(iv) / len(iv) if iv else None,
                          index_videos=len(iv) if cap in idx else None))
    print("\nper arm: accuracy; median input tokens; mean output tokens (incl. thinking) and thinking tokens; mean "
          "list $ per question; median seconds per call (CLI wall clock); one-time captioning $ per video:")
    print(pd.DataFrame(stats).set_index("arm").round({"acc": 3, "out_tok": 1, "think": 1, "usd_per_q": 6,
                                                      "index_usd_per_video": 5}).to_string())
    cor = df.pivot_table(index="question_item", columns="arm", values="correct", aggfunc="first").astype(float)
    usd = df.pivot_table(index="question_item", columns="arm", values="usd", aggfunc="first")
    print("\nbreak-even Q* (questions per video after which captioning once costs less in total), 95% CI over "
          "questions; with the accuracy difference, captions - frames:")
    for a, b in QSTAR_PAIRS:
        cap = a.split(":")[1]
        if a not in usd or b not in usd:
            print(f"  {a} vs. {b}: not run yet")
            continue
        q = [i for i in usd.index if pd.notna(usd.at[i, a]) and pd.notna(usd.at[i, b])
             and idx[cap].get(vid[i]) is not None]
        est, lo, hi, never = qstar_boot([idx[cap][vid[i]] for i in q], usd.loc[q, a].tolist(), usd.loc[q, b].tolist())
        dacc, dlo, dhi = paired_boot((cor.loc[q, a] - cor.loc[q, b]).tolist())
        print(f"  {a} vs. {b}: Q* = {est:.1f} [{lo:.1f}, {hi:.1f}] (never in {never:.0%} of resamples; n={len(q)}); "
              f"$/question {usd.loc[q, a].mean():.6f} vs. {usd.loc[q, b].mean():.6f}; accuracy {dacc:+.3f} "
              f"[{dlo:+.3f}, {dhi:+.3f}]")
    qstar_by_length(df, vid)


# Window counts for qstar_by_length: TempCompass's mean is added at run time; 12 = 1 min, 120 = 10 min
LENGTHS = [12, 120, float("inf")]


def qstar_by_length(df, vid: dict[str, str]):
    """D5: Q* and the per-question cost ratio projected to longer videos, for the QSTAR_PAIRS with a per-window
    caption index. A reader's cost per question is its measured mean plus its input tokens per window (the slope of
    input tokens on the window count, which is exact by construction) times its input price per extra window; output
    and thinking tokens stay at their mean (an assumption: they do not grow with W on these clips). The index costs
    its mean per window times W, since every window is a separate call. Assumes the reader reads every window."""
    import numpy as np
    wins = {v: len(caption_windows(v, CAPTIONERS["captions"])) for v in set(vid.values())}
    df = df.assign(W=[wins[vid[i]] for i in df.question_item])
    w_bar = df.W.mean()
    per_win = {}
    print(f"\nD5: projected to longer videos (W windows of 5 s; TempCompass mean W = {w_bar:.2f}). Input tokens per "
          "window (slope) and Q*, per-question cost ratio captions / frames:")
    for a, b in QSTAR_PAIRS:
        cap = a.split(":")[1]
        if cap in ("sonnetwhole", "sonnetaspects") or a not in set(df.arm) or b not in set(df.arm):
            continue
        cost = {}
        for arm in (a, b):
            d = df[df.arm == arm]
            slope = np.polyfit(d.W, d.in_tok, 1)[0]
            cost[arm] = (d.usd.mean(), slope * PRICES[arm.split(":")[0]][0] / 1e6, slope)
        if cap not in per_win:
            per_win[cap] = float(np.mean(list(index_usd(cap).values())))
        k = per_win[cap]
        cells = []
        for w in [w_bar] + LENGTHS:
            if w == float("inf"):
                save = cost[b][1] - cost[a][1]
                cells.append(f"W=inf: Q* {k / save:.1f}" if save > 0 else "W=inf: never")
                continue
            ca, cb = (cost[x][0] + cost[x][1] * (w - w_bar) for x in (a, b))
            qs = k * w / (cb - ca) if cb > ca else float("inf")
            cells.append(f"W={w:.1f}: Q* {qs:.1f}, ratio {ca / cb:.2f}")
        print(f"  {a} ({cost[a][2]:.0f} tok/window) vs. {b} ({cost[b][2]:.0f}); index ${k:.5f}/window: "
              + "; ".join(cells))


# E1 (research log): predicted terms as (label, arm, arm subtracted or None for an accuracy, prediction)
E1_TERMS = [("question only, Haiku", "haiku:blind", None, 0.40),
            ("question only, Sonnet", "sonnet:blind", None, 0.40),
            ("Haiku, 5 frames", "haiku:frames5", None, 0.60),
            ("Sonnet, 5 frames", "sonnet:frames5", None, 0.72),
            ("captioner effect (Haiku: Sonnet - own captions)", "haiku:sonnetcaptions", "haiku:selfcaptions", 0.06),
            ("modality, Haiku (own captions - frames)", "haiku:selfcaptions", "haiku:frames5", 0.03),
            ("modality, Sonnet (own captions - frames)", "sonnet:sonnetcaptions", "sonnet:frames5", -0.04),
            ("the cap (Haiku - Sonnet on Sonnet captions)", "haiku:sonnetcaptions", "sonnet:sonnetcaptions", 0.00),
            ("whole-video - per-window Sonnet captions, Haiku", "haiku:sonnetwhole", "haiku:sonnetcaptions", 0.00),
            ("1 frame - 5 frames, Haiku", "haiku:frame1", "haiku:frames5", -0.03)]
E1_QSTAR = [("haiku:sonnetcaptions", "sonnet:frames5"), ("haiku:selfcaptions", "sonnet:frames5"),
            ("haiku:sonnetwhole", "sonnet:frames5"), ("sonnet:sonnetcaptions", "sonnet:frames5"),
            ("haiku:sonnetcaptions", "haiku:frames5"), ("haiku:selfcaptions", "haiku:frames5")]
E1_QSTAR_PRED = 7.5  # D5's projection at W = 36 for the first pair


def call_frame(set_: str):
    """One row per (model:arm, question) on a set: correct, tokens and list-price $ (as in cmd_cost)."""
    import pandas as pd
    keep = {q["item"] for q in tc.items(set_)}
    rows = [r for m in PRICES for r in latest(m).values() if r["question_item"] in keep]
    df = pd.DataFrame(rows)
    df["arm"] = df.model + ":" + df.arm
    df["in_tok"] = [(u or {}).get("input_tokens", 0) for u in df.usage]
    df["out_tok"] = [(u or {}).get("output_tokens", 0) for u in df.usage]
    df["think"] = [thinking_tokens(r) for r in rows]
    df["usd"] = [list_usd(r["model"], r["usage"]) for r in rows]
    return df


def index_per_video(arm: str, videos) -> dict[str, float | None]:
    """One-time captioning $ per video for a caption arm (None if a window has no logged call)."""
    per_win = index_usd(arm)
    out = {}
    for v in videos:
        w = [per_win.get((v, p)) for p in range(len(caption_windows(v, CAPTIONERS[arm])))]
        out[v] = sum(w) if all(c is not None for c in w) else None
    return out


def cmd_e1(args):
    """E1 (research log): EgoSchema development run on ego_dev. Accuracy per arm, the predicted terms with 95% paired
    bootstrap CIs, Q_BE at W = 36 against D5, output and thinking tokens against TempCompass, and spend."""
    import pandas as pd
    qs = tc.items("ego_dev")
    vid = {q["item"]: q["video_id"] for q in qs}
    df = call_frame("ego_dev")
    print(f"{len(qs)} ego_dev questions (excluded: {sorted(k for k in tc.EXCLUDED if k.startswith('egd'))}); "
          f"{len(df)} reads; status counts: {df.status.value_counts().to_dict()}")
    cor = df.pivot_table(index="question_item", columns="arm", values="correct", aggfunc="first").astype(float)
    usd = df.pivot_table(index="question_item", columns="arm", values="usd", aggfunc="first")
    print("\naccuracy per arm (95% bootstrap CI over questions; chance 0.20):")
    for arm in cor:
        e, lo, hi = paired_boot(cor[arm].dropna().tolist())
        print(f"  {arm:24s} {e:.3f} [{lo:.3f}, {hi:.3f}] (n={cor[arm].notna().sum()})")
    print("\npredicted terms (prediction; result with 95% paired bootstrap CI; inside the CI?):")
    for label, a, b, pred in E1_TERMS:
        d = (cor[a] - cor[b]) if b else cor[a]
        e, lo, hi = paired_boot(d.dropna().tolist())
        print(f"  {label}: predicted {pred:+.2f}; {e:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={d.notna().sum()}); "
              f"{'inside' if lo <= pred <= hi else 'OUTSIDE'}")
    idx = {arm: index_per_video(arm, set(vid.values())) for arm in ("selfcaptions", "sonnetcaptions", "sonnetwhole")}
    print("\none-time captioning list $ per video (mean over videos):")
    for arm, d in idx.items():
        c = [x for x in d.values() if x is not None]
        print(f"  {arm}: ${sum(c) / len(c):.4f} ({len(c)} videos)")
    print(f"\nQ_BE at W = 36, 95% CI over questions (D5's projection for the first pair: {E1_QSTAR_PRED}); with the "
          "accuracy difference, captions - frames:")
    for a, b in E1_QSTAR:
        cap = a.split(":")[1]
        q = [i for i in usd.index if pd.notna(usd.at[i, a]) and pd.notna(usd.at[i, b])
             and idx[cap].get(vid[i]) is not None]
        est, lo, hi, never = qstar_boot([idx[cap][vid[i]] for i in q], usd.loc[q, a].tolist(), usd.loc[q, b].tolist())
        dacc, dlo, dhi = paired_boot((cor.loc[q, a] - cor.loc[q, b]).tolist())
        print(f"  {a} vs. {b}: Q_BE = {est:.1f} [{lo:.1f}, {hi:.1f}] (never in {never:.0%} of resamples; n={len(q)}); "
              f"$/question {usd.loc[q, a].mean():.5f} vs. {usd.loc[q, b].mean():.5f}; accuracy {dacc:+.3f} "
              f"[{dlo:+.3f}, {dhi:+.3f}]")
    print("\ntokens per read, EgoSchema (E1) against TempCompass (confirmatory): median input; mean output (incl. "
          "thinking) and thinking; mean list $:")
    print(token_table(df))
    reads = df.usd.sum()
    caps = sum(x for d in idx.values() for x in d.values() if x is not None)
    print(f"\nlist-price spend on ego_dev: reads ${reads:.2f} (incl. egd017's: "
          f"${sum(list_usd(m, r['usage']) for m in PRICES for r in latest(m).values() if r['item'].startswith('egd017')):.2f}"
          f"), captions ${caps:.2f} (the {len(vid)} kept videos)")


def token_table(df) -> str:
    """Per arm, tokens and list $ per read on df against the same arm on TempCompass's confirmatory set."""
    import pandas as pd
    tcf = call_frame("confirm")
    rows = []
    for arm in sorted(df.arm.unique()):
        e, t = df[df.arm == arm], tcf[tcf.arm == arm]
        rows.append(dict(arm=arm, in_ego=e.in_tok.median(), in_tc=t.in_tok.median() if len(t) else None,
                         out_ego=e.out_tok.mean(), out_tc=t.out_tok.mean() if len(t) else None,
                         think_ego=e.think.mean(), think_tc=t.think.mean() if len(t) else None,
                         usd_ego=e.usd.mean(), usd_tc=t.usd.mean() if len(t) else None))
    tok = pd.DataFrame(rows).set_index("arm")
    tok["out_ratio"] = tok.out_ego / tok.out_tc
    return tok.round({"out_ego": 1, "out_tc": 1, "think_ego": 1, "think_tc": 1, "usd_ego": 5, "usd_tc": 5,
                      "out_ratio": 2}).to_string()


# E2 (research log): primary hypotheses at 98.33% (Bonferroni over three), as (label, arm, arm subtracted,
# prediction); a pass needs the CI to exclude 0 on the predicted side. H3 is Q_BE's CI inside E2_QSTAR_BAND.
E2_LEVEL = 1 - 0.05 / 3
E2_PRIMARY = [("H1 captioner effect (Haiku: Sonnet - own captions)", "haiku:sonnetcaptions", "haiku:selfcaptions",
               0.08),
              ("H2 modality, Haiku (own captions - frames)", "haiku:selfcaptions", "haiku:frames5", -0.05)]
E2_QSTAR = ("haiku:sonnetcaptions", "sonnet:frames5", 6.3)
E2_QSTAR_BAND = (5.0, 10.0)
E2_SECONDARY = [("S1 question only, Haiku", "haiku:blind", None, 0.41),
                ("S1 question only, Sonnet", "sonnet:blind", None, 0.68),
                ("S2 video use, Sonnet (5 frames - question only)", "sonnet:frames5", "sonnet:blind", 0.00),
                ("S3 video use, Haiku (5 frames - question only)", "haiku:frames5", "haiku:blind", 0.25),
                ("S4 modality, Sonnet (own captions - frames)", "sonnet:sonnetcaptions", "sonnet:frames5", -0.03),
                ("S5 the cap (Haiku - Sonnet on Sonnet captions)", "haiku:sonnetcaptions", "sonnet:sonnetcaptions",
                 0.00),
                ("S6 whole-video - per-window Sonnet captions, Haiku", "haiku:sonnetwhole", "haiku:sonnetcaptions",
                 0.00),
                ("S7 1 frame - 5 frames, Haiku", "haiku:frame1", "haiku:frames5", -0.04),
                ("S8 Haiku on Sonnet captions - Sonnet on frames", "haiku:sonnetcaptions", "sonnet:frames5", -0.02)]
E2_VIDEO_TERMS = [t for t in E2_PRIMARY + E2_SECONDARY if t[2] and "sonnet:blind" not in t[1:3]]


def term_line(cor, label: str, a: str, b: str | None, pred: float | None, level: float = 0.95) -> str:
    d = ((cor[a] - cor[b]) if b else cor[a]).dropna()
    e, lo, hi = paired_boot(d.tolist(), level=level)
    p = f"predicted {pred:+.2f}; " if pred is not None else ""
    return f"  {label}: {p}{e:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(d)})"


def cmd_e2(args):
    """E2 (research log): EgoSchema confirmatory run. H1-H3 at 98.33%, secondary terms at 95%, Q_BE, tokens against
    TempCompass, spend, and the exploratory video-needed subset (Sonnet's question-only answer wrong)."""
    import pandas as pd
    qs = tc.items(args.set_)
    vid = {q["item"]: q["video_id"] for q in qs}
    df = call_frame(args.set_)
    big = df[df.arm.isin(["haiku:frames5", "sonnet:frames5"])]
    print(f"{len(qs)} {args.set_} questions (excluded: {sorted(k for k in tc.EXCLUDED if k[:3] == qs[0]['item'][:3])}); "
          f"{len(df)} reads; status counts: {df.status.value_counts().to_dict()}; 180-image reads failed: "
          f"{(big.status != 'ok').sum()} of {len(big)}")
    cor = df.pivot_table(index="question_item", columns="arm", values="correct", aggfunc="first").astype(float)
    usd = df.pivot_table(index="question_item", columns="arm", values="usd", aggfunc="first")
    print("\naccuracy per arm (95% bootstrap CI over questions; chance 0.20):")
    for arm in cor:
        print(term_line(cor, f"{arm:24s}", arm, None, None))
    print(f"\nprimary (98.33% paired bootstrap CIs; pass = CI excludes 0 on the predicted side):")
    for label, a, b, pred in E2_PRIMARY:
        d = (cor[a] - cor[b]).dropna()
        _, lo, hi = paired_boot(d.tolist(), level=E2_LEVEL)
        ok = lo > 0 if pred > 0 else hi < 0
        print(term_line(cor, label, a, b, pred, E2_LEVEL) + f": {'PASS' if ok else 'not passed'}")
    idx = {arm: index_per_video(arm, set(vid.values())) for arm in ("selfcaptions", "sonnetcaptions", "sonnetwhole")}

    def qbe(a, b, level=0.95):
        cap = a.split(":")[1]
        q = [i for i in usd.index if pd.notna(usd.at[i, a]) and pd.notna(usd.at[i, b])
             and idx[cap].get(vid[i]) is not None]
        est, lo, hi, never = qstar_boot([idx[cap][vid[i]] for i in q], usd.loc[q, a].tolist(), usd.loc[q, b].tolist(),
                                        level=level)
        dacc, dlo, dhi = paired_boot((cor.loc[q, a] - cor.loc[q, b]).tolist(), level=level)
        return est, lo, hi, (f"Q_BE = {est:.1f} [{lo:.1f}, {hi:.1f}] (never in {never:.0%} of resamples; n={len(q)}); "
                             f"$/question {usd.loc[q, a].mean():.5f} vs. {usd.loc[q, b].mean():.5f}; accuracy "
                             f"{dacc:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")

    a, b, pred = E2_QSTAR
    _, lo, hi, line = qbe(a, b, E2_LEVEL)
    ok = E2_QSTAR_BAND[0] <= lo and hi <= E2_QSTAR_BAND[1]
    print(f"  H3 Q_BE, {a} vs. {b}: predicted {pred}, 98.33% CI inside {list(E2_QSTAR_BAND)}; {line}: "
          f"{'PASS' if ok else 'not passed'}")
    print("\nsecondary (95% CIs):")
    for t in E2_SECONDARY:
        print(term_line(cor, *t))
    print("\none-time captioning list $ per video (mean over videos):")
    for arm, d in idx.items():
        c = [x for x in d.values() if x is not None]
        print(f"  {arm}: ${sum(c) / len(c):.4f} ({len(c)} videos)")
    print("\nQ_BE, 95% CI over questions; with the accuracy difference, captions - frames:")
    for a, b in E1_QSTAR:
        print(f"  {a} vs. {b}: {qbe(a, b)[3]}")
    print("\ntokens per read, EgoSchema against TempCompass (confirmatory): median input; mean output (incl. "
          "thinking) and thinking; mean list $:")
    print(token_table(df))
    need = cor.index[cor["sonnet:blind"] == 0]
    print(f"\nexploratory, video-needed subset (Sonnet's question-only answer wrong; n={len(need)}), 95% CIs:")
    for label, a, b, pred in E2_VIDEO_TERMS:
        print(term_line(cor.loc[need], label, a, b, None))
    caps = sum(x for d in idx.values() for x in d.values() if x is not None)
    print(f"\nlist-price spend on {args.set_}: reads ${df.usd.sum():.2f}, captions ${caps:.2f} ({len(vid)} kept videos)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=list(MODELS), required=True)
    r.add_argument("--set", dest="set_", choices=list(tc.SETS), default="dev")
    r.add_argument("--arms", nargs="+", choices=ARMS, default=K0_ARMS)
    r.add_argument("--per-dim", type=int, help="only the first N questions of each aspect (K1/K2: 10)")
    r.add_argument("--effort", default=EFFORT)
    r.add_argument("--limit", type=int)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--max-5h", type=float, default=0.8)
    r.add_argument("--retry", nargs="+", choices=["invalid", "error", "timeout"])
    r.add_argument("--skip-missing", action="store_true",
                   help="skip caption-arm items with a window the captioner gave no caption (C2's rule)")
    c = sub.add_parser("check")
    c.add_argument("--models", nargs="+", choices=list(MODELS), default=["haiku"])
    c.add_argument("--set", dest="set_", choices=list(tc.SETS), default="dev")
    c.add_argument("--per-dim", type=int)
    c.add_argument("--pairs", nargs="+", help="differences to report, as model:arm-model:arm")
    sub.add_parser("confirm")
    sub.add_parser("c2")
    sub.add_parser("c3")
    th = sub.add_parser("thinking")
    th.add_argument("--set", dest="set_", choices=list(tc.SETS), default="confirm")
    sub.add_parser("cost")
    ty = sub.add_parser("types")
    ty.add_argument("--set", dest="set_", choices=list(tc.SETS), default="confirm")
    for name in ("c4", "c5", "c6", "c8", "c9", "c10", "c11", "c12", "headroom", "tables", "e1"):
        sub.add_parser(name)
    e2 = sub.add_parser("e2")
    e2.add_argument("--set", dest="set_", choices=list(tc.SETS), default="ego_confirm")
    args = ap.parse_args()
    {"run": cmd_run, "check": cmd_check, "confirm": cmd_confirm, "c2": cmd_c2, "c3": cmd_c3,
     "thinking": cmd_thinking, "cost": cmd_cost, "types": cmd_types, "c4": cmd_c4, "c5": cmd_c5,
     "c6": cmd_c6, "c8": cmd_c8, "c9": cmd_c9, "c10": cmd_c10, "c11": cmd_c11, "c12": cmd_c12,
     "headroom": cmd_headroom, "tables": cmd_tables, "e1": cmd_e1, "e2": cmd_e2}[args.cmd](args)


if __name__ == "__main__":
    main()
