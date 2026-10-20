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
from index_qa.caption import CAPTIONS, MODE, captions, visual_tokens  # noqa: E402
from unshuffle.claude_reader import EFFORT, MODELS, call_claude  # noqa: E402

ARMS = ["blind", "captions", "frame1", "frames5", "selfcaptions", "framecaptions", "frameclipcaptions",
        "sonnetcaptions"]
K0_ARMS = ARMS[:4]
# Caption arms -> the captioner whose window captions they read (Claude's: self_caption.captioner)
CAPTIONERS = {"captions": "gemini-3.8-flash", "selfcaptions": "claude-haiku-5-5-self",
              "framecaptions": "gemini-3.8-flash-frames", "frameclipcaptions": "gemini-3.8-flash-frameclip",
              "sonnetcaptions": "claude-sonnet-5-5-frames"}
# caption.py --input that writes each Gemini caption arm, and self_caption.py --model for each Claude one
CAPTION_INPUT = {"captions": "video", "framecaptions": "frames", "frameclipcaptions": "frameclip"}
CLAUDE_CAPTIONER = {"selfcaptions": "haiku", "sonnetcaptions": "sonnet"}
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


def build(item: dict) -> tuple[list[dict], str]:
    """The user message's content blocks, and a loggable copy of the prompt (images as file paths)."""
    P, arm, v = parts(), item["arm"], item["video_id"]
    dur = tc.duration_s(v)
    wins = tc.windows(dur)
    tpl = (PROMPTS / "mcq.txt").read_text().strip()
    head, tail = tpl.split("{evidence}")
    fill = dict(duration=f"{dur:.1f}".rstrip("0").rstrip("."), modality=P["modality_captions" if arm in CAPTIONERS else f"modality_{arm}"], stem=item["stem"],
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
                     + (f"self_caption.py --model {CLAUDE_CAPTIONER[arm]} first" if arm in CLAUDE_CAPTIONER
                        else f"caption.py --input {CAPTION_INPUT[arm]} first"))
        body = "\n".join(f"{P['window_label'].format(start=s, end=round(e, 1))} {c}" for (s, e), c in zip(wins, caps))
        text = f"{head}\n\n{body}\n\n{tail}"
        return [{"type": "text", "text": text}], text
    blocks, log = [{"type": "text", "text": head}], [head]
    for (s, e), frames in zip(wins, tc.window_frames(v)):
        shown = [tc.middle(frames)] if arm == "frame1" else frames
        if arm == "frames5":
            label = P["window_label"].format(start=s, end=round(e, 1))
            blocks.append({"type": "text", "text": label})
            log.append(label)
        for f in shown:
            if arm == "frame1":
                label = P["frame_label"].format(t=s + int(f.stem.split("_")[1]))
                blocks.append({"type": "text", "text": label})
                log.append(label)
            blocks.append(_image(f))
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
    return OUT / f"{model}.jsonl"


def latest(model: str) -> dict[str, dict]:
    rows = {}
    if rows_path(model).exists():
        for line in rows_path(model).read_text().splitlines():
            r = json.loads(line)
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


def retention_boot(cap: list[float], frames: list[float], blind: list[float], n_boot: int = 10_000, seed: int = 0,
                   level: float = 0.95) -> tuple[float, float, float]:
    """(acc(captions) - acc(blind)) / (acc(frames) - acc(blind)) over the same questions, with a percentile CI from
    resampling questions."""
    import numpy as np
    x = np.column_stack([cap, frames, blind]).astype(float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
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
    rows = [r for m in ("haiku", "sonnet") for r in latest(m).values() if r["question_item"] in keep]
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
    tok = {m: caption_tokens(CAPTIONERS[a]) for a, m in CLAUDE_CAPTIONER.items()}
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


# Claude list prices, $ per million input / output tokens (thinking bills as output), from the claude-api skill's
# model table (cached 2026-10-06); Haiku's hold for prompts up to 100K tokens. The CLI's costUSD should match them.
PRICES = {"haiku": (0.10, 0.50), "sonnet": (2.00, 10.00)}
# Where each caption arm's one-time captioning cost comes from: Gemini's call log (by caption.py mode), or a Claude
# captioner's caption files at the list prices
INDEX_SOURCE = {"captions": ("gemini", MODE), "framecaptions": ("gemini", f"{MODE}_frames"),
                "frameclipcaptions": ("gemini", f"{MODE}_frameclip"), "selfcaptions": ("claude", "haiku"),
                "sonnetcaptions": ("claude", "sonnet")}
# Break-even pairs (Prop. 2 for one caption arm against one frame arm): (caption arm, frame arm)
QSTAR_PAIRS = [("haiku:captions", "haiku:frames5"), ("sonnet:captions", "sonnet:frames5"),
               ("haiku:captions", "sonnet:frames5"), ("haiku:framecaptions", "sonnet:frames5"),
               ("haiku:frameclipcaptions", "sonnet:frames5"), ("haiku:sonnetcaptions", "sonnet:frames5")]


def list_usd(model: str, usage: dict | None) -> float:
    """A Claude call's list-price cost from its usage (prompt caching is off in every call)."""
    u = usage or {}
    p_in, p_out = PRICES[model]
    return (u.get("input_tokens", 0) * p_in + u.get("output_tokens", 0) * p_out) / 1e6


def index_usd(arm: str) -> dict[tuple[str, int], float]:
    """$ per (video_id, window) to caption a caption arm's window once: the latest successful call per window."""
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
            w = [per_win.get((v, p)) for p in range(len(tc.windows(tc.duration_s(v))))]
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
    args = ap.parse_args()
    {"run": cmd_run, "check": cmd_check, "confirm": cmd_confirm, "c2": cmd_c2, "c3": cmd_c3,
     "thinking": cmd_thinking, "cost": cmd_cost}[args.cmd](args)


if __name__ == "__main__":
    main()
