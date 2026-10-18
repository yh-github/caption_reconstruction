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
Calls are isolated exactly as in scripts/unshuffle/claude_reader.py (whose call_claude this reuses): a fresh
`claude -p` per item, no tools, empty MCP config, empty working directory, prompt caching off, pinned model ID.
A format failure gets one retry; after that the item scores as wrong.

    .venv/bin/python scripts/index_qa/reader.py run --model haiku --dry-run
    .venv/bin/python scripts/index_qa/reader.py run --model haiku --limit 8
    .venv/bin/python scripts/index_qa/reader.py run --model sonnet --arms captions frames5 --per-dim 10
    .venv/bin/python scripts/index_qa/reader.py check [--models haiku sonnet] [--per-dim 10]
    .venv/bin/python scripts/index_qa/reader.py run --set confirm --model haiku --arms blind captions ...
    .venv/bin/python scripts/index_qa/reader.py confirm      # C1's pre-registered statistics (research log)
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

from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import captions  # noqa: E402
from unshuffle.claude_reader import EFFORT, MODELS, call_claude  # noqa: E402

ARMS = ["blind", "captions", "frame1", "frames5", "selfcaptions", "framecaptions"]
K0_ARMS = ARMS[:4]
# Caption arms -> the captioner whose window captions they read (selfcaptions: scripts/index_qa/self_caption.py)
CAPTIONERS = {"captions": "gemini-3.8-flash", "selfcaptions": "claude-haiku-5-5-self",
              "framecaptions": "gemini-3.8-flash-frames"}
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
                     f"{'self_caption' if arm == 'selfcaptions' else 'caption'}.py first"
                     f"{' with --input frames' if arm == 'framecaptions' else ''}")
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


def cmd_confirm(args):
    """C1's pre-registered statistics on the confirmatory set, exactly as listed in the research log. Primary tests
    (P1-P3) use 98.33% CIs (Bonferroni over three); secondary ones 95%. A failed call counts as wrong."""
    import pandas as pd
    qs = tc.items("confirm")
    keep = {q["item"] for q in qs}
    rows = [r for m in ("haiku", "sonnet") for r in latest(m).values() if r["question_item"] in keep]
    if not rows:
        sys.exit("no confirmatory rows")
    df = pd.DataFrame(rows)
    df["arm"] = df.model + ":" + df.arm
    wide = df.pivot_table(index="question_item", columns="arm", values="correct", aggfunc="first").astype(float)
    dim = {q["item"]: q["dim"] for q in qs}
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
    c = sub.add_parser("check")
    c.add_argument("--models", nargs="+", choices=list(MODELS), default=["haiku"])
    c.add_argument("--set", dest="set_", choices=list(tc.SETS), default="dev")
    c.add_argument("--per-dim", type=int)
    c.add_argument("--pairs", nargs="+", help="differences to report, as model:arm-model:arm")
    sub.add_parser("confirm")
    args = ap.parse_args()
    {"run": cmd_run, "check": cmd_check, "confirm": cmd_confirm}[args.cmd](args)


if __name__ == "__main__":
    main()
