#!/usr/bin/env python3
"""
Run frontier text-only arms through the Claude Code CLI (`claude -p`) on the user's subscription, one fresh,
isolated session per item, so the model can neither remember earlier items nor read anything but its prompt.

Isolation, per call:
    - a new `claude -p` process (no conversation carries over; --no-session-persistence)
    - --tools "" (no built-in tools) and --strict-mcp-config with an empty MCP config (no connectors)
    - cwd is a fixed, empty dir outside the repo, checked before every call (no repo files, no project CLAUDE.md or
      auto-memory)
    - --effort pinned (default medium) and recorded, since the models think adaptively
    - prompt caching disabled: nothing was ever read back from the cache, so cache writes were pure overhead
    About 5k input tokens per call.
    - --system-prompt replaces Claude Code's agent prompt with a short fixed one (also smaller and cacheable)
    - refuses to run if ANTHROPIC_API_KEY is set (that would bill the API instead of the subscription)
`probe` checks this on one call: the session's tool list must be empty and the run must take a single turn.

Tasks (text only; the model sees captions, never frames):
    recon   Fill a masked gap, same prompt as the Llama runs: prompts/dense_window via JSONPromptBuilder,
            FixedFillMasking(w, i=29). Videos: a seeded sample of N_RECON_VIDEOS, W in RECON_WIDTHS.
    choice  Forced choice (scripts/forced_choice_gap.py items): all K spans masked as in the Llama run, the K
            candidate spans shown in a seeded random order, answer one letter. Items: those Llama scored.
            Llama was scored by log-prob (PMI); this is direct choice, so report the protocol difference.

Usage (from repo root):
    .venv/bin/python scripts/blind_llm_runner.py probe --model haiku
    .venv/bin/python scripts/blind_llm_runner.py run --task recon --model haiku --limit 3 [--dry-run]
    .venv/bin/python scripts/blind_llm_runner.py check --task recon --model haiku
    .venv/bin/python scripts/blind_llm_runner.py run --task choice --model sonnet --limit 40 [--retry error timeout]
Outputs (append-only; a retry adds a row, the latest row per item counts):
    results/blind_llm/<task>__<model>.jsonl          one row per attempt: status (ok / invalid / error / timeout),
                                                     raw answer, parsed answer, thinking text, usage
    results/blind_llm/<task>__<model>__events.jsonl  everything the CLI emitted per attempt: the prompt, every
                                                     stream-json event (thinking blocks included), stderr, return code
Every item with a saved row is skipped, whatever its status; --retry re-runs only the chosen statuses.
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUT = Path("results/blind_llm")
PROMPT_DIR = Path("prompts/dense_window")
CAP_DIRS = [Path("datasets/wildQA/captions__wild4"), Path("datasets/wildQA/captions__wild5")]
RECON_WIDTHS = [1, 4, 8, 16]
N_RECON_VIDEOS = 100
SEED = 2026
TIMEOUT_S = 300
EFFORT = "medium"  # the models think adaptively; pin the level and record it with every row
SYSTEM_PROMPT = ("You complete one self-contained text task. Use only the text in the user's message. "
                 "Follow its output format exactly and output nothing else.")
EMPTY_MCP = '{"mcpServers": {}}'


# ----------------------------------------------------------------------------- items
def load_videos():
    from data.data_loaders import WildLoader
    return {v.video_id: v for d in CAP_DIRS for v in WildLoader(d).load()}


def recon_items(videos) -> list[dict]:
    from llm.prompting import JSONPromptBuilder
    from reconstruction.masking import FixedFillMasking
    builder = JSONPromptBuilder.from_path(PROMPT_DIR)
    vids = sorted(v for v in videos if len(videos[v].clips) >= 60)
    vids = random.Random(SEED).sample(vids, N_RECON_VIDEOS)
    items = []
    for vid in vids:
        video = videos[vid].model_copy(update={"clips": videos[vid].clips[:60]})
        for w in RECON_WIDTHS:
            masked, idx = FixedFillMasking(width=w, start_ind=29).mask_video(video)
            items.append(dict(item=f"{vid}|w{w}", vid=vid, w=w, gap=sorted(idx), prompt=builder.build_prompt(masked)))
    return items


def choice_items(videos, hint: bool = True) -> list[dict]:
    """hint=False drops the sentence telling the model that the other candidates fill the other masked intervals,
    which turns the task into a jigsaw (match each candidate to its slot) that Llama's PMI scoring never used."""
    import pandas as pd
    from forced_choice_gap import K, build_items
    from llm.prompting import JSONPromptBuilder
    from data_models.captions_only import CaptionedVideo
    builder = JSONPromptBuilder.from_path(PROMPT_DIR)
    merged = pd.read_csv("results/forced_choice/forced_choice_merged.csv")
    scored = set(merged.loc[merged["llama-3.1-8b_pmi_ok"].notna(), "item"])
    letters = "ABCDEFGH"[:K]
    items = []
    for it in build_items():
        if it["vid"] not in videos:
            continue
        v, gap = videos[it["vid"]], it["spans"][0]
        mask = {t for sp in it["spans"] for t in sp}
        clips = [c.masked_copy() if c.index in mask else c for c in v.clips[:60]]
        context = builder.build_prompt(CaptionedVideo(video_id=v.video_id, clips=clips))
        context = context.split("\n\n", 1)[1] if "\n\n" in context else context  # keep the caption JSON only
        order = random.Random(f"{SEED}|{it['item']}").sample(range(K), K)  # order[j] = span shown as letter j
        cands = "\n".join(f"{letters[j]}: " + json.dumps([v.clips[t].caption for t in it["spans"][order[j]]],
                                                         ensure_ascii=False) for j in range(K))
        prompt = (f"Below is a sequence of 1-second video captions. Several intervals are missing "
                  f"(caption: null). Which candidate is the true content of the missing interval at indices "
                  f"{gap}? Each candidate lists its captions in time order"
                  + ("; the other candidates belong to the other missing intervals of the same video" if hint else "")
                  + f".\n\nCaptions:\n{context}\n\nCandidates for "
                  f"indices {gap}:\n{cands}\n\n"
                  f'Answer with JSON only: {{"choice": "<one of {", ".join(letters)}>"}}')
        items.append(dict(item=it["item"], vid=it["vid"], w=it["w"], gap=gap, llama_scored=it["item"] in scored,
                          answer=letters[order.index(0)], prompt=prompt))
    return items


TASKS = ["recon", "choice", "choice_nohint"]


def task_items(task: str, videos) -> list[dict]:
    if task == "recon":
        return recon_items(videos)
    return choice_items(videos, hint=task != "choice_nohint")


def parse(task: str, text: str, item: dict):
    """Returns the parsed answer, or None if the output does not match the format."""
    # The last JSON value in the text that is a valid answer wins: models sometimes write a draft, then
    # "Correction, here is the full array" and the final answer.
    dec, best = json.JSONDecoder(), None
    for m in re.finditer(r"[\[{]", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        ans = answer_from(task, obj, item)
        if ans is not None:
            best = ans
    return best


def answer_from(task: str, obj, item: dict):
    try:
        if task == "recon":
            got = {int(e["index"]): str(e["caption"]) for e in obj
                   if isinstance(e, dict) and "index" in e and e.get("caption")}
            return {str(t): got[t] for t in item["gap"]} if all(t in got for t in item["gap"]) else None
        c = str(obj.get("choice", "")).strip().upper()
        return c if len(c) == 1 and c in "ABCDEFGH" else None
    except (TypeError, ValueError, AttributeError):  # any malformed answer is 'invalid', never a crash
        return None


# ----------------------------------------------------------------------------- calls
def empty_cwd() -> Path:
    """A fixed, empty directory outside the repo; refuses to run if anything appears in it."""
    d = Path(tempfile.gettempdir()) / "blind_llm_cwd"
    d.mkdir(exist_ok=True)
    if any(d.iterdir()):
        sys.exit(f"{d} is not empty; refusing to run (the model must see no files)")
    return d


def call_claude(prompt: str, model: str, effort: str) -> dict:
    """One isolated call. Returns everything the CLI emitted: every stream-json event (init, assistant messages
    including thinking blocks, result), stderr, the return code, and whether it timed out."""
    if "ANTHROPIC_API_KEY" in os.environ:
        sys.exit("ANTHROPIC_API_KEY is set: claude would bill the API, not the subscription. Unset it first.")
    cmd = ["claude", "-p", "--model", model, "--effort", effort, "--system-prompt", SYSTEM_PROMPT, "--tools", "",
           "--strict-mcp-config", "--mcp-config", EMPTY_MCP, "--no-session-persistence",
           "--output-format", "stream-json", "--verbose"]
    # Caching off: Claude Code sets its own cache breakpoints, and the shared prefix (system prompt) is below the
    # minimum cacheable length, so each call only wrote a 1-hour cache entry that was never read (pilot 2026-10-08).
    env = {**os.environ, "DISABLE_PROMPT_CACHING": "1"}
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=empty_cwd(), env=env,
                           timeout=TIMEOUT_S)
        stdout, stderr, rc, timed_out = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, timed_out = None, True
    events, junk = [], []
    for line in stdout.splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                junk.append(line)
    return dict(events=events, stdout_unparsed=junk, stderr=stderr, returncode=rc, timed_out=timed_out,
                seconds=round(time.time() - t0, 1))


def record(item: dict, task: str, model: str, effort: str, out: dict) -> dict:
    """Compact per-item row. The full call (all events, thinking included) goes to the events file."""
    res = next((e for e in out["events"] if e.get("type") == "result"), {})
    text = res.get("result", "") or ""
    parsed = parse(task, text, item) if text else None
    if out["timed_out"]:
        status = "timeout"
    elif not res or res.get("is_error") or out["returncode"] not in (0, None):
        status = "error"
    else:
        status = "ok" if parsed is not None else "invalid"
    thinking = [b.get("thinking", "") for e in out["events"] if e.get("type") == "assistant"
                for b in e.get("message", {}).get("content", []) if b.get("type") == "thinking"]
    return dict(item=item["item"], vid=item["vid"], w=item["w"], gap=item["gap"], task=task, model=model, effort=effort,
                answer_key=item.get("answer"), status=status, raw=text, parsed=parsed, thinking=thinking,
                error=(res.get("subtype") if status == "error" else None) or (out["stderr"][-1000:] or None
                                                                              if status != "ok" else None),
                num_turns=res.get("num_turns"), usage=res.get("usage"), model_usage=res.get("modelUsage"),
                seconds=out["seconds"], time=time.strftime("%Y-%m-%dT%H:%M:%S"))


def out_path(task: str, model: str) -> Path:
    return OUT / f"{task}__{model}.jsonl"


def events_path(task: str, model: str) -> Path:
    return OUT / f"{task}__{model}__events.jsonl"


def latest(path: Path) -> dict[str, dict]:
    """item -> its most recent row (a retry appends a new row; earlier attempts stay in the file)."""
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            r.setdefault("status", "ok" if r.get("parsed") is not None else "invalid")  # pilot rows predate status
            rows[r["item"]] = r
    return rows


def cmd_run(args):
    videos = load_videos()
    items = task_items(args.task, videos)
    path = out_path(args.task, args.model)
    prev = latest(path)
    retry = set(args.retry or [])
    todo = [it for it in items if it["item"] not in prev or prev[it["item"]]["status"] in retry]
    if args.pool == "llama":  # forced-choice items that Llama-3.1-8B also scored (180 items, 14 channels)
        todo = [it for it in todo if it.get("llama_scored", True)]
    if args.shuffle:  # seeded, so a --limit sample spreads over videos and channels and is reproducible
        random.Random(f"{SEED}|shuffle").shuffle(todo)
    todo = todo[: args.limit]
    print(f"{args.task}: {len(items)} items, {len(prev)} with a saved result "
          f"({sum(r['status'] == 'ok' for r in prev.values())} ok), running {len(todo)} with {args.model}"
          + (f" (retrying {sorted(retry)})" if retry else ""))
    if args.dry_run:
        for it in todo[:2]:
            print(f"--- {it['item']} (answer {it.get('answer')})\n{it['prompt']}\n")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f, open(events_path(args.task, args.model), "a") as fe:
        for n, it in enumerate(todo):
            out = call_claude(it["prompt"], args.model, args.effort)
            # Save the raw call first, so nothing after this point can lose a paid-for output.
            fe.write(json.dumps(dict(item=it["item"], time=time.strftime("%Y-%m-%dT%H:%M:%S"), model=args.model,
                                     effort=args.effort, prompt=it["prompt"], **out), ensure_ascii=False) + "\n")
            fe.flush()
            r = record(it, args.task, args.model, args.effort, out)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            u = r["usage"] or {}
            print(f"  [{n + 1}/{len(todo)}] {it['item']}: {r['status']} turns={r['num_turns']} "
                  f"in={u.get('input_tokens')} out={u.get('output_tokens')} "
                  f"thinking_blocks={len(r['thinking'])} {r['seconds']}s", flush=True)
            if r["status"] in ("error", "timeout"):
                print(f"    {r['status']}: {r['error'] or r['raw'][:300]}")
                if n == 0:
                    sys.exit("first call failed; stopping before spending more quota")


def cmd_reparse(args):
    """Re-derive rows from the saved raw calls (no new calls), e.g. after a parser fix. Appends a row, marked
    reparsed, for each item whose latest call now parses differently; also recovers calls that never got a row."""
    items = {it["item"]: it for it in task_items(args.task, load_videos())}
    rows = latest(out_path(args.task, args.model))
    calls = {}
    for line in events_path(args.task, args.model).read_text().splitlines():
        e = json.loads(line)
        calls[e["item"]] = e  # latest call per item
    changed = 0
    with open(out_path(args.task, args.model), "a") as f:
        for item_id, e in calls.items():
            out = {k: e[k] for k in ["events", "stdout_unparsed", "stderr", "returncode", "timed_out", "seconds"]}
            r = record(items[item_id], args.task, e["model"], e["effort"], out)
            old = rows.get(item_id)
            if old is None or (old["status"], old["parsed"]) != (r["status"], r["parsed"]):
                r.update(reparsed=True, time=e["time"])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                changed += 1
                print(f"  {item_id}: {old['status'] if old else 'no row'} -> {r['status']}")
    print(f"{changed} rows updated from {len(calls)} saved calls")


def cmd_probe(args):
    """One tiny call, to show the session's tools, MCP servers and token usage."""
    out = call_claude('Reply with JSON only: {"ok": true}', args.model, args.effort)
    if not out["events"]:
        sys.exit(f"probe failed: {out['stderr'][-1000:]}")
    init = next((e for e in out["events"] if e.get("type") == "system" and e.get("subtype") == "init"), {})
    res = next((e for e in out["events"] if e.get("type") == "result"), {})
    print(f"model: {init.get('model')}  cwd: {init.get('cwd')}")
    print(f"tools: {init.get('tools')}")
    print(f"mcp_servers: {init.get('mcp_servers')}")
    print(f"result: {res.get('result')!r}  turns: {res.get('num_turns')}  is_error: {res.get('is_error')}")
    print(f"usage: {res.get('usage')}")
    ok = not init.get("tools") and not init.get("mcp_servers") and res.get("num_turns") == 1
    print("ISOLATION OK" if ok else "ISOLATION NOT CONFIRMED: inspect the fields above")


def cmd_check(args):
    path = out_path(args.task, args.model)
    rows = list(latest(path).values())
    valid = [r for r in rows if r["status"] == "ok"]
    from collections import Counter
    print(f"{path}: {len(rows)} items, status {dict(Counter(r['status'] for r in rows))}, "
          f"turns != 1: {sum(r['num_turns'] != 1 for r in rows)}")
    tot = {k: sum((r["usage"] or {}).get(k) or 0 for r in rows)
           for k in ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"]}
    print(f"tokens (latest attempt per item): {tot}")
    if args.task.startswith("choice") and valid:
        import pandas as pd
        m = pd.read_csv("results/forced_choice/forced_choice_merged.csv").set_index("item")
        s = m.loc[[r["item"] for r in valid]].copy()
        s[args.model] = [float(r["parsed"] == r["answer_key"]) for r in valid]
        cols = [args.model, "llama-3.1-8b_pmi_ok", "text_copy_ok", "text_copy_assign_ok", "vis_copy_ok"]
        print(f"accuracy on the same {len(s)} items ({s.vid.nunique()} videos, {s.chan.nunique()} channels; "
              f"chance 25%; Llama only where scored, n={s['llama-3.1-8b_pmi_ok'].notna().sum()}):")
        print(s[cols].mean().round(3).to_string())
    for r in rows[: args.show]:
        print(f"--- {r['item']} parsed={r['parsed']!r} key={r['answer_key']}\n{r['raw'][:600]}")


def llama_fills() -> dict:
    """(vid, W) -> {slot: text} for Llama-3.1-8B at i=29 (the runs behind the master CSV)."""
    from caption_lag_robustness import LLAMA_RUNS, RECON
    out = {}
    for run in LLAMA_RUNS:
        for sub in (RECON / run).glob("*fixed_fill(w=*, i=29)"):
            w = int(re.search(r"w=(\d+)", sub.name).group(1))
            for jf in sub.glob("*.json"):
                if not (jf.name.startswith("skip__") or jf.name.endswith("metadata.json")):
                    d = json.load(open(jf))
                    out[(d["video_id"].replace("Olly_s", "Olly's"), w)] = {
                        int(k): v for k, v in d["reconstructed_captions"].items()}
    return out


def cmd_score(args):
    """Per (video, W): mean rank of the true second among the video's 60 (1 = best, chance 30.5).
    text: MPNet, fill vs. the 60 true captions (as the Llama runs were scored, pool_scope video).
    frames: SigLIP 2 text of the fill vs. the 60 frames. Rows pair Claude with Llama and caption copy."""
    import numpy as np
    import pandas as pd
    from caption_vs_siglip_audit import channel, unit, VIS_DIR
    from llm.local_embedder import LocalEmbedder, SiglipTextEmbedder
    rows = [r for r in latest(out_path("recon", args.model)).values() if r["status"] == "ok"]
    videos, llama = load_videos(), llama_fills()
    mp, sg = LocalEmbedder("all-mpnet-base-v2"), SiglipTextEmbedder("google/siglip2-base-patch16-224")

    def ranks(E, T, gap):
        sim = E @ T.T
        return [1 + int((sim[j] > sim[j, t]).sum()) for j, t in enumerate(gap)]

    res = []
    for r in rows:
        vid, w, gap = r["vid"], r["w"], r["gap"]
        caps = [c.caption for c in videos[vid].clips[:60]]
        Mt = unit(mp.get_embeddings(f"{vid}_dense_caps", caps))
        V = unit(np.load(VIS_DIR / f"{vid.replace(chr(39), '_')}.npy")[:60])
        lo, hi = gap[0] - 1, gap[-1] + 1
        fills = {"claude": [r["parsed"][str(t)] for t in gap],
                 "caption_copy": [caps[lo if t - lo <= hi - t else hi] for t in gap]}
        if (vid, w) in llama:
            fills["llama"] = [llama[(vid, w)][t] for t in gap]
        for m, texts in fills.items():
            key = f"{vid}_blind_{m}_{args.model if m == 'claude' else ''}_w{w}"
            res.append(dict(vid=vid, chan=channel(vid), w=w, method=m,
                            text_rank=np.mean(ranks(unit(mp.get_embeddings(key, texts)), Mt, gap)),
                            frame_rank=np.mean(ranks(unit(sg.get_embeddings(key, texts)), V, gap))))
    df = pd.DataFrame(res)
    df.to_csv(OUT / f"recon__{args.model}__scores.csv", index=False)
    n = df[df.method == "claude"].shape[0]
    print(f"{n} (video, W) gaps scored for {args.model}; mean rank of the true second (1 = best, chance 30.5)")
    print(df.pivot_table(index="method", columns="w", values=["text_rank", "frame_rank"]).round(1).to_string())
    if args.check_llama:  # this scorer on 60 Llama gaps vs. the pipeline's ranks in the master CSV
        m = pd.read_csv("results/unified_benchmark_master.csv")
        m = m[(m.method == "Llama-3.1-8B") & (m["index"] == 29)].set_index(["video_id", "width"]).mean_rank.to_dict()
        diffs = []
        for (vid, w), rec in list(llama.items())[:: max(1, len(llama) // 60)]:
            if (vid, w) not in m or vid not in videos:
                continue
            gap = sorted(rec)
            Mt = unit(mp.get_embeddings(f"{vid}_dense_caps", [c.caption for c in videos[vid].clips[:60]]))
            E = unit(mp.get_embeddings(f"{vid}_blind_llama__w{w}", [rec[t] for t in gap]))
            diffs.append(abs(np.mean(ranks(E, Mt, gap)) - m[(vid, w)]))
        print(f"Llama rank check vs. master CSV on {len(diffs)} gaps: max |diff| {max(diffs):.2f}, "
              f"mean {np.mean(diffs):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--model", default="haiku")
    p.add_argument("--effort", default=EFFORT)
    r = sub.add_parser("run")
    r.add_argument("--task", choices=TASKS, required=True)
    r.add_argument("--model", required=True, help="claude --model value, e.g. haiku, sonnet, opus")
    r.add_argument("--effort", default=EFFORT, help="claude --effort (thinking budget); pinned for reproducibility")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--dry-run", action="store_true", help="print prompts, make no calls")
    r.add_argument("--pool", choices=["llama", "all"], default="llama",
                   help="choice tasks: only the 180 items Llama scored (default), or all 1,325 (40 channels)")
    r.add_argument("--shuffle", action="store_true",
                   help="seeded shuffle before --limit (items are otherwise in video order: few channels per batch)")
    r.add_argument("--retry", nargs="+", choices=["invalid", "error", "timeout"],
                   help="re-run items whose latest saved status is one of these (default: never re-run saved items)")
    c = sub.add_parser("check")
    c.add_argument("--task", choices=TASKS, required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--show", type=int, default=3)
    rp = sub.add_parser("reparse", help="re-derive rows from the saved raw calls after a parser fix (no calls)")
    rp.add_argument("--task", choices=TASKS, required=True)
    rp.add_argument("--model", required=True)
    s = sub.add_parser("score", help="score recon outputs against Llama and caption copy (no calls)")
    s.add_argument("--model", required=True)
    s.add_argument("--check-llama", action="store_true", help="verify the scorer reproduces the master CSV")
    args = ap.parse_args()
    {"probe": cmd_probe, "run": cmd_run, "check": cmd_check, "reparse": cmd_reparse,
     "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
