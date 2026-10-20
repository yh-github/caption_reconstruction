#!/usr/bin/env python3
"""C7 (research log): caption audit. For each wrong answer of a reader on a caption arm (confirmatory set), a judge
labels the captions the reader saw, given the question and its correct answer: "correct" (the captions state or
clearly imply the correct answer, so the reader erred), "wrong" (they state or imply another option) or "silent"
(they don't decide it). A label belongs to the captions of one captioner for one question, so readers that read the
same captions share it.

The judge is Sonnet (prompts/index_qa/audit.txt), one isolated `claude -p` call per (caption arm, question), as in
reader.py. A format failure gets one retry. A hand-labelled sample checks the judge: `sheet` writes 40 items without
the judge's label for a person to label, and `agree` compares the two.

    .venv/bin/python scripts/index_qa/audit.py run --dry-run
    .venv/bin/python scripts/index_qa/audit.py run [--limit 3]
    .venv/bin/python scripts/index_qa/audit.py report
    .venv/bin/python scripts/index_qa/audit.py sheet      # results/index_qa/audit/hand_labels.csv
    .venv/bin/python scripts/index_qa/audit.py agree      # once the label column is filled
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import captions  # noqa: E402
from index_qa.reader import CAPTIONERS, PROMPTS, latest, parts  # noqa: E402
from unshuffle.claude_reader import EFFORT, MODELS, call_claude  # noqa: E402

OUT = tc.OUT / "audit"
JUDGE = "sonnet"
LABELS = ("correct", "wrong", "silent")
# Reader arms whose wrong answers are audited: model:arm, with the arm naming the captions read
READER_ARMS = ["haiku:selfcaptions", "haiku:sonnetcaptions", "haiku:frameclipcaptions", "haiku:framecaptions",
               "haiku:captions", "sonnet:captions", "sonnet:sonnetcaptions"]
SHEET = OUT / "hand_labels.csv"


def jobs() -> list[dict]:
    """One job per (caption arm, question) that some reader in READER_ARMS answered wrongly from those captions."""
    qs = {q["item"]: q for q in tc.items("confirm")}
    rows = {m: latest(m) for m in ("haiku", "sonnet")}
    out = {}
    for ma in READER_ARMS:
        model, arm = ma.split(":")
        for r in rows[model].values():
            if r["arm"] == arm and r["question_item"] in qs and not r["correct"]:
                out.setdefault((arm, r["question_item"]), dict(arm=arm, **qs[r["question_item"]]))
    return [out[k] for k in sorted(out)]


def prompt(job: dict) -> str | None:
    caps = captions(job["video_id"], CAPTIONERS[job["arm"]])
    if caps is None:
        return None
    wins = tc.windows(tc.duration_s(job["video_id"]))
    label = parts()["window_label"]
    body = "\n".join(f"{label.format(start=s, end=round(e, 1))} {c}" for (s, e), c in zip(wins, caps))
    dur = tc.duration_s(job["video_id"])
    return (PROMPTS / "audit.txt").read_text().strip().format(
        duration=f"{dur:.1f}".rstrip("0").rstrip("."), captions=body, stem=job["stem"],
        options="\n".join(f"{L}. {o}" for L, o in job["options"].items()), answer=job["answer"],
        answer_text=job["options"][job["answer"]])


def parse_label(text: str) -> tuple[str | None, str]:
    """The label and evidence of the last JSON object in the text with a valid label."""
    dec, best = json.JSONDecoder(), (None, "")
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and str(obj.get("label", "")).strip().lower() in LABELS:
            best = (obj["label"].strip().lower(), str(obj.get("evidence", "")))
    return best


def key(arm: str, item: str) -> str:
    return f"{arm}|{item}"


def labels() -> dict[str, dict]:
    path = OUT / f"{JUDGE}.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[r["key"]] = r
    return out


def cmd_run(args):
    done = labels()
    todo = [j for j in jobs() if key(j["arm"], j["item"]) not in done or done[key(j["arm"], j["item"])]["label"] is None]
    todo = todo[: args.limit]
    print(f"{len(todo)} (caption arm, question) labels to get from {MODELS[JUDGE]}")
    if args.dry_run:
        if todo:
            print(prompt(todo[0]))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{JUDGE}.jsonl", "a") as f:
        for n, j in enumerate(todo):
            text = prompt(j)
            if text is None:
                print(f"  {j['arm']} {j['item']}: captions missing, skipped")
                continue
            for attempt in (1, 2):
                out = call_claude([{"type": "text", "text": text}], MODELS[JUDGE], EFFORT)
                init = next((e for e in out["events"] if e.get("type") == "system" and e.get("subtype") == "init"), {})
                res = next((e for e in out["events"] if e.get("type") == "result"), {})
                label, evidence = parse_label(res.get("result") or "")
                rl = next((e.get("rate_limit_info", {}) for e in out["events"] if e.get("type") == "rate_limit_event"),
                          {}).get("unifiedWindows", {})
                row = dict(key=key(j["arm"], j["item"]), arm=j["arm"], question_item=j["item"], video_id=j["video_id"],
                           dim=j["dim"], label=label, evidence=evidence, raw=res.get("result"), attempt=attempt,
                           model_reported=init.get("model"), tools=init.get("tools"),
                           mcp_servers=init.get("mcp_servers"), num_turns=res.get("num_turns"), usage=res.get("usage"),
                           util_5h=rl.get("five_hour", {}).get("utilization"),
                           util_7d=rl.get("seven_day", {}).get("utilization"), seconds=out["seconds"],
                           time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if label:
                    break
            print(f"  [{n + 1}/{len(todo)}] {row['key']}: {label} 5h={row['util_5h']} 7d={row['util_7d']}", flush=True)
            if row["model_reported"] and row["model_reported"] != MODELS[JUDGE]:
                sys.exit(f"the CLI reports model {row['model_reported']}, not {MODELS[JUDGE]}; stopping")
            if row["tools"] or row["mcp_servers"] or (row["num_turns"] or 1) != 1:
                sys.exit("isolation not confirmed; stopping")
            if row["util_5h"] is not None and row["util_5h"] >= args.max_5h:
                sys.exit(f"5-hour quota use reached {row['util_5h']:.0%}; stopping")
            if row["util_7d"] is not None and row["util_7d"] >= args.max_7d:
                sys.exit(f"7-day quota use reached {row['util_7d']:.0%}; stopping")


def cmd_report(args):
    """Per reader arm, its wrong answers split by the judge's label of the captions it read, overall and by video
    type; and, per caption arm, the labels of the questions it was audited on."""
    import pandas as pd
    lab = labels()
    qs = {q["item"]: q for q in tc.items("confirm")}
    rows = []
    for ma in READER_ARMS:
        model, arm = ma.split(":")
        for r in latest(model).values():
            if r["arm"] == arm and r["question_item"] in qs and not r["correct"]:
                lb = lab.get(key(arm, r["question_item"]), {}).get("label")
                rows.append(dict(reader=ma, label=lb or "unlabelled",
                                 vtype=tc.video_type(qs[r["question_item"]]["video_id"])))
    if not rows:
        sys.exit("no wrong answers on caption arms")
    df = pd.DataFrame(rows)
    print("wrong answers per reader arm, by the judge's label of the captions (share; n wrong):")
    t = pd.crosstab(df.reader, df.label, normalize="index").round(2)
    t["n_wrong"] = df.reader.value_counts()
    print(t.reindex([a for a in READER_ARMS if a in t.index]).to_string())
    for v in tc.VIDEO_TYPES:
        d = df[df.vtype == v]
        if len(d):
            print(f"\n{v}:")
            t = pd.crosstab(d.reader, d.label, normalize="index").round(2)
            t["n_wrong"] = d.reader.value_counts()
            print(t.reindex([a for a in READER_ARMS if a in t.index]).to_string())


def cmd_sheet(args):
    """40 labelled jobs, drawn at random (seed 0), stratified over the judge's three labels as far as they go, for a
    person to label without seeing the judge's label. Refuses to overwrite a sheet."""
    if SHEET.exists():
        sys.exit(f"{SHEET} exists; a hand-label sample is fixed once drawn")
    lab = [r for r in labels().values() if r["label"]]
    rng = random.Random(0)
    by = {L: [r for r in lab if r["label"] == L] for L in LABELS}
    for v in by.values():
        rng.shuffle(v)
    pick = []
    while len(pick) < args.n and any(by.values()):
        for L in LABELS:
            if by[L] and len(pick) < args.n:
                pick.append(by[L].pop())
    rng.shuffle(pick)
    qs = {q["item"]: q for q in tc.items("confirm")}
    with SHEET.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "prompt", "label"])
        for r in pick:
            w.writerow([r["key"], prompt(dict(arm=r["arm"], **qs[r["question_item"]])), ""])
    print(f"{len(pick)} items -> {SHEET}; fill the label column with correct, wrong or silent")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_agree(args):
    lab = labels()
    with SHEET.open() as f:
        hand = [(r["key"], r["label"].strip().lower()) for r in csv.DictReader(f) if r["label"].strip()]
    if not hand:
        sys.exit(f"no hand labels in {SHEET} yet")
    a = [h for _, h in hand]
    b = [lab[k]["label"] for k, _ in hand]
    print(f"{len(hand)} hand labels; agreement {sum(x == y for x, y in zip(a, b)) / len(a):.2f}, "
          f"Cohen's kappa {cohen_kappa(a, b):.2f}")
    for L in LABELS:
        print(f"  hand {L}: judge " + ", ".join(f"{M} {sum(x == L and y == M for x, y in zip(a, b))}" for M in LABELS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--max-5h", type=float, default=0.8)
    r.add_argument("--max-7d", type=float, default=0.6)
    sub.add_parser("report")
    s = sub.add_parser("sheet")
    s.add_argument("--n", type=int, default=40)
    sub.add_parser("agree")
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report, "sheet": cmd_sheet, "agree": cmd_agree}[args.cmd](args)


if __name__ == "__main__":
    main()
