#!/usr/bin/env python3
"""Build the puzzles and run the free arms (no reader, learned) on the pilot videos. No API calls.

    .venv/bin/python scripts/unshuffle/run_free.py [--videos V ...] [--n-jobs 8]

Writes (results/unshuffle/, gitignored):
    puzzles.json            every puzzle with its two shuffles, shared by all arms (pre-registration §2)
    free/orders.jsonl       one row per (puzzle, variant, arm): the predicted order and its metrics
    free/scores.csv         the same without the orders, for analysis
    free/alphas.csv         the regularisation chosen in each leave-one-channel-out fold
Variants: "all" pieces, and "notext" without the pieces whose isolated caption quotes on-screen text.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from unshuffle import arms, core, data  # noqa: E402

ENCODERS = ["siglip2_text", "mpnet"]


def write_puzzles(videos: list[str]) -> list[dict]:
    puzzles = core.make_puzzles(videos)
    data.OUT.mkdir(parents=True, exist_ok=True)
    path = data.OUT / "puzzles.json"
    if path.exists():
        old = {p["puzzle"]: p for p in json.loads(path.read_text())}
        clash = [p["puzzle"] for p in puzzles if p["puzzle"] in old and old[p["puzzle"]] != p]
        if clash:
            sys.exit(f"puzzles.json disagrees with the puzzle generator on {len(clash)} puzzles, e.g. {clash[0]}")
        merged = {**old, **{p["puzzle"]: p for p in puzzles}}
        puzzles_all = list(merged.values())
    else:
        puzzles_all = puzzles
    path.write_text(json.dumps(puzzles_all, indent=0))
    return puzzles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", help="default: the 30 pilot videos")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    videos = args.videos or data.pilot_videos()
    t0 = time.time()
    puzzles = write_puzzles(videos)
    tagged = data.text_tagged()

    # embeddings: one (12, d) matrix per video and representation
    reps: dict[tuple, dict[str, object]] = {("frames", "-", "siglip2"): {v: data.piece_means(v) for v in videos}}
    for cs in data.CAPTION_SETS:
        for enc in ENCODERS:
            reps[("captions", cs, enc)] = {v: data.caption_embeddings(v, cs, enc) for v in videos}
    edges = {v: data.piece_edges(v) for v in videos}
    print(f"embeddings ready for {len(videos)} videos, {len(reps)} representations ({time.time() - t0:.0f} s)")

    learned, alphas = {}, []
    for rep, E in reps.items():
        for kind in ("probe", "pairwise"):
            learned[(rep, kind)] = arms.loco_scores(kind, E, n_jobs=args.n_jobs)
            alphas += [dict(modality=rep[0], caption_set=rep[1], encoder=rep[2], method=kind, video=v, alpha=a)
                       for v, (_, a) in learned[(rep, kind)].items()]
        print(f"  learned arms done for {rep} ({time.time() - t0:.0f} s)")

    out = data.OUT / "free"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pz in puzzles:
        v = pz["video"]
        variants = {"all": pz["pieces"]}
        drop = tagged.get(v, set())
        if drop:
            reduced = core.reduce_puzzle(pz["pieces"], drop)
            if len(reduced) >= 2:
                variants["notext"] = reduced
        else:
            variants["notext"] = pz["pieces"]
        for var, pieces in variants.items():
            base = dict(video=v, channel=core.channel(v), puzzle=pz["puzzle"], set=pz["set"], k=pz["k"],
                        variant=var, gap_s=pz["gap_s"])
            presented = core.restrict(pz["shuffles"][0], pieces)  # neutral input order (see arms.py)
            preds = {}
            for rep, E in reps.items():
                key = ":".join(rep)
                preds[(rep, "chain")] = arms.chain(E[v], presented, (key, pz["puzzle"], var))
                for kind in ("probe", "pairwise"):
                    preds[(rep, kind)] = arms.order_by_score(learned[(rep, kind)][v][0], presented)
            first, last = edges[v]
            preds[(("frames", "-", "siglip2"), "edge")] = arms.edge_chain(first, last, presented)
            for (rep, method), order in preds.items():
                rows.append(dict(**base, modality=rep[0], caption_set=rep[1], encoder=rep[2], method=method,
                                 arm=f"{rep[0]}:{rep[1]}:{rep[2]}:{method}", pieces=pieces, order=order,
                                 **core.score(order)))
    with open(out / "orders.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows).drop(columns=["pieces", "order"])
    df.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(alphas).to_csv(out / "alphas.csv", index=False)
    print(f"{len(rows)} rows, {df.arm.nunique()} arms, {df.puzzle.nunique()} puzzles -> {out.relative_to(data.ROOT)} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
