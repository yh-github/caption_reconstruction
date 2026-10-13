#!/usr/bin/env python3
"""
Check 5 s piece captions before scaling up (kickoff plan, Step 1.1). No API calls.

For each video and caption set:
  SigLIP retrieval  rank of the true piece among the video's 12 (SigLIP 2 text tower vs the mean of
                    frame rows 5p..5p+4), MRR and R@1. Chance MRR for 12 pieces is 0.259.
  length            words per caption, share over 45 words, share over 64 SigLIP tokens (truncated)
  order cues        share of captions with a cue word, per group (CUES below)
  on-screen text    share of captions with digits, quotes or text words (ONSCREEN below), each listed
  repetition        mean pairwise MPNet cosine between the video's captions (high = setting repeated)
  lead              SigLIP similarity of caption p to piece p+1 minus to piece p-1 (p = 1..10). Above 0 =
                    captions run ahead into the next piece, as the original captions do by 1-2 s

Caption sets:
  isolated, in_context     results/piece_captions/<model>/{isolated/<v>/<pp>.json, in_context/<v>.json}
  <mode>_<tag>             the same under a set-aside folder results/piece_captions/_<tag>/<model>/
                           (_with_audio: made with audio; _prompt_v1: the first prompt)
  original                 per-second captions (datasets/wildQA/captions__wild{4,5}), seconds 5p..5p+4 joined

The cue lists from the 2026-10-12 count were not saved, so CUES is a reconstruction; the original
captions' share is printed for comparison with that count (25.4% of 20,101 captions).

Usage (from repo root):
    .venv/bin/python scripts/piece_caption_check.py [--videos V ...] [--model gemini-3.8-flash]
Outputs: printed tables, results/piece_captions/check/{per_video,flagged}.csv
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm.local_embedder import LocalEmbedder, SiglipTextEmbedder

N_PIECES, PIECE_S = 12, 5
PC = ROOT / "results/piece_captions"
VIS_DIR = ROOT / "local/wild_videos_embs_siglip"
CAP_DIRS = [ROOT / "datasets/wildQA/captions__wild4", ROOT / "datasets/wildQA/captions__wild5"]
CUES = {
    "continuity": r"\b(continu\w*|still|again|another|keeps?|keeping|remains?|resum\w*)\b",
    "onset": r"\b(begins?|beginning|began|starts?|starting|started|finish\w*|complet\w*|proceeds?)\b",
    "adverb": r"\b(after|afterwards?|before|later|earlier|previous\w*|next|meanwhile|finally|eventually|then|now)\b",
}
ONSCREEN = r"\d|[\"“”]|\b(text|reads?|subtitles?|captions?|timestamps?|clock|date|title|logo|watermark|sign)\b"


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def load_sets(v: str, model: str) -> dict[str, dict[int, str]]:
    """Caption set -> {piece: caption}, for the sets this video has."""
    sets = {}
    for prefix, base in [("", PC / model), *((d.name, d / model) for d in sorted(PC.glob("_*")) if d.is_dir())]:
        iso = {}
        for f in sorted((base / "isolated" / v).glob("[0-9][0-9].json")):
            p = json.loads(f.read_text())["pieces"][0]
            iso[p["piece"]] = p["caption"]
        if iso:
            sets[f"isolated{prefix}"] = iso
        f = base / "in_context" / f"{v}.json"
        if f.exists():
            sets[f"in_context{prefix}"] = {p["piece"]: p["caption"] for p in json.loads(f.read_text())["pieces"]}
    cf = next((d / f"{v}.json" for d in CAP_DIRS if (d / f"{v}.json").exists()), None)
    if cf:
        caps = [c["caption"] for c in json.loads(cf.read_text())["captions"]]
        sets["original"] = {p: " ".join(caps[p * PIECE_S:(p + 1) * PIECE_S]) for p in range(N_PIECES)}
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", help="default: every video with new piece captions for --model")
    ap.add_argument("--model", default="gemini-3.8-flash")
    args = ap.parse_args()
    videos = args.videos or sorted({f.name for f in (PC / args.model / "isolated").glob("*") if f.is_dir()}
                                   | {f.stem for f in (PC / args.model / "in_context").glob("*.json")})
    if not videos:
        sys.exit(f"no piece captions under {PC / args.model}")

    # The stored frames are SigLIP 2 (timm v2_webli), so captions must use the SigLIP 2 text tower.
    sg, mp = SiglipTextEmbedder("google/siglip2-base-patch16-224"), LocalEmbedder("all-mpnet-base-v2")
    rows, flagged = [], []
    for v in videos:
        V = unit(np.load(VIS_DIR / f"{v.replace(chr(39), '_')}.npy"))
        P = unit(np.stack([V[p * PIECE_S:(p + 1) * PIECE_S].mean(0) for p in range(N_PIECES)]))
        for name, caps in load_sets(v, args.model).items():
            pieces = sorted(caps)
            texts = [caps[p] for p in pieces]
            S = unit(sg.get_embeddings(f"{v}_pieces_{name}", texts))
            sim = S @ P.T
            ranks = np.array([1 + (sim[i] > sim[i, p]).sum() for i, p in enumerate(pieces)])
            n_tok = [len(sg.tokenizer(t.lower())["input_ids"]) for t in texts]
            words = [len(t.split()) for t in texts]
            M = unit(mp.get_embeddings(f"{v}_pieces_{name}", texts))
            rep = np.mean([M[i] @ M[j] for i, j in itertools.combinations(range(len(texts)), 2)]) if len(texts) > 1 else np.nan
            inner = [(i, p) for i, p in enumerate(pieces) if 0 < p < N_PIECES - 1]
            lead = np.mean([sim[i, p + 1] - sim[i, p - 1] for i, p in inner]) if inner else np.nan
            row = dict(video_id=v, set=name, n=len(texts), mrr=np.mean(1 / ranks), r1=np.mean(ranks == 1),
                       mean_rank=ranks.mean(), words=np.mean(words), over45w=np.mean(np.array(words) > 45),
                       over64tok=np.mean(np.array(n_tok) > 64), repetition=rep, lead=lead)
            for g, rx in CUES.items():
                row[f"cue_{g}"] = np.mean([bool(re.search(rx, t, re.I)) for t in texts])
            row["cue_any"] = np.mean([any(re.search(rx, t, re.I) for rx in CUES.values()) for t in texts])
            hits = [(p, t) for p, t in zip(pieces, texts) if re.search(ONSCREEN, t, re.I)]
            row["onscreen"] = len(hits) / len(texts)
            flagged += [dict(video_id=v, set=name, piece=p, caption=t) for p, t in hits]
            rows.append(row)

    df = pd.DataFrame(rows)
    out = PC / "check"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_video.csv", index=False)
    pd.DataFrame(flagged, columns=["video_id", "set", "piece", "caption"]).to_csv(out / "flagged.csv", index=False)

    pd.set_option("display.width", 200, "display.max_colwidth", 200)
    cols = ["mrr", "r1", "mean_rank", "words", "over45w", "over64tok", "cue_any", "cue_continuity",
            "cue_onset", "cue_adverb", "onscreen", "repetition", "lead"]
    summary = df.groupby("set").agg(videos=("video_id", "nunique"), captions=("n", "sum"), **{c: (c, "mean") for c in cols})
    print(f"Per caption set, mean over videos (chance MRR for {N_PIECES} pieces = {np.mean([1 / k for k in range(1, N_PIECES + 1)]):.3f}):")
    print(summary.round(3).to_string())
    print("\nPer video:")
    print(df.set_index(["video_id", "set"])[["n", *cols]].round(3).to_string())
    print(f"\nOn-screen text candidates (for reading), new sets only:")
    for f in flagged:
        if f["set"] in ("isolated", "in_context"):
            print(f"  {f['set']:<10} {f['video_id']} piece {f['piece']:>2}: {f['caption']}")
    print(f"\nwrote {(out / 'per_video.csv').relative_to(ROOT)}, {(out / 'flagged.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
