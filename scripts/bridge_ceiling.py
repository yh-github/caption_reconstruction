#!/usr/bin/env python3
"""
Bridge ceiling: is the weak near-second score of the Gemini captions (shared-target near pool, c ~ 0.18) a property
of the captions, or of SigLIP 2's text-to-image link? Captions written for a single frame are time-aligned by
construction. If they also score about 0.18 against nearby frames, the limit is the encoder; if they score well
above it, the Gemini captions lack second-level information.

Steps (videos: one per channel, seeded, up to N_VIDEOS):
    extract  (local, ffmpeg)  frame at each exact second t = 0..59 -> results/bridge_ceiling/frames/<vid>/<t>.jpg
    caption  (GPU)            Florence-2 <DETAILED_CAPTION> per frame -> results/bridge_ceiling/captions.jsonl
                              (~30 s per frame on the local CPU, so run this step on a GPU)
    eval     (local)          SigLIP 2 text of each caption vs. the video's frames:
                              - near c: target frame t vs. frames 2..10 s away (both sides), t in [10, 50)
                              - other-video c: target vs. 20 frames from other channels (topic)
                              - Q1 MRR among the 60 frames, and the lag profile peak
                              for: per-frame captions, Gemini captions, Gemini shifted by +1/+2 s, and frame copy
                              (frame t-1). 95% CIs: bootstrap over videos (one video per channel).

Usage (from repo root):
    .venv/bin/python scripts/bridge_ceiling.py extract
    .venv/bin/python scripts/bridge_ceiling.py push-frames          # zip -> private HF repo, for Kaggle
    (Kaggle: paste scripts/kaggle_bridge_ceiling.py into a notebook; it runs pull-frames, caption, push-captions)
    .venv/bin/python scripts/bridge_ceiling.py pull-captions
    .venv/bin/python scripts/bridge_ceiling.py eval
    Any GPU machine with the frames folder: python scripts/bridge_ceiling.py caption [--device cuda] [--batch 16]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

OUT = Path("results/bridge_ceiling")
FRAMES = OUT / "frames"
CAPTIONS = OUT / "captions.jsonl"
N_VIDEOS = 30
SEED = 2026
T = 60
MAX_SIDE = 768
FLORENCE = "florence-community/Florence-2-base"
TASK = "<DETAILED_CAPTION>"


def channel(vid: str) -> str:
    return re.match(r"^(.*?)_\d+", vid.replace("'", "_")).group(1)


def pick_videos() -> dict[str, Path]:
    """One video per channel among those with a raw mp4, SigLIP 2 frames and 60 Gemini captions."""
    raw = {p.stem: p for p in Path("local/wild_videos_raw").rglob("*.mp4")}
    ok = []
    for d in ["wild4", "wild5"]:
        for cf in Path(f"datasets/wildQA/captions__{d}").glob("*.json"):
            stem = cf.stem.replace("'", "_")
            caps = json.load(open(cf)).get("captions", [])
            if stem in raw and Path(f"local/wild_videos_embs_siglip/{stem}.npy").exists() and len(caps) >= T:
                ok.append(stem)
    rng = random.Random(SEED)
    by_chan: dict[str, list[str]] = {}
    for v in sorted(ok):
        by_chan.setdefault(channel(v), []).append(v)
    chans = sorted(by_chan)
    rng.shuffle(chans)
    return {v: raw[v] for v in (rng.choice(by_chan[c]) for c in chans[:N_VIDEOS])}


def cmd_extract(_):
    vids = pick_videos()
    print(f"{len(vids)} videos from {len({channel(v) for v in vids})} channels")
    for vid, path in vids.items():
        d = FRAMES / vid
        d.mkdir(parents=True, exist_ok=True)
        for t in range(T):
            f = d / f"{t:02d}.jpg"
            if not f.exists():  # -ss before -i seeks to the exact second, matching the stored embedding rows
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                                "-vf", f"scale='min({MAX_SIDE},iw)':-2", "-q:v", "3", str(f)], check=True)
        print(f"  {vid}: {len(list(d.glob('*.jpg')))} frames")
    print(f"frames in {FRAMES} ({sum(f.stat().st_size for f in FRAMES.rglob('*.jpg')) / 1e6:.0f} MB)")


def cmd_caption(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoProcessor.from_pretrained(FLORENCE)
    model = Florence2ForConditionalGeneration.from_pretrained(
        FLORENCE, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    done = set()
    if CAPTIONS.exists():
        done = {(r["vid"], r["t"]) for r in map(json.loads, CAPTIONS.read_text().splitlines())}
    todo = [(f.parent.name, int(f.stem), f) for f in sorted(FRAMES.glob("*/*.jpg"))
            if (f.parent.name, int(f.stem)) not in done]
    print(f"{len(done)} captioned, {len(todo)} to go on {device}")
    with open(CAPTIONS, "a") as out:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            imgs = [Image.open(f).convert("RGB") for _, _, f in chunk]
            inp = proc(text=[TASK] * len(imgs), images=imgs, return_tensors="pt").to(device)
            if device == "cuda":
                inp["pixel_values"] = inp["pixel_values"].half()
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=96, num_beams=3)
            for (vid, t, _), txt in zip(chunk, proc.batch_decode(ids, skip_special_tokens=True)):
                out.write(json.dumps(dict(vid=vid, t=t, model=FLORENCE, task=TASK, caption=txt.strip())) + "\n")
            out.flush()
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)


HF_REPO, HF_DIR = "Y3/dense_video_captions", "bridge_ceiling"  # private dataset repo


def cmd_push_frames(_):
    """Zip the extracted frames and upload them to the private HF dataset repo (for the Kaggle GPU run)."""
    import shutil
    from huggingface_hub import HfApi
    zp = shutil.make_archive(str(OUT / "frames"), "zip", root_dir=FRAMES)
    HfApi().upload_file(path_or_fileobj=zp, path_in_repo=f"{HF_DIR}/frames.zip", repo_id=HF_REPO,
                        repo_type="dataset")
    print(f"uploaded {zp} ({Path(zp).stat().st_size / 1e6:.0f} MB) to {HF_REPO}/{HF_DIR}/frames.zip")


def cmd_pull_frames(_):
    """On the GPU machine: download and unpack the frames from HF."""
    import shutil
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download(HF_REPO, f"{HF_DIR}/frames.zip", repo_type="dataset")
    FRAMES.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zp, FRAMES)
    print(f"{len(list(FRAMES.glob('*/*.jpg')))} frames in {FRAMES}")


def cmd_push_captions(_):
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(CAPTIONS), path_in_repo=f"{HF_DIR}/captions.jsonl", repo_id=HF_REPO,
                        repo_type="dataset")
    print(f"uploaded {CAPTIONS} to {HF_REPO}/{HF_DIR}/captions.jsonl")


def cmd_pull_captions(_):
    import shutil
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HF_REPO, f"{HF_DIR}/captions.jsonl", repo_type="dataset", force_download=True)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, CAPTIONS)
    print(f"{sum(1 for _ in open(CAPTIONS))} captions in {CAPTIONS}")


def cmd_eval(_):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from llm.local_embedder import SiglipTextEmbedder
    sg = SiglipTextEmbedder("google/siglip2-base-patch16-224")
    unit = lambda x: (lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12))(np.asarray(x, np.float32))

    fl: dict[str, dict[int, str]] = {}
    for r in map(json.loads, CAPTIONS.read_text().splitlines()):
        fl.setdefault(r["vid"], {})[r["t"]] = r["caption"]
    vids = [v for v, c in fl.items() if len(c) == T]
    gem = {}
    for v in vids:
        cf = next(p for d in ["wild4", "wild5"] for p in [Path(f"datasets/wildQA/captions__{d}/{v}.json"),
                  Path(f"datasets/wildQA/captions__{d}/{v.replace('Olly_s', chr(39).join(['Olly', 's']))}.json")]
                  if p.exists())
        gem[v] = [c["caption"] for c in json.load(open(cf))["captions"][:T]]
    V = {v: unit(np.load(f"local/wild_videos_embs_siglip/{v}.npy")[:T]) for v in vids}
    F = {v: unit(sg.get_embeddings(f"{v}_florence2_detailed", [fl[v][t] for t in range(T)])) for v in vids}
    G = {v: unit(sg.get_embeddings(f"{v}_dense_caps", gem[v])) for v in vids}
    rng = np.random.default_rng(SEED)

    def c_score(q, target, pool):
        s_t, s_p = q @ target, pool @ q
        r = 1 + (s_p > s_t).sum() + 0.5 * (s_p == s_t).sum()
        return (len(s_p) + 2 - 2 * r) / len(s_p)  # (N + 1 - 2r) / (N - 1) with N = len(pool) + 1

    rows, lags = [], []
    for v in vids:
        others = np.concatenate([V[o][rng.choice(T, 4, replace=False)] for o in vids if channel(o) != channel(v)])
        queries = {"florence_per_frame": lambda t: F[v][t], "gemini": lambda t: G[v][t],
                   "gemini_shift_+1": lambda t: G[v][t - 1], "gemini_shift_+2": lambda t: G[v][t - 2],
                   "frame_copy": lambda t: V[v][t - 1]}
        for m, q in queries.items():
            near, other = [], []
            for t in range(10, 50):
                pool = [s for s in range(T) if 2 <= abs(s - t) <= 10]
                near.append(c_score(q(t), V[v][t], V[v][pool]))
                other.append(c_score(q(t), V[v][t], others[rng.choice(len(others), 20, replace=False)]))
            rows.append(dict(vid=v, method=m, near_c=np.mean(near), other_video_c=np.mean(other)))
        for m, E in [("florence_per_frame", F[v]), ("gemini", G[v])]:
            sim = E @ V[v].T
            z = (sim - sim.mean(1, keepdims=True)) / (sim.std(1, keepdims=True) + 1e-9)
            mrr = np.mean([1 / (1 + (sim[t] > sim[t, t]).sum()) for t in range(T)])
            prof = {k: np.mean([z[t, t + k] for t in range(T) if 0 <= t + k < T]) for k in range(-5, 6)}
            lags.append(dict(vid=v, method=m, q1_mrr=mrr, peak_lag=max(prof, key=prof.get), **{str(k): x for k, x in prof.items()}))
    df, lg = pd.DataFrame(rows), pd.DataFrame(lags)
    df.to_csv(OUT / "eval_per_video.csv", index=False)
    lg.to_csv(OUT / "eval_lag.csv", index=False)

    def ci(x):
        b = [np.mean(rng.choice(x, len(x))) for _ in range(2000)]
        return f"{np.mean(x):.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"

    print(f"{len(vids)} videos, {len({channel(v) for v in vids})} channels; calibrated c (0 = chance), 95% CI")
    for m, g in df.groupby("method", sort=False):
        print(f"  {m:20s} near (2-10 s) {ci(g.near_c.values)}   other video {ci(g.other_video_c.values)}")
    d = df.pivot(index="vid", columns="method", values="near_c")
    print(f"  paired near_c, per-frame minus Gemini: {ci((d.florence_per_frame - d.gemini).values)}")
    print(f"  paired near_c, per-frame minus Gemini +2 s: {ci((d.florence_per_frame - d['gemini_shift_+2']).values)}")
    for m, g in lg.groupby("method", sort=False):
        prof = g[[str(k) for k in range(-5, 6)]].mean()
        print(f"  {m:20s} Q1 MRR {ci(g.q1_mrr.values)} (chance 0.079); profile peak {prof.idxmax()}; "
              f"per-video peak lags {g.peak_lag.value_counts().sort_index().to_dict()}")


def main():
    ap = argparse.ArgumentParser(description="Bridge ceiling for the caption timing claim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    c = sub.add_parser("caption")
    c.add_argument("--device", default=None)
    c.add_argument("--batch", type=int, default=16)
    sub.add_parser("eval")
    for name in ["push-frames", "pull-frames", "push-captions", "pull-captions"]:
        sub.add_parser(name)
    args = ap.parse_args()
    {"extract": cmd_extract, "caption": cmd_caption, "eval": cmd_eval, "push-frames": cmd_push_frames,
     "pull-frames": cmd_pull_frames, "push-captions": cmd_push_captions,
     "pull-captions": cmd_pull_captions}[args.cmd](args)


if __name__ == "__main__":
    main()
