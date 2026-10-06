#!/usr/bin/env python3
"""
Forced-choice gap filling: which of K candidate spans belongs in the masked gap?

For each video and gap width W, the gap (centred at i=29, as in the benchmark) and K-1 distractor
spans of the same length from the same video are masked *jointly*. Every method sees the same K
candidates (the true gap content plus the distractors) and picks one. Chance is 1/K for every
method, so text and frame methods are directly comparable.

Methods
    text_copy        caption candidate most similar to the gap's boundary captions (MPNet)
    text_copy_assign text_copy minus the candidate's best similarity to any *other* masked slot's boundaries
    vis_copy         frame candidate most similar to the gap's boundary frames (SigLIP)
    vis_copy_assign  vis_copy minus the candidate's best similarity to any other masked slot's boundaries
    llm_pmi          log P(candidate | masked context) - log P(candidate | no context)   [primary]
    llm_sum, llm_mean  raw / per-token conditional log-prob (secondary)

The LLM sees the production prompt (prompts/dense_window) with all K spans set to null, and the
candidate is scored as the JSON answer for the gap's indices. No text is generated.

Usage (from repo root)
    # local, CPU: persistence baselines (needs local/wild_videos_embs_siglip/)
    .venv/bin/python scripts/forced_choice_gap.py baselines
    # GPU: LLM scoring (resumable; --limit for a pilot)
    python scripts/forced_choice_gap.py llm --model-key llama-3.1-8b [--limit 40] [--upload]
    # local: fetch GPU scores from HF (if uploaded) and report
    .venv/bin/python scripts/forced_choice_gap.py report [--download]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

WIDTHS = [1, 2, 4, 8]
K = 4
SEED = 2025
MARGIN = 3  # distractor spans keep this many seconds away from the gap's boundaries
OUT = Path("results/forced_choice")
HF_REPO, HF_DIR = "Y3/dense_video_captions", "forced_choice"
PROMPT_DIR = Path("prompts/dense_window")


def channel(vid: str) -> str:
    return re.match(r"^(.*?)_\d+", vid.replace("Olly's", "Olly_s")).group(1)


def load_videos() -> list[tuple[str, str, list[str]]]:
    vids = []
    for ds in ["wild4", "wild5"]:
        for cf in sorted(Path(f"datasets/wildQA/captions__{ds}").glob("*.json")):
            if cf.name == "categories.json":
                continue
            caps = [c["caption"] for c in json.load(open(cf)).get("captions", [])[:60]]
            if len(caps) == 60 and all(caps):
                vids.append((ds, cf.stem, caps))
    return vids


def build_items() -> list[dict]:
    """Deterministic item set (same on every machine): gap + K-1 non-overlapping distractor spans."""
    rng = np.random.default_rng(SEED)
    items = []
    for ds, vid, _ in load_videos():
        for w in WIDTHS:
            s0 = 30 - w // 2
            gap = list(range(s0, s0 + w))
            L, R = s0 - 1, s0 + w
            spans = [gap]
            # spans never touch second 0 or 59, so every slot has two visible boundaries
            starts = [s for s in range(1, 60 - w) if s + w - 1 < L - MARGIN + 1 or s > R + MARGIN - 1]
            for s in rng.permutation(starts):
                cand = list(range(int(s), int(s) + w))
                # at least one unmasked second between spans, so each slot keeps its own boundaries
                if all(min(abs(a - b) for a in cand for b in sp) >= 2 for sp in spans):
                    spans.append(cand)
                if len(spans) == K:
                    break
            if len(spans) == K:
                items.append(dict(item=f"{vid}|w{w}", ds=ds, vid=vid, w=w, spans=spans))
    return items


# ----------------------------------------------------------------------------- baselines
def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def slot_boundary(X, span, masked):
    L, R = span[0] - 1, span[-1] + 1
    b = [X[t] for t in (L, R) if 0 <= t < len(X) and t not in masked]
    return unit(np.mean(b, axis=0))


def run_baselines():
    from llm.local_embedder import LocalEmbedder
    emb = LocalEmbedder("all-mpnet-base-v2")
    caps = {vid: c for _, vid, c in load_videos()}
    rows = []
    for it in build_items():
        vf = Path(f"local/wild_videos_embs_siglip/{it['vid'].replace(chr(39), '_')}.npy")
        if not vf.exists():
            continue
        spaces = {"text": unit(emb.get_embeddings(f"{it['vid']}_dense_caps", caps[it["vid"]])),
                  "vis": unit(np.load(vf)[:60])}
        masked = {t for sp in it["spans"] for t in sp}
        row = dict(item=it["item"], vid=it["vid"], w=it["w"])
        for name, X in spaces.items():
            if len(X) < 60:
                break
            gap_b = slot_boundary(X, it["spans"][0], masked)
            cand = [unit(X[sp].mean(axis=0)) for sp in it["spans"]]
            plain = [float(c @ gap_b) for c in cand]
            # assignment: does the candidate fit the gap better than any *other* masked slot?
            # (uses only visible boundaries of all masked slots, never a candidate's origin)
            others = [slot_boundary(X, sp, masked) for sp in it["spans"][1:]]
            row[f"{name}_copy"] = plain
            row[f"{name}_copy_assign"] = [p - max(float(c @ o) for o in others) for p, c in zip(plain, cand)]
            row[f"{name}_bd"] = 1 - float(X[it["spans"][0][0] - 1] @ X[it["spans"][0][-1] + 1])
        else:
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "baselines.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"baselines: {len(rows)} items -> {OUT / 'baselines.jsonl'}")


# ----------------------------------------------------------------------------- LLM
def run_llm(model_key: str, limit: int | None, upload: bool):
    import torch
    from data.data_loaders import WildLoader
    from llm.local_llm import HuggingFaceModelAdapter
    from llm.prompting import JSONPromptBuilder
    from data_models.captions_only import CaptionedVideo

    builder = JSONPromptBuilder.from_path(PROMPT_DIR)
    videos = {}
    for ds in ["wild4", "wild5"]:
        for v in WildLoader(Path(f"datasets/wildQA/captions__{ds}")).load():
            videos[v.video_id] = v
    adapter = HuggingFaceModelAdapter(model_key=model_key)
    adapter._ensure_loaded()
    tok, model = adapter.tokenizer, adapter.model
    dev = model.device

    out_path = OUT / f"llm_{model_key}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["item"] for l in open(out_path)}
    items = [it for it in build_items() if it["item"] not in done and it["vid"] in videos]
    if limit:
        items = items[:limit]
    print(f"{len(done)} already scored; scoring {len(items)} items with {model_key}")

    def prompt_ids(video: CaptionedVideo, mask: set[int], gap: list[int]) -> list[int]:
        clips = [c.masked_copy() if c.index in mask else c for c in video.clips[:60]]
        mv = CaptionedVideo(video_id=video.video_id, clips=clips)
        text = builder.build_prompt(mv)
        # The template lists every masked index; ask only for the gap (other null spans stay unknown).
        text = text.replace(str(sorted(mask)), str(gap))
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        return list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "input_ids") else ids)

    def answer(captions: list[str], gap: list[int]) -> str:
        return json.dumps([{"index": t, "caption": c} for t, c in zip(gap, captions)], ensure_ascii=False)

    @torch.no_grad()
    def score(prefix: list[int], answers: list[str]) -> list[tuple[float, int]]:
        res = []
        for a in answers:  # one at a time keeps memory flat for 4-bit models on small GPUs
            ans = tok(a, add_special_tokens=False).input_ids
            ids = torch.tensor([prefix + ans], device=dev)
            logits = model(ids).logits[0, len(prefix) - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(ans, device=dev)[:, None]).sum().item()
            res.append((lp, len(ans)))
        return res

    with open(out_path, "a") as f:
        for n, it in enumerate(items):
            v = videos[it["vid"]]
            gap = it["spans"][0]
            mask = {t for sp in it["spans"] for t in sp}
            answers = [answer([v.clips[t].caption for t in sp], gap) for sp in it["spans"]]
            cond = score(prompt_ids(v, mask, gap), answers)
            uncond = score(prompt_ids(v, set(range(60)), gap), answers)
            f.write(json.dumps(dict(item=it["item"], model=model_key,
                                    lp_cond=[c[0] for c in cond], lp_uncond=[u[0] for u in uncond],
                                    n_tok=[c[1] for c in cond])) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    print(f"done -> {out_path}")
    if upload:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(out_path), path_in_repo=f"{HF_DIR}/{out_path.name}",
                            repo_id=HF_REPO, repo_type="dataset")
        print(f"uploaded to {HF_REPO}/{HF_DIR}/{out_path.name}")


# ----------------------------------------------------------------------------- report
def report(download: bool):
    if download:
        from huggingface_hub import HfApi, hf_hub_download
        for p in HfApi().list_repo_files(HF_REPO, repo_type="dataset"):
            if p.startswith(f"{HF_DIR}/llm_"):
                local = hf_hub_download(HF_REPO, p, repo_type="dataset")
                (OUT / Path(p).name).write_bytes(Path(local).read_bytes())
    base = pd.read_json(OUT / "baselines.jsonl", lines=True)
    methods = ["text_copy", "text_copy_assign", "vis_copy", "vis_copy_assign"]
    for lf in sorted(OUT.glob("llm_*.jsonl")):
        llm = pd.read_json(lf, lines=True)
        tag = lf.stem.replace("llm_", "")
        llm[f"{tag}_pmi"] = [list(np.subtract(c, u)) for c, u in zip(llm.lp_cond, llm.lp_uncond)]
        llm[f"{tag}_sum"] = llm.lp_cond
        llm[f"{tag}_mean"] = [list(np.divide(c, n)) for c, n in zip(llm.lp_cond, llm.n_tok)]
        base = base.merge(llm[["item", f"{tag}_pmi", f"{tag}_sum", f"{tag}_mean"]], on="item", how="left")
        methods += [f"{tag}_pmi", f"{tag}_sum", f"{tag}_mean"]
    for m in methods:
        base[m + "_ok"] = [np.nan if not isinstance(s, list) else float(int(np.argmax(s)) == 0) for s in base[m]]
    base["chan"] = base.vid.map(channel)
    base["scene_change"] = base.groupby("w").vis_bd.transform(lambda x: pd.qcut(x, 3, labels=["low", "mid", "high"]))

    rng = np.random.default_rng(SEED)
    chans = base.chan.unique()
    groups = {c: base.index[base.chan == c].values for c in chans}
    boots = [np.concatenate([groups[c] for c in rng.choice(chans, len(chans))]) for _ in range(2000)]
    ok = [m + "_ok" for m in methods]

    def table(df_idx, label):
        sub = base.loc[df_idx]
        point = sub[ok].mean()
        bs = pd.DataFrame([base.loc[np.intersect1d(b, df_idx)][ok].mean() for b in boots])
        t = pd.DataFrame({"n": sub[ok].notna().sum(), "acc": point, "lo": bs.quantile(0.025),
                          "hi": bs.quantile(0.975)}).round(3)
        print(f"\n{label} (chance {1 / K:.2f})")
        print(t.to_string())

    table(base.index.values, "All widths")
    for w in WIDTHS:
        table(base.index[base.w == w].values, f"W={w}")
    for s in ["low", "mid", "high"]:
        table(base.index[base.scene_change == s].values, f"Visual boundary disagreement: {s}")
    base.to_csv(OUT / "forced_choice_merged.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baselines", "llm", "report"])
    ap.add_argument("--model-key", default="llama-3.1-8b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    {"baselines": run_baselines, "llm": lambda: run_llm(a.model_key, a.limit, a.upload),
     "report": lambda: report(a.download)}[a.mode]()
