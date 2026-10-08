#!/usr/bin/env python3
"""
E1: question-conditioned evidence retrieval with content controls, text arms vs. frame arms.

For each WildQA question whose single evidence span (<= 15 s) was masked and reconstructed by
Llama-3.1-8B, the question retrieves the evidence span from a 60-second index. Each arm fills the
evidence span differently; everything else in the index is the unmasked video.

Text arms (captions, embedded with MPNet; SigLIP-text reported for reference):
    oracle, masked, repeat (nearest boundary), lerp (positional), recon (Llama),
    recon_other_video (Llama text from a different video of the same domain),
    recon_same_video (Llama text for a different, non-overlapping gap of the same video)
Frame arms (SigLIP 2 frames, queried with SigLIP 2 text; the "sg_" text arms also use SigLIP 2 text):
    oracle, masked, repeat, lerp, other_video (frames of a different same-domain video)

Two normalisations make text and frame arms comparable:
    recovery     = (arm - masked) / (oracle - masked)   on MRR, within the arm's own modality
    content gain = arm - its content-free control       (other-video fill of the same modality)

All CIs are cluster bootstraps over YouTube channels.

Usage (from repo root):
    .venv/bin/python scripts/e1_evidence_retrieval.py
Outputs: results/analysis_controls/e1_evidence_retrieval_rows.csv and _summary.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.wildqa_loader import load_wildqa_dataset
from evaluations.evidence_retrieval import calculate_evidence_retrieval_metrics
from llm.local_embedder import LocalEmbedder, SiglipTextEmbedder

HF_SNAPS = Path.home() / ".cache/huggingface/hub/datasets--Y3--dense_video_captions/snapshots"
MD = Path("results/recon/manual_download/reconstruction")
EVIDENCE_RUNS = {"dev": "wild4_llama_evidence", "test": "wild5_llama_evidence"}
SAME_VIDEO_RUNS = ["wild4_llama_multi_width", "wild5_llama_multi_width", "wild5_llama_w3_w6",
                   "wild4_llama_w3_window_v3_videopool", "wild4_llama_w6"]
VIS_DIR = Path("local/wild_videos_embs_siglip")
OUT = Path("results/analysis_controls")
B = 2000


def run_dirs(run: str) -> list[Path]:
    """All local copies of a run: manual_download, results/reconstruction, and every HF snapshot."""
    roots = [MD / run, Path("results/reconstruction") / run] + [s / "reconstruction" / run for s in HF_SNAPS.iterdir()]
    return [d for r in roots if r.exists() for d in r.iterdir() if d.is_dir()]


def load_recon(run: str, max_w: int = 16) -> dict[str, list[dict[int, str]]]:
    out: dict[str, dict[tuple, dict[int, str]]] = {}
    for d in run_dirs(run):
        m = re.search(r"w=(\d+)", d.name)
        if m and int(m.group(1)) > max_w:
            continue
        for f in d.glob("*.json"):
            if f.name.startswith("skip__"):
                continue
            rc = {int(k): v for k, v in json.load(open(f)).get("reconstructed_captions", {}).items() if v}
            if rc:
                vid = f.stem.replace("Olly_s-Farm", "Olly's-Farm")
                out.setdefault(vid, {})[tuple(sorted(rc))] = rc
    return {v: list(g.values()) for v, g in out.items()}


def channel(vid: str) -> str:
    return re.match(r"^(.*?)_\d+", vid.replace("Olly's", "Olly_s")).group(1)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > 0, x / np.where(n > 0, n, 1), 0)


def fill(index: np.ndarray, tg: list[int], vecs: np.ndarray) -> np.ndarray:
    out = index.copy()
    out[tg] = vecs
    return out


def boundary_fills(index: np.ndarray, tg: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-boundary repeat and positional LERP between the two boundaries."""
    L, R = tg[0] - 1, tg[-1] + 1
    has_l, has_r = L >= 0, R < len(index)
    rep, lerp = [], []
    for k, t in enumerate(tg):
        if has_l and has_r:
            near = L if (t - L) <= (R - t) else R
            a = (k + 1) / (len(tg) + 1)
            rep.append(index[near])
            lerp.append((1 - a) * index[L] + a * index[R])
        else:
            b = index[L] if has_l else index[R]
            rep.append(b)
            lerp.append(b)
    return unit(rep), unit(lerp)


def cycle(texts: list[str], n: int) -> list[str]:
    return [texts[j % len(texts)] for j in range(n)]


def main():
    rng = np.random.default_rng(2025)
    # SigLIP 2 text: the stored frames are SigLIP 2 (timm v2_webli). SigLIP 1 text made the frame arms meaningless.
    mpnet, sig = LocalEmbedder("all-mpnet-base-v2"), SiglipTextEmbedder("google/siglip2-base-patch16-224")
    same_video = {}
    for run in SAME_VIDEO_RUNS:
        for vid, recs in load_recon(run).items():
            same_video.setdefault(vid, []).extend(recs)

    items = []
    for split, run in EVIDENCE_RUNS.items():
        recon = {v: r[0] for v, r in load_recon(run, max_w=60).items()}
        cap_dir = Path(f"datasets/wildQA/captions__wild{4 if split == 'dev' else 5}")
        qas = load_wildqa_dataset(Path(f"datasets/wildQA/{split}.json"), filter_scene_only=True, max_duration=60.0,
                                  single_evidence_only=True, max_evidence_duration=15.0)
        for qi, qa in enumerate(qas):
            vid = qa.video_id.replace("Olly_s-Farm", "Olly's-Farm")
            tg = sorted(qa.evidence_indices(max_duration=60))
            cf = cap_dir / f"{qa.video_id}.json"
            vf = VIS_DIR / f"{qa.video_id.replace(chr(39), '_')}.npy"
            if not tg or vid not in recon or not cf.exists() or not vf.exists():
                continue
            caps = [c["caption"] for c in json.load(open(cf))["captions"][:60]]
            rt = [recon[vid].get(t, "") for t in tg]
            frames = np.load(vf)[:60]
            if len(caps) != 60 or len(frames) != 60 or not all(rt):
                continue
            # same-video control: a reconstructed gap at least 2 s away from the evidence span
            far = [r for r in same_video.get(vid, []) if min(abs(t - e) for t in r for e in tg) >= 2]
            sv = None
            if far:
                best = min(far, key=lambda r: abs(len(r) - len(tg)))
                sv = cycle([best[t] for t in sorted(best)], len(tg))
            items.append(dict(split=split, qid=f"{split}_{qi}", vid=vid, file_vid=qa.video_id, domain=qa.domain,
                              q=qa.question, qtype=",".join(qa.question_type), tg=tg, caps=caps,
                              recon=rt, same_video=sv, frames=unit(frames)))
    print(f"{len(items)} questions, {len({i['vid'] for i in items})} videos, "
          f"{len({channel(i['vid']) for i in items})} channels; same-video control for "
          f"{sum(i['same_video'] is not None for i in items)}")

    by_domain: dict[str, list[int]] = {}
    for k, it in enumerate(items):
        by_domain.setdefault(it["domain"], []).append(k)

    rows = []
    for k, it in enumerate(items):
        tg, n = it["tg"], len(it["tg"])
        others = [j for j in by_domain[it["domain"]] if items[j]["vid"] != it["vid"]]
        other = items[others[rng.integers(len(others))]]
        row = dict(split=it["split"], qid=it["qid"], vid=it["vid"], chan=channel(it["vid"]), domain=it["domain"],
                   qtype=it["qtype"], n=n)

        def rank(q, index):
            return calculate_evidence_retrieval_metrics(q, index, tg)["rank"]

        for name, emb in [("mp", mpnet), ("sg", sig)]:
            idx = unit(emb.get_embeddings(f"{it['file_vid']}_dense_caps", it["caps"]))
            q = unit(emb.get_embeddings(f"{it['file_vid']}_q_{it['qid']}", [it["q"]]))[0]
            rep, lerp = boundary_fills(idx, tg)
            row[f"{name}_oracle"] = rank(q, idx)
            row[f"{name}_masked"] = rank(q, fill(idx, tg, np.zeros((n, idx.shape[1]))))
            row[f"{name}_repeat"] = rank(q, fill(idx, tg, rep))
            row[f"{name}_lerp"] = rank(q, fill(idx, tg, lerp))
            row[f"{name}_recon"] = rank(q, fill(idx, tg, unit(emb.get_embeddings("e1", it["recon"]))))
            row[f"{name}_recon_other_video"] = rank(q, fill(idx, tg, unit(emb.get_embeddings("e1", cycle(other["recon"], n)))))
            if it["same_video"] is not None:
                row[f"{name}_recon_same_video"] = rank(q, fill(idx, tg, unit(emb.get_embeddings("e1", it["same_video"]))))
            if name == "sg":
                V = it["frames"]
                vrep, vlerp = boundary_fills(V, tg)
                row["vis_oracle"] = rank(q, V)
                row["vis_masked"] = rank(q, fill(V, tg, np.zeros((n, V.shape[1]))))
                row["vis_repeat"] = rank(q, fill(V, tg, vrep))
                row["vis_lerp"] = rank(q, fill(V, tg, vlerp))
                row["vis_other_video"] = rank(q, fill(V, tg, other["frames"][tg]))
        rows.append(row)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "e1_evidence_retrieval_rows.csv", index=False)

    # ---- channel-clustered summary ----
    rr = {c: 1.0 / df[c] for c in df.columns if c.startswith(("mp_", "sg_", "vis_"))}
    chans = df.chan.values
    uniq = np.unique(chans)
    members = [np.where(chans == c)[0] for c in uniq]
    boots = [np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))]) for _ in range(B)]

    def stat(fn):
        point = fn(np.arange(len(df)))
        vals = np.array([fn(b) for b in boots])
        return point, np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)

    def mrr(col):
        return lambda ix: np.nanmean(rr[col].values[ix])

    def recovery(arm, oracle, masked):
        return lambda ix: (mrr(arm)(ix) - mrr(masked)(ix)) / (mrr(oracle)(ix) - mrr(masked)(ix))

    def diff(a, b):
        return lambda ix: np.nanmean(rr[a].values[ix] - rr[b].values[ix])

    summ = []
    for fam, arms in [("mp", ["oracle", "masked", "repeat", "lerp", "recon", "recon_other_video", "recon_same_video"]),
                      ("sg", ["oracle", "masked", "repeat", "lerp", "recon", "recon_other_video", "recon_same_video"]),
                      ("vis", ["oracle", "masked", "repeat", "lerp", "other_video"])]:
        for a in arms:
            col = f"{fam}_{a}"
            if col not in rr:
                continue
            p, lo, hi = stat(mrr(col))
            rp, rlo, rhi = stat(recovery(col, f"{fam}_oracle", f"{fam}_masked"))
            summ.append(dict(space=fam, arm=a, n=int(rr[col].notna().sum()), mrr=p, mrr_lo=lo, mrr_hi=hi,
                             recovery=rp, rec_lo=rlo, rec_hi=rhi))
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "e1_evidence_retrieval_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))

    print("\nPaired MRR differences (cluster-bootstrap 95% CI):")
    for a, b in [("mp_recon", "mp_recon_other_video"), ("mp_recon", "mp_recon_same_video"), ("mp_recon", "mp_repeat"),
                 ("mp_recon", "mp_lerp"), ("sg_recon", "sg_recon_other_video"), ("sg_recon", "sg_recon_same_video"),
                 ("vis_repeat", "vis_other_video"), ("vis_lerp", "vis_other_video"),
                 ("sg_recon", "vis_repeat"), ("sg_recon", "vis_lerp")]:
        p, lo, hi = stat(diff(a, b))
        print(f"  {a:>22} - {b:<24} {p:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
