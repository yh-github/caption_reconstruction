#!/usr/bin/env python
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
WILD4_DIR = PROJECT_ROOT / "datasets" / "wildQA" / "captions__wild4"
V2_DIR = PROJECT_ROOT / "results" / "recon" / "wild4_sim_text_v2_direct" / "reconstruction" / "wild4_sim_text_v2"

def load_wild4_gt():
    gt = {}
    for p in WILD4_DIR.glob("*.json"):
        if p.name == "categories.json": continue
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        gt[p.stem] = [c["caption"] for c in d.get("captions", [])]
    return gt

def compute_word_length(text):
    return len(text.strip().split()) if text else 0

def check_flowery_language(text):
    flowery_keywords = [
        "breathtaking", "cascades", "cascading", "dappled", "dappling", "lush",
        "tranquil", "bathed", "golden hour", "serene", "majestic", "unfolds",
        "tapestry", "glistening", "hues", "symphony", "peaceful pastoral"
    ]
    cliche_starters = [
        "the video begins", "the video opens", "the scene opens",
        "a stunning view", "a breathtaking view"
    ]
    t_lower = text.lower()
    has_flowery = any(w in t_lower for w in flowery_keywords)
    has_cliche = any(t_lower.startswith(c) for c in cliche_starters)
    return has_flowery, has_cliche

def main():
    gt_map = load_wild4_gt()
    print(f"Loaded ground truth for {len(gt_map)} Wild4 videos.")

    # Parse all experiment directories
    strategy_dirs = [d for d in V2_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(strategy_dirs)} configuration folders.")

    # Structure: results[strategy_name][w][i] -> list of metrics
    agg_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "lens": [],
        "flowery_cnt": 0,
        "cliche_cnt": 0,
        "total_caps": 0,
        "cos_sims": [],
        "mrrs": [],
        "r1s": [],
        "r5s": []
    })))

    for s_dir in strategy_dirs:
        folder_name = s_dir.name
        # Match pattern: (phi-3_v2__t=..._rp=...)__fixed_fill(w=..., i=...)
        m = re.match(r"(phi-3_v2__t=[\d\.]+_rp=[\d\.]+)__fixed_fill\(w=(\d+),\s*i=(\d+)\)", folder_name)
        if not m:
            continue
        strat_name, w_str, i_str = m.groups()
        w = int(w_str)
        i_pos = int(i_str)

        entry = agg_data[strat_name][w][i_pos]

        for jf in s_dir.glob("*.json"):
            try:
                with open(jf, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception:
                continue

            recons = data.get("reconstructed_captions", {})
            metrics = data.get("metrics", {})

            for idx_str, cap in recons.items():
                if cap and cap.strip():
                    entry["lens"].append(compute_word_length(cap))
                    flow, clich = check_flowery_language(cap)
                    if flow: entry["flowery_cnt"] += 1
                    if clich: entry["cliche_cnt"] += 1
                    entry["total_caps"] += 1

            if "cos_sim" in metrics and isinstance(metrics["cos_sim"], list):
                entry["cos_sims"].extend(metrics["cos_sim"])
            if "mrr" in metrics and metrics["mrr"] is not None:
                entry["mrrs"].append(metrics["mrr"])
            if "recall_at_1" in metrics and metrics["recall_at_1"] is not None:
                entry["r1s"].append(metrics["recall_at_1"])
            if "recall_at_5" in metrics and metrics["recall_at_5"] is not None:
                entry["r5s"].append(metrics["recall_at_5"])

    print("\n" + "="*100)
    print("=== SUMMARY METRICS ACROSS WINDOW SIZES (W=3, 6, 9, 12) ===")
    print("="*100)

    # Let's focus on the best performing strategy: phi-3_v2__t=1.0_rp=1.0 vs others
    strats = sorted(agg_data.keys())
    for strat in strats:
        print(f"\n Strategy: {strat}")
        print(f"  {'W':<4} | {'Pos':<5} | {'Mean Words':<12} | {'Flowery':<8} | {'Cliché':<8} | {'Mean CosSim':<12} | {'MRR':<8} | {'R@1':<8} | {'R@5':<8}")
        print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for w in [3, 6, 9, 12]:
            for i_pos in [0, 29, 59]:
                if i_pos in agg_data[strat][w]:
                    e = agg_data[strat][w][i_pos]
                    if e["total_caps"] == 0: continue
                    mean_len = f"{np.mean(e['lens']):.1f}±{np.std(e['lens']):.1f}"
                    flow_pct = f"{(e['flowery_cnt']/e['total_caps']*100):.1f}%"
                    clich_pct = f"{(e['cliche_cnt']/e['total_caps']*100):.1f}%"
                    mean_cos = f"{np.mean(e['cos_sims']):.4f}" if e["cos_sims"] else "N/A"
                    mrr_val = f"{np.mean(e['mrrs']):.4f}" if e["mrrs"] else "N/A"
                    r1_val = f"{np.mean(e['r1s']):.4f}" if e["r1s"] else "N/A"
                    r5_val = f"{np.mean(e['r5s']):.4f}" if e["r5s"] else "N/A"
                    print(f"  W={w:<2} | i={i_pos:<3} | {mean_len:<12} | {flow_pct:<8} | {clich_pct:<8} | {mean_cos:<12} | {mrr_val:<8} | {r1_val:<8} | {r5_val:<8}")

    # Qualitative comparison across W=3, 6, 12 on sample videos
    print("\n" + "="*100)
    print("=== QUALITATIVE COMPARISON ACROSS WINDOW WIDTHS (W=3, 6, 12) ===")
    print("="*100)

    best_strat = "phi-3_v2__t=1.0_rp=1.0"
    sample_videos = ["BC-Bushcraft_10-clip-8", "MilitaryNotes_8-clip-0", "John-Suscovich_3-manual"]

    for vid in sample_videos:
        if vid not in gt_map: continue
        print(f"\n🎥 Video: {vid}")
        gt = gt_map[vid]

        for w in [3, 6, 12]:
            print(f"\n  ▶ Gap Width W={w} (Middle gap i=29):")
            # Load json for this video
            target_f = V2_DIR / f"{best_strat}__fixed_fill(w={w}, i=29)" / f"{vid}.json"
            if target_f.exists():
                with open(target_f) as fp:
                    d = json.load(fp)
                recons = d.get("reconstructed_captions", {})
                for idx_str in sorted(recons.keys(), key=lambda x: int(x)):
                    idx = int(idx_str)
                    gt_str = gt[idx] if idx < len(gt) else "[No GT]"
                    rec_str = recons[idx_str]
                    print(f"    [{idx:02d}] GT  : \"{gt_str}\"")
                    print(f"         REC : \"{rec_str}\"")

if __name__ == "__main__":
    main()
