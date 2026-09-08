#!/usr/bin/env python
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
WILD4_DIR = PROJECT_ROOT / "datasets" / "wildQA" / "captions__wild4"
V1_DIR = PROJECT_ROOT / "results" / "recon" / "wild4_sim_text"
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

    # Calculate average ground-truth caption word count
    all_gt_lens = [compute_word_length(cap) for caps in gt_map.values() for cap in caps]
    print(f"Ground Truth: Mean word length = {np.mean(all_gt_lens):.2f} words (std={np.std(all_gt_lens):.2f})")

    # Assess V1 vs V2 results
    print("\n" + "="*80)
    print("=== QUANTITATIVE COMPARISON: V1 (OLD PROMPT) vs V2 (NEW PROMPT) ===")
    print("="*80)

    # Let's inspect t=0.1, t=0.6, t=1.0 at i=0, i=29, i=59
    configs_to_check = [
        ("t=0.1", "phi-3__t=0.1_rp=1.2", "phi-3_v2__t=0.1_rp=1.2"),
        ("t=0.6", "phi-3__t=0.6_rp=1.2", "phi-3_v2__t=0.6_rp=1.2"),
        ("t=1.0", "phi-3__t=1.0_rp=1.2", "phi-3_v2__t=1.0_rp=1.2"),
    ]

    for label, v1_prefix, v2_prefix in configs_to_check:
        print(f"\n--- Strategy {label} (w=3) ---")
        for i_pos in [0, 29, 59]:
            v1_pattern = f"{v1_prefix}__fixed_fill(w=3, i={i_pos})"
            v2_pattern = f"{v2_prefix}__fixed_fill(w=3, i={i_pos})"

            v1_files = list(V1_DIR.glob(f"{v1_pattern}/*.json"))
            v2_files = list(V2_DIR.glob(f"{v2_pattern}/*.json"))

            # Length & Style stats
            def get_stats(file_list):
                lens = []
                flowery_cnt = 0
                cliche_cnt = 0
                total_caps = 0
                for f in file_list:
                    with open(f) as fp:
                        d = json.load(fp)
                    recons = d.get("reconstructed_captions", {})
                    for idx_str, cap in recons.items():
                        if cap:
                            lens.append(compute_word_length(cap))
                            flow, clich = check_flowery_language(cap)
                            if flow: flowery_cnt += 1
                            if clich: cliche_cnt += 1
                            total_caps += 1
                return {
                    "count": total_caps,
                    "mean_len": np.mean(lens) if lens else 0,
                    "std_len": np.std(lens) if lens else 0,
                    "flowery_pct": (flowery_cnt / total_caps * 100) if total_caps else 0,
                    "cliche_pct": (cliche_cnt / total_caps * 100) if total_caps else 0
                }

            v1_s = get_stats(v1_files)
            v2_s = get_stats(v2_files)

            print(f"  Pos i={i_pos:2d} | V1 (Old): Mean Words = {v1_s['mean_len']:.1f} ± {v1_s['std_len']:.1f} | Flowery = {v1_s['flowery_pct']:.1f}% | Clichés = {v1_s['cliche_pct']:.1f}%")
            print(f"           | V2 (New): Mean Words = {v2_s['mean_len']:.1f} ± {v2_s['std_len']:.1f} | Flowery = {v2_s['flowery_pct']:.1f}% | Clichés = {v2_s['cliche_pct']:.1f}%")

    # Side-by-side Qualitative Comparison
    print("\n" + "="*80)
    print("=== QUALITATIVE SIDE-BY-SIDE EXAMPLES ===")
    print("="*80)

    sample_videos = [
        "MilitaryNotes_8-clip-0",
        "4k-Relaxation_12-clip-6",
        "Climate-Change_6-clip-7",
        "BC-Bushcraft_10-clip-8",
        "John-Suscovich_3-manual"
    ]

    for vid in sample_videos:
        if vid not in gt_map: continue
        print(f"\n🎥 Video: {vid}")
        gt_caps = gt_map[vid]

        # Check i=0 (opening scene)
        print(f"  ▶ Opening Scene (i=0, w=3):")
        print(f"    - Ground Truth [0-2]:")
        for k in range(min(3, len(gt_caps))):
            print(f"        [{k:02d}] {gt_caps[k]}")

        # V1 i=0
        v1_f = V1_DIR / "phi-3__t=1.0_rp=1.2__fixed_fill(w=3, i=0)" / f"{vid}.json"
        if v1_f.exists():
            with open(v1_f) as fp:
                d1 = json.load(fp)
            print(f"    - V1 Reconstructed [0-2] (Old Prompt):")
            for k, txt in sorted(d1.get("reconstructed_captions", {}).items()):
                print(f"        [{int(k):02d}] {txt}")

        # V2 i=0
        v2_f = V2_DIR / "phi-3_v2__t=1.0_rp=1.2__fixed_fill(w=3, i=0)" / f"{vid}.json"
        if v2_f.exists():
            with open(v2_f) as fp:
                d2 = json.load(fp)
            print(f"    - V2 Reconstructed [0-2] (New Prompt):")
            for k, txt in sorted(d2.get("reconstructed_captions", {}).items()):
                print(f"        [{int(k):02d}] {txt}")

        # Context around clip 3
        if len(gt_caps) > 3:
            print(f"    - Visible Unmasked Context [03]: \"{gt_caps[3]}\"")

        # Check i=29 (middle bridge scene)
        if len(gt_caps) >= 32:
            print(f"\n  ▶ Middle Bridge (i=29, w=3):")
            print(f"    - Preceding Context [28]: \"{gt_caps[28]}\"")
            print(f"    - Ground Truth [29-31]:")
            for k in range(29, 32):
                print(f"        [{k:02d}] {gt_caps[k]}")

            v1_f29 = V1_DIR / "phi-3__t=1.0_rp=1.2__fixed_fill(w=3, i=29)" / f"{vid}.json"
            if v1_f29.exists():
                with open(v1_f29) as fp:
                    d1_29 = json.load(fp)
                print(f"    - V1 Reconstructed [29-31] (Old Prompt):")
                for k, txt in sorted(d1_29.get("reconstructed_captions", {}).items()):
                    print(f"        [{int(k):02d}] {txt}")

            v2_f29 = V2_DIR / "phi-3_v2__t=1.0_rp=1.2__fixed_fill(w=3, i=29)" / f"{vid}.json"
            if v2_f29.exists():
                with open(v2_f29) as fp:
                    d2_29 = json.load(fp)
                print(f"    - V2 Reconstructed [29-31] (New Prompt):")
                for k, txt in sorted(d2_29.get("reconstructed_captions", {}).items()):
                    print(f"        [{int(k):02d}] {txt}")
            print(f"    - Following Context [32]: \"{gt_caps[32]}\"")

if __name__ == "__main__":
    main()
