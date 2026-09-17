#!/usr/bin/env python3
"""
scripts/build_unified_master_and_plots.py

Builds unified master benchmark tables and publication-quality figures
comparing Generative Llama-3.1-8B against standardized Caption Vector
and SigLIP Video Vector baselines across Wild4 and Wild5.
"""

import os
import re
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import diskcache
import matplotlib.pyplot as plt
import seaborn as sns
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from evaluations.eval_vectors import calculate_retrieval_metrics

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9,
    'axes.labelsize': 9.5,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.0,
    'figure.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,
})

def load_categories():
    cat_file = REPO_ROOT / "local" / "categories_wild2_vs_wild4.json"
    known = {}
    if cat_file.exists():
        with open(cat_file) as f:
            d = json.load(f)
        for k, v in d.items():
            if isinstance(v, dict):
                known[k] = v.get("ground_truth", "Unknown")
            else:
                known[k] = str(v)
    return known

KNOWN_CATS = load_categories()

def get_category(vid: str) -> str:
    if vid in KNOWN_CATS and KNOWN_CATS[vid] != "Unknown":
        c = KNOWN_CATS[vid]
        if c == "Geography": return "Nature & Scenery"
        if c == "Human Survival": return "Survival"
        return c
    v = vid.lower()
    if any(k in v for k in ["military", "army", "defense", "air-force", "navy", "war", "soldier", "gung-ho", "sandboxx", "aiirsource"]):
        return "Military"
    elif any(k in v for k in ["farm", "agriculture", "tractor", "harvest", "suscovich", "crop", "cattle"]):
        return "Farming"
    elif any(k in v for k in ["survival", "primitive", "bushcraft", "robinet", "instinct", "zuber", "gaillard", "wilderness", "craft"]):
        return "Survival"
    elif any(k in v for k in ["disaster", "tornado", "storm", "hurricane", "tsunami", "flood", "earthquake", "climate", "skip-talbot", "dan-robinson", "weathershot"]):
        return "Natural Disaster"
    elif any(k in v for k in ["relaxation", "nature", "earth", "treadmill", "scenery", "landscape", "amazon", "safari", "explorer", "flying-dutchman", "virtual-running"]):
        return "Nature & Scenery"
    elif any(k in v for k in ["vehicle", "aviation", "airplane", "chase", "train", "flight", "car", "speed", "hinshaw"]):
        return "Action & Vehicle"
    return "Other"

def parse_masked_w_i(masked_str):
    try:
        inds = json.loads(masked_str)
        if not inds: return None, None
        w = len(inds)
        min_idx, max_idx = min(inds), max(inds)
        if min_idx <= 2:
            i_pos = 0
        elif max_idx >= 57:
            i_pos = 59
        else:
            i_pos = 29
        return w, i_pos
    except:
        return None, None

def parse_folder_params(folder_name):
    m = re.search(r"w=(\d+),\s*i=(\d+)", folder_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def load_gt_dict(dataset_name: str) -> dict[str, list[str]]:
    gt_dir = REPO_ROOT / f"datasets/wildQA/captions__{dataset_name.lower()}"
    gt_map = {}
    for p in gt_dir.glob("*.json"):
        if p.name == "categories.json": continue
        try:
            with open(p) as f:
                d = json.load(f)
            clips = d.get("captions") or d.get("clips") or []
            gt_map[p.stem] = [c["caption"] for c in clips if "caption" in c]
        except Exception:
            pass
    return gt_map

def main():
    print("Collecting records across all benchmarks and modalities...")
    cache = diskcache.Cache(str(REPO_ROOT / "disk_cache/local_all-mpnet-base-v2"))
    gt_wild4 = load_gt_dict("wild4")
    gt_wild5 = load_gt_dict("wild5")
    records = []

    # -------------------------------------------------------------
    # 1. Vector Baselines (Wild4 & Wild5)
    # -------------------------------------------------------------
    baseline_files = [
        ("Wild4", "Vector_Caption", "Caption_MeanClosest", REPO_ROOT / "results/recon/wild4_sim_vec/wild4_sim_vec.csv"),
        ("Wild4", "Vector_Video", "Visual_SigLIP_MeanClosest", REPO_ROOT / "results/recon/wild4_siglip_sim_vec_vid/wild4_siglip_sim_vec_vid.csv"),
        ("Wild5", "Vector_Caption", "Caption_MeanClosest", REPO_ROOT / "results/recon/wild5_sim_vec/wild5_sim_vec.csv"),
        ("Wild5", "Vector_Video", "Visual_SigLIP_MeanClosest", REPO_ROOT / "results/recon/wild5_siglip_sim_vec_vid/wild5_siglip_sim_vec_vid.csv"),
    ]

    for dataset_name, m_type, m_name, path in baseline_files:
        if not path.exists():
            print(f"Warning: {path} not found!")
            continue
        print(f"Parsing {dataset_name} {m_name} from {path.name}...")
        df = pd.read_csv(path)
        for strat in ["MeanClosestVectors", "RepeatClosestVector"]:
            sub = df[df["recon_strategy"] == strat]
            strat_label = m_name if "Mean" in strat else m_name.replace("MeanClosest", "RepeatClosest")
            for _, row in sub.iterrows():
                vid = row["video_id"]
                w, i_pos = parse_masked_w_i(str(row["masked"]))
                if w is None: continue
                records.append({
                    "dataset": dataset_name,
                    "video_id": vid,
                    "category": get_category(vid),
                    "method": strat_label,
                    "method_family": m_type,
                    "width": w,
                    "index": i_pos,
                    "mrr": row.get("mrr_mean"),
                    "recall_at_1": row.get("recall_at_1_mean"),
                    "recall_at_5": row.get("recall_at_5_mean"),
                    "cos_sim": row.get("cos_sim_mean"),
                    "cos_sim_residual": row.get("cos_sim_residual_mean"),
                    "mean_rank": row.get("mean_rank_mean"),
                })

    # -------------------------------------------------------------
    # 2. Llama-3.1-8B on Wild4 (Harmonized to pool_scope: video)
    # -------------------------------------------------------------
    llama_w4_paths = [
        Path("/home/yoavh/.cache/huggingface/hub/datasets--Y3--dense_video_captions/snapshots/70175ef8c0aa921a322a023057b4176c9381aae4/reconstruction/wild4_llama_w3_window_v3"),
        REPO_ROOT / "results/reconstruction/wild4_llama_w6"
    ]
    print("Parsing Wild4 Llama-3.1-8B results...")
    for root_p in llama_w4_paths:
        if not root_p.exists(): continue
        for s_dir in root_p.iterdir():
            if not s_dir.is_dir(): continue
            w, i_pos = parse_folder_params(s_dir.name)
            if w is None: continue
            for jf in s_dir.glob("*.json"):
                if jf.name.startswith("skip__") or jf.name.endswith("metadata.json"): continue
                vid = jf.stem
                try:
                    with open(jf) as fp:
                        d = json.load(fp)
                    metrics = d.get("metrics", {})
                    cos_list = metrics.get("cos_sim", [])
                    recon = d.get("reconstructed_captions", {})
                    
                    # Harmonize retrieval metrics to full-video pool (scope: video)
                    mrr_val = metrics.get("mrr")
                    r1_val = metrics.get("recall_at_1")
                    r5_val = metrics.get("recall_at_5")
                    mean_rank_val = metrics.get("mean_rank")

                    if vid in gt_wild4 and recon:
                        gt_caps = gt_wild4[vid]
                        sorted_inds = sorted([int(k) for k in recon.keys()])
                        if all(recon[str(i)] in cache for i in sorted_inds) and all(c in cache for c in gt_caps):
                            pred_vecs = np.array([cache[recon[str(i)]] for i in sorted_inds])
                            true_vecs = np.array([cache[gt_caps[i]] for i in sorted_inds])
                            dist_pool = np.array([cache[c] for c in gt_caps])
                            res = calculate_retrieval_metrics(
                                reconstructed_vectors=pred_vecs,
                                ground_truth_vectors=true_vecs,
                                distractor_pool=dist_pool,
                                gt_indices_in_pool=sorted_inds
                            )
                            mrr_val = res["mrr"]
                            r1_val = res["recall_at_1"]
                            r5_val = res["recall_at_5"]
                            mean_rank_val = res["mean_rank"]

                    records.append({
                        "dataset": "Wild4",
                        "video_id": vid,
                        "category": get_category(vid),
                        "method": "Llama-3.1-8B",
                        "method_family": "SLM_Text",
                        "width": w,
                        "index": i_pos,
                        "mrr": mrr_val,
                        "recall_at_1": r1_val,
                        "recall_at_5": r5_val,
                        "cos_sim": float(np.mean(cos_list)) if cos_list else None,
                        "cos_sim_residual": float(np.mean(metrics.get("cos_sim_residual", []))) if "cos_sim_residual" in metrics else None,
                        "mean_rank": mean_rank_val,
                    })
                except Exception:
                    pass

    # -------------------------------------------------------------
    # 3. Llama-3.1-8B on Wild5 (W=3 & W=6)
    # -------------------------------------------------------------
    llama_w5_path = Path("/home/yoavh/.cache/huggingface/hub/datasets--Y3--dense_video_captions/snapshots/62959be22d810cfdca58265c05f8ed0a77cdebc0/reconstruction/wild5_llama_w3_w6")
    if llama_w5_path.exists():
        print("Parsing Wild5 Llama-3.1-8B results...")
        for s_dir in llama_w5_path.iterdir():
            if not s_dir.is_dir(): continue
            w, i_pos = parse_folder_params(s_dir.name)
            if w is None: continue
            for jf in s_dir.glob("*.json"):
                if jf.name.startswith("skip__") or jf.name.endswith("metadata.json"): continue
                vid = jf.stem
                try:
                    with open(jf) as fp:
                        d = json.load(fp)
                    metrics = d.get("metrics", {})
                    cos_list = metrics.get("cos_sim", [])
                    records.append({
                        "dataset": "Wild5",
                        "video_id": vid,
                        "category": get_category(vid),
                        "method": "Llama-3.1-8B",
                        "method_family": "SLM_Text",
                        "width": w,
                        "index": i_pos,
                        "mrr": metrics.get("mrr"),
                        "recall_at_1": metrics.get("recall_at_1"),
                        "recall_at_5": metrics.get("recall_at_5"),
                        "cos_sim": float(np.mean(cos_list)) if cos_list else None,
                        "cos_sim_residual": float(np.mean(metrics.get("cos_sim_residual", []))) if "cos_sim_residual" in metrics else None,
                        "mean_rank": metrics.get("mean_rank"),
                    })
                except Exception:
                    pass

    master_df = pd.DataFrame(records)
    print(f"\nTotal master records compiled: {len(master_df)}")
    
    # Save master dataset
    master_csv = REPO_ROOT / "results" / "unified_benchmark_master.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"Saved master dataset to: {master_csv}")

    # Aggregated Summary
    summary_df = master_df.groupby(["dataset", "method_family", "method", "width"])[
        ["mrr", "recall_at_1", "recall_at_5", "cos_sim", "cos_sim_residual", "mean_rank"]
    ].agg(["mean", "std", "count"]).reset_index()
    
    summary_csv = REPO_ROOT / "results" / "unified_benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary to: {summary_csv}")

    # Display clean table
    print("\n" + "="*80)
    print("UNIFIED BENCHMARK SUMMARY TABLE (FULL 60-CLIP VIDEO DISTRACTOR POOL)")
    print("="*80)
    clean_summary = master_df.groupby(["dataset", "method", "width"])[
        ["mrr", "recall_at_1", "recall_at_5", "cos_sim", "cos_sim_residual"]
    ].mean().reset_index()
    print(clean_summary.to_string(index=False))

    # -------------------------------------------------------------
    # 4. Generate Publication Figures
    # -------------------------------------------------------------
    plot_dirs = [
        REPO_ROOT / "results" / "plots",
        REPO_ROOT / "docs" / "paper" / "LaTeX" / "figures"
    ]
    for pd_dir in plot_dirs:
        pd_dir.mkdir(parents=True, exist_ok=True)

    palette = {
        "Llama-3.1-8B": "#1f77b4",              # Navy Blue
        "Caption_MeanClosest": "#9467bd",       # Purple
        "Caption_RepeatClosest": "#c5b0d5",     # Light Purple
        "Visual_SigLIP_MeanClosest": "#2ca02c", # Forest Green
        "Visual_SigLIP_RepeatClosest": "#98df8a"# Light Green
    }

    # FIGURE 1: MRR vs Gap Width (Side-by-side Wild4 and Wild5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    
    for ax, dset, title in [(ax1, "Wild4", "Wild4 Benchmark (100 Videos)"), 
                            (ax2, "Wild5", "Wild5 Benchmark (235 Videos)")]:
        sub_d = master_df[master_df["dataset"] == dset]
        methods_to_plot = ["Llama-3.1-8B", "Visual_SigLIP_MeanClosest", "Caption_MeanClosest"]
            
        for m in methods_to_plot:
            m_sub = sub_d[sub_d["method"] == m]
            if len(m_sub) == 0: continue
            stats_w = m_sub.groupby("width")["mrr"].agg(["mean", "sem"]).reset_index()
            ci95 = stats_w["sem"] * 1.96
            ax.errorbar(
                stats_w["width"], stats_w["mean"], yerr=ci95,
                label=m.replace("_", " "), color=palette.get(m, "#333333"),
                marker='o', markersize=5, linewidth=1.8, capsize=3, capthick=1.0
            )
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Masked Gap Width W (seconds)")
        ax.set_xticks([3, 6, 9, 12])
        ax.grid(True)
    
    ax1.set_ylabel("Reconstruction MRR (Pool Size = 60)")
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    fig.suptitle("Retrieval MRR vs. Gap Width Across Modalities", fontsize=11, fontweight="bold", y=1.02)
    
    for p_dir in plot_dirs:
        fig.savefig(p_dir / "fig_mrr_vs_width_wild4_wild5.png")
    plt.close(fig)
    print("Saved fig_mrr_vs_width_wild4_wild5.png")

    # FIGURE 2: Recall@1 vs Gap Width
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, dset, title in [(ax1, "Wild4", "Wild4 Benchmark (100 Videos)"), 
                            (ax2, "Wild5", "Wild5 Benchmark (235 Videos)")]:
        sub_d = master_df[master_df["dataset"] == dset]
        methods_to_plot = ["Llama-3.1-8B", "Visual_SigLIP_MeanClosest", "Caption_MeanClosest"]
            
        for m in methods_to_plot:
            m_sub = sub_d[sub_d["method"] == m]
            if len(m_sub) == 0: continue
            stats_w = m_sub.groupby("width")["recall_at_1"].agg(["mean", "sem"]).reset_index()
            ci95 = stats_w["sem"] * 1.96
            ax.errorbar(
                stats_w["width"], stats_w["mean"], yerr=ci95,
                label=m.replace("_", " "), color=palette.get(m, "#333333"),
                marker='s', markersize=5, linewidth=1.8, capsize=3, capthick=1.0
            )
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Masked Gap Width W (seconds)")
        ax.set_xticks([3, 6, 9, 12])
        ax.grid(True)
        
    ax1.set_ylabel("Recall@1")
    ax1.axhline(1/60, color='#888888', linestyle=':', linewidth=1.2, label='Chance (1/60 ≈ 0.017)')
    ax2.axhline(1/60, color='#888888', linestyle=':', linewidth=1.2, label='Chance (1/60 ≈ 0.017)')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    fig.suptitle("Recall@1 vs. Gap Width Across Modalities", fontsize=11, fontweight="bold", y=1.02)
    
    for p_dir in plot_dirs:
        fig.savefig(p_dir / "fig_recall1_vs_width_wild4_wild5.png")
    plt.close(fig)
    print("Saved fig_recall1_vs_width_wild4_wild5.png")

    # FIGURE 3: Grouped Bar Chart at W=3 and W=6
    fig, (ax_mrr, ax_r1) = plt.subplots(1, 2, figsize=(8.8, 3.6))
    bar_df = master_df[master_df["width"].isin([3, 6]) & master_df["method"].isin(["Llama-3.1-8B", "Visual_SigLIP_MeanClosest", "Caption_MeanClosest"])].copy()
    bar_df["Condition"] = bar_df["dataset"] + " (W=" + bar_df["width"].astype(str) + ")"
    bar_df["CleanMethod"] = bar_df["method"].str.replace("_", " ")
    
    order = ["Wild4 (W=3)", "Wild4 (W=6)", "Wild5 (W=3)", "Wild5 (W=6)"]
    clean_palette = {k.replace("_", " "): v for k, v in palette.items()}
    
    sns.barplot(
        data=bar_df, x="Condition", y="mrr", hue="CleanMethod",
        order=order, palette=clean_palette, ax=ax_mrr, errorbar=('ci', 95), capsize=0.08, edgecolor='#333333', linewidth=0.8
    )
    ax_mrr.set_title("Mean Reciprocal Rank (MRR)", fontweight="bold", fontsize=10)
    ax_mrr.set_ylabel("MRR")
    ax_mrr.set_xlabel("")
    ax_mrr.grid(True, axis='y')
    ax_mrr.legend().remove()
    
    sns.barplot(
        data=bar_df, x="Condition", y="recall_at_1", hue="CleanMethod",
        order=order, palette=clean_palette, ax=ax_r1, errorbar=('ci', 95), capsize=0.08, edgecolor='#333333', linewidth=0.8
    )
    ax_r1.axhline(1/60, color='#888888', linestyle=':', label='Chance Level')
    ax_r1.set_title("Recall@1 (Top-1 Identification)", fontweight="bold", fontsize=10)
    ax_r1.set_ylabel("Recall@1")
    ax_r1.set_xlabel("")
    ax_r1.grid(True, axis='y')
    ax_r1.legend(title="", frameon=True, facecolor='white', framealpha=0.9)
    
    fig.suptitle("Performance Comparison across Datasets & Gap Widths", fontsize=11, fontweight="bold", y=1.02)
    
    for p_dir in plot_dirs:
        fig.savefig(p_dir / "fig_modality_comparison_bar.png")
    plt.close(fig)
    print("Saved fig_modality_comparison_bar.png")

    # FIGURE 4: Semantic Category Performance Breakdown (Wild5 W=3)
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    cat_df = master_df[(master_df["dataset"] == "Wild5") & (master_df["width"] == 3) & (master_df["method"].isin(["Llama-3.1-8B", "Visual_SigLIP_MeanClosest", "Caption_MeanClosest"]))].copy()
    valid_cats = ["Farming", "Survival", "Military", "Nature & Scenery", "Natural Disaster"]
    cat_df = cat_df[cat_df["category"].isin(valid_cats)]
    cat_df["CleanMethod"] = cat_df["method"].str.replace("_", " ")
    
    sns.barplot(
        data=cat_df, x="category", y="recall_at_1", hue="CleanMethod",
        order=valid_cats, palette=clean_palette, ax=ax, errorbar=('ci', 95), capsize=0.08, edgecolor='#333333', linewidth=0.8
    )
    ax.axhline(1/60, color='#888888', linestyle=':', label='Chance Level (1/60)')
    ax.set_title("Wild5 (W=3): Top-1 Identification (Recall@1) by Domain Category", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Recall@1")
    ax.set_xlabel("Video Domain / Category")
    ax.grid(True, axis='y')
    ax.legend(title="", frameon=True, facecolor='white', framealpha=0.9)
    
    for p_dir in plot_dirs:
        fig.savefig(p_dir / "fig_category_breakdown_wild5.png")
    plt.close(fig)
    print("Saved fig_category_breakdown_wild5.png")

if __name__ == "__main__":
    main()
