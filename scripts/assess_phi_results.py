import json
import glob
import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
WILD_QA_DIR = PROJECT_ROOT / "datasets" / "wildQA"
RESULTS_DIR = PROJECT_ROOT / "results" / "recon"

def load_all_ground_truth():
    """Build map of video_id -> list of clip captions."""
    gt_map = {}
    for wild_dir in WILD_QA_DIR.glob("captions__*"):
        for json_file in wild_dir.glob("*.json"):
            video_id = json_file.stem
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    continue
            
            captions = []
            if isinstance(data, dict):
                if 'captions' in data and isinstance(data['captions'], list):
                    for item in data['captions']:
                        if isinstance(item, dict) and 'caption' in item:
                            captions.append(item['caption'])
                        elif isinstance(item, str):
                            captions.append(item)
                elif 'segments' in data and isinstance(data['segments'], list):
                    for seg in data['segments']:
                        if isinstance(seg, dict):
                            if 'key_moments' in seg and isinstance(seg['key_moments'], list):
                                for km in seg['key_moments']:
                                    if isinstance(km, dict) and 'caption' in km:
                                        captions.append(km['caption'])
                            elif 'segment_summary' in seg:
                                captions.append(seg['segment_summary'])
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'caption' in item:
                        captions.append(item['caption'])
            
            if captions:
                gt_map[video_id] = captions
    return gt_map

def find_phi3_results():
    """Find all phi-3 reconstruction JSON files and group by (window_size, temp, start_ind)."""
    all_files = [p for p in RESULTS_DIR.glob("**/*.json") if "phi-3" in str(p) or "phi" in str(p)]
    
    grouped = defaultdict(list)
    for p in all_files:
        parent_dir = p.parent.name
        match = re.search(r"phi-3__t=([\d\.]+)_rp=[\d\.]+__fixed_fill\(w=(\d+),\s*i=(\d+)\)", parent_dir)
        if match:
            temp = float(match.group(1))
            w = int(match.group(2))
            i = int(match.group(3))
            grouped[(w, temp, i)].append(p)
    
    return grouped

def analyze_qualitative_sample(gt_map, grouped_files, target_videos, target_w_list=[3, 6], target_t_list=[0.1, 0.6, 1.0, 1.5]):
    """Extract side-by-side reconstruction comparisons for target videos."""
    results = []
    
    for vid in target_videos:
        gt_caps = gt_map.get(vid, [])
        if not gt_caps:
            continue
        
        vid_summary = {
            "video_id": vid,
            "gt_total_clips": len(gt_caps),
            "gt_sample_masked": {},
            "reconstructions": {}
        }
        
        # Look for results for this video across (w, temp, i)
        for (w, temp, i), file_list in grouped_files.items():
            if w not in target_w_list or temp not in target_t_list or i != 0:
                continue
            
            target_file = None
            for f in file_list:
                if f.name == f"{vid}.json":
                    target_file = f
                    break
            
            if not target_file:
                continue
            
            with open(target_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            recon_caps = data.get("reconstructed_captions", {})
            metrics = data.get("metrics", {})
            
            key = (w, temp)
            if key not in vid_summary["reconstructions"]:
                vid_summary["reconstructions"][key] = {
                    "recon_caps": recon_caps,
                    "metrics": metrics,
                    "file": str(target_file.relative_to(PROJECT_ROOT))
                }
            
            # Record GT for the masked range (0 to w-1)
            if w not in vid_summary["gt_sample_masked"]:
                vid_summary["gt_sample_masked"][w] = gt_caps[:w]
        
        if vid_summary["reconstructions"]:
            results.append(vid_summary)
            
    return results

def compute_aggregate_metrics(grouped_files):
    """Aggregate metrics by (W, Temp)."""
    agg = defaultdict(lambda: defaultdict(list))
    
    for (w, temp, i), file_list in grouped_files.items():
        for f in file_list:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    d = json.load(fp)
                m = d.get("metrics", {})
                if not m:
                    continue
                
                if "mrr" in m and m["mrr"] is not None:
                    agg[(w, temp)]["mrr"].append(m["mrr"])
                if "recall_at_1" in m and m["recall_at_1"] is not None:
                    agg[(w, temp)]["r1"].append(m["recall_at_1"])
                if "recall_at_5" in m and m["recall_at_5"] is not None:
                    agg[(w, temp)]["r5"].append(m["recall_at_5"])
                if "cos_sim" in m and m["cos_sim"]:
                    mean_cos = float(np.mean(m["cos_sim"]))
                    agg[(w, temp)]["cos_sim"].append(mean_cos)
            except Exception:
                continue
                
    summary = {}
    for (w, temp), mdict in sorted(agg.items()):
        summary[(w, temp)] = {
            "count": len(mdict["mrr"]),
            "mrr": float(np.mean(mdict["mrr"])) if mdict["mrr"] else 0.0,
            "r1": float(np.mean(mdict["r1"])) if mdict["r1"] else 0.0,
            "r5": float(np.mean(mdict["r5"])) if mdict["r5"] else 0.0,
            "cos_sim": float(np.mean(mdict["cos_sim"])) if mdict["cos_sim"] else 0.0,
        }
    return summary

def generate_report(gt_map, grouped_files):
    target_vids = [
        "Survival-Instinct_9-clip-2",
        "Army-military-2018_8-clip-73",
        "Climate-Change_6-clip-7",
        "Olly's-Farm_1-clip-5",
        "4k-Relaxation_12-clip-6",
        "Chad-Zuber_10-clip-20",
        "Joe-Robinet_2-clip-5"
    ]
    
    qualitative = analyze_qualitative_sample(gt_map, grouped_files, target_vids)
    agg_metrics = compute_aggregate_metrics(grouped_files)
    
    print("=" * 80)
    print("PHI-3 SLM CAPTION RECONSTRUCTION ASSESSMENT REPORT")
    print("=" * 80)
    
    print("\n--- 1. QUANTITATIVE METRICS SUMMARY ---")
    print(f"{'W':<4} | {'Temp':<6} | {'Files':<6} | {'Cos Sim':<8} | {'MRR':<8} | {'R@1':<8} | {'R@5':<8}")
    print("-" * 65)
    for (w, temp), stats in sorted(agg_metrics.items()):
        if w in [3, 6, 9, 12]:
            print(f"{w:<4} | {temp:<6.1f} | {stats['count']:<6} | {stats['cos_sim']:<8.4f} | {stats['mrr']:<8.4f} | {stats['r1']:<8.4f} | {stats['r5']:<8.4f}")
            
    print("\n--- 2. QUALITATIVE CASE STUDIES ---")
    for sample in qualitative:
        vid = sample["video_id"]
        print(f"\n==================================================")
        print(f" VIDEO: {vid}")
        print(f"==================================================")
        
        for w in [3, 6]:
            gt_sub = sample["gt_sample_masked"].get(w, [])
            print(f"\n  [Window Size W={w}] (Ground Truth for masked indices 0..{w-1}):")
            for idx, gt_text in enumerate(gt_sub):
                print(f"    - Clip {idx} (GT): \"{gt_text}\"")
                
            for temp in [0.1, 0.6, 1.0, 1.5]:
                recon_info = sample["reconstructions"].get((w, temp))
                if not recon_info:
                    continue
                recon_caps = recon_info["recon_caps"]
                m = recon_info["metrics"]
                cos_list = m.get("cos_sim", [])
                mean_c = np.mean(cos_list) if cos_list else 0.0
                
                print(f"\n    > Phi-3 [t={temp:.1f}] (Mean CosSim={mean_c:.3f}):")
                for k_str in sorted(recon_caps.keys(), key=lambda x: int(x)):
                    k_int = int(k_str)
                    if k_int < w:
                        print(f"      - Clip {k_str}: \"{recon_caps[k_str]}\"")

if __name__ == "__main__":
    print("Loading ground truth maps...")
    gt_map = load_all_ground_truth()
    print(f"Loaded GT for {len(gt_map)} videos.")
    
    print("Scanning Phi-3 result files...")
    grouped = find_phi3_results()
    total_found = sum(len(v) for v in grouped.values())
    print(f"Found {total_found} result files across {len(grouped)} strategy configurations.")
    
    generate_report(gt_map, grouped)
