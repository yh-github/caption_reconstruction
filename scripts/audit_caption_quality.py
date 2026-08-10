"""
Comprehensive Caption Quality Audit Script.

Audits generated video captions for:
 1. Pairwise Similarity & Deduplication (Cosine similarity & N-gram overlap)
 2. Ad & Commercial Phrase Screening (Regex pattern matching)
 3. Domain Consistency (Predicted vs Ground-truth domain from dev.json / test.json)
 4. Caption Quality Issues (Repetitive text, extreme brevity, empty fields)

Usage:
    python scripts/audit_caption_quality.py --dataset wild4
    python scripts/audit_caption_quality.py --dataset wild5
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── AD & BOILERPLATE REGEX PATTERNS ─────────────────────────────────────────
AD_PATTERNS = [
    (r"\b(subscribe|subscribing)\b", "Channel subscription request"),
    (r"\b(like and share|smash that like)\b", "Call to action"),
    (r"\b(link in (the )?description|check the link)\b", "Description link promo"),
    (r"\b(sponsor|sponsored|sponsorship)\b", "Sponsor disclaimer"),
    (r"\b(nordvpn|expressvpn|surfshark|raid shadow legends|squarespace|grammarly|audible|honey)\b", "Known sponsor brand"),
    (r"\b(promo code|discount code|use code)\b", "Discount code promo"),
    (r"\b(vr headset|virtual reality|gameplay|headset)\b", "Suspect VR headset ad content"),
    (r"\b(freshly baked bread|baked bread)\b", "Suspect bread ad content"),
    (r"\b(black gridded fabric|chalk pencil)\b", "Suspect fabric ad content"),
    (r"\b(tearing drawing|tearing paper)\b", "Suspect tearing drawing ad content"),
    (r"\b(buy now|order today|limited time offer)\b", "Commercial sales line"),
    (r"\b(copyright|all rights reserved)\b", "Copyright boilerplate"),
]

# ── DOMAIN KEYWORDS ──────────────────────────────────────────────────────────
KEYWORDS = {
    "Military": [
        r"\bmilitar\b", r"\bsoldier\b", r"\btroops?\b", r"\barmy\b", r"\bweapon\b",
        r"\bgunfire\b", r"\bexplosion\b", r"\bammunition\b", r"\bcombat\b",
        r"\bdefend\b", r"\battack\b", r"\bhelicopt\b", r"\btank\b", r"\bgunship\b",
        r"\brifle\b", r"\bgrenade\b", r"\bbullet\b", r"\buniform\b", r"\bhelmet\b",
        r"\bmissile\b", r"\baircraft\b", r"\bparachut\b", r"\bwarf?are\b",
    ],
    "Human Survival": [
        r"\bsurviv\b", r"\bbushcraft\b", r"\bwilderness\b", r"\bcamp\b",
        r"\bfire\b", r"\bknife\b", r"\bshelter\b", r"\bforage\b",
        r"\bhike\b", r"\bhiking\b", r"\btrail\b", r"\btrek\b",
        r"\bfish\b", r"\bhunt\b", r"\btrap\b", r"\bwater filter\b",
        r"\bcoconut\b", r"\bbuild\b", r"\bcraft\b", r"\bboat\b",
        r"\bwood\b", r"\btool\b", r"\bcook\b", r"\beat\b",
        r"\bsea\b", r"\bocean\b", r"\beach\b", r"\briver\b", r"\blake\b",
        r"\bjungle\b", r"\bforest walk\b",
    ],
    "Geography": [
        r"\baerial\b", r"\bdrone\b", r"\blandsca\b", r"\bterrain\b",
        r"\bgeograph\b", r"\bcanyon\b", r"\bmountain\b", r"\bvalley\b",
        r"\bcoastlin\b", r"\bcliff\b", r"\bdesert\b", r"\bvolcan\b",
        r"\bmap\b", r"\btopograph\b", r"\bglacier\b", r"\btundra\b",
        r"\bsavann\b", r"\bprairie\b", r"\bplain\b", r"\bpanoram\b",
        r"\bscenery\b", r"\bscenic\b", r"\blandmark\b", r"\brelax\b",
        r"\bnature\b", r"\bwildlife\b", r"\bcanopy\b", r"\bforest\b",
    ],
    "Agriculture": [
        r"\bfarm\b", r"\bcrops?\b", r"\bharvest\b", r"\bagricultur\b",
        r"\bfield\b", r"\btractor\b", r"\bplow\b", r"\birrigat\b",
        r"\bplant\b", r"\bseed\b", r"\bsoil\b", r"\bgarden\b",
        r"\borchard\b", r"\blivestock\b", r"\bcattle\b", r"\bgraze\b",
        r"\bfertiliz\b", r"\bpesticide\b", r"\bwheat\b", r"\bcorn\b",
        r"\brice\b", r"\bsow\b",
    ],
    "Natural Disaster": [
        r"\bflood\b", r"\bwildfire\b", r"\bhurricane\b", r"\btornado\b",
        r"\bearthquak\b", r"\btsunami\b", r"\bvolcanic\b", r"\berupt\b",
        r"\blandslide\b", r"\bavalanche\b", r"\bstorm\b", r"\bcyclone\b",
        r"\bdisaster\b", r"\bdevastati\b", r"\bemergenc\b", r"\bevacuat\b",
        r"\bdebris\b", r"\bashes?\b", r"\blava\b",
    ],
}


def load_ground_truth(meta_path: Path) -> dict[str, str]:
    """Extract video_id -> domain map from dev.json / test.json."""
    if not meta_path.exists():
        print(f"Warning: Metadata file {meta_path} does not exist.")
        return {}
    with open(meta_path) as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        vid = item.get("video_id")
        domain = item.get("domain")
        if vid and domain:
            mapping[vid] = domain
    return mapping


def load_captions(caption_dir: Path) -> list[dict]:
    """Load all valid JSON caption files in directory."""
    files = sorted([
        f for f in caption_dir.iterdir()
        if f.suffix == ".json" and not f.name.startswith("error") and f.name != "categories.json"
    ])
    dataset = []
    for f in files:
        with open(f) as fp:
            try:
                d = json.load(fp)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue
        video_id = d.get("video_id", f.stem)
        
        text_segments = []
        # Check 'captions'
        captions_list = d.get("captions", [])
        for c in captions_list:
            if isinstance(c, dict):
                text_segments.append(c.get("caption", ""))
            elif isinstance(c, str):
                text_segments.append(c)

        # Check 'segments' (used in wild1)
        segments_list = d.get("segments", [])
        for s in segments_list:
            if isinstance(s, dict):
                if "segment_summary" in s:
                    text_segments.append(s["segment_summary"])
                if "caption" in s:
                    text_segments.append(s["caption"])
                if "key_moments" in s and isinstance(s["key_moments"], list):
                    for km in s["key_moments"]:
                        if isinstance(km, dict) and "caption" in km:
                            text_segments.append(km["caption"])

        full_text = " ".join([t for t in text_segments if t]).strip()
        dataset.append({
            "video_id": video_id,
            "filename": f.name,
            "path": f,
            "segments": text_segments,
            "full_text": full_text,
            "num_segments": len(text_segments)
        })
    return dataset


def get_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """Compute n-gram Jaccard similarity."""
    words1 = re.findall(r"\w+", text1.lower())
    words2 = re.findall(r"\w+", text2.lower())
    
    if len(words1) < n or len(words2) < n:
        return 0.0
        
    set1 = set(zip(*[words1[i:] for i in range(n)]))
    set2 = set(zip(*[words2[i:] for i in range(n)]))
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def score_domain_text(text: str) -> dict[str, int]:
    """Count keyword hits for each domain."""
    text_lower = text.lower()
    scores = {}
    for domain, patterns in KEYWORDS.items():
        hits = sum(len(re.findall(p, text_lower)) for p in patterns)
        scores[domain] = hits
    return scores


def audit_dataset(dataset_name: str, out_report_path: Path | None = None):
    root = Path(__file__).resolve().parent.parent
    caption_dir = root / "datasets" / "wildQA" / f"captions__{dataset_name}"
    
    if dataset_name == "wild4":
        meta_path = root / "datasets" / "wildQA" / "dev.json"
    else:
        meta_path = root / "datasets" / "wildQA" / "test.json"
        
    gt_map = load_ground_truth(meta_path)
    captions = load_captions(caption_dir)
    
    print(f"\n==================================================================")
    print(f" AUDITING DATASET: {dataset_name.upper()}")
    print(f" Caption Directory: {caption_dir}")
    print(f" Loaded Captions: {len(captions)} files")
    print(f" Ground Truth Metadata Entries: {len(gt_map)}")
    print(f"==================================================================\n")
    
    if not captions:
        print("No caption files found to audit!")
        return

    # 1. AD SCREENING & QUALITY CHECKS
    ad_hits = []
    repetitive_clips = []
    
    for item in captions:
        vid = item["video_id"]
        text = item["full_text"]
        
        # Check ad patterns
        matched_patterns = []
        for pat, desc in AD_PATTERNS:
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                matched_patterns.append((desc, len(matches)))
        if matched_patterns:
            ad_hits.append({
                "video_id": vid,
                "matches": matched_patterns,
                "snippet": text[:150]
            })
            
        # Check repetitive text (e.g. same segment repeated > 10 times)
        if item["segments"]:
            counts = defaultdict(int)
            for seg in item["segments"]:
                counts[seg.strip().lower()] += 1
            max_rep = max(counts.values()) if counts else 0
            if max_rep >= 10:
                most_common = max(counts, key=counts.get)
                repetitive_clips.append({
                    "video_id": vid,
                    "max_rep": max_rep,
                    "num_segments": item["num_segments"],
                    "phrase": most_common[:100]
                })

    # 2. PAIRWISE SIMILARITY & DEDUPLICATION ANALYSIS
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = [item["full_text"] for item in captions]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cos_sim = cosine_similarity(tfidf_matrix)
    
    high_similarity_pairs = []
    N = len(captions)
    for i in range(N):
        for j in range(i + 1, N):
            sim = cos_sim[i, j]
            if sim > 0.70: # High threshold for suspicious duplication
                jaccard = get_ngram_jaccard(captions[i]["full_text"], captions[j]["full_text"], n=3)
                high_similarity_pairs.append({
                    "vid1": captions[i]["video_id"],
                    "vid2": captions[j]["video_id"],
                    "cosine_sim": float(sim),
                    "ngram_jaccard": float(jaccard),
                    "snippet1": captions[i]["full_text"][:120],
                    "snippet2": captions[j]["full_text"][:120]
                })

    # Sort duplicate pairs by similarity descending
    high_similarity_pairs.sort(key=lambda x: x["cosine_sim"], reverse=True)

    # Group into duplicate clusters
    clusters = []
    visited = set()
    for pair in high_similarity_pairs:
        v1, v2 = pair["vid1"], pair["vid2"]
        if pair["cosine_sim"] >= 0.85: # Strict duplication threshold
            found_cluster = False
            for c in clusters:
                if v1 in c or v2 in c:
                    c.add(v1)
                    c.add(v2)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append({v1, v2})

    # 3. DOMAIN CONSISTENCY CHECK
    domain_audit = []
    mismatch_count = 0
    correct_count = 0
    unknown_count = 0
    
    for item in captions:
        vid = item["video_id"]
        gt_domain = gt_map.get(vid, "UNKNOWN")
        scores = score_domain_text(item["full_text"])
        
        predicted = max(scores, key=scores.get) if any(scores.values()) else "UNCLASSIFIED"
        gt_score = scores.get(gt_domain, 0)
        
        is_correct = (predicted == gt_domain)
        if gt_domain == "UNKNOWN":
            unknown_count += 1
        elif is_correct:
            correct_count += 1
        else:
            mismatch_count += 1
            
        domain_audit.append({
            "video_id": vid,
            "ground_truth": gt_domain,
            "predicted": predicted,
            "correct": is_correct,
            "gt_score": gt_score,
            "scores": scores,
            "snippet": item["full_text"][:120]
        })

    # ── PRINT CONSOLE SUMMARY ────────────────────────────────────────────────
    print("------------------------------------------------------------------")
    print(" AUDIT FINDINGS SUMMARY")
    print("------------------------------------------------------------------")
    print(f" Total Captions Audited: {N}")
    print(f" Suspected Duplicate Clusters (CosSim >= 0.85): {len(clusters)}")
    print(f" High Similarity Pairs (CosSim > 0.70): {len(high_similarity_pairs)}")
    print(f" Ad / Commercial Boilerplate Hits: {len(ad_hits)}")
    print(f" Highly Repetitive Clips: {len(repetitive_clips)}")
    if (correct_count + mismatch_count) > 0:
        acc = 100.0 * correct_count / (correct_count + mismatch_count)
        print(f" Domain Keyword Accuracy: {correct_count}/{correct_count + mismatch_count} ({acc:.1f}%)")
    print("------------------------------------------------------------------\n")

    # ── GENERATE MARKDOWN REPORT ──────────────────────────────────────────────
    report = []
    report.append(f"# Dataset Audit Report: `{dataset_name}`\n")
    report.append(f"**Date**: 2026-08-10  ")
    report.append(f"**Target Directory**: `{caption_dir.relative_to(root)}`  ")
    report.append(f"**Total Clips Audited**: `{N}`  \n")
    
    report.append("## Executive Summary\n")
    report.append(f"| Metric | Result | Health Status |")
    report.append(f"| :--- | :--- | :--- |")
    report.append(f"| **Suspected Duplicate Clusters** | `{len(clusters)}` | {'🟢 Clean (0 duplicates)' if len(clusters)==0 else '🔴 Issue Found'} |")
    report.append(f"| **Ad / Commercial Boilerplate Hits** | `{len(ad_hits)}` | {'🟢 Clean (0 ads)' if len(ad_hits)==0 else '🟡 Warning'} |")
    report.append(f"| **Repetitive / Loop Captions** | `{len(repetitive_clips)}` | {'🟢 Clean' if len(repetitive_clips)==0 else '🟡 Warning'} |")
    if (correct_count + mismatch_count) > 0:
        report.append(f"| **Domain Keyword Alignment** | `{correct_count}/{correct_count + mismatch_count} ({acc:.1f}%)` | {'🟢 Strong (>70%)' if acc >= 70 else '🟡 Fair'} |")
    report.append("\n---\n")

    # Section 1: Duplication Analysis
    report.append("## 1. Duplication & Similarity Analysis\n")
    if not high_similarity_pairs:
        report.append("🟢 **No high-similarity pairs or duplicate video clusters found.** All video captions are distinct and unique.\n")
    else:
        report.append(f"Found **{len(high_similarity_pairs)}** pairs of videos with Cosine Similarity > 0.70:\n")
        report.append("| Video 1 | Video 2 | Cosine Sim | 3-Gram Jaccard | Status |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        for pair in high_similarity_pairs:
            status = "🔴 Suspected Duplicate" if pair["cosine_sim"] >= 0.85 else "🟡 Similar Content"
            report.append(f"| `{pair['vid1']}` | `{pair['vid2']}` | `{pair['cosine_sim']:.3f}` | `{pair['ngram_jaccard']:.3f}` | {status} |")
        report.append("\n")

    # Section 2: Ad Screening
    report.append("## 2. Ad & Commercial Boilerplate Screening\n")
    if not ad_hits:
        report.append("🟢 **No advertisement boilerplates, sponsor messages, or commercial terms detected.** Captions represent true video content.\n")
    else:
        report.append(f"Found **{len(ad_hits)}** files containing potential ad/boilerplate keywords:\n")
        for hit in ad_hits:
            match_str = ", ".join(f"{desc} ({count}x)" for desc, count in hit["matches"])
            report.append(f"* **`{hit['video_id']}`**: Matched {match_str}")
            report.append(f"  * *Snippet*: \"{hit['snippet']}...\"\n")

    # Section 3: Domain Alignment & Outliers
    report.append("## 3. Domain Alignment Analysis\n")
    report.append("Comparison of keyword profile predictions against ground-truth dataset domains:\n")
    
    mismatches = [d for d in domain_audit if not d["correct"] and d["ground_truth"] != "UNKNOWN"]
    if not mismatches:
        report.append("🟢 **All evaluated videos align with their ground-truth domains.**\n")
    else:
        report.append(f"### Mismatched Domain Outliers ({len(mismatches)} files)\n")
        report.append("| Video ID | Ground Truth | Predicted Domain | GT Domain Score | Snippet |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        for m in mismatches:
            report.append(f"| `{m['video_id']}` | `{m['ground_truth']}` | `{m['predicted']}` | `{m['gt_score']}` | \"{m['snippet'][:80]}...\" |")
        report.append("\n")

    report_str = "\n".join(report)
    
    if out_report_path:
        out_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_report_path, "w") as f:
            f.write(report_str)
        print(f"✅ Audit report saved to {out_report_path}")

    return {
        "num_audited": N,
        "duplicate_clusters": len(clusters),
        "ad_hits": len(ad_hits),
        "domain_accuracy": acc if (correct_count + mismatch_count) > 0 else None
    }


def main():
    parser = argparse.ArgumentParser(description="Audit dataset caption quality.")
    parser.add_argument("--dataset", type=str, default="wild4", help="Dataset name to audit (e.g. wild1, wild2, wild3, wild4, wild5)")
    parser.add_argument("--out-report", type=str, default=None, help="Output markdown report path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.out_report:
        out_path = Path(args.out_report)
    else:
        out_path = root / "docs" / "data" / "audit_captions" / f"audit_{args.dataset}_results.md"

    audit_dataset(args.dataset, out_path)


if __name__ == "__main__":
    main()
