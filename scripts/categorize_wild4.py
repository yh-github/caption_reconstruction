"""
Categorize Wild4 videos using their per-second captions.

Strategy:
 - Concatenate all captions for each video into a summary text.
 - Apply keyword/heuristic rules matching the 5 WildQA domain labels:
     Agriculture, Geography, Human Survival, Military, Natural Disaster
 - Compare predicted categories against ground truth from dev.json (via jq).
 - Output a summary table.
"""

import json
import os
import re
import subprocess
from pathlib import Path

CAPTIONS_DIR = Path("datasets/wildQA/captions__wild4")
DEV_JSON = Path("datasets/wildQA/dev.json")
DOMAINS = ["Agriculture", "Geography", "Human Survival", "Military", "Natural Disaster"]

# Ground truth from dev.json (pre-fetched via jq to avoid slow JSON parsing)
GROUND_TRUTH = {
    "4k-Relaxation_12-clip-6": "Geography",
    "4k-Relaxation_3-clip-4": "Geography",
    "AiirSource-Military_1-clip-0": "Military",
    "AiirSource-Military_12-manual": "Military",
    "AiirSource-Military_7-clip-1": "Military",
    "Army-military-2018_8-clip-73": "Military",
    "BC-Bushcraft_11-clip-31": "Human Survival",
    "BC-Bushcraft_2-clip-2": "Human Survival",
    "BC-Bushcraft_9-clip-5": "Human Survival",
    "Bertram-Craft_12-clip-25": "Human Survival",
    "Bertram-Craft_2-clip-3": "Human Survival",
    "Bertram-Craft_5-clip-33": "Human Survival",
    "Chad-Zuber_10-clip-20": "Human Survival",
}

# ── Keyword lists for each domain ───────────────────────────────────────────
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
        r"\bdisaster\b", r"\bdevastati\b", r"\bemergenc\b", r"\brevacuat\b",
        r"\bdebris\b", r"\bashes?\b", r"\blava\b",
    ],
}


def load_captions(path: Path) -> tuple[str, str]:
    """Return (video_id, concatenated_caption_text)."""
    with open(path) as f:
        data = json.load(f)
    video_id = data.get("video_id", path.stem)
    captions = data.get("captions", [])
    text = " ".join(c["caption"] for c in captions)
    return video_id, text


def score_text(text: str) -> dict[str, int]:
    """Count keyword hits per domain."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for domain, patterns in KEYWORDS.items():
        hits = sum(len(re.findall(p, text_lower)) for p in patterns)
        scores[domain] = hits
    return scores


def predict_domain(scores: dict[str, int]) -> str:
    return max(scores, key=scores.get)


def summarize_captions(text: str, n: int = 5) -> str:
    """Return first n sentences of caption text as a readable summary."""
    sentences = [s.strip() for s in re.split(r'\.(?:\s|$)', text) if s.strip()]
    return ". ".join(sentences[:n]) + ("..." if len(sentences) > n else ".")


def main():
    json_files = sorted([
        f for f in CAPTIONS_DIR.iterdir()
        if f.suffix == ".json" and not f.name.startswith("error")
    ])

    results = []
    for path in json_files:
        video_id, text = load_captions(path)
        scores = score_text(text)
        predicted = predict_domain(scores)
        ground_truth = GROUND_TRUTH.get(video_id, "UNKNOWN")
        correct = "✓" if predicted == ground_truth else "✗"
        summary = summarize_captions(text)
        results.append({
            "video_id": video_id,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": correct,
            "scores": scores,
            "summary": summary,
        })

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'VIDEO ID':<40} {'PREDICTED':<18} {'GROUND TRUTH':<18} {'OK?':<5} {'TOP SCORES'}")
    print("=" * 100)

    n_correct = 0
    for r in results:
        top2 = sorted(r["scores"].items(), key=lambda x: -x[1])[:2]
        score_str = ", ".join(f"{d}:{s}" for d, s in top2)
        print(f"{r['video_id']:<40} {r['predicted']:<18} {r['ground_truth']:<18} {r['correct']:<5} {score_str}")
        if r["correct"] == "✓":
            n_correct += 1

    print("=" * 100)
    print(f"\nAccuracy: {n_correct}/{len(results)} ({100*n_correct/len(results):.0f}%)")

    # ── Print per-video caption summaries & scores ───────────────────────────
    print("\n" + "=" * 100)
    print("DETAILED SCORES & CAPTION SUMMARIES")
    print("=" * 100)
    for r in results:
        print(f"\n▶ {r['video_id']}  [{r['correct']}  predicted={r['predicted']}  gt={r['ground_truth']}]")
        sorted_scores = sorted(r["scores"].items(), key=lambda x: -x[1])
        score_line = "  Scores: " + " | ".join(f"{d}={s}" for d, s in sorted_scores)
        print(score_line)
        print(f"  Summary: {r['summary']}")

    # ── Save JSON output ─────────────────────────────────────────────────────
    out_path = CAPTIONS_DIR / "categories.json"
    output = {
        r["video_id"]: {
            "predicted_domain": r["predicted"],
            "ground_truth_domain": r["ground_truth"],
            "correct": r["correct"] == "✓",
            "scores": r["scores"],
        }
        for r in results
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
