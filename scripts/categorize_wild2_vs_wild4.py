"""
Run the same keyword-based categorization on wild2 counterparts of the 13 wild4 files.
Outputs a side-by-side comparison: wild2 prediction vs wild4 prediction vs ground truth.
"""

import json
import os
import re
from pathlib import Path

WILD2_DIR = Path("datasets/wildQA/captions__wild2")
WILD4_DIR = Path("datasets/wildQA/captions__wild4")

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


def load_captions(path: Path) -> str:
    with open(path) as f:
        data = json.load(f)
    captions = data.get("captions", [])
    return " ".join(c["caption"] for c in captions)


def score_text(text: str) -> dict[str, int]:
    text_lower = text.lower()
    return {
        domain: sum(len(re.findall(p, text_lower)) for p in patterns)
        for domain, patterns in KEYWORDS.items()
    }


def predict(scores: dict[str, int]) -> str:
    return max(scores, key=scores.get)


def summarize(text: str, n: int = 3) -> str:
    sentences = [s.strip() for s in re.split(r'\.(?:\s|$)', text) if s.strip()]
    return ". ".join(sentences[:n]) + ("..." if len(sentences) > n else ".")


def ok(predicted: str, gt: str) -> str:
    return "✓" if predicted == gt else "✗"


def main():
    wild4_ids = sorted(GROUND_TRUTH.keys())

    rows = []
    for vid in wild4_ids:
        w2_path = WILD2_DIR / f"{vid}.json"
        w4_path = WILD4_DIR / f"{vid}.json"
        gt = GROUND_TRUTH[vid]

        w2_pred = w2_scores = None
        w4_pred = w4_scores = None

        if w2_path.exists():
            w2_text = load_captions(w2_path)
            w2_scores = score_text(w2_text)
            w2_pred = predict(w2_scores)
        else:
            w2_text = ""

        if w4_path.exists():
            w4_text = load_captions(w4_path)
            w4_scores = score_text(w4_text)
            w4_pred = predict(w4_scores)
        else:
            w4_text = ""

        rows.append({
            "video_id": vid,
            "gt": gt,
            "w2_pred": w2_pred or "N/A",
            "w2_scores": w2_scores or {},
            "w4_pred": w4_pred or "N/A",
            "w4_scores": w4_scores or {},
            "w2_text": w2_text,
            "w4_text": w4_text,
        })

    # ── Side-by-side summary table ────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"{'VIDEO ID':<40} {'GT':<18} {'WILD2 PRED':<18} {'W2?':<4} {'WILD4 PRED':<18} {'W4?':<4}")
    print("=" * 110)

    w2_correct = w4_correct = 0
    for r in rows:
        w2_ok = ok(r["w2_pred"], r["gt"])
        w4_ok = ok(r["w4_pred"], r["gt"])
        if w2_ok == "✓": w2_correct += 1
        if w4_ok == "✓": w4_correct += 1
        print(f"{r['video_id']:<40} {r['gt']:<18} {r['w2_pred']:<18} {w2_ok:<4} {r['w4_pred']:<18} {w4_ok:<4}")

    n = len(rows)
    print("=" * 110)
    print(f"\nWild2 accuracy: {w2_correct}/{n} ({100*w2_correct/n:.0f}%)")
    print(f"Wild4 accuracy: {w4_correct}/{n} ({100*w4_correct/n:.0f}%)")

    # ── Per-video detailed breakdown ──────────────────────────────────────────
    print("\n" + "=" * 110)
    print("DETAILED COMPARISON (top 3 domain scores)")
    print("=" * 110)

    for r in rows:
        w2_ok = ok(r["w2_pred"], r["gt"])
        w4_ok = ok(r["w4_pred"], r["gt"])
        print(f"\n▶ {r['video_id']}  [GT: {r['gt']}]")

        def top3(scores):
            return " | ".join(
                f"{d}={s}" for d, s in sorted(scores.items(), key=lambda x: -x[1])[:3]
            )

        print(f"  Wild2 [{w2_ok}] {r['w2_pred']:<18}  Scores: {top3(r['w2_scores'])}")
        print(f"  Wild4 [{w4_ok}] {r['w4_pred']:<18}  Scores: {top3(r['w4_scores'])}")

        # Show caption summary for each, highlighting differences
        if r["w2_text"] and r["w4_text"]:
            print(f"  W2 summary: {summarize(r['w2_text'])}")
            print(f"  W4 summary: {summarize(r['w4_text'])}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {}
    for r in rows:
        out[r["video_id"]] = {
            "ground_truth": r["gt"],
            "wild2": {
                "predicted": r["w2_pred"],
                "correct": r["w2_pred"] == r["gt"],
                "scores": r["w2_scores"],
            },
            "wild4": {
                "predicted": r["w4_pred"],
                "correct": r["w4_pred"] == r["gt"],
                "scores": r["w4_scores"],
            },
        }

    out_path = Path("datasets/wildQA/categories_wild2_vs_wild4.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
