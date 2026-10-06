#!/usr/bin/env python3
"""
Re-scores a Llama run that was evaluated with the legacy `pool_scope: "window"` against the
current standard `pool_scope: "video"` (all 60 timestamps), without regenerating any text.

The wild4 W=3 run (`wild4_llama_w3_window_v3`) is the only wild4 W=3 Llama run at i=29, but its
stored metrics rank each prediction against the 3 masked captions only (chance MRR ~0.61).
This writes corrected copies to `<run>_videopool/`, locally only (no HF upload).

Usage (from repo root):
    .venv/bin/python scripts/rescore_window_pool_run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loaders import WildLoader
from evaluations.evaluation import ReconstructionEvaluator_Retrieval
from evaluations.metrics import round_metrics
from llm.local_embedder import LocalEmbedder
from reconstruction.text_reconstruction import Reconstructed

SRC = Path("results/recon/manual_download/reconstruction/wild4_llama_w3_window_v3")
DST = SRC.with_name(SRC.name + "_videopool")
CAPTIONS = Path("datasets/wildQA/captions__wild4")


def main():
    videos = {v.video_id: v for v in WildLoader(CAPTIONS).load()}
    evaluator = ReconstructionEvaluator_Retrieval(LocalEmbedder("all-mpnet-base-v2"), pool_scope="video")
    for run_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        out_dir = DST / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        done = skipped = 0
        for jf in sorted(run_dir.glob("*.json")):
            if jf.name.startswith("skip__"):
                continue
            r = Reconstructed.model_validate_json(jf.read_text())
            video = videos.get(r.video_id)
            if video is None:
                skipped += 1
                continue
            metrics = evaluator.evaluate(r, video)
            if "mean_rank" not in metrics:
                skipped += 1
                continue
            (out_dir / jf.name).write_text(r.with_metrics(round_metrics(metrics)).json_str())
            done += 1
        print(f"{run_dir.name}: rescored {done}, skipped {skipped}")


if __name__ == "__main__":
    main()
