# src/experiment_runner.py

import statistics
import logging
from data_loaders import BaseDataLoader
from masking import MaskingStrategy
from reconstruction_strategies import ReconstructionStrategy
from evaluation import ReconstructionEvaluator, metrics_to_json



class ExperimentRunner:
    """
    Encapsulates and runs a single, atomic experiment.
    It is a pure "doer" that receives all its dependencies via injection.
    """
    def __init__(
        self,
        run_name: str,
        data_loader: BaseDataLoader,
        masking_strategy: MaskingStrategy,
        reconstruction_strategy: ReconstructionStrategy,
        evaluator: ReconstructionEvaluator
    ):
        self.run_name = run_name
        self.data_loader = data_loader
        self.masking_strategy = masking_strategy
        self.reconstruction_strategy = reconstruction_strategy
        self.evaluator = evaluator

    def run(self):
        """Runs the full experiment from data loading to evaluation."""
        all_videos = self.data_loader.load()
        all_metrics = []

        for video in all_videos:
            logging.debug(f"--- Processing Video: {video.video_id} ---")

            masked_video, masked_indices = self.masking_strategy.mask_video(video)
            if not masked_video:
                logging.warning(f"Not masking video {video.video_id} size={len(video.clips)} with {self.masking_strategy}")
                continue

            reconstructed_video = self.reconstruction_strategy.reconstruct(masked_video)
            if not reconstructed_video:
                logging.error(f"Reconstruction failed for video: {video.video_id}")
                # mlflow.log_metric("reconstruction_failed", 1)
                continue

            video_metrics = self.evaluator.evaluate(reconstructed_video.clips, video.clips, masked_indices)
            logging.info(f"Evaluation complete for video_id={video.video_id} metrics={metrics_to_json(video_metrics)}")
            all_metrics.append(video_metrics)
            logging.debug(f"Successfully processed video: {video.video_id}")


        if not all_metrics:
            logging.warning("No metrics were generated to log.")
            return {}
        # Calculate the mean for each metric across all videos
        mean_f1 = statistics.mean([m['bs_f1'].mean().item() for m in all_metrics])
        mean_precision = statistics.mean([m['bs_p'].mean().item() for m in all_metrics])
        mean_recall = statistics.mean([m['bs_r'].mean().item() for m in all_metrics])

        return {
            "num_of_instances": len(all_metrics),
            "mean_f1_score": mean_f1,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall
        }
