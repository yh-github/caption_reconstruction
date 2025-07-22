# src/experiment_runner.py
import logging
import mlflow
from data_loaders import BaseDataLoader
from masking import MaskingStrategy
from reconstruction_strategies import ReconstructionStrategy

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
        reconstruction_strategy: ReconstructionStrategy
    ):
        self.run_name = run_name
        self.data_loader = data_loader
        self.masking_strategy = masking_strategy
        self.reconstruction_strategy = reconstruction_strategy

    def run(self):
        """Runs the full experiment from data loading to evaluation."""
        all_videos = self.data_loader.load()
        
        for video in all_videos:
            logging.info(f"--- Processing Video: {video.video_id} ---")
            masked_clips = self.masking_strategy.apply(video.clips)
            
            masked_video = video.model_copy(update={'clips': masked_clips})

            reconstructed_video = self.reconstruction_strategy.reconstruct(masked_video)
            
            if reconstructed_video:
                # metrics = evaluate_reconstruction(reconstructed_video, video)
                # mlflow.log_metrics(metrics)
                logging.info(f"Successfully processed video: {video.video_id}")
            else:
                logging.error(f"Reconstruction failed for video: {video.video_id}")
                mlflow.log_metric("reconstruction_failed", 1)
