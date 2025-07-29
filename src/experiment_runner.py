import statistics
import logging

from data_loaders import BaseDataLoader
from masking import MaskingStrategy
from reconstruction_strategies import ReconstructionStrategy
from evaluation import ReconstructionEvaluator, metrics_to_json, round_metrics
from data_models import CaptionedVideo


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
        all_videos:list[CaptionedVideo] = self.data_loader.load()
        all_metrics:list[dict] = []
        all_recon_videos:list[str] = []

        for video in all_videos:
            logging.debug(f"--- Processing Video: {video.video_id} ---")

            masked_video, masked_indices = self.masking_strategy.mask_video(video)
            if not masked_video:
                logging.warning(f"Not masking video {video.video_id} size={len(video.clips)} with {self.masking_strategy}")
                all_recon_videos.append(f"SKIP {video.video_id=} NOT_MASKING")
                continue

            reconstructed = self.reconstruction_strategy.reconstruct(masked_video)
            if not reconstructed or not reconstructed.reconstructed_clips:
                logging.error(f"Reconstruction failed for video: {video.video_id}")
                all_recon_videos.append(f"SKIP {video.video_id=} FAIL")
                continue

            if reconstructed.reconstructed_clips.keys() != masked_indices and not reconstructed.debug_data:
                crit_msg = f"Reconstruction failed for video: {video.video_id}, {reconstructed.reconstructed_clips.keys()=} != {masked_indices=}"
                logging.critical(crit_msg)
                raise Exception(crit_msg)

            if reconstructed.debug_data and reconstructed.debug_data.get('failed',0):
                logging.warning(f'Masked data found in reconstructed_video {video.video_id}, skipping')
                all_recon_videos.append(reconstructed.skip('failed>0').model_dump_json())
                continue
            elif reconstructed.reconstructed_clips.keys() != masked_indices:
                logging.warning(f'Bad indices found in reconstructed_video {video.video_id}, {reconstructed.indices=}, {masked_indices=}, skipping')
                all_recon_videos.append(reconstructed.skip(f"{masked_indices=}").model_dump_json())
                continue
            elif reconstructed.debug_data:
                logging.warning(f'Problems found in reconstructed_video {video.video_id}, proceeding anyway')

            video_metrics = self.evaluator.evaluate(reconstructed, video)

            all_metrics.append(video_metrics)

            metrics = round_metrics(video_metrics)
            all_recon_videos.append(reconstructed.with_metrics(metrics).model_dump_json())

            metrics.update({
                "num_captions": len(video.clips),
                "masked": list(masked_indices)
            })

            logging.info(f"Evaluation complete for "
                         f"video_id={video.video_id} "
                         f"metrics={metrics_to_json(metrics)}")

            logging.debug(f"Successfully processed video: {video.video_id}")

        if not all_metrics:
            logging.warning("No metrics were generated to log.")
            return {}

        # TODO: keep only the sums (NA as 0)
        mean_f1 = statistics.mean([m['bs_f1'].min().item() for m in all_metrics])
        mean_precision = statistics.mean([m['bs_p'].min().item() for m in all_metrics])
        mean_recall = statistics.mean([m['bs_r'].min().item() for m in all_metrics])

        agg_metrics = {
            "num_of_instances": len(all_metrics),
            "mean_f1_score": mean_f1,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall
        }

        return agg_metrics, all_recon_videos
