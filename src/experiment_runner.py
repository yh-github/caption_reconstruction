import logging
from typing import Any

from data_loaders import BaseDataLoader
from data_models.captions_only import CaptionedVideo
from evaluation import ReconstructionEvaluator_BertScore, metrics_to_json, round_metrics
from masking import MaskingStrategy
from reconstruction_strategies import ReconstructionStrategy, Reconstructed


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
        evaluator: ReconstructionEvaluator_BertScore,
        conf_for_log:dict[str, Any]
    ):
        self.run_name = run_name
        self.data_loader = data_loader
        self.masking_strategy = masking_strategy
        self.reconstruction_strategy = reconstruction_strategy
        self.evaluator = evaluator
        self.conf_for_log = conf_for_log

    def run(self) -> tuple[dict, list[Reconstructed]]:
        """Runs the full experiment from data loading to evaluation."""
        all_videos:list[CaptionedVideo] = self.data_loader.load()
        all_metrics:list[dict] = []
        all_recon_videos:list[Reconstructed] = []

        for video in all_videos:
            logging.debug(f"--- Processing Video: {video.video_id} ---")

            def err(message:str, extra:dict|None=None):
                return ReconstructionStrategy.create_error_result(
                    video_id=video.video_id,
                    error_message=message,
                    extra_debug_data=extra
                )

            masked_video, masked_indices = self.masking_strategy.mask_video(video)
            if not masked_video:
                logging.warning(f"Not masking video {video.video_id} size={len(video.clips)} with {self.masking_strategy}")
                all_recon_videos.append(err("NOT_MASKING"))
                continue

            reconstructed:Reconstructed = self.reconstruction_strategy.reconstruct(masked_video)
            # if not reconstructed or not reconstructed.reconstructed_captions:
            #     logging.error(f"Reconstruction failed for video: {video.video_id}")
            #     all_recon_videos.append(err("RECONSTRUCTION_FAILED"))
            #     continue

            if not reconstructed.debug_data and reconstructed.reconstructed_captions.keys() != masked_indices:
                crit_msg = f"Reconstruction failed for video: {video.video_id}, {reconstructed.reconstructed_captions.keys()=} != {masked_indices=}"
                logging.critical(crit_msg)
                raise Exception(crit_msg)

            if reconstructed.debug_data and reconstructed.debug_data.get('failed',0):
                logging.warning(f'Masked data found in reconstructed_video {video.video_id}, skipping')
                all_recon_videos.append(reconstructed.skip('failed>0'))
                continue
            elif reconstructed.reconstructed_captions.keys() != masked_indices:
                logging.warning(f'Bad indices found in reconstructed_video {video.video_id}, {reconstructed.reconstructed_captions.keys()=}, {masked_indices=}, skipping')
                all_recon_videos.append(reconstructed.skip(f"{masked_indices=}"))
                continue
            elif reconstructed.debug_data:
                logging.warning(f'Problems found in reconstructed_video {video.video_id}, proceeding anyway')

            video_metrics = self.evaluator.evaluate(reconstructed, video)

            all_metrics.append(video_metrics)

            metrics = round_metrics(video_metrics)
            all_recon_videos.append(reconstructed.with_metrics(metrics))

            metrics.update({
                "num_captions": len(video.clips),
                "masked": list(masked_indices)
            })

            logging.info(f"Evaluation complete for "
                         f"video_id={video.video_id} "
                         f"metrics={metrics_to_json(metrics)}")

            logging.debug(f"Successfully processed video: {video.video_id}")

        if not all_metrics:
            return {}, all_recon_videos

        # TODO: keep only the sums (NA as 0)

        return self.evaluator.agg_metrics(all_metrics), all_recon_videos
