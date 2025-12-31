import logging
from pathlib import Path
from typing import Any
from data.data_loaders import BaseDataLoader
from data_models.captions_only import CaptionedVideo
from evaluations.evaluation import ReconstructionEvaluator
from evaluations.metrics import metrics_to_json, round_metrics, MetricsMetadata, MetricsRecordRaw
from reconstruction.masking import MaskingStrategy
from reconstruction.text_reconstruction import ReconstructionStrategy, Reconstructed


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
        evaluator: ReconstructionEvaluator,
        save_path: Path,
        conf_for_log:dict[str, Any]
    ):
        self.run_name = run_name
        self.data_loader = data_loader
        self._masking_strategy = masking_strategy
        self._reconstruction_strategy = reconstruction_strategy
        self.evaluator = evaluator
        self._save_path = save_path/run_name
        self.conf_for_log = conf_for_log

    @staticmethod
    def _filename(video_id:str) -> str:
        return f"{video_id}.json"

    def _save_result(self, r:Reconstructed):
        filename = self._filename(r.video_id)
        if r.skip_reason:
            filename = f"skip__{filename}"

        with open(self._save_path / filename, "w") as f:
            f.write(r.json_str())

    def run(self) -> list[MetricsRecordRaw]:
        """Runs the full experiment from data loading to evaluation."""
        self._save_path.mkdir(parents=True, exist_ok=True)
        all_videos:list[CaptionedVideo] = self.data_loader.load()
        all_metrics:list[MetricsRecordRaw] = []

        for video in all_videos:
            if metric := self._process_single_video(video):
                all_metrics.append(metric)

        # TODO: keep only the sums (NA as 0)

        return all_metrics

    def _process_single_video(self, video: CaptionedVideo) -> MetricsRecordRaw | None:
        """
        Orchestrates processing for a single video:
        - Helper to `run`
        - Checks for existing results (Resumption)
        - Runs new experiment if needed
        """
        result_file = self._save_path / self._filename(video.video_id)

        if result_file.exists():
            return self._load_existing_result(video, result_file)

        return self._run_new_experiment(video)

    def _load_existing_result(self, video: CaptionedVideo, result_file: Path) -> MetricsRecordRaw | None:
        """
        Loads an existing result from disk.
        CRITICAL: We must still run the masking strategy to advance the PRN state
        to ensure determinism for subsequent videos.
        """
        try:
            with open(result_file, "r") as f:
                content = f.read()

            reconstructed = Reconstructed.model_validate_json(content)
            logging.info(f"Video {video.video_id} result found, loading...")

            # Advance PRN state and get mask info
            masked_video, masked_indices = self._masking_strategy.mask_video(video)

            if not reconstructed.metrics:
                logging.warning(f"Found result for {video.video_id} but no metrics in it. Skipping inclusion.")
                return None
            
            # If masking failed during this "dry run" but we have a result file,
            # it implies a configuration change or non-determinism issue.
            if masked_indices is None:
                logging.warning(f"Masking strategy returned None for {video.video_id} during resumption, but result file exists.")
                return None

            return MetricsRecordRaw(
                raw_metrics=reconstructed.metrics,
                metadata=MetricsMetadata(
                    video_id=video.video_id,
                    size=len(video.clips),
                    masked=list(masked_indices),
                    recon_strategy=str(self._reconstruction_strategy),
                    data_type=self.data_loader.get_data_type_name()
                )
            )

        except Exception as e:
            logging.warning(f"Failed to load existing result for {video.video_id}: {e}")
            return None

    def _run_new_experiment(self, video: CaptionedVideo) -> MetricsRecordRaw | None:
        """
        Runs the actual experiment logic for a new video.
        """
        logging.debug(f"--- Processing Video: {video.video_id} ---")

        def err(message:str, extra:dict|None=None):
            return ReconstructionStrategy.create_error_result(
                video_id=video.video_id,
                error_message=message,
                extra_debug_data=extra
            )

        masked_video, masked_indices = self._masking_strategy.mask_video(video)
        if not masked_video:
            logging.warning(f"Not masking video {video.video_id} size={len(video.clips)} with {self._masking_strategy}")
            self._save_result(err("NOT_MASKING"))
            return None

        reconstructed:Reconstructed = self._reconstruction_strategy.reconstruct(masked_video)

        if not reconstructed.debug_data and reconstructed.reconstructed_captions.keys() != masked_indices:
            crit_msg = f"Reconstruction failed for video: {video.video_id}, {reconstructed.reconstructed_captions.keys()=} != {masked_indices=}"
            logging.critical(crit_msg)
            raise Exception(crit_msg)

        if reconstructed.debug_data and reconstructed.debug_data.get('failed',0):
            logging.warning(f'Masked data found in reconstructed_video {video.video_id}, skipping')
            self._save_result(reconstructed.skip('failed>0'))
            return None
        elif reconstructed.reconstructed_captions.keys() != masked_indices:
            logging.warning(f'Bad indices found in reconstructed_video {video.video_id}, {reconstructed.reconstructed_captions.keys()=}, {masked_indices=}, skipping')
            self._save_result(reconstructed.skip(f"mismatch with {masked_indices=}"))
            return None
        elif reconstructed.debug_data:
            logging.warning(f'Problems found in reconstructed_video {video.video_id}, proceeding anyway')

        video_metrics = self.evaluator.evaluate(reconstructed, video)

        raw_record = MetricsRecordRaw(
            raw_metrics=video_metrics,
            metadata=MetricsMetadata(
                video_id=video.video_id,
                size=len(video.clips),
                masked=list(masked_indices),
                recon_strategy=str(self._reconstruction_strategy),
                data_type=self.data_loader.get_data_type_name()
            )
        )

        rounded = round_metrics(video_metrics)
        self._save_result(reconstructed.with_metrics(rounded))

        logging.info(f"Evaluation metrics {metrics_to_json(rounded)}")
        logging.debug(f"Successfully processed video: {video.video_id}")

        return raw_record
