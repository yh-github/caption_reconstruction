from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional
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
        conf_for_log:dict[str, Any],
        hf_manager: Any = None, # Optional HFFileManager
        config_stem: str = "",
        eval_only: bool = False,
        no_download_existing: bool = False,
        worker_id: int = 0,
        total_workers: int = 1,
        max_runtime_hours: float | None = None
    ):
        if total_workers < 1:
            raise ValueError(f"total_workers must be >= 1, got {total_workers}")
        if not (0 <= worker_id < total_workers):
            raise ValueError(f"worker_id must be in range [0, {total_workers-1}], got {worker_id}")

        self.run_name = run_name
        self.data_loader = data_loader
        self._masking_strategy = masking_strategy
        self._reconstruction_strategy = reconstruction_strategy
        self.evaluator = evaluator
        self._save_path = save_path/run_name
        self.conf_for_log = conf_for_log
        self.hf_manager = hf_manager
        self.eval_only = eval_only
        self.no_download_existing = no_download_existing
        self.worker_id = worker_id
        self.total_workers = total_workers
        self.max_runtime_hours = max_runtime_hours
        
        self.remote_run_path = f"reconstruction/{config_stem}/{run_name}"
        
        self.remote_files: set[str] = set()
        
        # Ensure directory exists immediately for monitoring
        self._save_path.mkdir(parents=True, exist_ok=True)

    def _sync_hf_state(self):
        """Lazy initialization of remote state."""
        if self.hf_manager:
            # 1. Ensure config matches
            self.hf_manager.ensure_config_match(self.remote_run_path, self.conf_for_log)
            # 2. Get list of already done files
            self.remote_files = self.hf_manager.list_files(self.remote_run_path)
            logging.info(f"HF Sync: Found {len(self.remote_files)} existing result files remotely.")

    @staticmethod
    def _filename(video_id:str) -> str:
        return f"{video_id}.json"

    def _save_result(self, r:Reconstructed):
        filename = self._filename(r.video_id)
        if r.skip_reason:
            filename = f"skip__{filename}"
            
        local_file_path = self._save_path / filename

        with open(local_file_path, "w") as f:
            f.write(r.json_str())
            
        # Trigger background upload
        if self.hf_manager:
            remote_file_path = f"{self.remote_run_path}/{filename}"
            self.hf_manager.upload_file_async(local_file_path, remote_file_path)

    def run(self) -> list[MetricsRecordRaw]:
        """Runs the full experiment from data loading to evaluation."""
        import time
        self._sync_hf_state()
        
        self._save_path.mkdir(parents=True, exist_ok=True)
        all_videos:list[CaptionedVideo] = self.data_loader.load()
        all_metrics:list[MetricsRecordRaw] = []

        start_time = time.time()
        my_videos = [v for idx, v in enumerate(all_videos) if (idx % self.total_workers) == self.worker_id]
        print(f"[Worker {self.worker_id}/{self.total_workers}] Assigned {len(my_videos)} out of {len(all_videos)} total videos.", flush=True)

        for i, video in enumerate(my_videos):
            if self.max_runtime_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= self.max_runtime_hours:
                    print(
                        f"[Worker {self.worker_id}/{self.total_workers}] Reached max runtime limit of "
                        f"{self.max_runtime_hours}h (elapsed: {elapsed_hours:.2f}h). Exiting loop cleanly.",
                        flush=True
                    )
                    break

            print(f"[Worker {self.worker_id}/{self.total_workers}] [{i+1}/{len(my_videos)}] Processing: {video.video_id}...", flush=True)
            if metric := self._process_single_video(video):
                all_metrics.append(metric)
                print(f"[Worker {self.worker_id}/{self.total_workers}] ✓ Completed: {video.video_id}", flush=True)

        # TODO: keep only the sums (NA as 0)

        return all_metrics

    def _process_single_video(self, video: CaptionedVideo) -> MetricsRecordRaw | None:
        """
        Orchestrates processing for a single video:
        - Helper to `run`
        - Checks for existing results (Resumption)
        - Runs new experiment if needed
        """
        filename = self._filename(video.video_id)
        result_file = self._save_path / filename

        # 1. Check Local
        if result_file.exists():
            return self._load_existing_result(video, result_file)
            
        # 2. Check Remote (HF)
        if self.hf_manager and filename in self.remote_files:
            if self.no_download_existing:
                logging.info(f"Video {video.video_id} found in HF Cache (Remote). Skipping download (--no-download-existing).")
                return None

            logging.info(f"Video {video.video_id} found in HF Cache. Downloading...")
            remote_path = f"{self.remote_run_path}/{filename}"
            if self.hf_manager.download_file(remote_path, result_file):
                 return self._load_existing_result(video, result_file)
            else:
                 logging.warning(f"Failed to download {video.video_id} from HF despite listing. Re-computing.")



        if self.eval_only:
             logging.warning(f"Eval-only mode: Skipping new experiment for {video.video_id} (not found).")
             return None

        return self._run_new_experiment(video)

    def _load_existing_result(self, video: CaptionedVideo, result_file: Path) -> MetricsRecordRaw | None:
        """
        Loads an existing result from disk.
        If eval_only is True, force re-evaluation of the loaded result.
        """
        try:
            with open(result_file, "r") as f:
                content = f.read()

            reconstructed = Reconstructed.model_validate_json(content)
            logging.info(f"Video {video.video_id} result found.")

            if self.eval_only:
                 logging.info(f"Eval-only: Re-evaluating {video.video_id}...")
                 # Force re-evaluation
                 # We need to make sure we have the proper ground truth (video)
                 # Note: evaluator.evaluate expects (reconstructed, orig_video)
                 
                 # Re-run evaluation
                 video_metrics = self.evaluator.evaluate(reconstructed, video)

                 # Update metrics in result
                 rounded = round_metrics(video_metrics)
                 reconstructed_with_metrics = reconstructed.with_metrics(rounded)
                 
                 # Save
                 self._save_result(reconstructed_with_metrics)
                 
                 # Return new record
                 # Advance PRN state just in case (e.g. for masking consistency if checked)
                 _, masked_indices = self._masking_strategy.mask_video(video)
                 
                 return MetricsRecordRaw(
                    raw_metrics=video_metrics,
                    metadata=MetricsMetadata(
                        video_id=video.video_id,
                        size=len(video.clips),
                        masked=list(masked_indices) if masked_indices else [],
                        recon_strategy=str(self._reconstruction_strategy),
                        data_type=self.data_loader.get_data_type_name()
                    )
                )

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
