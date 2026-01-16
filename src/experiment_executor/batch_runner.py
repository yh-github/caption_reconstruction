import logging
from pathlib import Path
from typing import Any
from data.data_loaders import BaseDataLoader
from data_models.captions_only import CaptionedVideo
from evaluations.evaluation import ReconstructionEvaluator
from evaluations.metrics import metrics_to_json, round_metrics, MetricsMetadata, MetricsRecordRaw
from reconstruction.masking import MaskingStrategy
from reconstruction.text_reconstruction import ReconstructionStrategy, Reconstructed, BatchGridSearchStrategy
from experiment_executor.experiment_runner import ExperimentRunner

class BatchExperimentRunner:
    """
    Orchestrates MULTIPLE experiments (for same video, same masking, but different model params)
    in a single parallel pass using BatchGridSearchStrategy.
    Wraps multiple "logical" ExperimentRunners but executes them physically together.
    """
    def __init__(
        self,
        base_run_name: str,
        runners: list[ExperimentRunner], 
        batch_strategy: BatchGridSearchStrategy,
        data_loader: BaseDataLoader,
        masking_strategy: MaskingStrategy,
        evaluator: ReconstructionEvaluator,
        no_download_existing: bool = False
    ):
        self.run_name = base_run_name
        self.runners = runners
        self.batch_strategy = batch_strategy
        self.data_loader = data_loader
        self._masking_strategy = masking_strategy
        self.evaluator = evaluator
        self.no_download_existing = no_download_existing
        self.conf_for_log = {'batch_size': len(runners), 'runners': [r.run_name for r in runners]}
        
        # Batch runner doesn't have a single remote path, as it manages multiple runners.
        # However, for logging/monitoring purposes that might inspect this property,
        # we can point to a common parent or the first runner's path.
        # Assuming all runners in a batch share the exact same config_stem:
        if runners:
            self.remote_run_path = runners[0].remote_run_path
        else:
            self.remote_run_path = f"reconstruction/batch_empty/{base_run_name}"

    def run(self) -> list[MetricsRecordRaw]:
        # Initialize all sub-runners (create dirs, sync HF, etc)
        for runner in self.runners:
            runner._sync_hf_state()
            runner._save_path.mkdir(parents=True, exist_ok=True)

        all_videos: list[CaptionedVideo] = self.data_loader.load()
        all_metrics: list[MetricsRecordRaw] = []

        for video in all_videos:
            metrics = self._process_single_video_batch(video)
            all_metrics.extend(metrics)

        return all_metrics

    def _process_single_video_batch(self, video: CaptionedVideo) -> list[MetricsRecordRaw]:
        # 1. Determine which configs actually need to run
        active_indices = []
        
        # Check each runner to see if it already has a result
        for i, runner in enumerate(self.runners):
             filename = runner._filename(video.video_id)
             result_file = runner._save_path / filename
             
             # If result exists locally, we don't need to run it (unless eval-only mode, but ignoring for now)
             # Also check remote? For now using runner's internal check logic would be complex 
             # because runner.run() does it all.
             # We reproduce the check logic here briefly:
             
             exists_locally = result_file.exists()
             exists_remotely = runner.hf_manager and filename in runner.remote_files
             
             if not exists_locally and not exists_remotely:
                 active_indices.append(i)
             elif exists_remotely and not exists_locally:
                 if self.no_download_existing:
                     # Treat as "done/exists" without downloading
                     continue
                 
                 # Try download
                 # If download fails, add to active
                 logging.info(f"BatchRunner: Downloading {video.video_id} for {runner.run_name}...")
                 remote_path = f"{runner.remote_run_path}/{filename}"
                 if runner.hf_manager.download_file(remote_path, result_file):
                     continue # Done
                 else:
                     active_indices.append(i)
        
        # 2. If no work needed, just load and return existing metrics
        if not active_indices:
             # Load all metrics to return them (for aggregation)
             return self._load_all_metrics(video)

        logging.info(f"BatchRunner: Running {len(active_indices)}/{len(self.runners)} active configs for {video.video_id}")
        
        # 3. Mask Video (Shared)
        masked_video, masked_indices = self._masking_strategy.mask_video(video)
        if not masked_video:
             # Logic for "skip" or "error" needs to apply to all active runners
             # For simplicity, we skip this video for now or return error record
             return []

        # 4. Run Batch Inference
        batch_results = self.batch_strategy.reconstruct(masked_video, active_indices=active_indices)
        
        results_metrics = []

        # 5. Distribute Results to Runners and Evaluate
        for i, result in enumerate(batch_results):
            # i corresponds to index in self.runners (mapped 1:1 with configs)
            # if i was NOT in active_indices, result is a skipped placeholder.
            
            runner = self.runners[i]
            
            if result.skip_reason == "batch_inactive":
                # Already exists, load it
                if m := runner._load_existing_result(video, runner._save_path / runner._filename(video.video_id)):
                    results_metrics.append(m)
                continue
                
            # Process new result
            # (Logic copied from ExperimentRunner._run_new_experiment)
            
            # Basic validation
            if not result.debug_data and result.reconstructed_captions.keys() != masked_indices:
                logging.warning(f"Batch Result mismatch for {runner.run_name}")
                # Save as skipped/error
                runner._save_result(result.skip(f"mismatch {result.reconstructed_captions.keys()}!={masked_indices}"))
                continue
                
            # Evaluate
            video_metrics = runner.evaluator.evaluate(result, video)
            
            raw_record = MetricsRecordRaw(
                raw_metrics=video_metrics,
                metadata=MetricsMetadata(
                    video_id=video.video_id,
                    size=len(video.clips),
                    masked=list(masked_indices),
                    recon_strategy=str(runner._reconstruction_strategy),
                    data_type=self.data_loader.get_data_type_name()
                )
            )
            
            rounded = round_metrics(video_metrics)
            runner._save_result(result.with_metrics(rounded))
            
            results_metrics.append(raw_record)
            
        return results_metrics

    def _load_all_metrics(self, video) -> list[MetricsRecordRaw]:
        res = []
        for runner in self.runners:
             if m := runner._load_existing_result(video, runner._save_path / runner._filename(video.video_id)):
                 res.append(m)
        return res
