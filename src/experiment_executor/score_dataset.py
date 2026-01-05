
import logging
import json
import hashlib
import argparse
import concurrent.futures

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

from experiment_executor.config_loader import config_from_args

from data_models.exec_args import ExecArgs, DEFAULT_SYSTEM_CONFIG_PATH
from data.data_loaders import get_data_loader
from reconstruction.masking import get_masking_strategies
from llm.prior_surprise_score import PriorSurpriseScorer
from llm.pmi_scorer import PMIScorer
import torch
from data.hf_sync import HFResultsSync
from common_utils.tracking import get_datetime_str

def log_gpu_stats(context=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"[GPU {context}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

@dataclass
class ScoringResources:
    config: dict[str, Any]
    prior_scorer: PriorSurpriseScorer
    pmi_scorer: PMIScorer | None
    data_loader: Any 
    masking_strategies: list
    syncer: HFResultsSync
    existing_data: dict[str, Any]
    model_key: str
    videos_to_process: list

def parse_scoring_args(arg_list: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Scoring dataset wrapper")
    
    # Standard arguments required for ExecArgs/ConfigLoader
    parser.add_argument("config_path", type=Path, help="Path to experiment config")
    parser.add_argument("--system_config_path", type=Path, default=DEFAULT_SYSTEM_CONFIG_PATH)
    parser.add_argument("--override", nargs='+', default=[], help="Config overrides (KEY=VALUE)")
    
    # Standard Execution flags
    parser.add_argument("--debug", action="store_true")
    
    # Script-specific flags
    parser.add_argument("--calc-pmi", action="store_true", help="Calculate PMI (slow, logic heavy)")
    parser.add_argument("--score-all", action="store_true", help="Score all clips instead of masked ones")
    parser.add_argument("--calc-attn-distance", action="store_true", help="Calculate Attention Distance (Memory heavy!)")
    parser.add_argument("--ignore-gpu", action="store_true", help="Allow running on CPU (checking logic only)")
    parser.add_argument("--hf-repo-id", type=str, default="Y3/dense_video_captions", help="HF Repo ID for sync")
    parser.add_argument("--upload-interval", type=int, default=10, help="Upload results every N videos")
    
    return parser.parse_args(arg_list)

def setup_scoring_resources(args, force_pull: bool = False) -> ScoringResources:
    """
    Initializes all resources (models, config, data, sync) needed for scoring.
    Useful for interactive use in Colab.
    """
    # Create ExecArgs for config loader, excluding script-specific args
    exec_args_dict = {k: v for k, v in vars(args).items() if k not in ['calc_pmi', 'score_all', 'ignore_gpu', 'hf_repo_id', 'upload_interval']}
    exec_args = ExecArgs.model_validate(exec_args_dict)

    logging.basicConfig(level=exec_args.log_level(logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = config_from_args(exec_args)
    
    # extract model key from override or config, default to phi-3
    model_key = config.get('scoring_model_key', 'phi-3')
    
    logging.info(f"Starting Difficulty Scoring with model: {model_key}")
    
    # 2. GPU Check
    if not args.ignore_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU not found. Aborting to prevent heavy CPU usage. Use --ignore-gpu to override.")

    # 3. HF Sync Setup & Existing Data Retrieval
    # Generate hyperparameter hash (moved up for early sync)
    hyperparams = {
        "model_key": model_key,
        "masking_configs": config.get("masking_configs"),
        "data_config": config.get("data_config"),
        "seed": config.get("base_params", {}).get("master_seed")
    }
    hyperparams_hash = hashlib.sha256(json.dumps(hyperparams, sort_keys=True, default=str).encode()).hexdigest()[:12]
    run_name = config['base_params'].get('run_name', 'default')

    syncer = HFResultsSync(
        repo_id=args.hf_repo_id,
        run_name=run_name, 
        hyperparams_hash=hyperparams_hash,
        output_dir=Path("results/scores")
    )
    
    if syncer.local_path.exists():
        logging.info(f"Found existing local results file: {syncer.local_path}")
    
    existing_data = syncer.pull(force_download=force_pull)
    existing_scores = existing_data.get("scores", {})
    
    # 4. Initialize Scorers
    logging.info(f"Initializing models for {model_key}. This may take a minute...")
    prior_scorer = PriorSurpriseScorer(model_key=model_key)
    
    # Check for PMI flag
    pmi_scorer = None
    if args.calc_pmi:
        pmi_scorer = PMIScorer(model_key=model_key)
    
    # 3. Load Data
    data_config = config["data_config"]
    data_loader = get_data_loader(data_config)
    all_videos = data_loader.load()
    
    # Filter videos that are already fully processed
    videos_to_process = []
    
    for v in all_videos:
        # Case 1: Video is completely new
        if v.video_id not in existing_scores:
            videos_to_process.append(v)
            continue
            
        # Case 2: Video exists, but maybe we need to add PMI data?
        if args.calc_pmi:
            existing_entry = existing_scores[v.video_id]
            # Check if PMI data is missing
            if not existing_entry.get("segments_pmi"):
                logging.info(f"Video {v.video_id} exists but missing PMI. Re-queueing.")
                videos_to_process.append(v)

    logging.info(f"Loaded {len(all_videos)} videos. Found {len(existing_scores)} existing scores. Video re-queueing determined {len(videos_to_process)} videos to process.")
    
    # 4. Initialize Masking Strategies (to identify what to score)
    masking_strategies = get_masking_strategies(
        masking_configs=config["masking_configs"],
        master_seed=config["base_params"]["master_seed"]
    )
    
    return ScoringResources(
        config=config,
        prior_scorer=prior_scorer,
        pmi_scorer=pmi_scorer,
        data_loader=data_loader,
        masking_strategies=masking_strategies,
        syncer=syncer,
        existing_data=existing_data,
        model_key=model_key,
        videos_to_process=videos_to_process
    )

def refresh_resources(resources: ScoringResources, args, force_pull: bool = False):
    """
    Refreshes the sync state and video queue without re-initializing models or scorers.
    Useful if you've deleted files on HF or want to pick up new work without a reload.
    """
    logging.info("Refreshing scoring state (re-syncing with HF and re-loading data)...")
    
    # 1. Re-pull from sync (forces a fresh check of the remote/disk)
    existing_data = resources.syncer.pull(force_download=force_pull)
    existing_scores = existing_data.get("scores", {})
    
    # 2. Re-load data and re-filter
    # Note: We use the existing data_loader
    all_videos = resources.data_loader.load()
    videos_to_process = []
    
    for v in all_videos:
        if v.video_id not in existing_scores:
            videos_to_process.append(v)
            continue
            
        if args.calc_pmi:
            existing_entry = existing_scores[v.video_id]
            if not existing_entry.get("segments_pmi"):
                videos_to_process.append(v)

    # 3. Update the resources object in-place
    resources.existing_data = existing_data
    resources.videos_to_process = videos_to_process
    
    logging.info(f"Refresh complete. Found {len(existing_scores)} existing scores. Queue now has {len(videos_to_process)} videos.")


def upload_checkpoint(syncer_ref, existing, current_results, curr_config, count):
    """Helper to run upload in background"""
    try:
        # Reconstruct the full data object to save
        merged = syncer_ref.merge_results(existing, current_results, curr_config)
        syncer_ref.push(merged, commit_message=f"Checkpoint {count} videos")
    except Exception as e:
        logging.warning(f"Background upload failed: {e}")

def run_scoring_loop(resources: ScoringResources, args):
    """
    Executes the main scoring loop. using resources prepared by setup_scoring_resources.
    """
    results = {}
    
    # Thread pool for background uploads
    upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # 5. Process Videos
    videos_to_process = resources.videos_to_process
    logging.info(f"run_scoring_loop: received {len(videos_to_process)} videos to process.")
    
    if not videos_to_process:
        logging.info("Nothing to process. Returning.")
        return

    prior_scorer = resources.prior_scorer
    pmi_scorer = resources.pmi_scorer
    masking_strategies = resources.masking_strategies
    syncer = resources.syncer
    existing_data = resources.existing_data
    config = resources.config
    
    try:
        for i, video in enumerate(videos_to_process):
            logging.info(f"[{i+1}/{len(videos_to_process)}] Scoring {video.video_id}...")
            if torch.cuda.is_available():
                log_gpu_stats("Start Video")
            
            video_result = {
                "video_id": video.video_id,
                "segments_pmi": [],
                "whole_video_surprisal": None
            }
            
            # A. Whole Video Surprisal (Constraint: Model context length. Captions are usually short enough)
            captions_text = [f"[{c.timestamp.start:02.0f}:{int((c.timestamp.start%60)*100)%100}] {c.caption}" for c in video.clips if c.caption]
            # Rough format approximation matching the scorer's expectation
            # The scorer expects list of strings.
            
            try:
                surprisal_scores = prior_scorer.calculate_whole_log_surprisal(
                    captions_text,
                    calc_attn_dist=args.calc_attn_distance
                )
                # We aggregate for the whole video

                if surprisal_scores:
                    avg_nll = sum(s.loss for s in surprisal_scores) / len(surprisal_scores)
                    avg_ppl = sum(s.perplexity for s in surprisal_scores) / len(surprisal_scores)
                    video_result['whole_video_surprisal'] = {
                        "avg_surprisal_nll": avg_nll,
                        "avg_perplexity": avg_ppl,
                        "measurements": [asdict(s) for s in surprisal_scores]
                    }
            except torch.cuda.OutOfMemoryError:
                logging.error(f"FATAL: CUDA OOM during Surprisal for video {video.video_id}.")
                log_gpu_stats("OOM State")
                torch.cuda.empty_cache()
                log_gpu_stats("Post-EmptyCache")
                logging.error("Halting execution to prevent unstable state.")
                raise
            except Exception as e:
                logging.error(f"Failed Surprisal for {video.video_id}: {e}")
                
            # B. PMI for Masked Segments
            if args.calc_pmi and pmi_scorer:
                # We iterate through all masking strategies to find coverage
                labels_processed = set()
                
                # Check command line for --score-all arg
                indices_to_score = []
                if args.score_all:
                     indices_to_score = [c.index for c in video.clips if c.caption]
                else:
                    for masker in masking_strategies:
                        _, masked_indices = masker.mask_video(video)
                        if masked_indices:
                            indices_to_score.extend(list(masked_indices))
                
                # Deduplicate
                indices_to_score = sorted(list(set(indices_to_score)))

                # BATC PREPARATION
                batch_indices = []
                batch_ctx_before = []
                batch_ctx_after = []
                batch_targets = []

                for idx in indices_to_score:
                    # Context Window: Effectively unlimited
                    WINDOW_SIZE = 500
                    start_before = max(0, idx - WINDOW_SIZE)
                    
                    # We need simple text
                    context_before_clips = video.clips[start_before:idx]
                    context_after_clips = video.clips[idx+1 : idx+1+WINDOW_SIZE] # Slice handles OOB
                    
                    def fmt(cl): return f"[{cl.timestamp.start:.0f}s] {cl.caption}"
                    
                    ctx_before = "\n".join(fmt(c) for c in context_before_clips if c.caption)
                    ctx_after = "\n".join(fmt(c) for c in context_after_clips if c.caption)
                    target_line = fmt(video.clips[idx])
                    
                    batch_indices.append(idx)
                    batch_ctx_before.append(ctx_before)
                    batch_ctx_after.append(ctx_after)
                    batch_targets.append(target_line)

                # BATCH EXECUTION
                if batch_indices:
                    try:
                        # Process in chunks of 16 to avoid OOM even with batching
                        CHUNK_SIZE = 16
                        for i in range(0, len(batch_indices), CHUNK_SIZE):
                            chunk_slice = slice(i, i + CHUNK_SIZE)
                            
                            chunk_res = pmi_scorer.calculate_informativeness_batch(
                                batch_ctx_before[chunk_slice], 
                                batch_ctx_after[chunk_slice], 
                                batch_targets[chunk_slice]
                            )
                            
                            for j, res in enumerate(chunk_res):
                                res['clip_index'] = batch_indices[i+j]
                                video_result['segments_pmi'].append(res)
                                
                    except torch.cuda.OutOfMemoryError:
                        logging.error(f"FATAL: CUDA OOM during PMI for video {video.video_id}. Halting execution.")
                        raise
                    except Exception as e:
                        logging.error(f"Failed Batch PMI for {video.video_id}: {e}")

            results[video.video_id] = video_result
            
            # Checkpoint Upload
            if (i + 1) % args.upload_interval == 0:
                logging.info(f"Checkpoint: triggering background upload for {len(results)} videos...")
                # We copy results to ensure thread safety while main loop continues
                upload_executor.submit(
                    upload_checkpoint, 
                    syncer, 
                    existing_data, 
                    results.copy(), 
                    config, 
                    len(results)
                )
            
            # Free up memory after each video
            torch.cuda.empty_cache()
    finally:
        # 6. Merge & Push Results
        logging.info("Shutting down scoring loop. Waiting for background uploads to finish...")
        upload_executor.shutdown(wait=True)
    
    if results:
        merged_data = syncer.merge_results(existing_data, results, config)
        syncer.push(merged_data)
        
        # 7. Verification / Sanity Check
        logging.info("Verifying upload integrity via forced pull...")
        try:
            remote_data = syncer.pull(force_download=True)
            local_count = len(merged_data.get("scores", {}))
            remote_count = len(remote_data.get("scores", {}))
            
            if local_count == remote_count:
                logging.info(f"VERIFICATION SUCCESS: Remote has {remote_count} videos (Matches local).")
            else:
                logging.error(f"VERIFICATION FAILED: Remote has {remote_count} videos, expected {local_count}.")
                # Don't raise crash exception to ensure we don't hide the fact we finished, but log error loudly.
        except Exception as e:
            logging.error(f"Verification process failed: {e}")

    else:
        logging.info("No new results to save.")

def main():
    args = parse_scoring_args()
    try:
        resources = setup_scoring_resources(args)
        run_scoring_loop(resources, args)
    except Exception as e:
        logging.exception(f"Fatal error in main loop: {e}")
        # We don't want to swallow errors, but we might want to ensure logs are flushed
        raise e

if __name__ == "__main__":
    main()
