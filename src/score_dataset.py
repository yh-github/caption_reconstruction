import logging
import json
import hashlib
import argparse
import concurrent.futures

from pathlib import Path

from dataclasses import asdict

from experiment_executor.config_loader import config_from_args

from data_models.exec_args import ExecArgs, DEFAULT_SYSTEM_CONFIG_PATH
from data.data_loaders import get_data_loader
from reconstruction.masking import get_masking_strategies
from llm.prior_surprise_score import PriorSurpriseScorer
from llm.pmi_scorer import PMIScorer
import torch
from data.hf_sync import HFResultsSync
from common_utils.tracking import get_datetime_str

def parse_scoring_args():
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
    parser.add_argument("--ignore-gpu", action="store_true", help="Allow running on CPU (checking logic only)")
    parser.add_argument("--hf-repo-id", type=str, default="Y3/dense_video_captions", help="HF Repo ID for sync")
    parser.add_argument("--upload-interval", type=int, default=10, help="Upload results every N videos")
    
    return parser.parse_args()

def main():
    # 1. Parse Args
    args = parse_scoring_args()
    
    # Create ExecArgs for config loader, excluding script-specific args
    exec_args_dict = {k: v for k, v in vars(args).items() if k not in ['calc_pmi', 'score_all']}
    exec_args = ExecArgs.model_validate(exec_args_dict)

    logging.basicConfig(level=exec_args.log_level(logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = config_from_args(exec_args)
    
    # extract model key from override or config, default to mistral
    model_key = config.get('scoring_model_key', 'mistral-v0.3')
    
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
    
    existing_data = syncer.pull()
    existing_scores = existing_data.get("scores", {})
    
    # 4. Initialize Scorers
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
    # Logic: If video_id in existing_scores, skip it.
    # Note: If we want to re-process partially done stuff, we'd need more logic. 
    # For now, simplistic "done is done".
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
    
    # videos_to_process = [v for v in all_videos if v.video_id not in existing_scores]
    
    logging.info(f"Loaded {len(all_videos)} videos. Found {len(existing_scores)} existing scores. Processing {len(videos_to_process)} new videos.")
    
    # 4. Initialize Masking Strategies (to identify what to score)
    masking_strategies = get_masking_strategies(
        masking_configs=config["masking_configs"],
        master_seed=config["base_params"]["master_seed"]
    )
    
    results = {}
    
    # Thread pool for background uploads
    upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def upload_checkpoint(syncer_ref, existing, current_results, curr_config, count):
        """Helper to run upload in background"""
        try:
            # Reconstruct the full data object to save
            merged = syncer_ref.merge_results(existing, current_results, curr_config)
            syncer_ref.push(merged, commit_message=f"Checkpoint {count} videos")
        except Exception as e:
            logging.warning(f"Background upload failed: {e}")

    # 5. Process Videos
    # 5. Process Videos
    for i, video in enumerate(videos_to_process):
        logging.info(f"[{i+1}/{len(videos_to_process)}] Scoring {video.video_id}...")
        
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
            surprisal_scores = prior_scorer.calculate_whole_log_surprisal(captions_text)
            # We aggregate for the whole video

            if surprisal_scores:
                avg_nll = sum(s.loss for s in surprisal_scores) / len(surprisal_scores)
                avg_ppl = sum(s.perplexity for s in surprisal_scores) / len(surprisal_scores)
                video_result['whole_video_surprisal'] = {
                    "avg_surprisal_nll": avg_nll,
                    "avg_perplexity": avg_ppl,
                    "measurements": [asdict(s) for s in surprisal_scores]
                }
        except Exception as e:
            logging.error(f"Failed Surprisal for {video.video_id}: {e}")
            
        # B. PMI for Masked Segments
        if args.calc_pmi:
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
        
    # 6. Merge & Push Results
    logging.info("Processing complete. Waiting for background uploads to finish...")
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

if __name__ == "__main__":
    main()
