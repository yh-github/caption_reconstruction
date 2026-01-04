import logging
import json
import argparse
from pathlib import Path
from typing import Any

from experiment_executor.config_loader import config_from_args
from data_models.exec_args import args_parser
from data.data_loaders import get_data_loader
from reconstruction.masking import get_masking_strategies
from llm.prior_surprise_score import PriorSurpriseScorer
from llm.pmi_scorer import PMIScorer
from common_utils.tracking import get_datetime_str

def main():
    # 1. Parse Args & Config
    # We extend the base parser to add model selection
    # Since args_parser returns a Pydantic model immediately, we might need a separate parser 
    # or just rely on 'override' to set a param? 
    # Let's just use the standard ExecArgs and hardcode the model selection or read from config if present.
    # To keep it simple for now, I'll default to 'mistral-v0.3' or allow override via env/config.
    # Actually, let's just use a separate arg parser for the score script parts + ExecArgs for the rest?
    # It's easier to just use the existing ExecArgs and maybe assume model_key is in config or just use a default.
    # Better: Let's just use argparse here manually for the model_key and then call args_parser() if needed, 
    # but args_parser() parses sys.argv.
    
    # Simpler approach: Just look at sys.argv for a --model-key if I want custom, 
    # but let's stick to the pattern: Config drives everything.
    # I'll check if 'scoring_model' is in the config, else default.
    
    exec_args = args_parser()
    logging.basicConfig(level=exec_args.log_level(logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = config_from_args(exec_args)
    
    # extract model key from override or config, default to mistral
    model_key = config.get('scoring_model_key', 'mistral-v0.3')
    
    logging.info(f"Starting Difficulty Scoring with model: {model_key}")
    
    # 2. Initialize Scorers
    prior_scorer = PriorSurpriseScorer(model_key=model_key)
    pmi_scorer = PMIScorer(model_key=model_key)
    
    # 3. Load Data
    data_config = config["data_config"]
    data_loader = get_data_loader(data_config)
    all_videos = data_loader.load()
    
    # 4. Initialize Masking Strategies (to identify what to score)
    masking_strategies = get_masking_strategies(
        masking_configs=config["masking_configs"],
        master_seed=config["base_params"]["master_seed"]
    )
    
    results = {}
    
    # 5. Process Videos
    for i, video in enumerate(all_videos):
        logging.info(f"[{i+1}/{len(all_videos)}] Scoring {video.video_id}...")
        
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
                avg_loss = sum(s['loss'] for s in surprisal_scores) / len(surprisal_scores)
                video_result['whole_video_surprisal'] = {
                    "avg_loss": avg_loss,
                    "measurements": surprisal_scores
                }
        except Exception as e:
            logging.error(f"Failed Surprisal for {video.video_id}: {e}")
            
        # B. PMI for Masked Segments
        # We iterate through all masking strategies to find coverage
        labels_processed = set()
        
        # Check command line for --score-all arg (primitive arg parsing for now)
        score_all = "--score-all" in sys.argv
        
        indices_to_score = []
        if score_all:
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
        
    # 6. Save Results
    output_dir = Path("results/scores")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = config['base_params'].get('run_name', 'default')
    timestamp = get_datetime_str()
    filename = f"scores_{run_name}_{model_key}_{timestamp}.json"
    
    final_output = {
        "metadata": {
            "run_name": run_name,
            "model_key": model_key,
            "timestamp": timestamp,
            "config": config  # Saving the exact config used
        },
        "scores": results
    }
    
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    logging.info(f"Saved scores to {output_path}")

if __name__ == "__main__":
    main()
