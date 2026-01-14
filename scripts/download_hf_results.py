
import argparse
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Add src to path to import project modules
# assuming script is in scripts/ and src is in src/ (sibling to scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from experiment_executor.config_loader import load_config

DEFAULT_REPO_ID = "Y3/dense_video_captions"
DEFAULT_TARGET_DIR = "results/recon/manual_download"

def main():
    parser = argparse.ArgumentParser(description="Download results from HF based on config.")
    parser.add_argument("--config", type=Path, help="Path to experiment config file.")
    args = parser.parse_args()

    repo_id = DEFAULT_REPO_ID
    target_dir = DEFAULT_TARGET_DIR
    allow_patterns = None # Default: download everything (or restricted if we want)

    if args.config:
        print(f"Loading config from {args.config}...")
        try:
            # We assume system config is at standard location
            config = load_config(args.config)
            
            # 1. Repo ID
            if config.get('paths') and config['paths'].get('hf_repo_id'):
                repo_id = config['paths']['hf_repo_id']
            
            # 2. Target Directory
            # format: results/recon/{parent_run_name}
            # We should probably respect the 'results' path in config
            results_base = config.get('paths', {}).get('results', 'results')
            parent_run_name = config.get('__parent_run_name__', args.config.stem)
            target_dir = str(Path(results_base) / "recon" / parent_run_name)
            
            # 3. Patterns
            # match: reconstruction/PARENT_RUN_NAME/**/*STRATEGY_NAME*/*.json
            # We accumulate patterns for all strategies
            patterns = []
            
            strategies = config.get('recon_strategy', [])
            for strat in strategies:
                s_name = strat.get('name')
                if s_name:
                    # e.g. reconstruction/wild_dev_sim_text/**/*phi-3__t=0.1_rp=1.2*/*.json
                    # Note: The run name on HF is usually {strategy_name}__{masking_strategy}
                    # checking wild_dev_sim_text.yaml, names are like "phi-3__t=0.1_rp=1.2"
                    # We want to match any folder containing this string.
                    patterns.append(f"reconstruction/{parent_run_name}/**/*{s_name}*/*.json")
            
            if patterns:
                allow_patterns = patterns
            
            print(f"Config loaded.")
            print(f" - Repo: {repo_id}")
            print(f" - Target: {target_dir}")
            print(f" - Patterns: {len(patterns)} strategies found.")
            if patterns:
                for p in patterns[:5]:
                     print(f"   e.g. {p}")

        except Exception as e:
            print(f"Error loading config: {e}")
            return
    else:
        print("No config provided. Using defaults.")
        # Default behavior: match the manual hardcoded logic from before?
        # Or download everything?
        # The previous version had: reconstruction/**/*phi-3*t=0.1*/*.json and fixed target
        allow_patterns = "reconstruction/**/*phi-3*t=0.1*/*.json"

    print(f"Starting sync from {repo_id} to {target_dir}...")
    
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            # local_dir_use_symlinks argument is deprecated and ignored in newer versions
            allow_patterns=allow_patterns,
            # ignore_patterns=["*"], # implicit if allow_patterns is set? No, allow_patterns whitelist.
            tqdm_class=None, # Use default tqdm
        )
        print(f"Sync complete. Files located in: {local_path}")
        
    except Exception as e:
        print(f"Sync failed: {e}")

if __name__ == "__main__":
    main()
