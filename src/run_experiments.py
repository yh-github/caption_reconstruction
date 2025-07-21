import logging
import sys
import mlflow
from filelock import FileLock

# Local imports
from utils import check_git_repository_is_clean, setup_mlflow
from config_loader import load_config
from reconstruction_strategies import build_reconstruction_strategy

from experiment_runner import ExperimentRunner
from exceptions import UserFacingError


def init():
    # --- 1. Pre-flight Checks and Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    git_commit_hash = check_git_repository_is_clean()

    if len(sys.argv) < 2:
        raise UserFacingError("Please provide the path to the experiment config file.")

    return load_config(sys.argv[1])

def main(config):
    data_loader = get_data_loader(config["data_config"])
    
    # --- 2. The Experiment Loops ---
    with FileLock("experiment.lock"):
        parent_run_name = config.get("batch_name", "ExperimentBatch")
        
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            logging.info(f"--- Starting Experiment Batch: {parent_run_name} ---")
            
            # --- Loop 1: Reconstruction Strategy ---
            for strategy_params in config.get("recon_strategy", []):
                
                # Build the strategy object once for this block
                recon_strategy = build_reconstruction_strategy(strategy_params)
                masking_strategies = get_masking_strategies(
                    masking_configs=config["masking_configs"],
                    master_seed=config["master_seed"]
                )

                # --- Loop 2: Iterate over the generated masking strategies ---
                for masking_strategy in masking_strategies:
                    
                    # Create a unique run name for this specific combination
                    run_name = f"{strategy_params.get('name', 'strategy')}_{masking_strategy}"
                    
                    with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                        logging.info(f"--- Starting Nested Run: {run_name} ---")
                        
                        # Assemble the final config for this specific run
                        run_config = {
                            **config.get("base_params", {}), 
                            "strategy": strategy_params, 
                            "masking": masking_params
                        }
                        
                        setup_mlflow(run_config, git_commit_hash)

                        # --- 3. The "Doing" Part (Innermost Loop) ---
                        # Build the final runner object with all components
                        runner = ExperimentRunner(data_loader=data_loader, masker=masking_strategy, recon_strategy=recon_strategy)
                        # The actual execution happens here
                        runner.run()

def done():
    logging.info(f'PID {os.getpid()} DONE.')
    print("\n✅ Finished successfully.")
    print("\nRun `mlflow ui` in your terminal to view the full results.")


if __name__ == "__main__":
    try:
        config = init()
        main(config)
        done()
    except UserFacingError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.error(f"Experiment failed with a critical error: {e}", exc_info=True)
        raise


