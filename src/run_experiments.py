import logging
import mlflow
from filelock import FileLock

# Local imports
from utils import check_git_repository_is_clean, setup_mlflow
from config_loader import load_config
from strategy_builder import setup_experiment_strategy
from experiment_runner import ExperimentRunner

def main():
    """
    A simple, single-script orchestrator for running a batch of experiments
    defined by a single configuration file and nested loops.
    """
    # --- 1. Pre-flight Checks and Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    git_commit_hash = check_git_repository_is_clean()
    
    # This single config file now defines the entire batch
    config = load_config("config/experiment_batch_01.yaml") 

    data_loader = get_data_loader(config["data_config"])
    all_videos = data_loader.load()
    
    # --- 2. The Experiment Loops ---
    with FileLock("experiment.lock"):
        parent_run_name = config.get("batch_name", "ExperimentBatch")
        
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            logging.info(f"--- Starting Experiment Batch: {parent_run_name} ---")
            
            # --- Loop 1: Reconstruction Strategy ---
            for strategy_params in config.get("strategies", []):
                
                # Build the strategy object once for this block
                strategy = setup_experiment_strategy(strategy_params)
                masking_strategies = get_masking_strategies(
                    masking_configs=config["masking_configs"],
                    master_seed=config["master_seed"]
                )

                # --- Loop 2: Iterate over the generated masking strategies ---
                for masking_strategy in masking_strategies:
                    
                    # Create a unique run name for this specific combination
                    run_name = f"{strategy_params.get('name', 'strategy')}_{masking_params.get('scheme', 'masking')}"
                    
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
                        runner = ExperimentRunner(run_config, strategy)
                        # The actual execution happens here
                        runner.run()

if __name__ == "__main__":
    main()
