import platform
from importlib.metadata import version
import logging
import os
import sys
import mlflow
from filelock import FileLock

# Local imports
from masking import get_masking_strategies
from evaluation import ReconstructionEvaluator
from utils import check_git_repository_is_clean, object_to_dict, setup_logging, get_notification_logger
from config_loader import load_config
from reconstruction_strategies import ReconstructionStrategyBuilder
from data_loaders import get_data_loader
from experiment_runner import ExperimentRunner
from exceptions import UserFacingError


def init():
    # --- 1. Pre-flight Checks and Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if len(sys.argv) < 2:
        raise UserFacingError("Please provide the path to the experiment config file.")

    return load_config(sys.argv[1])

def main(config):
    # mlflow_uri = config['paths']['mlflow_tracking_uri']
    # experiment_name=config['base_params']['experiment_name']

    notifier = get_notification_logger()

    git_commit_hash = check_git_repository_is_clean()

    # setup_mlflow(experiment_name=experiment_name, tracking_uri=mlflow_uri)
    # --- 2. The Experiment Loops ---
    with FileLock(".lock"):
        parent_run_name = config.get("batch_name", "ExperimentBatch")

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            log_path = setup_logging(log_dir=config['paths']['log_dir'], run_id=parent_run.info.run_id)
            print(f'{log_path = }')
            logging.info(f"--- Starting Experiment Batch: {parent_run_name} ---")

            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            logging.getLogger("transformers").setLevel(logging.ERROR)

            # Log reproducibility parameters
            mlflow.log_param("git_commit_hash", git_commit_hash)
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("mlflow_version", version('mlflow'))

            for runner, run_params in build_experiments(config):
                run_name = runner.run_name
                with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                    logging.info(f"--- Starting Nested Run: {run_name} ---")
                    mlflow.log_params(run_params)
                    metrics = runner.run()
                    mlflow.log_metrics(metrics)
                    log_message = (f"{run_name} Logged aggregated metrics on"
                                   f" {metrics['num_of_instances']} instances."
                                   f" Mean F1: {metrics['mean_f1_score']:.4f}"
                                   f" Mean P: {metrics['mean_precision']:.4f}"
                                   f" Mean R: {metrics['mean_recall']:.4f}")
                    logging.info(log_message)
                    notifier.info(log_message)
    return log_path
            
def build_experiments(config):
    data_loader = get_data_loader(config["data_config"])
    # --- Loop 1: Reconstruction Strategy ---
    eval_conf = config.get('evaluation', {})

    evaluator = ReconstructionEvaluator(
        model_type=eval_conf.get('model'),
        verbose=len(sys.argv) > 2 and sys.argv[2] == '--verbose'
    )
    evaluator.calc_idf(sents=[c.data.description for x in data_loader.load() for c in x.clips])

    rs_builder = ReconstructionStrategyBuilder(config)
    for strategy_params in config.get("recon_strategy", []):
        
        # Build the strategy object once for this block
        recon_strategy = rs_builder.get_strategy(strategy_params)
        masking_strategies = get_masking_strategies(
            masking_configs=config["masking_configs"],
            master_seed=config["base_params"]["master_seed"]
        )

        # --- Loop 2: Iterate over the generated masking strategies ---
        for masker in masking_strategies:
            # Build the final runner object with all components
            run_conf = {
                **config.get('base_params'),
                'data_config': config["data_config"],
                'masking': object_to_dict(masker),
                'recon_strategy': object_to_dict(recon_strategy)
            }
            runner = ExperimentRunner(
                run_name=f"{recon_strategy}__{masker}",
                data_loader=data_loader,
                masking_strategy=masker,
                reconstruction_strategy=recon_strategy,
                evaluator=evaluator
            )
            yield runner, run_conf

def done(log_path):
    logging.info(f'PID {os.getpid()} DONE.')
    print("\n✅ Finished successfully.")
    print("\nRun `mlflow ui` in your terminal to view the full results.")
    print("\nView log in", log_path)
    print()


if __name__ == "__main__":
    try:
        config = init()
        if len(sys.argv) > 2 and sys.argv[2]=='--dry-run':
            print(f"prepared {len(list(build_experiments(config)))} experiments")
        else:
            log_path = main(config)
            done(log_path)
    except UserFacingError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Experiment batch cancelled by user. Shutting down gracefully.")
        sys.exit(130) # 130 is the standard exit code for Ctrl+C
    except Exception as e:
        logging.error(f"Experiment failed with a critical error: {e}", exc_info=True)
        raise

