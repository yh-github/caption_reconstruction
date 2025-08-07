import platform
from importlib.metadata import version
import logging
import os
import sys

import diskcache
import mlflow
from filelock import FileLock

# Local imports
from masking import get_masking_strategies
from evaluation import ReconstructionEvaluator
from utils import check_git_repository_is_clean, setup_logging, flush_loggers, \
    setup_mlflow, get_datetime_str, flat_dict
from config_loader import load_config
from reconstruction_strategies import ReconstructionStrategyBuilder
from data_loaders import get_data_loader
from experiment_runner import ExperimentRunner
from exceptions import UserFacingError

cache:diskcache.Cache|None=None

def init():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if len(sys.argv) < 2:
        raise UserFacingError("Please provide the path to the experiment config file.")

    config = load_config(sys.argv[1])

    global cache
    cache = diskcache.Cache(directory=config['paths']['disk_cache'])

    return config

def main(config:dict) -> str:
    experiment_name = get_datetime_str(config.get('tz'))
    parent_run_name = config["__parent_run_name__"]+f" ({experiment_name})"
    mlflow_uri = config['paths']['mlflow_tracking_uri']

    git_commit_hash = check_git_repository_is_clean()

    with FileLock(".lock"):
        setup_mlflow(experiment_name=experiment_name, tracking_uri=mlflow_uri)
        with mlflow.start_run(run_name=parent_run_name) as parent_run, cache:
            log_path, notifier = setup_logging(
                log_dir=config['paths']['log_dir'],
                run_id=parent_run.info.run_id,
                tz_str=config.get('tz', None)
            )
            print(f'{log_path = }')
            start_msg = f"--- Starting Experiment Batch: {parent_run_name=} experiment_id={parent_run.info.experiment_id} ---"
            logging.info(start_msg)
            notifier.info(start_msg)

            # Log reproducibility parameters
            mlflow.log_param("git_commit_hash", git_commit_hash)
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("mlflow_version", version('mlflow'))

            for runner, run_params in build_experiments(config):
                run_name = runner.run_name
                with mlflow.start_run(run_name=run_name, nested=True):
                    logging.info(f"--- Starting Nested Run: {run_name} ---")
                    mlflow.log_params(run_params)
                    metrics, all_recon_videos = runner.run()

                    if all_recon_videos:
                        mlflow.log_text(text="\n".join(all_recon_videos), artifact_file='all_recon_videos.jsonl')

                    if metrics:
                        mlflow.log_metrics(metrics)
                        log_message = (f"{run_name} Logged aggregated metrics on"
                                       f" {metrics['num_of_instances']} instances."
                                       f" Mean F1: {metrics['mean_f1_score']:.4f}"
                                       f" Mean P: {metrics['mean_precision']:.4f}"
                                       f" Mean R: {metrics['mean_recall']:.4f}")
                        logging.info(log_message)
                        notifier.info(log_message)
                    else:
                        logging.error("No metrics were generated")
                    flush_loggers()
    return log_path
            
def build_experiments(config):
    data_loader = get_data_loader(config["data_config"])
    # --- Loop 1: Reconstruction Strategy ---
    eval_conf = config.get('evaluation', {})

    evaluator = ReconstructionEvaluator(
        model_type=eval_conf.get('model', 'microsoft/deberta-large-mnli'),
        verbose=len(sys.argv) > 2 and sys.argv[2] == '--verbose',
        idf=eval_conf.get('idf', True)
    )
    evaluator.calc_idf(sents=data_loader.load_all_sentences())

    rs_builder = ReconstructionStrategyBuilder(llm_cache=cache, master_seed=config["base_params"]["master_seed"])
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
            run_conf = flat_dict({
                '':config.get('base_params'),
                'data_config': config["data_config"],
                'masking': masker.get_params_for_repr(),
                'recon_strategy': strategy_params
            })
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
