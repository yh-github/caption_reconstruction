import logging
import os
import platform
from importlib.metadata import version
import diskcache
import mlflow
from filelock import FileLock
from google import genai

from config_loader import load_config
from data_loaders import get_data_loader
from data_models.exec_args import ExecArgs
from evaluation import ReconstructionEvaluator
from experiment_runner import ExperimentRunner
# Local imports
from masking import get_masking_strategies
from reconstruction_strategies import ReconstructionStrategyBuilder
from utils import check_git_repository_is_clean, setup_logging, flush_loggers, \
    setup_mlflow, get_datetime_str, flat_dict

from unittest.mock import Mock

class ExperimentPipeline:

    def __init__(self, exec_args:ExecArgs):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.exec_args = exec_args
        self.config = None

        self.config = load_config(self.exec_args.config_path)
        self.cache = diskcache.Cache(directory=self.config['paths']['disk_cache'])

        if self.exec_args.dry_run or self.exec_args.validate_cache:
            logging.info("Running in dry-run mode. Mocking LLM client.")
            llm_client = self._create_mock_llm_client()
        else:
            llm_client = genai.Client()

        self.rs_builder = ReconstructionStrategyBuilder(
            llm_cache=self.cache,
            master_seed=self.config["base_params"]["master_seed"],
            llm_client=llm_client
        )

        self.log_path = None
        self.mlflow_run_path = None

    @staticmethod
    def _create_mock_llm_client():
        """
        Creates a mock for `llm_client`, raising exceptions with method name
        and parameters when methods are called.
        """
        llm_mock = Mock()

        # Dynamically simulate an exception when any attribute or method is accessed
        def mock_getattr(name):
            def raise_exception(*args, **kwargs):
                raise RuntimeError(
                    f"llm_client: Attempted to call method '{name}' with args: {args}, kwargs: {kwargs}"
                )
            return raise_exception

        llm_mock.__getattr__.side_effect = mock_getattr

        return llm_mock

    def main(self):
        experiment_name = get_datetime_str(self.config.get('tz'))
        parent_run_name = self.config["__parent_run_name__"]+f" ({experiment_name})"
        mlflow_uri = self.config['paths']['mlflow_tracking_uri']

        git_commit_hash = check_git_repository_is_clean()

        with FileLock(".lock"):
            setup_mlflow(experiment_name=experiment_name, tracking_uri=mlflow_uri)
            with mlflow.start_run(run_name=parent_run_name) as parent_run, self.cache:
                log_path, notifier = setup_logging(
                    log_dir=self.config['paths']['log_dir'],
                    run_id=parent_run.info.run_id,
                    tz_str=self.config.get('tz', None)
                )
                self.log_path = log_path

                print(f'{log_path = }')
                start_msg = f"--- Starting Experiment Batch: {parent_run_name=} experiment_id={parent_run.info.experiment_id} ---"
                self.mlflow_run_path = str(os.path.join(mlflow_uri.removeprefix("file:"),parent_run.info.experiment_id))

                logging.info(start_msg)
                notifier.info(start_msg)

                # Log reproducibility parameters
                mlflow.log_param("git_commit_hash", git_commit_hash)
                mlflow.log_param("python_version", platform.python_version())
                mlflow.log_param("mlflow_version", version('mlflow'))

                for runner, run_params in self.build_experiments():
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
            
    def build_experiments(self):
        config = self.config
        data_loader = get_data_loader(config["data_config"])
        # --- Loop 1: Reconstruction Strategy ---
        eval_conf = config.get('evaluation', {})

        evaluator = ReconstructionEvaluator(
            model_type=eval_conf.get('model', 'microsoft/deberta-large-mnli'),
            verbose=self.exec_args.verbose,
            idf=eval_conf.get('idf', True)
        )
        evaluator.calc_idf(sents=data_loader.load_all_sentences())

        for strategy_params in config.get("recon_strategy", []):

            # Build the strategy object once for this block
            recon_strategy = self.rs_builder.get_strategy(strategy_params)

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

    def done(self):
        logging.info(f'PID {os.getpid()} DONE.')
        print(f"\n✅ Finished successfully.")
        print(f"\nRun `mlflow ui` in your terminal to view the full results.")
        print(f"\nRun `python scripts/mlflow_runs.py {self.mlflow_run_path}` for command-line access.")
        print(f"\nView log in {self.log_path}")
        print()
