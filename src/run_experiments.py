import logging
import os
import platform
from abc import abstractmethod, ABC
from importlib.metadata import version
from pathlib import Path
from typing import Any

import diskcache
import mlflow
from filelock import FileLock
from google import genai

from config_loader import load_config
from data_loaders import get_data_loader
from data_models.exec_args import ExecArgs
from embedder import Embedder
from evaluation import ReconstructionEvaluator_BertScore, EvaluatorNOP, ReconstructionEvaluator, metrics_to_json, \
    ReconstructionEvaluator_EmbSimilarity
from experiment_runner import ExperimentRunner
# Local imports
from masking import get_masking_strategies
from reconstruction_strategies import ReconstructionStrategyBuilder
from utils import check_git_repository_is_clean, setup_logging, flush_loggers, \
    setup_mlflow, get_datetime_str, flat_dict, UserFacingError, ExceptionStr

from unittest.mock import Mock

class ExperimentPipeline(ABC):

    @staticmethod
    def build(exec_args:ExecArgs):
        logging.basicConfig(level=exec_args.log_level(logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
        config = load_config(exec_args.config_path)
        experiment_type = config['base_params'].get('experiment_type', 'recon').upper()
        if experiment_type == 'RECON':
            return ExperimentPipeline_Reconstruction(exec_args, config)
        # elif experiment_type == 'QA':
        #     return ExperimentPipeline_QA(exec_args, config)
        else:
            raise UserFacingError(f"Unknown {experiment_type=}")

    def __init__(self, exec_args:ExecArgs, config:dict[str, Any]):
        self.experiment_type = config['base_params'].get('experiment_type', 'recon').upper()
        self.exec_args = exec_args
        self.config = config

        self.cache = diskcache.Cache(directory=self.config['paths']['disk_cache'])

        self.data_loader = get_data_loader(self.config["data_config"])

        self.evaluator: ReconstructionEvaluator = EvaluatorNOP()
        if self.exec_args.dry_run or self.exec_args.validate_cache:
            logging.info("Running in dry-run mode. Blocking LLM client and Evaluator set to NOP.")
            self._llm_client = self._create_mock_llm_client()
            self.evaluator = EvaluatorNOP()
        else:
            self._llm_client = genai.Client()

            eval_conf = self.config.get('evaluation', {})
            if self.experiment_type == 'RECON':
                eval_type = eval_conf.get('type', 'bert_score')
                if eval_type == 'bert_score':
                    self.evaluator = ReconstructionEvaluator_BertScore(
                        model_type=eval_conf.get('model', 'microsoft/deberta-large-mnli'),
                        verbose=self.exec_args.verbose or eval_conf.get('verbose', False),
                        idf=eval_conf.get('idf', True)
                    ).calc_idf(sents=self.data_loader.load_all_sentences())
                elif eval_type == 'emb_sim':
                    self.evaluator = ReconstructionEvaluator_EmbSimilarity(Embedder())
                else:
                    raise UserFacingError(f"Unknown evaluation type '{eval_type}'")
            else:
                logging.warning("No evaluation config found. Setting evaluator to NOP.")
                self.evaluator = EvaluatorNOP()

        self.experiment_name:str = get_datetime_str(self.config.get('tz'))
        self.parent_run_name:str = self.config["__parent_run_name__"]+f"__{self.experiment_name}"

        self.log_path: str | None = None
        self.mlflow_run_path: str | None = None


    def main(self):
        experiment_name = self.experiment_name
        parent_run_name = self.parent_run_name

        mlflow_uri = self.config['paths']['mlflow_tracking_uri']

        git_commit_hash = check_git_repository_is_clean()

        with FileLock(".lock"):
            setup_mlflow(experiment_name=experiment_name, tracking_uri=mlflow_uri)
            with mlflow.start_run(run_name=parent_run_name) as parent_run, self.cache:
                log_path, notifier = setup_logging(
                    log_dir=self.config['paths']['log_dir'],
                    run_id=parent_run.info.run_id,
                    tz_str=self.config.get('tz', None),
                    console_level=self.exec_args.log_level(logging.WARNING),
                    base_level=self.exec_args.log_level(logging.INFO)
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

                for runner in self.build_experiments():
                    with mlflow.start_run(run_name=runner.run_name, nested=True):
                        logging.info(f"--- Starting Nested Run: {runner.run_name} ---")
                        mlflow.log_params(runner.conf_for_log)

                        ###
                        metrics = runner.run()

                        if metrics:
                            mlflow.log_metrics(metrics)
                            log_message = f"{runner.run_name} Logged aggregated metrics {metrics_to_json(metrics)}"
                            logging.info(log_message)
                            notifier.info(log_message)
                        else:
                            logging.error("No metrics were generated")

                        ###

                        flush_loggers()

    def done(self, exception:Exception | None = None):
        logging.info(f'PID {os.getpid()} DONE.')
        if not exception:
            print(f"\n✅ Finished successfully.")
        else:
            print(ExceptionStr(exception).model_dump_json(indent=4, exclude_none=True))
        if self.mlflow_run_path:
            print(f"\nRun `mlflow ui` in your terminal to view the full results.")
            print(f"\nRun `python scripts/mlflow_runs.py {self.mlflow_run_path}` for command-line access.")
        if self.log_path:
            print(f"\nView log in {self.log_path}")
        else:
            print(f"\nNo log generated.")
        print()

    @abstractmethod
    def build_experiments(self):
        pass

    @staticmethod
    def _create_mock_llm_client():
        """
        Creates a mock for `llm_client` that raises exceptions for any accessed attribute
        or method.
        """

        # Dynamically handle all attribute/method access
        def raise_exception(name):
            def _raise(*args, **kwargs):
                raise RuntimeError(
                    f"llm_client: Attempted to call method '{name}' with args: {args}, kwargs: {kwargs}"
                )

            return _raise

        llm_mock = Mock()
        llm_mock.side_effect = lambda name: raise_exception(name)

        return llm_mock


# class ExperimentPipeline_QA(ExperimentPipeline):
#
#     def __init__(self, exec_args: ExecArgs, config: dict[str, Any]):
#         super().__init__(exec_args, config)
#         self.rs_builder = ReconstructionStrategyBuilder(
#             llm_cache=self.cache,
#             master_seed=self.config["base_params"]["master_seed"],
#             llm_client=self._llm_client
#         )
#
#     def run_and_eval(self, runner: ExperimentRunner):
#         pass
#
#     def build_experiments(self):
#         config = self.config
#         runner = ExperimentRunner(
#             run_name=f"{recon_strategy}__{masker}",
#             data_loader=self.data_loader,
#             evaluator=self.evaluator,
#             conf_for_log=conf_for_log
#         )
#         yield runner


class ExperimentPipeline_Reconstruction(ExperimentPipeline):

    def __init__(self, exec_args: ExecArgs, config: dict[str, Any]):
        super().__init__(exec_args, config)
        self.rs_builder = ReconstructionStrategyBuilder(
            llm_cache=self.cache,
            master_seed=self.config["base_params"]["master_seed"],
            llm_client=self._llm_client
        )

    def build_experiments(self):
        config = self.config

        # --- Loop 1: Reconstruction Strategy ---
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
                run_conf:dict[str,Any] = flat_dict({
                    '':config.get('base_params'),
                    'data_config': config["data_config"],
                    'masking': masker.get_params_for_repr(),
                    'recon_strategy': strategy_params
                })
                runner = ExperimentRunner(
                    run_name=f"{recon_strategy}__{masker}",
                    data_loader=self.data_loader,
                    masking_strategy=masker,
                    reconstruction_strategy=recon_strategy,
                    evaluator=self.evaluator,
                    #TODO add result path to config
                    save_path=Path("results/recon/" + self.parent_run_name),
                    conf_for_log=run_conf
                )
                yield runner
