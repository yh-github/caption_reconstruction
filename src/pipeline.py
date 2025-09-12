import logging
from abc import abstractmethod, ABC
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import diskcache
from google import genai

from common_utils.error_handling import UserFacingError
from common_utils.jsonables import flat_dict
from common_utils.tracking import get_datetime_str
from config_loader import load_config
from data.data_loaders import get_data_loader
from data_models.exec_args import ExecArgs
from evaluations.evaluation import ReconstructionEvaluator
from experiment_runner import ExperimentRunner
from reconstruction.masking import get_masking_strategies
from reconstruction.reconstruction_strategies import ReconstructionStrategyBuilder
from data.vector_dataloaders import VectorDataLoader
from vectors.reconstruction_startegies import VectorReconstructionStrategyBuilder
from vectors.vector_runner import VectorRunner


class ConfigError(Exception):

    def __init__(self, key: str, exec_args: ExecArgs, config: dict|None):
        self.key = key
        self.exec_args = exec_args
        self.config = config


class ExperimentPipeline(ABC):

    @staticmethod
    def build(exec_args:ExecArgs, config_override:Callable[[dict], None]|None=None):
        config = None
        try:
            logging.basicConfig(level=exec_args.log_level(logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
            config = load_config(exec_args.config_path)
            if config_override:
                print('config_override')
                config_override(config)
                # print(json.dumps(config, indent=4))
            experiment_type = config['base_params'].get('experiment_type', 'recon').upper()
            if experiment_type in {'RECON', 'RECON_VECTORS'}:
                return ExperimentPipeline_Reconstruction(exec_args, config)
            # elif experiment_type == 'QA':
            #     return ExperimentPipeline_QA(exec_args, config)
            else:
                raise UserFacingError(f"Unknown {experiment_type=}")
        except KeyError as e:
            raise ConfigError(str(e), exec_args, config) from e

    def _init_experiment_type(self):
        return self.config['base_params'].get('experiment_type', 'recon').upper()

    def _init_cache(self):
        return diskcache.Cache(directory=self.config['paths']['disk_cache'])

    def _init_llm_client(self):
        if self.exec_args.dry_run or self.exec_args.validate_cache:
            logging.info("Blocking LLM client.")
            return self._create_mock_llm_client()
        return genai.Client()

    def _get_eval_conf(self):
        eval_conf = self.config.get("evaluation", {}).copy()
        if self.exec_args.dry_run or self.exec_args.validate_cache:
            eval_conf['type'] = 'NOP'
        if self.exec_args.verbose:
            eval_conf['verbose'] = True
        eval_conf['data_type'] = self.data_loader.get_data_type_name()
        return eval_conf

    def __init__(self, exec_args:ExecArgs, config:dict[str, Any]):
        self.exec_args = exec_args
        self.config = config
        self.experiment_type = self._init_experiment_type()
        self.cache = self._init_cache()

        data_config = self.config["data_config"]

        if self.experiment_type == 'RECON':
            self.data_loader = get_data_loader(data_config)
            self.experiment_runner_factory = ExperimentRunner
            self.rs_builder = ReconstructionStrategyBuilder(
                llm_cache=self.cache,
                master_seed=self.config["base_params"]["master_seed"],
                llm_client=self._init_llm_client()
            )
        elif self.experiment_type == 'RECON_VECTORS':
            self.data_loader = VectorDataLoader.from_config(data_config)
            self.experiment_runner_factory = VectorRunner
            self.rs_builder = VectorReconstructionStrategyBuilder()
        else:
            raise Exception(f"Unknown {self.experiment_type=}")

        self.evaluator = ReconstructionEvaluator.from_config(self._get_eval_conf())
        if hasattr(self.evaluator, 'idf') and self.evaluator.idf:
            self.evaluator.calc_idf(self.data_loader.load_all_sentences())

        self.experiment_name:str = get_datetime_str(self.config.get('tz'))
        self.parent_run_name:str = self.config["__parent_run_name__"]+f"__{self.experiment_name}"
        results_path = self.config["paths"].get("results", "results")
        self.result_path = Path(f"{results_path}/" + self.parent_run_name)

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

    def dry_run(self):
        return list(self.build_experiments()), self.data_loader.count()


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
                runner = self.experiment_runner_factory(
                    run_name=f"{recon_strategy}__{masker}",
                    data_loader=self.data_loader,
                    masking_strategy=masker,
                    reconstruction_strategy=recon_strategy,
                    evaluator=self.evaluator,
                    #TODO add result path to config
                    save_path=self.result_path,
                    conf_for_log=run_conf
                )
                yield runner
