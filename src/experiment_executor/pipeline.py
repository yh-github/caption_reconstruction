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
from experiment_executor.config_loader import load_config, config_from_args
from data.data_loaders import get_data_loader
from data_models.exec_args import ExecArgs
from evaluations.evaluation import ReconstructionEvaluator
from experiment_executor.experiment_runner import ExperimentRunner
from reconstruction.masking import get_masking_strategies
from reconstruction.text_reconstruction import TextReconstructionStrategyBuilder
from data.vector_dataloaders import VectorDataLoader
from reconstruction.vector_reconstruction import VectorReconstructionStrategyBuilder
from experiment_executor.vector_runner import VectorRunner
from llm.embedder import CacheMissError


class ConfigError(Exception):

    def __init__(self, key: str, exec_args: ExecArgs, config: dict|None):
        self.key = key
        self.exec_args = exec_args
        self.config = config




class ExperimentPipeline(ABC):

    @staticmethod
    def build(exec_args:ExecArgs):
        config = None
        try:
            logging.basicConfig(level=exec_args.log_level(logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
            config = config_from_args(exec_args)
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
        # Block LLM if dry-run, validate-cache, OR cached-execution-only is set
        if self.exec_args.dry_run or self.exec_args.validate_cache or self.exec_args.cached_execution_only:
            logging.info("Blocking LLM client (Mock Mode).")
            return self._create_mock_llm_client()
        try:
            return genai.Client()
        except ValueError as e:
            # If API key is missing, we might still be running a local-only experiment.
            # Return None, so we fail later ONLY if the client is actually used.
            logging.warning("Failed to initialize Google GenAI Client (missing API key?). Proceeding with client=None. "
                            "Local experiments will behave normally; Gemini-based ones will fail on use.")
            return None

    def _get_eval_conf(self):
        eval_conf = self.config.get("evaluation", {}).copy()
        # NOP only for dry-run or validation. cached_execution_only SHOULD run evaluation.
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
        self.llm_client = self._init_llm_client()
        
        # Initialize HF Sync Manager if repo ID is provided
        self.hf_manager = None
        hf_repo_id = config['paths'].get('hf_repo_id') 
        if hf_repo_id:
             from data.hf_sync import HFFileManager
             read_only = self.exec_args.dry_run
             if read_only:
                 logging.info(f"Initializing HF Sync Manager (ReadOnly: DryRun) for repo: {hf_repo_id}")
             else:
                 logging.info(f"Initializing HF Sync Manager (Active) for repo: {hf_repo_id}")
             
             self.hf_manager = HFFileManager(repo_id=hf_repo_id, read_only=read_only)

        data_config = self.config["data_config"]

        if self.experiment_type == 'RECON':
            self.data_loader = get_data_loader(data_config)
            self.experiment_runner_factory = ExperimentRunner
            block_llm = self.exec_args.dry_run or self.exec_args.validate_cache or self.exec_args.cached_execution_only
            self.rs_builder = TextReconstructionStrategyBuilder(
                llm_cache=self.cache,
                master_seed=self.config["base_params"]["master_seed"],
                llm_client=self.llm_client,
                block_llm=block_llm
            )
        elif self.experiment_type == 'RECON_VECTORS':
            self.data_loader = VectorDataLoader.from_config(data_config, llm_client=self.llm_client)
            self.experiment_runner_factory = VectorRunner
            self.rs_builder = VectorReconstructionStrategyBuilder()
        else:
            raise Exception(f"Unknown {self.experiment_type=}")

        self.evaluator = ReconstructionEvaluator.from_config(self._get_eval_conf(), llm_client=self.llm_client)
        if hasattr(self.evaluator, 'idf') and self.evaluator.idf:
            self.evaluator.calc_idf(self.data_loader.load_all_sentences())

        self.experiment_name:str = get_datetime_str(self.config.get('tz'))
        # We remove the timestamp from the directory name to ensure persistent caching across runs.
        # This allows resuming without re-downloading or re-computing existing results.
        self.parent_run_name:str = self.config["__parent_run_name__"]
        results_path = self.config["paths"].get("results", "results")
        self.result_path = Path(f"{results_path}/recon/" + self.parent_run_name)

        for_analysis_path = Path(self.config["paths"].get("for_analysis", "results/for_analysis"))
        for_analysis_path.mkdir(parents=True, exist_ok=True)
        self.for_analysis_path = for_analysis_path

            # check if we need to prefetch
        # check if we need to prefetch
        if self.hf_manager:
            if self.exec_args.dry_run:
                logging.info("Dry-run mode: Skipping automatic HF prefetch/sync.")
            else:
                # We always attempt to sync/prefetch to ensure we have the latest/complete set of files from HF.
                # This handles cases where the local folder exists but is partial (e.g. previous run crashed).
                
                # We download the ENTIRE parent run folder to ensure we have everything (masking variations, etc.)
                base_run_name = self.config.get("__parent_run_name__", "default")
                
                # Match everything under reconstruction/base_run_name
                patterns = [f"reconstruction/{base_run_name}/**"]
                
                if patterns:
                    # Move temp file usage inside the block to avoid premature cleanup
                    import shutil
                    import tempfile
                    logging.info(f"Checking for remote results to sync at {self.result_path}...")
                    
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_path = Path(tmp_dir)
                        downloaded_root = self.hf_manager.prefetch_folder(tmp_path, patterns)
                        
                        if downloaded_root:
                            # Remote structure is reconstruction/base_run_name
                            source_dir = downloaded_root / "reconstruction" / base_run_name
                            if source_dir.exists():
                                # Move/Copy to self.result_path
                                self.result_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                logging.info(f"Moving downloaded results from {source_dir} to {self.result_path}")
                                shutil.copytree(source_dir, self.result_path, dirs_exist_ok=True)
                            else:
                                root_content = list(downloaded_root.iterdir()) if downloaded_root.exists() else "DIR_GONE"
                                logging.warning(f"Prefetch completed but expected source dir {source_dir} not found. (Downloaded root: {root_content} )")

        # Warmup models to ensure they are downloaded/cached before starting the loop
        if not self.exec_args.dry_run: # Skip in dry-run as it might block/OOM
            self._warmup_models()
    
    def _warmup_models(self):
        """Forces loading of local LLM models to fail fast if cache/connection is missing."""
        from reconstruction.text_reconstruction import IterativeReconstructionStrategy, BatchGridSearchStrategy
        
        strategies = self.strategies if isinstance(self.strategies, list) else [self.strategies]
        
        for strat in strategies:
            # Check for IterativeReconstructionStrategy
            if isinstance(strat, IterativeReconstructionStrategy):
                if hasattr(strat, 'model_adapter'):
                    logging.info(f"Warming up model for strategy {strat.name}...")
                    strat.model_adapter._ensure_loaded()
            
            # Check for BatchGridSearchStrategy
            elif isinstance(strat, BatchGridSearchStrategy):
                 if hasattr(strat, 'model_adapter'):
                    logging.info(f"Warming up model for batch strategy {strat.name}...")
                    strat.model_adapter._ensure_loaded()
        
    def shutdown(self):
        if self.hf_manager:
            self.hf_manager.shutdown()

    def __str__(self):
        return (
            f"Pipeline: "
            f"dataloader={self.data_loader.__class__.__name__} "
            f"runner={self.experiment_runner_factory.__name__} "
            f"evaluator={self.evaluator}"
        )

    @abstractmethod
    def build_experiments(self):
        pass

    @staticmethod
    def _create_mock_llm_client():
        """
        Creates a mock for `llm_client` that raises exceptions for any accessed attribute
        or method.
        """
        class BlockingClient:
            def __init__(self, name="LLM_Client", max_misses=20):
                self.name = name
                self.miss_count = 0
                self.max_misses = max_misses
            
            def _check_limit(self):
                self.miss_count += 1
                if self.miss_count > self.max_misses:
                    # Critical error: Raise RuntimeError to crash the pipeline
                    raise RuntimeError(
                        f"CRITICAL: Too many cache misses ({self.miss_count} > {self.max_misses}) "
                        "in blocked mode! This indicates a severe mismatch between cache and experiment."
                    )

            def __getattr__(self, name):
                self._check_limit()
                # Raise CacheMissError so callers can handle gracefully
                raise CacheMissError(
                    f"Cache miss: Attempted to access '{self.name}.{name}' in blocked mode."
                )
            
            def __call__(self, *args, **kwargs):
                self._check_limit()
                raise CacheMissError(
                    f"Cache miss: Attempted to call '{self.name}' in blocked mode."
                )

        return BlockingClient()

    def dry_run(self):
        try:
            return list(self.build_experiments()), self.data_loader.count()
        finally:
            self.shutdown()


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
# 


class ExperimentPipeline_Reconstruction(ExperimentPipeline):

    def __init__(self, exec_args: ExecArgs, config: dict[str, Any]):
        super().__init__(exec_args, config)

    def build_experiments(self):
        try:
            config = self.config

            # --- Loop 1: Reconstruction Strategy ---
            # Group strategies by (model_key, prompt_dir) to enable batching
            # We assume strategy_params has 'model_key' etc.
            
            # Map: (ModelKey, PromptDir) -> List[StrategyParams]
            strategies_by_model: dict[tuple, list[dict]] = {}
            
            for strategy_params in config.get("recon_strategy", []):
                stype = strategy_params.get("type")
                if stype == "local_llm":
                    key = (strategy_params.get("model_key", "phi-3"), strategy_params.get("prompt_dir", "iterative_cloze"))
                    strategies_by_model.setdefault(key, []).append(strategy_params)
                else:
                    # Non-batchable strategies go to a "mixed" bucket or handled individually
                    strategies_by_model.setdefault(("__n/a__", "__n/a__"), []).append(strategy_params)

            
            masking_strategies = get_masking_strategies(
                masking_configs=config["masking_configs"],
                master_seed=config["base_params"]["master_seed"]
            )

            # --- Process Groups ---
            from reconstruction.text_reconstruction import BatchGridSearchStrategy
            from experiment_executor.batch_runner import BatchExperimentRunner

            for (m_key, p_dir), group_configs in strategies_by_model.items():
                
                # --- Loop 2: Illuminate/Iterate over masking strategies ---
                for masker in masking_strategies:
                    
                    # Can we batch this group?
                    # Only if > 1 config and valid local_llm key
                    if m_key != "__n/a__" and len(group_configs) > 1:
                        logging.info(f"Using Batch Processing for {m_key} with {len(group_configs)} configs.")
                        
                        # 1. Build Shared Components (Model, PromptBuilder)
                        # We need to access the internal adapter from builder to create the BatchStrategy
                        # This is a bit hacky, but consistent with the factory pattern usage here.
                        
                        # We instantiate the first strategy just to trigger caching/loading in rs_builder
                        _ = self.rs_builder.get_strategy(group_configs[0])
                        
                        # Retrieve cached adapter
                        # Builder cache key logic: f"{model_key}_{backend}"
                        # We assume default backend for now or check system component
                        # A better way: expose a get_adapter method in builder? 
                        # Or just rely on the fact we know how builder works.
                        from common_utils import device_setup
                        backend = device_setup.get_llm_backend()
                        adapter_key = f"{m_key}_{backend}"
                        adapter = self.rs_builder._local_model_cache.get(adapter_key)
                        
                        # Retrieve prompt builder (re-create finding path logic or cache it?)
                        # Re-creating is cheap
                        prompt_path = self.rs_builder.prompts_dir / p_dir
                        from llm.prompting import ClozePromptBuilder
                        if prompt_path.is_dir():
                            prompt_builder = ClozePromptBuilder.from_directory(prompt_path)
                        else:
                             raise UserFacingError(f"Prompt directory '{prompt_path}' does not exist.")
                             
                        # 2. Create Batch Strategy
                        batch_strategy = BatchGridSearchStrategy(
                            name=f"Batch_{m_key}_{len(group_configs)}",
                            model_adapter=adapter,
                            prompt_builder=prompt_builder,
                            configs=group_configs
                        )
                        
                        # 3. Create Individual Runners (Container-only, stripped of heavy logic)
                        runners = []
                        for strategy_params in group_configs:
                            recon_strategy = self.rs_builder.get_strategy(strategy_params)
                            run_conf = flat_dict({
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
                                save_path=self.result_path,
                                conf_for_log=run_conf,
                                hf_manager=self.hf_manager,
                                config_stem=config.get("__parent_run_name__", "default"),
                                eval_only=self.exec_args.eval_only
                            )
                            runners.append(runner)
                            
                        # 4. Yield BatchRunner
                        yield BatchExperimentRunner(
                            base_run_name=f"Batch_{m_key}",
                            runners=runners,
                            batch_strategy=batch_strategy,
                            data_loader=self.data_loader,
                            masking_strategy=masker,
                            evaluator=self.evaluator
                        )

                    else:
                        # Fallback to standard sequential processing
                        for strategy_params in group_configs:
                            recon_strategy = self.rs_builder.get_strategy(strategy_params)
                            run_conf:dict[str,Any] = flat_dict({
                                '':config.get('base_params'),
                                'data_config': config["data_config"],
                                'masking': masker.get_params_for_repr(),
                                'recon_strategy': strategy_params
                            })
                            
                            try: 
                                runner = self.experiment_runner_factory(
                                    run_name=f"{recon_strategy}__{masker}",
                                    data_loader=self.data_loader,
                                    masking_strategy=masker,
                                    reconstruction_strategy=recon_strategy,
                                    evaluator=self.evaluator,
                                    #TODO add result path to config
                                    save_path=self.result_path,
                                    conf_for_log=run_conf,
                                    hf_manager=self.hf_manager,
                                    config_stem=config.get("__parent_run_name__", "default"),
                                    eval_only=self.exec_args.eval_only
                                )
                                yield runner
                            except TypeError:
                                 # Fallback for VectorRunner which doesn't support hf_manager yet
                                 runner = self.experiment_runner_factory(
                                    run_name=f"{recon_strategy}__{masker}",
                                    data_loader=self.data_loader,
                                    masking_strategy=masker,
                                    reconstruction_strategy=recon_strategy,
                                    evaluator=self.evaluator,
                                    save_path=self.result_path,
                                    conf_for_log=run_conf
                                )
                                 yield runner
        finally:
             if not self.exec_args.dry_run:
                 self.shutdown()
