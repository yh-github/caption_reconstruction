import logging
import os
import platform
from importlib.metadata import version
from logging import Logger
from pathlib import Path
from typing import Iterator
import mlflow
import pandas as pd
from filelock import FileLock
from mlflow.entities import RunInfo
from pydantic import BaseModel
from evaluations.evaluation import ReconstructionEvaluator
from evaluations.metrics import MetricsRecordRaw, metrics_to_json
from experiment_executor.pipeline import ExperimentPipeline
from common_utils.tracking import check_git_repository_is_clean, setup_logging, flush_loggers, \
    setup_mlflow
from common_utils.path_handling import add_suffix_to_path
from common_utils.error_handling import ExceptionStr
import shutil


class ExecutionResult(BaseModel):
    log_path:str
    mlflow_run_path:str
    num_of_raw_results:int
    current_run:str
    results_paths:list[Path]

    def is_done(self) -> bool:
        return self.current_run=="DONE"

    @staticmethod
    def empty():
        return ExecutionResult(
            log_path="",
            mlflow_run_path="",
            num_of_raw_results=0,
            current_run="INIT",
            results_paths=[]
        )


class Executor:

    def __init__(self, pipeline:ExperimentPipeline):
        self.pipeline = pipeline
        self.log_path = ""
        self.mlflow_run_path = ""
        self.notifier:Logger=logging.getLogger()
        self.exec_status = ExecutionResult.empty()

    def _execute(self) -> Iterator[ExecutionResult]:
        check_git_repository_is_clean(ignore_risk=self.pipeline.exec_args.should_ignore_unsafe())
        mlflow_uri = self.pipeline.config['paths']['mlflow_tracking_uri']

        with FileLock(self.pipeline.config['paths'].get("lock", ".lock")):
            setup_mlflow(experiment_name=self.pipeline.experiment_name, tracking_uri=mlflow_uri)
            with mlflow.start_run(run_name=self.pipeline.parent_run_name) as parent_run, self.pipeline.cache:
                self._setup_execution(parent_run.info, mlflow_uri)

                all_results = []
                for runner in self.pipeline.build_experiments():
                    yield self._partial_results(all_results, runner.run_name)
                    all_results.extend(self._exec_runner(runner))

                yield self._final_results(all_results)

    def _exec_runner(self, runner) -> list[MetricsRecordRaw]:
        with mlflow.start_run(run_name=runner.run_name, nested=True):
            logging.info(f"--- Starting Nested Run: {runner.run_name} ---")
            mlflow.log_params(runner.conf_for_log)

            ###
            run_metrics: list[MetricsRecordRaw] = runner.run()
            agg_metrics = runner.evaluator.agg_metrics(run_metrics)

            if agg_metrics:
                mlflow.log_metrics(agg_metrics)
                log_message = f"{runner.run_name} Logged aggregated metrics {metrics_to_json(agg_metrics)}"
                logging.info(log_message)
                self.notifier.info(log_message)
            else:
                logging.error("No metrics were generated")

            flush_loggers()
            return run_metrics

    def _setup_execution(self, parent_run_info:RunInfo, mlflow_uri:str):
        log_path, self.notifier = setup_logging(
            log_dir=self.pipeline.config['paths']['log_dir'],
            run_id=parent_run_info.run_id,
            tz_str=self.pipeline.config.get('tz', None),
            console_level=logging.INFO if (self.pipeline.exec_args.verbose or self.pipeline.exec_args.debug) else logging.WARNING,
            base_level=self.pipeline.exec_args.log_level(logging.INFO)
        )
        self.log_path = log_path
        
        if self.pipeline.hf_manager:
            self.pipeline.hf_manager.register_log_file(Path(log_path))

        print(f'{log_path = }')
        start_msg = (
            f"--- Starting: "
             f"parent_run_name={self.pipeline.parent_run_name} "
             f"experiment_id={parent_run_info.experiment_id} "
             f"pipeline={self.pipeline} "
             f"---"
        )
        self.mlflow_run_path = str(os.path.join(mlflow_uri.removeprefix("file:"), parent_run_info.experiment_id))

        logging.info(start_msg)
        self.notifier.info(start_msg)

        # Log reproducibility parameters
        mlflow.log_param("git_commit_hash", check_git_repository_is_clean(ignore_risk=self.pipeline.exec_args.should_ignore_unsafe()))
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("mlflow_version", version('mlflow'))

    def _final_results(self, all_results:list[MetricsRecordRaw]) -> ExecutionResult:
        self.pipeline.result_path.mkdir(parents=True, exist_ok=True)
        CSV_PATH = self.pipeline.result_path / (self.pipeline.config["__parent_run_name__"] + ".csv")
        CSV_PATH2 = add_suffix_to_path(CSV_PATH, "_z_score")
        pd.DataFrame([x.stats().to_flat_dict() for x in all_results]).to_csv(CSV_PATH)
        global_stats = ReconstructionEvaluator.global_stats(all_results)
        pd.DataFrame([x.stats_z_score(global_stats).to_flat_dict() for x in all_results]).to_csv(CSV_PATH2)

        self.pipeline.for_analysis_path.mkdir(parents=True, exist_ok=True)

        shutil.copy(CSV_PATH, self.pipeline.for_analysis_path)
        shutil.copy(CSV_PATH2, self.pipeline.for_analysis_path)
        
        return ExecutionResult(
            mlflow_run_path=self.mlflow_run_path,
            log_path=self.log_path,
            num_of_raw_results=len(all_results),
            current_run="DONE",
            results_paths=[CSV_PATH, CSV_PATH2]
        )

    def _partial_results(self, all_results:list[MetricsRecordRaw], current_run:str) -> ExecutionResult:
        return ExecutionResult(
            mlflow_run_path=self.mlflow_run_path,
            log_path=self.log_path,
            num_of_raw_results=len(all_results),
            current_run=current_run,
            results_paths=[]
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done(exc_val)

    def main(self):
        for exec_status in self._execute():
            self.exec_status = exec_status
        return self.exec_status

    def done(self, exception:Exception | None = None):
        logging.info(f'PID {os.getpid()} DONE.')

        if not exception and self.exec_status.is_done():
            print(f"\n✅ Finished successfully.")
        else:
            if not self.exec_status.is_done():
                print(f"\n Current Run: {self.exec_status.current_run}")
            if exception:
                print(ExceptionStr(exception).model_dump_json(indent=4, exclude_none=True))

        if self.pipeline:
             self.pipeline.shutdown()

        if self.exec_status.mlflow_run_path:
            print(f"\nRun `mlflow ui` in your terminal to view the full results.")
            print(f"\nRun `python scripts/mlflow_runs.py {self.exec_status.mlflow_run_path}` for command-line access.")
        if self.pipeline.result_path:
            print(f"\nResults: {self.pipeline.result_path}")
        if self.exec_status.log_path:
            print(f"\nView log in {self.exec_status.log_path}")
        else:
            print(f"\nNo log generated.")
        print()
