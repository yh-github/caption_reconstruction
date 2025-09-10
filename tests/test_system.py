import sys

import pandas as pd
import pytest
from pathlib import Path
from data_models.exec_args import ExecArgs
from run_experiments import ExperimentPipeline, ConfigError
from utils import add_suffix_to_path


@pytest.mark.parametrize(
    "config_filename, expected_num_exps, expected_data_count",
    [
        ("toy_llm2.yaml", 1, 2),
        ("toy_baseline2.yaml", 3, 2),
        ("toy_llm.yaml", 4, 2),
        ("toy_baseline.yaml", 5, 2),
    ]
)
def test_toy_data_dry_run_results(
    config_filename: str,
    expected_num_exps: int,
    expected_data_count: int
):
    """
    Tests that the dry_run method produces the correct number of experiments
    and data count for each of the toy configuration files.
    """
    # Assume the config files are in a 'config/' subdirectory relative to the project root.
    # Adjust this path if your structure is different.
    config_path = Path("config") / config_filename

    # This is the same logic as your original script.
    ep = ExperimentPipeline.build(ExecArgs(
        config_path=config_path,
        dry_run=True
    ))
    exps, data_count = ep.dry_run()
    num_exps = len(exps)

    # Instead of printing, we use `assert` to verify the results.
    # Pytest provides detailed output if an assertion fails.
    assert num_exps == expected_num_exps, f"Mismatch in experiment count for {config_filename}"
    assert data_count == expected_data_count, f"Mismatch in data count for {config_filename}"

all_results = {}
@pytest.mark.parametrize(
    "config_filename, expected_num_exps, expected_data_count",
    [
        ("toy_llm2.yaml", 1, 2),
        ("toy_baseline2.yaml", 3, 2),
        ("toy_llm.yaml", 4, 2),
        ("toy_baseline.yaml", 5, 2),
        ("*", 5,5)
    ]
)
def test_toy_data(config_filename:str, expected_num_exps:int, expected_data_count:int):
    if config_filename == "*":
        print("###########", file=sys.stderr)
        print("\n\n")
        for k,v in all_results.items():
            print(f" ===> {k} <===")
            print(v)
        print()
        return

    config_path = Path("config") / config_filename
    def set_results_path(conf:dict):
        conf["paths"]["results"] = "test_results" # TODO freeze results
        conf["paths"]["log_dir"] = "test_logs"

    try:
        ep = ExperimentPipeline.build(
            ExecArgs(config_path=config_path, debug=True),
            config_override=set_results_path
        )

        csv_path = ep.main()
        df = pd.read_csv(csv_path, index_col=0)
        all_results[config_filename] = df

        csv_path2 = add_suffix_to_path(csv_path, "_z_score")
        df2 = pd.read_csv(csv_path2, index_col=0)

        all_results[config_filename+"(Z_SCORE)"] = df2

        assert len(df) == expected_num_exps*expected_data_count, f"Mismatch in counts for {config_filename}"
        assert len(df2) == expected_num_exps * expected_data_count, f"Mismatch in counts for {config_filename}"
        # assert data_count == , f"Mismatch in data count for {config_filename}"
    except ConfigError as e:
        print(e.config)
        print()
