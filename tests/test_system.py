import pytest
from pathlib import Path
from data_models.exec_args import ExecArgs
from run_experiments import ExperimentPipeline


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
