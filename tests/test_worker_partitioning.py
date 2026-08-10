import pytest
from data_models.exec_args import ExecArgs, args_parser
from experiment_executor.experiment_runner import ExperimentRunner

def test_exec_args_worker_defaults():
    args = ExecArgs(config_path="config/test.yaml")
    assert args.worker_id == 0
    assert args.total_workers == 1
    assert args.max_runtime_hours is None

def test_experiment_runner_worker_bounds():
    # Invalid total_workers
    with pytest.raises(ValueError, match="total_workers must be >= 1"):
        ExperimentRunner(
            run_name="test",
            data_loader=None,
            masking_strategy=None,
            reconstruction_strategy=None,
            evaluator=None,
            save_path=None,
            conf_for_log={},
            worker_id=0,
            total_workers=0
        )

    # Invalid worker_id >= total_workers
    with pytest.raises(ValueError, match="worker_id must be in range"):
        ExperimentRunner(
            run_name="test",
            data_loader=None,
            masking_strategy=None,
            reconstruction_strategy=None,
            evaluator=None,
            save_path=None,
            conf_for_log={},
            worker_id=2,
            total_workers=2
        )

def test_dataset_strided_partitioning():
    all_items = list(range(100))
    total_workers = 3
    
    worker0_items = [x for idx, x in enumerate(all_items) if (idx % total_workers) == 0]
    worker1_items = [x for idx, x in enumerate(all_items) if (idx % total_workers) == 1]
    worker2_items = [x for idx, x in enumerate(all_items) if (idx % total_workers) == 2]

    # Verify no overlaps
    assert set(worker0_items).isdisjoint(set(worker1_items))
    assert set(worker0_items).isdisjoint(set(worker2_items))
    assert set(worker1_items).isdisjoint(set(worker2_items))

    # Verify complete reconstruction
    reconstructed = sorted(worker0_items + worker1_items + worker2_items)
    assert reconstructed == all_items
