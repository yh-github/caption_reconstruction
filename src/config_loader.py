# src/config_loader.py
import yaml
import logging

def load_config(
    system_config_path="config/system.yaml",
    experiment_config_path="config/base.yaml"
):
    """
    Loads both system and experiment configurations and merges them.
    Experiment-specific configs will override system-level configs.
    """
    logging.info(f"Loading system config from: {system_config_path}")
    with open(system_config_path, 'r') as f:
        system_config = yaml.safe_load(f)

    logging.info(f"Loading experiment config from: {experiment_config_path}")
    with open(experiment_config_path, 'r') as f:
        experiment_config = yaml.safe_load(f)
    
    # Merge the two dictionaries
    merged_config = {**system_config, **experiment_config}
    return merged_config
