import logging
from pathlib import Path
from typing import Any
import yaml

def load_yaml(path:Path|str) -> dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_config(
    experiment_config_path:Path|str,
    system_config_path:Path|str="config/system.yaml"
) -> dict[str, Any]:
    """
    Loads both system and experiment configurations and merges them.
    Experiment-specific configs will override system-level configs.
    """
    experiment_config_path = Path(experiment_config_path)
    system_config_path = Path(system_config_path)

    logging.info(f"Loading system config from: {system_config_path}")
    system_config = load_yaml(system_config_path)

    logging.info(f"Loading experiment config from: {experiment_config_path}")
    experiment_config = load_yaml(experiment_config_path)

    # TODO: add master_seed here to all llm configs and all configs with seed

    for k, orig_v in experiment_config.copy().items():
        if k.startswith("IMPORT "):
            k = k.split()[1]
            imp_path = Path(orig_v)
            if not imp_path.is_absolute():
                imp_path = experiment_config_path.parent/Path(orig_v)
            logging.info(f"importing {k} from {imp_path}")
            v = load_yaml(imp_path)
            if k in experiment_config:
                experiment_config[k].extend(v[k])
            else:
                experiment_config[k] = v[k]
    
    #TODO think about the order
    return {
        "__parent_run_name__": experiment_config_path.stem,
        **system_config,
        **experiment_config
    }

def get_llm_config(config:dict[str, Any]) -> dict[str, Any]:
    llm_config = config['llm'].copy()
    llm_config['seed'] = config['base_params']['master_seed'] + llm_config.get('seed',0)
    return llm_config


if __name__ == "__main__":
    conf = load_config("/home/yoavh/code/research/caption_reconstruction/config/embs_vs_llms/wild_dev_sim.yaml")
    import json
    print(json.dumps(conf, indent=4))

