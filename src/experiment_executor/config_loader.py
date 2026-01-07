import logging
from pathlib import Path
from typing import Any
import yaml
from data_models.exec_args import DEFAULT_SYSTEM_CONFIG_PATH, ExecArgs

logger = logging.getLogger(__name__)


def load_yaml(path:Path|str) -> dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_config(
    experiment_config_path:Path|str,
    system_config_path:Path|str=DEFAULT_SYSTEM_CONFIG_PATH
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


from dict_path import DictPath
def config_from_args(exec_args:ExecArgs) -> dict[str, Any]:
    con = DictPath(load_config(exec_args.config_path, exec_args.system_config_path))
    if exec_args.override:
        for x in exec_args.override:
            k, v = x.split('=', 1)
            old_v = con.get(k)
            logger.info(f"Overriding {k} from {old_v} to {v}, type: {type(old_v)}")
            if old_v is not None and type(old_v) != type(v):
                con.set(k,type(old_v)(v)) # TODO use eval instead of cast? Support lists?
            else:
                con.set(k, v)

    return con.dict


if __name__ == "__main__":
    conf = config_from_args(ExecArgs(
        config_path=Path("config/embs_vs_llms/wild_dev_sim.yaml"),
        override=[
            "paths/results=test_res/",
            "base_params/master_seed=4455"
        ]
    ))

    import json
    print(json.dumps(conf, indent=4))

