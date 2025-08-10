import sys

from data_loaders import get_data_loader
from config_loader import load_config
from prompting import JSONPromptBuilder

config_path = sys.argv[1]
ind = int(sys.argv[2])

config = load_config(config_path)
dl = get_data_loader(config["data_config"])

data = dl.load()

for recon_strategy in config["recon_strategy"]:
    if recon_strategy["type"] != "llm":
        continue
    builder = JSONPromptBuilder.from_config(recon_strategy['llm'])
    prompt = builder.build_prompt(data[ind])

    print()
    print(f"==> {recon_strategy['name']} <==")
    print(prompt)
    print()

