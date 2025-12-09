# Execution Guide

This document describes how to execute the project's scripts and experiments.

## Requirements

Ensure you have a Python environment set up with the dependencies installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Main Entry Point

The primary script for running experiments is `src/main.py`.

```bash
python src/main.py [CONFIG_PATH] [OPTIONS]
```

### Arguments

*   `CONFIG_PATH`: Path to the experiment configuration YAML file (e.g., `config/toy_baseline.yaml`).
*   `--system_config_path`: Path to the system configuration file (default: `config/system.yaml`).
*   `--override`: Override configuration parameters (e.g., `base_params.master_seed=123`).
*   `--verbose`: Display detailed information.
*   `--debug`: Display debug information.
*   `--dry-run`: Prepare experiments without executing them.
*   `--validate-cache`: Run experiments to populate/validate cache without full execution? (Check implementation).

### Examples

**Dry Run:**
Check what experiments will be run:
```bash
python src/main.py config/toy_baseline.yaml --dry-run
```

**Run Experiment:**
Execute the baseline experiment:
```bash
python src/main.py config/toy_baseline.yaml
```

## Helper Scripts

### `scripts/execute_on_data.py`
Execute Python code against data files (JSON/YAML) from the command line.

```bash
python scripts/execute_on_data.py "print(data[0]['caption'])" datasets/toy_dataset/data.json
```

### `scripts/check_recon.py`
Inspect and evaluate reconstruction results stored in MLflow.

```bash
# List all reconstructions in the artifact file
python scripts/check_recon.py config/system.yaml file:///path/to/mlruns/EXPERIMENT_ID ls

# View a specific reconstruction by Video ID
python scripts/check_recon.py config/system.yaml file:///path/to/mlruns/EXPERIMENT_ID VIDEO_ID
```

### `scripts/mlflow_runs.py`
View the hierarchy of MLflow runs in the terminal.

```bash
python scripts/mlflow_runs.py ./mlruns
```

### `scripts/try_prompts.py`
Load a config and data, then print the generated prompt for a specific data index. Useful for debugging prompt construction.
```bash
python scripts/try_prompts.py config/path.yaml <index>
```

### `scripts/parse_results.py`
Development script for parsing and analyzing specific result output text. **Note:** Contains hardcoded input text and paths; likely needs modification for general use.
```bash
python scripts/parse_results.py ...
```

### `scripts/explore_wild.py`
Analyze and download data from "wild" datasets (e.g., from Dropbox). This script seems to contain hardcoded paths and specific logic for a dataset.

### `scripts/download_data.py`
Downloads and sets up the project data/cache from a remote source (Google Drive).
It handles downloading zip files, extracting them, and merging them with local data safely.
```bash
python scripts/download_data.py
```

## Advanced Analysis & Utility Scripts

### `scripts/load_and_repl.py`
Load a JSON or YAML file into a Python variable and start an interactive REPL session to inspect it.
```bash
python scripts/load_and_repl.py path/to/file.json
```

### `src/data/pooling.py`
Perform dimensionality reduction (PCA, t-SNE) on vector datasets compared with various pooling strategies.
```bash
python src/data/pooling.py config/path.yaml
```

### `src/qa/dev_qa_video_main.py`
Run Video QA experiments using a VLM (Video Language Model) against specific video datasets.
```bash
python src/qa/dev_qa_video_main.py [CONFIG_PATH] [RUN_ID]
```

### `src/qa/dev_qa_main.py`
Run QA experiments on text captions (development script). Note that this script may contain hardcoded paths (e.g., specific user directories) in its `__main__` block and might need adjustment.
```bash
python src/qa/dev_qa_main.py
```

### `src/experiment_executor/yt_video_processing.py`
Process YouTube video links (defined in a dataset) to generate captions or analysis using Gemini.
```bash
python src/experiment_executor/yt_video_processing.py [CONFIG_PATH]
```

### `src/experiment_executor/config_loader.py`
Debug utility to load and print the resolved configuration (merging system, experiment, and overrides).
```bash
python src/experiment_executor/config_loader.py path/to/config.yaml [override=value ...]
```

### Analysis plotting & Tools
Various scripts in `src/analysis/` are used to generate specific plots/analysis from experiment results.
*   `src/analysis/visual_dim_reduction.py`: Plot 2D scatter plots of reduced vector embeddings.
    ```bash
    python src/analysis/visual_dim_reduction.py results/pooling/DATASET_NAME
    ```
*   `src/analysis/ranking.py`, `ranking2.py`, `ranking3.py`: Analyze and visualize the stability of reconstruction rankings using various metrics.
    ```bash
    python src/analysis/ranking.py [metric_name]
    python src/analysis/ranking2.py [metric_name]
    python src/analysis/ranking3.py [metric_name]
    ```
    *   `metric_name` defaults to `cos_sim_mean`.
*   `src/analysis/llm_based.py`: helper script to identify "interesting" videos (high rank difference between methods) and generate a prompt for an LLM to analyze them.
    ```bash
    python src/analysis/llm_based.py <metric_name> [--z]
    ```

### Internal / Testing
*   `src/experiment_executor/cli_parser.py`: Contains a self-test block for the argument parser utility.
    ```bash
    python src/experiment_executor/cli_parser.py
    ```



