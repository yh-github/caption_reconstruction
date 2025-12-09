# Project Setup Guide

This guide details the initial setup required to run the Caption Reconstruction project, covering directory creation, environment setup, and data management.

## 1. Environment Setup

As described in `overview.md`, ensure you have a Python virtual environment set up and dependencies installed:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (ensure you possess requirements.txt)
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

## 2. Directory Structure Setup

The project relies on specific directories for caching, local data, and results. You must create these manually if they do not exist:

```bash
# Create core data and result directories
mkdir -p disk_cache
mkdir -p local
mkdir -p results/upload
```

### Directory Roles

*   **`disk_cache/`**: Stores cached LLM responses and embeddings to save costs and time. The system will automatically populate this.
*   **`local/`**: Intended for large, local-only assets (e.g., video embeddings) that are not kept in git.
    *   **Action Required**: If you are planning to run analysis on the "Wild" dataset, create `local/wild_videos_embs` and populate it with the necessary `.npy` files.
*   **`results/`**: The output destination for experiment runs.
*   **`results/upload/`**: A staging area for aggregate analysis. You will manually copy result CSVs here to compare them.

## 3. Configuration

Key configuration files are located in `config/`.

*   **`config/system.yaml`**: Defines system-wide paths.
    *   Verify that `disk_cache` points to `"disk_cache/"` (or your preferred location).
    *   Verify `mlflow_tracking_uri` if you plan to use MLflow.

## 4. Verification

To verify your setup is correct and the code can run, execute a dry run of the main pipeline:

```bash
python src/main.py config/embs_vs_llms/wild_dev_sim.yaml --dry-run
```

If this completes without error, your environment and basic directory structure are correctly configured.
