# Project Overview

The **Caption Reconstruction** project is designed to experiment with reconstructing masked parts of video captions using Large Language Models (LLMs) and vector embeddings. It provides a framework for:

1. **Loading Data**: Handling video caption datasets (JSON/JSONL formats) and vector matrices (`.npy` format).
2. **Masking**: Applying configurable masking strategies (contiguous, fixed fill, random, partition) to hide parts of captions or embeddings.
3. **Reconstruction**: Reconstructing masked content using baseline heuristics, remote LLMs (Gemini), local SLMs (Phi-3), or vector interpolation strategies.
4. **Evaluation**: Measuring reconstruction quality using cosine similarity, residual similarity, retrieval ranking metrics (MRR, Recall@k), and BERTScore.
5. **Tracking**: Managing experiments, metrics, and CSV outputs using MLflow.

---

## 📚 Documentation Guides

Explore the detailed documentation guides:

* **[Setup Guide](setup.md)**: Environment initialization, dependencies, and remote dataset download.
* **[Execution Guide](execution.md)**: Command-line usage, flags (`--dry-run`, `--block-llm`, `--eval-only`), and helper scripts.
* **[Configuration Reference Guide](configuration.md)**: Complete guide to YAML configuration parameters, data loaders, masking schemes, reconstruction strategies, and evaluation types.
* **[Outputs and Results Guide](outputs_and_results.md)**: Guide to output directories, CSV column definitions, z-score calculations, and master aggregated result files.
* **[Caching & Reproducibility](caching.md)**: Disk cache system, file locking, and reproduction without API costs.
* **[Batch Processing](batch_processing.md)**: Details on heterogeneous batch processing and parameter sweeps.

---

## Directory Structure

* `src/`: Source code for the project.
  * `src/main.py`: The main entry point for running experiments.
  * `src/experiment_executor/`: Pipeline orchestration and experiment runners.
  * `src/reconstruction/`: Implementation of text and vector reconstruction strategies.
  * `src/evaluations/`: Evaluation metrics and scoring logic.
  * `src/llm/`: Interfaces for Gemini, local LLMs, and embedding models.
* `scripts/`: Utility scripts for data analysis, quick execution, and result inspection.
* `config/`: Configuration files (YAML) for experiments and system settings.
* `datasets/`: Input caption datasets.
* `local/`: Local pre-computed video embedding datasets.
* `results/`: Output directories for timestamped run results, analysis CSVs, and master aggregated datasets.
* `disk_cache/`: Persistent disk caching for LLM responses and embeddings.
* `mlruns/`: MLflow experiment tracking database.
