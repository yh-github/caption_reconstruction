# Project Overview

The **Caption Reconstruction** project is designed to experiment with reconstructing masked parts of video captions using Large Language Models (LLMs) and vector embeddings. It provides a framework for:

1.  **Loading Data**: Handling video caption datasets (JSON/JSONL formats).
2.  **Masking**: Applying various masking strategies to hide parts of the captions.
3.  **Reconstruction**: Using different strategies (baselines, LLMs, vector retrieval) to reconstruct the masked content.
4.  **Evaluation**: Measuring the quality of reconstruction using metrics like BERTScore.
5.  **Tracking**: Managing experiments and results using MLflow.

## Quick Start

1.  **Setup**: Follow the [Setup Guide](setup.md) to initialize your environment and data directories.
2.  **Run**: Execute the main pipeline:

## Directory Structure

*   `src/`: Source code for the project.
    *   `src/main.py`: The main entry point for running experiments.
    *   `src/experiment_executor/`: Logic for running experiment pipelines.
    *   `src/reconstruction/`: Implementations of reconstruction strategies.
    *   `src/evaluations/`: Evaluation metrics and logic.
    *   `src/llm/`: Interactions with LLMs (Gemini) and embeddings.
*   `scripts/`: Utility scripts for data analysis, quick execution, and result inspection.
*   `config/`: Configuration files (YAML) for experiments and system settings.
*   `datasets/`: Directory for input datasets.
*   `disk_cache/`: Default location for persistent caching of LLM responses and embeddings.
*   `mlruns/`: Default location for MLflow experiment tracking.
