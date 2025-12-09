# Caching & Reproducibility

To avoid redundant API calls and processing, the project uses a caching system.

## Mechanisms

### 1. `diskcache`
The project uses the `diskcache` library to store:
*   **LLM Responses**: Responses from the Gemini API.
*   **Embeddings**: Vector embeddings generated for content.

### 2. File Lock
`filelock` is used to ensure safe concurrent access to shared resources, presumably for MLflow tracking or cache access during parallel execution.

## Directories

*   **`disk_cache/`**: This is the default directory for the disk cache.
    *   Subdirectories are created based on the model and task type (e.g., `gemini-embedding-001__512__SEMANTIC_SIMILARITY`).
    *   **Recommendation**: Copy this directory between runs or environments to save on API costs and time.

*   **`.lock`**: A lock file created in the project root (configurable via `paths.lock` in `system.yaml`).

## Configuration

Cache paths are defined in `config/system.yaml`:

```yaml
paths:
  disk_cache: "disk_cache/"
```

### `disk_cache` Structure
The `disk_cache` directory serves as the persistence layer for LLM interactions to avoid redundant API calls and costs.

*   **LLM Responses (`cache.db`)**: The root of `disk_cache/` (specifically the `cache.db` file) is used by `LLM_Manager` to store text responses from the LLM. Keys are a hash of the model configuration and the prompt.
*   **Embeddings**: Subdirectories (e.g., `gemini-embedding-001__512__SEMANTIC_SIMILARITY/`) are created by the `Embedder` class. Each acts as a separate `diskcache` instance for storing vector embeddings, isolated by model, dimensionality, and task type.

### Version Control
To ensure reproducibility, the system tracks the Git commit hash of the code used to generate cached entries. This allows for verifying which version of the logic produced a specific result.

## Component Testing: `validate_cache`

The project includes a `validate_cache` mode, accessible via the `--validate-cache` flag in `src/main.py`. This mode serves as a component test for the caching system.

*   **Purpose:** To verify that the system correctly utilizes cached data and to populate the cache without triggering actual LLM API calls.
*   **Behavior:**
    *   It initializes the pipeline and data loaders.
    *   It mocks the LLM client to ensure no external API calls are made.
    *   It skips time-consuming evaluation steps.
    *   It is useful for verifying that a set of experiments can run fully from cache (e.g., for reproduction or debugging) without network dependency or cost.

## Local Data as Cache

Files stored in `local/` directories are treated as a form of "local cache" for heavy assets that are not committed to version control.

*   **Typical Contents**:
    *   `wild_videos_embs/`: Stores NumPy (`.npy`) files containing pre-computed video embeddings (e.g., for the "Wild" dataset).
*   **Usage**: Scripts like `src/data/pooling.py` or specific dataloaders look for these local resources to avoid re-downloading or re-computing expensive processing steps.
*   **Setup**: Users must ensure these directories exist and are populated with the necessary data (often downloaded from an external source or generated via a one-time script) before running dependent experiments.

## Manual Results Staging: `results/upload/`

The directory `results/upload/` serves as a **manual staging area** for aggregate analysis.

*   **Workflow:**
    1.  Experiments (via `src/main.py`) generate results in `results/<RUN_ID>/` (containing CSV files with metrics).
    2.  To perform comparative analysis (e.g., ranking stability), the specific CSV files of interest must be **manually copied** from their respective run directories to `results/upload/`.
    3.  Analysis scripts (e.g., `src/analysis/ranking.py`, `src/analysis/llm_based.py`) scan `results/upload/` to ingest all present results for plotting and reporting.
*   **Purpose:** This decouples individual experiment runs from the aggregate analysis, allowing users to cureate exactly which runs to compare.

## Reproducibility

*   **Seeds**: Experiments use a `master_seed` (defined in experiment configs) to control randomness for masking and other stochastic processes.
*   **Git Hash**: The executor records the current Git commit hash in MLflow. It warns or fails if the repository is not clean (unless `--ignore_unsafe` or `--debug` is used).
