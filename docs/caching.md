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

Files stored in `local/` directories (e.g., `local/wild_videos_embs`) are treated as a form of "local cache".

*   **Usage:** These directories often contain pre-computed heavy assets like video embeddings or downloaded raw data.
*   **Structure:** Scripts may look for these local resources to avoid re-downloading or re-computing expensive processing steps.
*   **Documentation:** When using scripts that rely on `local/` data, ensure these expected paths are documented or configured, as they are essential for the script's independent execution.

## Reproducibility

*   **Seeds**: Experiments use a `master_seed` (defined in experiment configs) to control randomness for masking and other stochastic processes.
*   **Git Hash**: The executor records the current Git commit hash in MLflow. It warns or fails if the repository is not clean (unless `--ignore_unsafe` or `--debug` is used).
