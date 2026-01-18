# Caption Reconstruction

**Dense caption reconstruction: video in-filling with language models**

This project investigates the boundary between **semantic inference** (what *must* happen) and **visual perception** (what *actually* happened) through a novel *Caption Reconstruction* comparative framework. By comparing how well an LLM (Text-Only) and a Visual-Interpolation model (Visual-Only) reconstruct missing segments of a video, we quantify "Multimodal Redundancy" and identify a "Predictability Spectrum" across different video domains.

## 📄 Abstract

Recent Video-LLMs typically treat video understanding as a continuous stream of visual encoding. However, real-world events often follow structured, semantic scripts that pre-trained language models can predict without immediate visual evidence.
Our analysis showcases a spectrum of narrative predictability: while *stochastic* events (e.g., nature) require visual grounding, *procedural* events (e.g., farming, manufacturing) allow text-only models to in-fill accurate reconstructions, rendering visual processing redundant for significant durations.

For more details, see the [paper draft](docs/paper/draft.md).

## 🔬 Reproduction

For a fully interactive reproduction of the paper's results (without needing API keys or heavy computation), check out the **[End-to-End Reproduction Notebook](notebooks/end_to_end_reproduction.ipynb)**.

This notebook allows you to:
1. Download all cached data.
2. Regenerate video embeddings.
3. Run the experiments using cached LLM responses.
4. Re-create the plots from the paper.

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- Access to Hugging Face (for model access if running inference)
- Google Gemini API Key (if running LLM inference)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/caption_reconstruction.git
   cd caption_reconstruction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## 💾 Data Setup

This project relies on specific datasets and pre-computed embeddings. Use the provided script to download the necessary data from the remote cache:

```bash
python scripts/download_data.py
```
This will download and extract datasets into the `local/` and `datasets/` directories.

## 🚀 Usage

The main entry point for running experiments is `src/main.py`.

### Basic Execution
To run a specific experiment configuration:

```bash
python src/main.py config/recon/wild_dev_sim_text.yaml
```

### Common Flags

- **`--dry-run`**: Simulate the execution without making API calls or heavy computations. Useful for verifying the pipeline steps.
  ```bash
  python src/main.py config/recon/wild_dev_sim_text.yaml --dry-run --verbose
  ```

- **`--verbose`**: Enable detailed logging to see exactly what the runner is doing.

- **`--block-llm`** (or `--cached-execution-only`): strictly use cached LLM responses. If a response is missing, the script will error out instead of calling the API. Ideal for reproduction without incurring costs.

- **`--eval-only`**: Skip the generation phase entirely and only run the evaluation/metrics calculation on existing results.

- **`--no-download-existing`**: Skip downloading results that already exist locally.

### Configuration
Experiments are defined using YAML files in the `config/` directory.
- `config/system.yaml`: Defines system-level paths (cache, datasets).
- `config/recon/`: Contains specific reconstruction experiment configurations.

## 📚 Documentation

For more detailed information, please refer to the documentation in `docs/code/`:

- **[Setup Guide](docs/code/setup.md)**: Detailed instructions on environment and directory setup.
- **[Execution Guide](docs/code/execution.md)**: Comprehensive guide on running experiments and using helper scripts.
- **[Caching & Reproducibility](docs/code/caching.md)**: Explanation of the disk cache system, file locking, and how to verify experiments without API costs.
- **[Batch Processing](docs/code/batch_processing.md)**: Details on heterogeneous batch processing for parameter sweeps.

## 📂 Project Structure

- **`src/`**: Core source code.
    - `experiment_executor/`: Pipeline orchestration and runners.
    - `data_models/`: Pydantic models for configuration and arguments.
    - `llm/`: Interfaces for Large Language Models (Gemini, etc.).
    - `reconstruction/`: Logic for text and visual reconstruction methods.
- **`scripts/`**: Utility scripts for data downloading, analysis, and plotting.
- **`config/`**: Configuration files for experiments and system settings.
- **`docs/`**: Documentation and paper drafts.
## 🎓 Citation

If you use this codebase or dataset in your research, please cite our work:

```bibtex
@inproceedings{haimovitch2026caption,
  title={Dense caption reconstruction: video in-filling with language models},
  author={Haimovitch, Yoav},
  year={2026}
}
```
