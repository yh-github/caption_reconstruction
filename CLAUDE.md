# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research codebase for a paper on *caption reconstruction*: mask a contiguous gap of 1-second clips in a 60-second video, then reconstruct it either from **text** (an LLM in-fills the missing captions from surrounding captions) or from **vision** (interpolate SigLIP frame embeddings across the gap). Comparing the two measures how predictable a video's events are without seeing them. The current experimental standard, which results are legacy, and the key findings so far are in `docs/experiments/experiment_protocol_current.md`. Read it before designing or interpreting experiments.

## Commands

All commands run from the repo root, because config paths such as `config/system.yaml` and `config/recon/...` are relative. Use the project venv (`.venv/bin/python`, Python 3.13; the package requires >=3.11).

```bash
pip install -r requirements.txt && pip install -e .     # setup
python scripts/download_data.py                          # fetch disk_cache/, local/, datasets from remote

python src/main.py <config.yaml> --dry-run --verbose     # list runs that would execute, no API/model calls
python src/main.py <config.yaml>                         # full run
python src/main.py <config.yaml> --block-llm             # cached LLM responses only; a cache miss errors
python src/main.py <config.yaml> --eval-only             # re-evaluate existing result JSONs only
python src/main.py <config.yaml> --override base_params/master_seed=7 paths/results=tmp_res/
python src/main.py <config.yaml> --worker-id 0 --total-workers 4   # shard the videos across workers

pytest tests/                                            # all tests (pyproject puts src/ on pythonpath)
pytest tests/test_masking.py::test_name                  # single test
streamlit run scripts/evaluation_explorer_app.py         # interactive results explorer
```

Gotchas:
- `--override` keys use `/` as the path separator, not `.` (see `set_nested_key` in `src/experiment_executor/config_loader.py`). List indices are numeric parts, e.g. `recon_strategy/0/temperature=0.6`. Some docs show `.`, which is wrong.
- A real (non-dry) run fails if the git tree is dirty, including untracked files, because the commit hash is logged to MLflow for reproducibility. Pass `--ignore-unsafe` (or `--debug`) to bypass this.
- `src/` modules import each other as top-level packages (`from experiment_executor...`, `from llm...`), not `src.xxx`. Scripts in `scripts/` add `src/` to `sys.path` themselves.
- There is no lint or CI config. CONTRIBUTING says loose PEP 8 with type hints; black/flake8 are optional.

## Architecture

**Config → Pipeline → Runners → per-video results → Evaluator → CSV/MLflow.**

- `src/main.py` parses `ExecArgs` (`src/data_models/exec_args.py`), builds an `ExperimentPipeline`, and hands it to `Executor` (`pipeline_executor.py`). The `Executor` takes a per-worker file lock, opens a parent MLflow run, and runs each runner as a nested MLflow run.
- **Config loading** (`config_loader.py`): `config/system.yaml` (paths, `hf_repo_id`, optional `llm_backend`) is merged with the experiment YAML, and experiment keys take precedence. A top-level key of the form `IMPORT <key>: <file.yaml>` pulls `<key>` from a file relative to the config and extends any list already there. This is how shared `masking_configs` are reused. The config file's stem becomes the parent run name and the results folder name.
- **Pipeline** (`pipeline.py`): `base_params.experiment_type` chooses between two paths:
  - `RECON`: indirect/text path. Data loader from `src/data/data_loaders.py`, `TextReconstructionStrategyBuilder` (`src/reconstruction/text_reconstruction.py`), and `ExperimentRunner`.
  - `RECON_VECTORS`: direct/vector path. `VectorDataLoader`, `VectorReconstructionStrategyBuilder` (`mean_closest`, `repeat_closest`), and `VectorRunner`.

  `build_experiments()` yields the cross product of `recon_strategy` × masking strategies (from `src/reconstruction/masking.py`). When several `local_llm` strategies share the same `(model_key, prompt_dir)`, they are grouped into one `BatchExperimentRunner` + `BatchGridSearchStrategy`, so a single model load serves the whole parameter sweep (`docs/code/batch_processing.md`).
- **Runners** (`experiment_runner.py`) write one `<video_id>.json` per video to `results/recon/<config_stem>/<run_name>/`. Videos whose files already exist are loaded and skipped, which is how resumption works. Results are also synced to and from the Hugging Face dataset repo (`paths.hf_repo_id`, see `src/data/hf_sync.py`). Unless `--no-download-existing` is set, the pipeline prefetches existing remote results before running.
- **LLM layer** (`src/llm/`):
  - `llm_interaction.py` handles Gemini through `google.genai`, with responses cached in `disk_cache/cache.db`.
  - `local_llm.py` holds the `MODELS` registry of HF local models (`phi-3`, `llama-3.1-8b`, `qwen-2.5-*`, ...) and uses 4-bit quantization on GPU.
  - `keras_llm.py` is the Keras/JAX backend for TPU. `common_utils/device_setup.get_llm_backend()` chooses it automatically, or from `llm_backend` in `system.yaml`.
  - `embedder.py` (Gemini) and `local_embedder.py` (`local:<hf_id>`, e.g. `local:all-mpnet-base-v2`) each keep a separate diskcache under `disk_cache/<model>__<dim>__<task>/`.
  - In blocked modes (`--dry-run`, `--validate-cache`, `--block-llm`), the LLM client is replaced by a `BlockingClient` that raises `CacheMissError` and aborts after 20 misses.
  - Prompts are text templates under `prompts/`. Local models use a prompt *directory* with `start.txt`, `default.txt`, and `end.txt` variants, chosen by where the gap falls (`llm/prompting.py`, `ClozePromptBuilder`).
- **Evaluation** (`src/evaluations/`): the `evaluation.type` config key selects the method. `emb_sim` gives cos_sim and context-projected `cos_sim_residual`. `retrieval` gives MRR, Recall@k, and mean rank against a distractor pool controlled by `pool_scope`. `bert_score` and `nop` are also available.
- **Shared-target evaluation** (`src/shared_target/` + `scripts/eval_shared_target.py`, config `configs/eval_shared_target.yaml`): a separate, newer evaluation that projects text and visual "arms" into SigLIP space. It uses stratified candidate pools, calibrated AUC, cluster bootstrap, and pre-registered sanity gates.
- `scripts/` holds about 100 one-off analysis, plotting, and audit scripts, and `scripts/scratch/` holds throwaway code. Many of these scripts contain hardcoded paths. `src/analysis/` holds older analysis and plotting modules.

## Research conventions that affect code

- **Current standard:** the text arm is `llama-3.1-8b` with prompt dir `prompts/dense_window/`, temperature 0.6, and repetition penalty 1.05. The evaluation embedder is `all-mpnet-base-v2`. The visual arm is SigLIP `google/siglip-base-patch16-224` (768-dim) with `Visual_SigLIP_MeanClosest`/`RepeatClosest`. The benchmark uses wild4 (100 videos) + wild5 (235 videos) = 335 videos, W ∈ {1,2,3,4,6,8,12,16}, and gap start i=29. W > 16 is excluded.
- Retrieval must use `pool_scope: "video"` (all 60 timestamps). `pool_scope: "window"` produced the bogus `wild4_llama_w3_window` result.
- **Legacy, do not use for new analyses:** Phi-3 runs, the 384-dim embeddings in `local/wild_videos_embs/`, `phi_vs_video_integration_summary.csv`, and `temporal_metrics_final.csv`.
- **Joining datasets:** captions use `Olly's-Farm` while files on disk use `Olly_s-Farm`, so normalize before joining.
- The 335 clips come from 40 YouTube channels (`movie_id`). Statistics must cluster by channel.
- `cos_sim_mean` is strongly confounded by caption repetitiveness (`APCS_T`). Prefer rank-based metrics such as calibrated AUC and paired ΔRank.
- **Main data products:** `results/unified_benchmark_master.csv` and `results/apriori_full_scores.csv`. Column meanings are in `docs/experiments/field_dictionary.md`.
- `results/`, `disk_cache/`, `local/`, `mlruns/`, and `logs/` are gitignored, apart from a few allowlisted CSVs.

## Repo workflow

When asked to commit and push (`.agents/workflows/push.md`), group the changed files into logically related commits with meaningful messages, then push.
