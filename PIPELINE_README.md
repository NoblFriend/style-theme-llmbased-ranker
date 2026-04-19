# Pipeline Quick Guide

This file is a short handoff guide: what the pipeline does, and how to run it on needed folders.

## What It Does

For each input folder with `.txt` files, pipeline runs 3 stages:

1. Scoring: evaluates each text independently.
2. Ranking: ranks texts row-by-row.
3. Metrics: computes metrics and joins scorer + ranker results.

Final output for each folder is a separate metrics file:

- `results/{folder_name}/metrics.csv`

No global combined CSV is produced by this script.

## Required Inputs

Each data folder should contain multiple `.txt` files with line-aligned generations.

Example:

- `data/09_02_agit/*.txt`
- `data/16_03_llama_sad/*.txt`

## Before Running

1. Activate virtual environment.
2. Fill API settings in `config.yaml` under `api`:
   - `base_url` (OpenAI-compatible, usually ends with `/v1`)
   - `model`
   - `api_key` if needed
3. Make sure endpoint is reachable.

## Main Command

Run from repository root.

```bash
# process all folders from data/
python src/process_pipeline.py --topic astrophysics
```

## Common Run Modes

```bash
# process only specific folders
python src/process_pipeline.py data/09_02_agit data/16_03_llama_sad --topic astrophysics

# custom output base directory
python src/process_pipeline.py --topic astrophysics --output-base my_results

# custom data source directory for auto-discovery
python src/process_pipeline.py --topic astrophysics --data-dir data
```

## What Appears In Output Folder

For folder `data/09_02_agit` output will be in `results/09_02_agit/`:

1. `scored.csv`
2. `scored_detailed.csv`
3. `ranked.csv`
4. `scored_metrics.csv`
5. `ranked_metrics.csv`
6. `metrics.csv` (final joined table)

## Minimal Step-by-Step 

1. Open repository root.
2. Activate venv.
3. Set `api.base_url` and `api.model` in `config.yaml` (and `api_key` if required).
4. Run:

```bash
python src/process_pipeline.py data/YOUR_FOLDER --topic YOUR_TOPIC --output-base new_results
```

5. Take final file from:

- `new_results/results/YOUR_FOLDER/metrics.csv`

## If Something Fails

- Config error about API: check `config.yaml` and env overrides.
- HTTP/auth error: verify `base_url`, `api_key`, and endpoint format.
- Empty output: confirm folder has `.txt` files and non-empty lines.
- Metrics missing: ensure `scored.csv` and `ranked.csv` were generated first.
