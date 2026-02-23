# local-llm-translation

Minimal scripts for multi-language translation runs with local Ollama models, plus token-efficiency analysis.

## What is here

- `scripts/run_jobs_multi_lang.py`: run translation jobs across language pairs and models.
- `run_translation_pipeline.sh`: main helper to run jobs across multiple local Ollama servers.
- `scripts/analyze_efficiency.py`: generate efficiency plots from a run `stats.json`.
- `run_efficiency_reports.sh`: batch analysis for `output/*` runs.

## Setup (Conda, recommended)

```bash
conda env create -f environment.yml
conda activate local-llm-translation
```

`OPENAI_API_KEY` is only needed for OpenAI API runs.

## Run (local Ollama, primary)

```bash
./run_translation_pipeline.sh -- \
  --source-file path/to/source.txt \
  --pair de --pair ru \
  --models "aya-expanse:32b,qwen2.5:32b" \
  --out-root output/local_run
```

## Run (OpenAI API, optional)

```bash
python scripts/run_jobs_multi_lang.py \
  --source-file path/to/source.txt \
  --pair de --pair ru \
  --models "gpt-5.2" \
  --out-root output/run1
```

## Stats output (brief)

Each run writes `stats.json` to `<out-root>/stats.json` (unless `--no-stats` is used).
It includes run metadata, totals, per-model/per-language aggregates, and per-task rows:

```json
{
  "source_file": "...",
  "models": ["..."],
  "total_tasks": 0,
  "failed_lines": 0,
  "model_stats": {"model_name": {"lines": 0, "failed_lines": 0}},
  "lang_stats": {"de": {"lines": 0, "failed_lines": 0}},
  "tasks": [
    {
      "model": "...",
      "target_lang": "...",
      "out_path": "...",
      "metrics_path": "...",
      "output_tokens_total": 0,
      "prompt_tokens_total": 0
    }
  ]
}
```

## Analyze

```bash
python scripts/analyze_efficiency.py \
  --stats output/run1/stats.json \
  --output-dir output/run1/visualizations_efficiency
```

```bash
./run_efficiency_reports.sh
```
