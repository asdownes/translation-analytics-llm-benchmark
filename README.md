# local-llm-translation

Minimal scripts for multi-language translation runs with local Ollama models, plus token-efficiency analysis.

## What is here

- `scripts/run_jobs_multi_lang.py`: run translation jobs across language pairs and models.
- `run_translation_pipeline.sh`: main helper to run jobs across multiple local Ollama servers.
- `scripts/analyze_efficiency.py`: generate efficiency plots from a run `stats.json`.
- `run_efficiency_reports.sh`: batch analysis for run directories with `stats.json`.
- `data/`: static benchmark source, outputs, and per-line metrics.

## Setup (Conda, recommended)

```bash
conda env create -f environment.yml
conda activate local-llm-translation
```

`OPENAI_API_KEY` is only needed for OpenAI API runs.

## Run (local Ollama, primary)

```bash
./run_translation_pipeline.sh -- \
  --source-file data/1143_en.txt \
  --pair de --pair ru \
  --models "aya-expanse:32b,qwen2.5:32b" \
  --out-root output/local_run
```

## Run (OpenAI API, optional)

```bash
python scripts/run_jobs_multi_lang.py \
  --source-file data/1143_en.txt \
  --pair de --pair ru \
  --models "gpt-5.2" \
  --out-root output/run1
```

## Data and output layout

The committed `data/` folder is static benchmark data, not the destination for new runs. It includes `1143_en.txt` plus translations to German, Japanese, Russian, and Simplified Chinese. `local_9_*_1143` contains nine local Ollama model runs per language; `gpt_5.2_all_pairs_1143` contains GPT-5.2 runs for all four languages.

Generated `--out-root` directories use the same layout:

- `<line_count>_<target_lang>_<model>_<prompt_id>.txt`: translated lines.
- `metrics/*.jsonl`: per-line request status, text, token counts, timing, and raw provider metadata.
- `thinking/*_thinking.jsonl`: captured reasoning summaries where available.
- `stats.json`: run metadata, totals, per-model/per-language aggregates, and task paths.

## Analyze

```bash
python scripts/analyze_efficiency.py \
  --stats output/run1/stats.json \
  --output-dir output/run1/visualizations_efficiency
```

```bash
./run_efficiency_reports.sh
```
