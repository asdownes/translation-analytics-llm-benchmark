# translation-analytics-llm-benchmark

Minimal scripts for multi-language translation runs with local Ollama models, plus token-efficiency analysis.

## Project Paper

This repository contains the code associated with the project described in the paper *Translation Analytics for Freelancers II: Benchmarking Local LLMs for Confidential Translation Workflows* by Yuri Balashov, Rex VanHorn, Austin Downes, and Mingxi Xu (EAMT-2026, forthcoming).

## Data Acknowledgment

The Christopher & Dana Reeve Foundation Multilingual Corpus (RFMC) used for these benchmarking experiments is made available with permission from the Christopher & Dana Reeve Foundation for **non-commercial, academic research purposes only**. Please cite the associated paper if you use this dataset.

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

## Citation

```bibtex
@inproceedings{balashov-etal-2026-translation,
    title = "Translation Analytics for Freelancers {II}: Benchmarking Local {LLM}s for Confidential Translation Workflows",
    author = "Balashov, Yuri  and
      VanHorn, Rex  and
      Downes, Austin  and
      Xu, Mingxi",
    booktitle = "Proceedings of the 26th Annual Conference of the European Association for Machine Translation",
    year = "2026",
    publisher = "European Association for Machine Translation (EAMT)",
    note = "Forthcoming"
}
```
