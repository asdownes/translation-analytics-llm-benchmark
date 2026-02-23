#!/usr/bin/env python3
"""
Analyze token efficiency from a run_jobs_multi_lang.py stats.json file.

The script measures tokens-per-output-word for each model/prompt and saves
visualizations compatible with prior naming:
  9_token_efficiency_by_model.png
  10_token_efficiency_by_prompt.png
  11_efficiency_heatmap.png
  12_efficiency_distributions.png
  13_output_word_counts.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze token efficiency from stats.json")
    parser.add_argument("--stats", type=Path, required=True, help="Path to stats.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for visualization PNGs")
    return parser.parse_args()


def resolve_output_path(raw_path: str, stats_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidate_from_stats = (stats_path.parent / path).resolve()
    if candidate_from_stats.exists():
        return candidate_from_stats

    return (Path.cwd() / path).resolve()


def count_words(file_path: Path) -> int:
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Warning: failed to read {file_path}: {exc}")
        return 0

    return len(text.split())


def build_efficiency_rows(stats: Dict, stats_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    tasks = stats.get("tasks", [])
    if not isinstance(tasks, list):
        return rows

    for task in tasks:
        if not isinstance(task, dict):
            continue

        raw_out_path = task.get("out_path")
        if not isinstance(raw_out_path, str) or not raw_out_path.strip():
            continue

        total_tokens = task.get("total_tokens")
        if not isinstance(total_tokens, (int, float)):
            total_tokens = task.get("output_tokens_total", 0)
        if not isinstance(total_tokens, (int, float)) or total_tokens <= 0:
            continue

        out_path = resolve_output_path(raw_out_path, stats_path)
        word_count = count_words(out_path)
        if word_count <= 0:
            continue

        model = str(task.get("model", "unknown"))
        prompt = str(task.get("prompt_label", "unknown"))
        tokens_per_word = float(total_tokens) / float(word_count)

        rows.append(
            {
                "model": model,
                "prompt": prompt,
                "word_count": word_count,
                "total_tokens": float(total_tokens),
                "tokens_per_word": tokens_per_word,
                "out_path": str(out_path),
            }
        )

    return rows


def aggregate_by_key(rows: List[Dict[str, object]], key: str) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, Dict[str, float]] = {}
    for row in rows:
        label = str(row[key])
        data = aggregated.setdefault(label, {"total_words": 0.0, "total_tokens": 0.0, "tasks": 0.0})
        data["total_words"] += float(row["word_count"])
        data["total_tokens"] += float(row["total_tokens"])
        data["tasks"] += 1.0

    for data in aggregated.values():
        if data["total_words"] > 0:
            data["tokens_per_word"] = data["total_tokens"] / data["total_words"]
        else:
            data["tokens_per_word"] = 0.0

    return aggregated


def plot_by_model(rows: List[Dict[str, object]], model_efficiency: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    models = list(model_efficiency.keys())
    if not models:
        return

    tokens_per_word = [model_efficiency[m]["tokens_per_word"] for m in models]
    avg_words = [model_efficiency[m]["total_words"] / model_efficiency[m]["tasks"] for m in models]

    sorted_pairs = sorted(zip(models, tokens_per_word), key=lambda x: x[1])
    models_sorted, tpw_sorted = zip(*sorted_pairs)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models_sorted)))
    ax1.barh(range(len(models_sorted)), tpw_sorted, color=colors)
    ax1.set_yticks(range(len(models_sorted)))
    ax1.set_yticklabels(models_sorted)
    ax1.set_xlabel("Tokens per Output Word")
    ax1.set_title("Model Token Efficiency\n(Lower = More Efficient)")
    ax1.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(tpw_sorted):
        ax1.text(value, idx, f" {value:.2f}", va="center", fontsize=9)

    words_per_token = [1.0 / tpw if tpw > 0 else 0.0 for tpw in tokens_per_word]
    wpt_sorted = sorted(zip(models, words_per_token), key=lambda x: x[1], reverse=True)
    models_wpt, wpt_values = zip(*wpt_sorted)
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models_wpt)))
    ax2.barh(range(len(models_wpt)), wpt_values, color=colors2)
    ax2.set_yticks(range(len(models_wpt)))
    ax2.set_yticklabels(models_wpt)
    ax2.set_xlabel("Output Words per Token")
    ax2.set_title("Model Output Efficiency\n(Higher = More Efficient)")
    ax2.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(wpt_values):
        ax2.text(value, idx, f" {value:.3f}", va="center", fontsize=9)

    ax3.scatter(avg_words, tokens_per_word, s=200, alpha=0.6, c=range(len(models)), cmap="viridis")
    for idx, model in enumerate(models):
        ax3.annotate(model, (avg_words[idx], tokens_per_word[idx]), fontsize=8, ha="right", va="bottom")
    ax3.set_xlabel("Average Output Words per Task")
    ax3.set_ylabel("Tokens per Word")
    ax3.set_title("Output Size vs Token Efficiency")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "9_token_efficiency_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_by_prompt(rows: List[Dict[str, object]], prompt_efficiency: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    prompts = list(prompt_efficiency.keys())
    if not prompts:
        return

    prompt_tpw = [prompt_efficiency[p]["tokens_per_word"] for p in prompts]

    sorted_pairs = sorted(zip(prompts, prompt_tpw), key=lambda x: x[1])
    prompts_sorted, ptpw_sorted = zip(*sorted_pairs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(prompts_sorted)))
    ax1.barh(range(len(prompts_sorted)), ptpw_sorted, color=colors)
    ax1.set_yticks(range(len(prompts_sorted)))
    ax1.set_yticklabels(prompts_sorted)
    ax1.set_xlabel("Tokens per Output Word")
    ax1.set_title("Prompt Token Efficiency\n(Lower = More Efficient)")
    ax1.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(ptpw_sorted):
        ax1.text(value, idx, f" {value:.2f}", va="center", fontsize=9)

    prompt_wpt = [1.0 / tpw if tpw > 0 else 0.0 for tpw in prompt_tpw]
    pwpt_sorted = sorted(zip(prompts, prompt_wpt), key=lambda x: x[1], reverse=True)
    prompts_wpt, pwpt_values = zip(*pwpt_sorted)
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(prompts_wpt)))
    ax2.barh(range(len(prompts_wpt)), pwpt_values, color=colors2)
    ax2.set_yticks(range(len(prompts_wpt)))
    ax2.set_yticklabels(prompts_wpt)
    ax2.set_xlabel("Output Words per Token")
    ax2.set_title("Prompt Output Efficiency\n(Higher = More Efficient)")
    ax2.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(pwpt_values):
        ax2.text(value, idx, f" {value:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "10_token_efficiency_by_prompt.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(rows: List[Dict[str, object]], output_dir: Path) -> None:
    model_prompt_efficiency: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        model = str(row["model"])
        prompt = str(row["prompt"])
        model_prompt_efficiency.setdefault(model, {}).setdefault(prompt, []).append(float(row["tokens_per_word"]))

    prompts = sorted({str(row["prompt"]) for row in rows})
    models = sorted(model_prompt_efficiency.keys())
    if not prompts or not models:
        return

    matrix = np.zeros((len(models), len(prompts)))
    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            values = model_prompt_efficiency[model].get(prompt, [])
            matrix[i, j] = float(np.mean(values)) if values else np.nan

    fig, ax = plt.subplots(figsize=(14, 8))
    center = float(np.nanmean(matrix)) if np.isfinite(np.nanmean(matrix)) else None
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        xticklabels=prompts,
        yticklabels=models,
        cbar_kws={"label": "Tokens per Word"},
        ax=ax,
        center=center,
    )
    ax.set_title("Token Efficiency Heatmap: Tokens per Output Word\n(Lower = Better)", fontsize=14, pad=20)
    ax.set_xlabel("Prompt Type", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "11_efficiency_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_distributions(rows: List[Dict[str, object]], output_dir: Path) -> None:
    all_tpw = [float(row["tokens_per_word"]) for row in rows]
    if not all_tpw:
        return

    models = sorted({str(row["model"]) for row in rows})
    prompts = sorted({str(row["prompt"]) for row in rows})

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.hist(all_tpw, bins=min(30, max(5, len(all_tpw) // 2)), color="steelblue", alpha=0.7, edgecolor="black")
    ax1.axvline(np.mean(all_tpw), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(all_tpw):.2f}")
    ax1.axvline(np.median(all_tpw), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(all_tpw):.2f}")
    ax1.set_xlabel("Tokens per Word")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Token Efficiency Across All Tasks")
    ax1.legend()
    ax1.grid(alpha=0.3)

    model_tpw_data = [[float(row["tokens_per_word"]) for row in rows if row["model"] == model] for model in models]
    ax2.boxplot(model_tpw_data, labels=models, vert=True)
    ax2.set_ylabel("Tokens per Word")
    ax2.set_title("Token Efficiency Distribution by Model")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    prompt_tpw_data = [[float(row["tokens_per_word"]) for row in rows if row["prompt"] == prompt] for prompt in prompts]
    ax3.boxplot(prompt_tpw_data, labels=prompts, vert=True)
    ax3.set_ylabel("Tokens per Word")
    ax3.set_title("Token Efficiency Distribution by Prompt")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(axis="y", alpha=0.3)

    word_counts = [float(row["word_count"]) for row in rows]
    token_counts = [float(row["total_tokens"]) for row in rows]
    colors = [models.index(str(row["model"])) for row in rows]
    ax4.scatter(word_counts, token_counts, c=colors, cmap="tab10", alpha=0.6, s=50)
    ax4.set_xlabel("Output Word Count")
    ax4.set_ylabel("Total Tokens Used")
    ax4.set_title("Word Count vs Token Count (All Tasks)")
    ax4.grid(alpha=0.3)

    if len(word_counts) >= 2:
        z = np.polyfit(word_counts, token_counts, 1)
        trend = np.poly1d(z)
        x_line = np.linspace(min(word_counts), max(word_counts), 100)
        ax4.plot(x_line, trend(x_line), "r--", alpha=0.8, linewidth=2, label=f"Trend: y={z[0]:.2f}x+{z[1]:.0f}")
        ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "12_efficiency_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_word_counts(rows: List[Dict[str, object]], output_dir: Path) -> None:
    models = sorted({str(row["model"]) for row in rows})
    prompts = sorted({str(row["prompt"]) for row in rows})
    if not models or not prompts:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    model_word_counts = {model: [float(row["word_count"]) for row in rows if row["model"] == model] for model in models}
    avg_words_by_model = {model: float(np.mean(values)) for model, values in model_word_counts.items() if values}
    std_words_by_model = {model: float(np.std(values)) for model, values in model_word_counts.items() if values}

    model_labels = list(avg_words_by_model.keys())
    model_avgs = [avg_words_by_model[label] for label in model_labels]
    model_stds = [std_words_by_model[label] for label in model_labels]

    ax1.barh(model_labels, model_avgs, xerr=model_stds, color="lightcoral", alpha=0.7, capsize=5)
    ax1.set_xlabel("Average Output Word Count")
    ax1.set_title("Output Word Count by Model\n(Should be roughly consistent)")
    ax1.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(model_avgs):
        ax1.text(value, idx, f" {value:.0f}+/-{model_stds[idx]:.0f}", va="center", fontsize=9)

    prompt_word_counts = {prompt: [float(row["word_count"]) for row in rows if row["prompt"] == prompt] for prompt in prompts}
    avg_words_by_prompt = {prompt: float(np.mean(values)) for prompt, values in prompt_word_counts.items() if values}
    std_words_by_prompt = {prompt: float(np.std(values)) for prompt, values in prompt_word_counts.items() if values}

    prompt_labels = list(avg_words_by_prompt.keys())
    prompt_avgs = [avg_words_by_prompt[label] for label in prompt_labels]
    prompt_stds = [std_words_by_prompt[label] for label in prompt_labels]

    ax2.barh(prompt_labels, prompt_avgs, xerr=prompt_stds, color="lightblue", alpha=0.7, capsize=5)
    ax2.set_xlabel("Average Output Word Count")
    ax2.set_title("Output Word Count by Prompt\n(Should be roughly consistent)")
    ax2.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(prompt_avgs):
        ax2.text(value, idx, f" {value:.0f}+/-{prompt_stds[idx]:.0f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "13_output_word_counts.png", dpi=300, bbox_inches="tight")
    plt.close()


def print_summary(rows: List[Dict[str, object]], model_efficiency: Dict[str, Dict[str, float]], prompt_efficiency: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    all_tpw = [float(row["tokens_per_word"]) for row in rows]
    word_counts = [float(row["word_count"]) for row in rows]

    print("=" * 60)
    print("TOKEN EFFICIENCY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Rows analyzed: {len(rows)}")

    print("\nOverall statistics:")
    print(f"  Average tokens per word: {np.mean(all_tpw):.2f}")
    print(f"  Median tokens per word: {np.median(all_tpw):.2f}")
    print(f"  Std deviation: {np.std(all_tpw):.2f}")
    print(f"  Range: {min(all_tpw):.2f} - {max(all_tpw):.2f}")

    model_sorted = sorted(
        ((model, data["tokens_per_word"]) for model, data in model_efficiency.items()),
        key=lambda x: x[1],
    )
    prompt_sorted = sorted(
        ((prompt, data["tokens_per_word"]) for prompt, data in prompt_efficiency.items()),
        key=lambda x: x[1],
    )

    print("\nMost efficient models (lowest tokens per word):")
    for idx, (model, tpw) in enumerate(model_sorted[:3], 1):
        wpt = 1.0 / tpw if tpw > 0 else 0.0
        print(f"  {idx}. {model}: {tpw:.2f} tokens/word ({wpt:.3f} words/token)")

    print("\nMost efficient prompts (lowest tokens per word):")
    for idx, (prompt, tpw) in enumerate(prompt_sorted[:3], 1):
        wpt = 1.0 / tpw if tpw > 0 else 0.0
        print(f"  {idx}. {prompt}: {tpw:.2f} tokens/word ({wpt:.3f} words/token)")

    print("\nOutput consistency check:")
    print(f"  Average output words per task: {np.mean(word_counts):.1f}")
    print(f"  Std deviation: {np.std(word_counts):.1f}")
    print(f"  Range: {min(word_counts):.0f} - {max(word_counts):.0f} words")
    if np.mean(word_counts) > 0:
        cv = (np.std(word_counts) / np.mean(word_counts)) * 100.0
        print(f"  Coefficient of variation: {cv:.1f}%")

    print("\nSaved visualizations to:")
    print(f"  {output_dir}")
    print("=" * 60)


def main() -> int:
    args = parse_args()

    if not args.stats.exists():
        print(f"Error: stats file not found: {args.stats}")
        return 1

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_efficiency_rows(stats, args.stats)
    if not rows:
        print("Error: no usable task data found in stats file")
        return 1

    model_efficiency = aggregate_by_key(rows, "model")
    prompt_efficiency = aggregate_by_key(rows, "prompt")

    plot_by_model(rows, model_efficiency, args.output_dir)
    plot_by_prompt(rows, prompt_efficiency, args.output_dir)
    plot_heatmap(rows, args.output_dir)
    plot_distributions(rows, args.output_dir)
    plot_word_counts(rows, args.output_dir)

    print_summary(rows, model_efficiency, prompt_efficiency, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
