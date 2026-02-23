#!/usr/bin/env python3
"""
Run multi-language translation batches via the OpenAI API (Responses or Chat Completions).

Key behavior:
- Single source file shared across all language pairs (aligned lines).
- Shared prompt template with {SOURCE_LANG}/{TARGET_LANG} placeholders; optional per-model prompt override.
- Work is scheduled per model/target task and distributed across worker threads.
- Per-line metrics are logged to JSONL files, plus optional reasoning summaries.

Output files:
  <out_root>/<line_count>_<output_lang>_<model_tag>_<prompt_id>.txt
  <metrics_dir>/<line_count>_<output_lang>_<model_tag>_<prompt_id>.jsonl
  <thinking_dir>/<line_count>_<output_lang>_<model_tag>_<prompt_id>_thinking.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


FAILED_SENTINEL = "[FAILED]"
PROMPT_ID_RE = re.compile(r"^(p\d+)")
DEFAULT_TEMPLATE_PROMPT_ID = "p0"
DEFAULT_TEMPLATE_PROMPT_PATH = Path(__file__).resolve().parents[1] / "main_prompt_template.txt"
DEFAULT_MODELS = ["gpt-5.2"]
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_API_MODE = "responses"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_REASONING_EFFORT = "high"

LANGUAGE_NAME_OVERRIDES = {
    "en-gb": "English (UK)",
    "en-us": "English (US)",
    "pt-br": "Portuguese (Brazil)",
    "pt-pt": "Portuguese (Portugal)",
    "zh-cn": "Simplified Chinese (Mainland China)",
    "zh-hans": "Simplified Chinese (Mainland China)",
    "zh-hant": "Chinese (Traditional)",
    "zh-tw": "Chinese (Traditional)",
}
LANGUAGE_NAME_MAP = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "mr": "Marathi",
    "ms": "Malay",
    "nb": "Norwegian Bokmal",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Simplified Chinese (Mainland China)",
}


class ProgressTracker:
    """Simple console progress renderer with optional live stats."""

    def __init__(self, total_tasks: int, total_lines: int, enabled: bool = True, bar_width: int = 28, min_interval: float = 0.1):
        self.total_tasks = total_tasks
        self.total_lines = total_lines
        self.enabled = enabled and total_lines >= 0
        self.bar_width = bar_width
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.active_tasks: Dict[int, Dict[str, object]] = {}
        self.completed_lines = 0
        self.completed_tasks = 0
        self.start_time = time.time()
        self._last_render_len = 0
        self._last_render_time = 0.0

    def task_started(self, task_id: int, job_name: str, model_name: str, total_lines: int) -> None:
        with self.lock:
            self.active_tasks[task_id] = {
                "job": job_name,
                "model": model_name,
                "total": total_lines,
                "done": 0,
            }
            self._render(force=True)

    def advance(self, task_id: int, amount: int = 1) -> None:
        with self.lock:
            state = self.active_tasks.get(task_id)
            if state is not None:
                state["done"] = min(state["total"], int(state["done"]) + amount)
            if self.total_lines > 0:
                self.completed_lines = min(self.total_lines, self.completed_lines + amount)
            else:
                self.completed_lines += amount
            self._render()

    def task_completed(self, task_id: int, success: bool = True) -> None:
        with self.lock:
            self.active_tasks.pop(task_id, None)
            if success:
                self.completed_tasks += 1
            self._render(force=True)

    def log(self, message: str) -> None:
        with self.lock:
            self._clear_line_locked()
            print(message, flush=True)
            self._render(force=True)

    def close(self) -> None:
        with self.lock:
            if self.enabled:
                self._render(force=True)
                self._write_line("", newline=True)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def _render(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self._last_render_time) < self.min_interval:
            return
        self._last_render_time = now

        total_lines = max(self.total_lines, 0)
        completed_lines = max(min(self.completed_lines, total_lines if total_lines else self.completed_lines), 0)
        percent = 1.0 if total_lines == 0 else min(1.0, completed_lines / total_lines)
        filled = int(round(percent * self.bar_width))
        filled = min(filled, self.bar_width)
        bar = "=" * filled + "." * (self.bar_width - filled)

        elapsed = now - self.start_time
        rate = completed_lines / elapsed if elapsed > 0 else 0.0
        remaining = max(total_lines - completed_lines, 0)
        eta = remaining / rate if rate > 0 else None

        msg = f"Progress [{bar}] {percent * 100:5.1f}% ({completed_lines}/{total_lines} lines)"
        if rate > 0:
            msg += f" | {rate:.2f} lines/s | ETA {self._format_duration(eta)}"
        else:
            msg += " | ETA --:--"

        active = self._active_summary()
        if active:
            msg += f" | Active {active}"

        self._write_line(msg, newline=False)

    def _active_summary(self) -> str:
        if not self.active_tasks:
            return ""
        entries = []
        for idx, state in enumerate(self.active_tasks.values()):
            if idx >= 2:
                break
            job = str(state["job"])
            model = str(state["model"])
            done = int(state["done"])
            total = int(state["total"])
            entries.append(f"{job}:{model} {done}/{total}")
        remaining = len(self.active_tasks) - len(entries)
        if remaining > 0:
            entries.append(f"+{remaining} more")
        return ", ".join(entries)

    def _format_duration(self, seconds: Optional[float]) -> str:
        if seconds is None or seconds < 0:
            return "--:--"
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _clear_line_locked(self) -> None:
        if not self.enabled or self._last_render_len == 0:
            return
        sys.stderr.write("\r" + " " * self._last_render_len + "\r")
        sys.stderr.flush()
        self._last_render_len = 0

    def _write_line(self, msg: str, newline: bool) -> None:
        if not self.enabled:
            return
        sys.stderr.write("\r" + msg)
        if len(msg) < self._last_render_len:
            sys.stderr.write(" " * (self._last_render_len - len(msg)))
        if newline:
            sys.stderr.write("\n")
            self._last_render_len = 0
        else:
            self._last_render_len = len(msg)
        sys.stderr.flush()


@dataclass(frozen=True)
class PromptFile:
    label: str
    text: str
    prompt_id: str
    path: Path


@dataclass(frozen=True)
class PairSpec:
    source_lang: str
    target_lang: str
    prompt: PromptFile


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    model: str
    source_lang: str
    target_lang: str
    prompt: PromptFile
    out_path: Path
    metrics_path: Path
    thinking_path: Path


@dataclass
class TaskResult:
    model: str
    target_lang: str
    prompt_label: str
    prompt_id: str
    out_path: Path
    metrics_path: Path
    thinking_path: Path
    elapsed: float
    total_lines: int
    blank_lines: int
    over_limit_lines: int
    error_lines: int
    token_estimate_lines: int
    output_tokens_total: int
    output_tokens_est_total: int
    prompt_tokens_total: int
    attempts_total: int
    request_count: int
    wall_time_s_total: float
    wall_time_s_max: float

    @property
    def failed_lines(self) -> int:
        return self.over_limit_lines + self.error_lines


def slugify_model(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def language_name_from_code(code: str) -> str:
    normalized = code.strip().lower().replace("_", "-")
    if not normalized:
        return code
    override = LANGUAGE_NAME_OVERRIDES.get(normalized)
    if override:
        return override
    base = re.split(r"[-_]", normalized, maxsplit=1)[0]
    name = LANGUAGE_NAME_MAP.get(base)
    return name or normalized


def load_prompt_template(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Prompt template is empty: {path}")
    return text


def render_prompt_template(template_text: str, source_lang_code: str, target_lang_code: str) -> str:
    source_name = language_name_from_code(source_lang_code)
    target_name = language_name_from_code(target_lang_code)
    return template_text.replace("{SOURCE_LANG}", source_name).replace("{TARGET_LANG}", target_name)


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    return text.splitlines()


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for i, line in enumerate(lines):
            if i:
                f.write("\n")
            f.write(line)


def clean_translation(s: str, keep_multiline: bool) -> str:
    if not keep_multiline:
        s = s.splitlines()[0] if s.splitlines() else ""
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`")
    s = re.sub(r"^(?:[-*]\s+|\d+\s*[.)]\s+)", "", s)
    s = s.strip("\"'")
    return s


def parse_models(models_csv: str) -> List[str]:
    return [m.strip() for m in models_csv.split(",") if m.strip()]


def build_options(options_json: str, temperature: float, max_output_tokens: int) -> Dict:
    try:
        options = json.loads(options_json) if options_json else {}
    except Exception as exc:
        raise ValueError(f"Invalid --options-json: {exc}") from exc
    if not isinstance(options, dict):
        raise ValueError("--options-json must decode to a JSON object")
    options["temperature"] = temperature
    if max_output_tokens > 0:
        options["max_output_tokens"] = max_output_tokens
    return options


def prompt_id_from_label(prompt_label: str) -> str:
    match = PROMPT_ID_RE.match(prompt_label)
    if not match:
        raise ValueError(f"Prompt label '{prompt_label}' must start with p<digits> (example: p1-baseline)")
    return match.group(1)


def prompt_id_from_label_or_default(prompt_label: str, default: str) -> str:
    match = PROMPT_ID_RE.match(prompt_label)
    if match:
        return match.group(1)
    return default


def sanitize_output_lang(output_lang: str) -> str:
    lang = output_lang.strip()
    if not lang:
        raise ValueError("Output language must be non-empty")
    return re.sub(r"[^A-Za-z0-9._-]", "_", lang)


def format_model_name(model: str) -> str:
    return slugify_model(model).replace("_", "-")


def build_output_base(line_count: int, output_lang: str, model: str, prompt_id: str) -> str:
    model_tag = format_model_name(model)
    return f"{line_count}_{output_lang}_{model_tag}_{prompt_id}"


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def load_prompt(path: Path, cache: Dict[Path, PromptFile]) -> PromptFile:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Prompt file is empty: {path}")
    label = path.stem
    prompt_id = prompt_id_from_label(label)
    prompt = PromptFile(label=label, text=text, prompt_id=prompt_id, path=path)
    cache[resolved] = prompt
    return prompt


def parse_pair_arg(
    arg: str,
    default_source_lang: str,
    cache: Dict[Path, PromptFile],
    template_text: str,
    template_label: str,
    template_prompt_id: str,
    template_path: Path,
) -> PairSpec:
    pair_spec = arg.strip()
    prompt_override: Optional[PromptFile] = None
    if "=" in arg:
        left, right = arg.split("=", 1)
        pair_spec = left.strip()
        prompt_path_str = right.strip()
        if not prompt_path_str:
            raise ValueError(
                f"Invalid --pair value '{arg}'. Expected <tgt>, <src->tgt>, or <src->tgt>=<prompt_path>."
            )
        prompt_override = load_prompt(Path(prompt_path_str), cache)
    if not pair_spec:
        raise ValueError(
            f"Invalid --pair value '{arg}'. Expected <tgt>, <src->tgt>, or <src->tgt>=<prompt_path>."
        )

    if "->" in pair_spec:
        src_lang, tgt_lang = pair_spec.split("->", 1)
    elif ":" in pair_spec:
        src_lang, tgt_lang = pair_spec.split(":", 1)
    else:
        src_lang, tgt_lang = default_source_lang, pair_spec
    src_lang = src_lang.strip().lower()
    tgt_lang = tgt_lang.strip().lower()
    if not src_lang or not tgt_lang:
        raise ValueError(f"Invalid language pair in --pair '{arg}'")

    prompt = prompt_override
    if prompt is None:
        prompt_text = render_prompt_template(template_text, src_lang, tgt_lang)
        prompt = PromptFile(
            label=template_label,
            text=prompt_text,
            prompt_id=template_prompt_id,
            path=template_path,
        )
    return PairSpec(source_lang=src_lang, target_lang=tgt_lang, prompt=prompt)


def parse_model_prompt_override(arg: str, cache: Dict[Path, PromptFile]) -> Tuple[str, str, PromptFile]:
    if "=" not in arg:
        raise ValueError(f"Invalid --model-prompt value '{arg}'. Expected model[[@lang]]=<prompt_path>.")
    left, right = arg.split("=", 1)
    model_spec = left.strip()
    prompt_path = Path(right.strip())
    if "@" in model_spec:
        model_name, lang = model_spec.split("@", 1)
        lang_key = lang.strip().lower()
    else:
        model_name = model_spec
        lang_key = "*"
    model_name = model_name.strip()
    if not model_name:
        raise ValueError(f"Invalid --model-prompt value '{arg}': missing model name.")
    prompt = load_prompt(prompt_path, cache)
    return model_name, lang_key, prompt


def _build_headers(api_key: str, organization: Optional[str], project: Optional[str]) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project
    return headers


def _post_json(session: requests.Session, url: str, headers: Dict[str, str], payload: Dict, timeout: float) -> Dict:
    resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
        message = None
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = error.get("message")
            if not message:
                message = data.get("message")
        if message:
            raise RuntimeError(f"OpenAI API error {resp.status_code}: {message}")
        resp.raise_for_status()
    return resp.json()


def _extract_openai_text(data: Dict) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output = data.get("output")
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type in ("output_text", "text", "refusal"):
                        text_value = block.get("text")
                        if isinstance(text_value, str):
                            parts.append(text_value)
            elif isinstance(content, str):
                parts.append(content)
        if parts:
            return "".join(parts)

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text_value = first.get("text")
            if isinstance(text_value, str):
                return text_value

    return ""


def _extract_usage_tokens(data: Dict) -> Tuple[Optional[int], Optional[int]]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None, None
    output_tokens = usage.get("output_tokens")
    if not isinstance(output_tokens, int):
        output_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("input_tokens")
    if not isinstance(prompt_tokens, int):
        prompt_tokens = usage.get("prompt_tokens")
    return output_tokens if isinstance(output_tokens, int) else None, prompt_tokens if isinstance(prompt_tokens, int) else None


def _extract_reasoning_summary(data: Dict) -> str:
    output = data.get("output")
    if not isinstance(output, list):
        return ""
    summaries: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if isinstance(summary, str):
            summaries.append(summary)
        elif isinstance(summary, list):
            for block in summary:
                if isinstance(block, str):
                    summaries.append(block)
                elif isinstance(block, dict):
                    text_value = block.get("text")
                    if isinstance(text_value, str):
                        summaries.append(text_value)
    return "\n".join(summaries).strip()


def _is_gpt5_family(model: str) -> bool:
    return model.strip().lower().startswith("gpt-5")


def _is_gpt5_1_or_2(model: str) -> bool:
    lowered = model.strip().lower()
    return lowered.startswith("gpt-5.1") or lowered.startswith("gpt-5.2")


def _strip_sampling_params_if_needed(model: str, options: Dict, reasoning_effort: Optional[str]) -> Tuple[Dict, bool]:
    if not _is_gpt5_family(model):
        return options, False
    effort = (reasoning_effort or "").strip().lower()
    must_strip = False
    if not _is_gpt5_1_or_2(model):
        must_strip = True
    elif effort and effort != "none":
        must_strip = True
    if not must_strip:
        return options, False
    cleaned = dict(options)
    removed = False
    for key in ("temperature", "top_p", "logprobs"):
        if key in cleaned:
            cleaned.pop(key, None)
            removed = True
    return cleaned, removed


def openai_chat_with_tokens(
    session: requests.Session,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    options: Optional[Dict] = None,
    timeout: float = 600.0,
    api_mode: str = DEFAULT_API_MODE,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
) -> Tuple[str, Optional[int], Optional[int], str, Dict]:
    base = api_base.rstrip("/")
    headers = _build_headers(api_key, organization, project)
    payload = dict(options or {})
    payload["model"] = model
    payload["stream"] = False

    if reasoning_effort:
        if api_mode == "responses":
            reasoning_payload: Dict[str, object] = {"effort": reasoning_effort}
            if reasoning_summary:
                reasoning_payload["summary"] = reasoning_summary
            payload["reasoning"] = reasoning_payload
        else:
            payload["reasoning_effort"] = reasoning_effort

    if api_mode == "chat":
        url = base + "/chat/completions"
        payload.pop("input", None)
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        if "max_output_tokens" in payload and "max_tokens" not in payload:
            payload["max_tokens"] = payload["max_output_tokens"]
        payload.pop("max_output_tokens", None)
    else:
        url = base + "/responses"
        payload.pop("messages", None)
        payload["input"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        if "max_tokens" in payload and "max_output_tokens" not in payload:
            payload["max_output_tokens"] = payload["max_tokens"]
        payload.pop("max_tokens", None)

    data = _post_json(session, url, headers, payload, timeout)
    content = _extract_openai_text(data)
    output_tokens, prompt_tokens = _extract_usage_tokens(data)
    summary_text = _extract_reasoning_summary(data) if api_mode == "responses" else ""
    return content, output_tokens, prompt_tokens, summary_text, data


def request_with_retries(
    session: requests.Session,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    options: Dict,
    timeout: float,
    retries: int,
    delay: float,
    api_mode: str,
    organization: Optional[str],
    project: Optional[str],
    reasoning_effort: Optional[str],
    reasoning_summary: Optional[str],
) -> Tuple[Optional[str], Optional[int], Optional[int], str, Optional[str], float, int, Optional[str]]:
    attempt = 0
    last_error: Optional[str] = None
    while True:
        attempt += 1
        start = time.perf_counter()
        try:
            content, output_tokens, prompt_tokens, summary_text, data = openai_chat_with_tokens(
                session=session,
                api_base=api_base,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                options=options,
                timeout=timeout,
                api_mode=api_mode,
                organization=organization,
                project=project,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            )
            wall_time = time.perf_counter() - start
            response_id = None
            if isinstance(data, dict):
                response_id = data.get("id")
                if not isinstance(response_id, str):
                    response_id = None
            return content, output_tokens, prompt_tokens, summary_text, response_id, wall_time, attempt, None
        except Exception as exc:
            wall_time = time.perf_counter() - start
            last_error = str(exc)
            if attempt > retries:
                return None, None, None, "", None, wall_time, attempt, last_error
            time.sleep(delay * attempt)


def _write_jsonl_line(handle, record: Dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _translate_task(
    task: TaskSpec,
    src_lines: List[str],
    api_base: str,
    api_key: str,
    options: Dict,
    timeout: float,
    retries: int,
    delay: float,
    keep_multiline: bool,
    max_output_tokens: int,
    api_mode: str,
    organization: Optional[str],
    project: Optional[str],
    reasoning_effort: Optional[str],
    reasoning_summary: Optional[str],
    progress: Optional[ProgressTracker],
) -> TaskResult:
    if progress is not None:
        progress.task_started(task.task_id, task.target_lang, task.model, len(src_lines))

    session = requests.Session()
    out_lines: List[str] = ["" for _ in src_lines]
    blank_lines = 0
    over_limit_lines = 0
    error_lines = 0
    token_estimate_lines = 0
    output_tokens_total = 0
    output_tokens_est_total = 0
    prompt_tokens_total = 0
    attempts_total = 0
    request_count = 0
    wall_time_s_total = 0.0
    wall_time_s_max = 0.0

    start = time.perf_counter()
    task.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    task.thinking_path.parent.mkdir(parents=True, exist_ok=True)

    with task.metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_f, task.thinking_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as thinking_f:
        for idx, line in enumerate(src_lines):
            record: Dict[str, object] = {
                "line_index": idx,
                "source_text": line,
                "model": task.model,
                "target_lang": task.target_lang,
                "prompt_label": task.prompt.label,
                "prompt_id": task.prompt.prompt_id,
                "prompt_text": task.prompt.text,
                "max_output_tokens": max_output_tokens,
                "reasoning_effort": reasoning_effort,
                "reasoning_summary": reasoning_summary,
            }

            if line.strip() == "":
                out_lines[idx] = ""
                blank_lines += 1
                record["blank"] = True
                _write_jsonl_line(metrics_f, record)
                _write_jsonl_line(
                    thinking_f,
                    {
                        "line_index": idx,
                        "model": task.model,
                        "target_lang": task.target_lang,
                        "reasoning_summary": "",
                    },
                )
                if progress is not None:
                    progress.advance(task.task_id)
                continue

            raw, output_tokens, prompt_tokens, summary_text, response_id, wall_time, attempts, error = request_with_retries(
                session=session,
                api_base=api_base,
                api_key=api_key,
                model=task.model,
                system_prompt=task.prompt.text,
                user_text=line,
                options=options,
                timeout=timeout,
                retries=retries,
                delay=delay,
                api_mode=api_mode,
                organization=organization,
                project=project,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
            )
            attempts_total += attempts
            request_count += 1
            wall_time_s_total += wall_time
            wall_time_s_max = max(wall_time_s_max, wall_time)

            record["response_id"] = response_id
            record["wall_time_s"] = wall_time
            record["attempts"] = attempts
            record["error"] = error

            if raw is None:
                out_lines[idx] = FAILED_SENTINEL
                error_lines += 1
                record["status"] = "error"
                _write_jsonl_line(metrics_f, record)
                _write_jsonl_line(
                    thinking_f,
                    {
                        "line_index": idx,
                        "model": task.model,
                        "target_lang": task.target_lang,
                        "reasoning_summary": "",
                    },
                )
                if progress is not None:
                    progress.advance(task.task_id)
                continue

            token_count = output_tokens
            if token_count is None:
                token_count = estimate_tokens(raw)
                token_estimate_lines += 1
                output_tokens_est_total += token_count
            else:
                output_tokens_total += token_count

            prompt_tokens_total += prompt_tokens or 0

            over_limit = max_output_tokens > 0 and token_count > max_output_tokens
            if over_limit:
                out_lines[idx] = FAILED_SENTINEL
                over_limit_lines += 1
            else:
                out_lines[idx] = clean_translation(raw, keep_multiline)

            record.update(
                {
                    "status": "over_limit" if over_limit else "ok",
                    "raw_content": raw,
                    "cleaned_content": out_lines[idx],
                    "output_tokens": output_tokens,
                    "output_tokens_est": token_count if output_tokens is None else None,
                    "prompt_tokens": prompt_tokens,
                    "over_limit": over_limit,
                    "reasoning_summary_text": summary_text,
                }
            )
            _write_jsonl_line(metrics_f, record)
            _write_jsonl_line(
                thinking_f,
                {
                    "line_index": idx,
                    "model": task.model,
                    "target_lang": task.target_lang,
                    "reasoning_summary": summary_text,
                },
            )

            if progress is not None:
                progress.advance(task.task_id)

    write_lines(task.out_path, out_lines)

    elapsed = time.perf_counter() - start
    success = (over_limit_lines + error_lines) == 0
    if progress is not None:
        progress.task_completed(task.task_id, success=success)

    return TaskResult(
        model=task.model,
        target_lang=task.target_lang,
        prompt_label=task.prompt.label,
        prompt_id=task.prompt.prompt_id,
        out_path=task.out_path,
        metrics_path=task.metrics_path,
        thinking_path=task.thinking_path,
        elapsed=elapsed,
        total_lines=len(src_lines),
        blank_lines=blank_lines,
        over_limit_lines=over_limit_lines,
        error_lines=error_lines,
        token_estimate_lines=token_estimate_lines,
        output_tokens_total=output_tokens_total,
        output_tokens_est_total=output_tokens_est_total,
        prompt_tokens_total=prompt_tokens_total,
        attempts_total=attempts_total,
        request_count=request_count,
        wall_time_s_total=wall_time_s_total,
        wall_time_s_max=wall_time_s_max,
    )


def summarize_results(results: List[TaskResult]) -> Dict[str, Dict[str, Dict[str, float]]]:
    model_stats: Dict[str, Dict[str, float]] = {}
    lang_stats: Dict[str, Dict[str, float]] = {}

    for result in results:
        model_entry = model_stats.setdefault(
            result.model,
            {
                "elapsed": 0.0,
                "tasks": 0,
                "lines": 0,
                "blank_lines": 0,
                "failed_lines": 0,
                "over_limit_lines": 0,
                "error_lines": 0,
                "token_estimate_lines": 0,
                "output_tokens_total": 0,
                "output_tokens_est_total": 0,
                "prompt_tokens_total": 0,
                "wall_time_s_total": 0.0,
            },
        )
        model_entry["elapsed"] += result.elapsed
        model_entry["tasks"] += 1
        model_entry["lines"] += result.total_lines
        model_entry["blank_lines"] += result.blank_lines
        model_entry["failed_lines"] += result.failed_lines
        model_entry["over_limit_lines"] += result.over_limit_lines
        model_entry["error_lines"] += result.error_lines
        model_entry["token_estimate_lines"] += result.token_estimate_lines
        model_entry["output_tokens_total"] += result.output_tokens_total
        model_entry["output_tokens_est_total"] += result.output_tokens_est_total
        model_entry["prompt_tokens_total"] += result.prompt_tokens_total
        model_entry["wall_time_s_total"] += result.wall_time_s_total

        lang_entry = lang_stats.setdefault(
            result.target_lang,
            {
                "elapsed": 0.0,
                "tasks": 0,
                "lines": 0,
                "blank_lines": 0,
                "failed_lines": 0,
                "over_limit_lines": 0,
                "error_lines": 0,
                "token_estimate_lines": 0,
                "output_tokens_total": 0,
                "output_tokens_est_total": 0,
                "prompt_tokens_total": 0,
                "wall_time_s_total": 0.0,
            },
        )
        lang_entry["elapsed"] += result.elapsed
        lang_entry["tasks"] += 1
        lang_entry["lines"] += result.total_lines
        lang_entry["blank_lines"] += result.blank_lines
        lang_entry["failed_lines"] += result.failed_lines
        lang_entry["over_limit_lines"] += result.over_limit_lines
        lang_entry["error_lines"] += result.error_lines
        lang_entry["token_estimate_lines"] += result.token_estimate_lines
        lang_entry["output_tokens_total"] += result.output_tokens_total
        lang_entry["output_tokens_est_total"] += result.output_tokens_est_total
        lang_entry["prompt_tokens_total"] += result.prompt_tokens_total
        lang_entry["wall_time_s_total"] += result.wall_time_s_total

    return {"model": model_stats, "lang": lang_stats}


def write_stats(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Translate one source file into multiple target languages via OpenAI API")
    ap.add_argument("--source-file", type=Path, required=True, help="Source file (one line per entry)")
    ap.add_argument("--source-lang", type=str, default="en", help="Source language code for metadata")
    ap.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Target pairs: <tgt> or <src->tgt> (uses --prompt-template); optional override <src->tgt>=<prompt_path>",
    )
    ap.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_TEMPLATE_PROMPT_PATH,
        help="Prompt template with {SOURCE_LANG} and {TARGET_LANG} placeholders",
    )
    ap.add_argument(
        "--model-prompt",
        action="append",
        default=[],
        help="Override prompt for a model: model[[@lang]]=<prompt_path>",
    )
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="Comma-separated list of model names")
    ap.add_argument("--out-root", type=Path, required=True, help="Root output directory")
    ap.add_argument("--metrics-dir", type=Path, default=None, help="Override metrics directory")
    ap.add_argument("--thinking-dir", type=Path, default=None, help="Override reasoning summary directory")
    ap.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    ap.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="OpenAI API base URL")
    ap.add_argument(
        "--api-mode",
        type=str,
        choices=("responses", "chat"),
        default=DEFAULT_API_MODE,
        help="OpenAI endpoint mode: responses or chat",
    )
    ap.add_argument("--organization", type=str, default=None, help="OpenAI organization ID")
    ap.add_argument("--project", type=str, default=None, help="OpenAI project ID")
    ap.add_argument(
        "--options-json",
        type=str,
        default='{}',
        help="JSON dict of OpenAI options for generation",
    )
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature override passed to OpenAI options")
    ap.add_argument(
        "--reasoning-effort",
        type=str,
        default=DEFAULT_REASONING_EFFORT,
        help="Reasoning effort for GPT-5 models (none, minimal, low, medium, high, xhigh)",
    )
    ap.add_argument(
        "--reasoning-summary",
        type=str,
        default=None,
        help="Include reasoning summaries (auto, concise, detailed) for Responses API",
    )
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Hard cap for output tokens; lines exceeding this become [FAILED]",
    )
    ap.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds")
    ap.add_argument("--concurrency", type=int, default=1, help="Task-level concurrency across (lang,model) pairs")
    ap.add_argument("--retries", type=int, default=3, help="Retries per line on network or API error")
    ap.add_argument("--retry-delay", type=float, default=1.0, help="Base delay seconds for exponential backoff")
    ap.add_argument("--keep-multiline", action="store_true", help="Do not collapse assistant output to the first line")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--stats-path", type=Path, default=None, help="Write JSON stats to this path")
    ap.add_argument("--no-stats", action="store_true", help="Disable stats file output")

    args = ap.parse_args(argv)

    if not args.pair:
        print("At least one --pair must be provided.", file=sys.stderr)
        return 2

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing OpenAI API key. Provide --api-key or set OPENAI_API_KEY.", file=sys.stderr)
        return 2

    if not args.source_file.is_file():
        print(f"Source file not found: {args.source_file}", file=sys.stderr)
        return 2

    source_lines = read_lines(args.source_file)
    if not source_lines:
        print(f"Source file is empty: {args.source_file}", file=sys.stderr)
        return 2

    try:
        options = build_options(args.options_json, args.temperature, args.max_output_tokens)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.api_mode == "chat" and args.reasoning_summary:
        print("Warning: --reasoning-summary is only supported for Responses API; ignoring.", file=sys.stderr)
        args.reasoning_summary = None

    try:
        template_text = load_prompt_template(args.prompt_template)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    template_label = args.prompt_template.stem
    template_prompt_id = prompt_id_from_label_or_default(template_label, DEFAULT_TEMPLATE_PROMPT_ID)

    prompt_cache: Dict[Path, PromptFile] = {}
    pair_specs: List[PairSpec] = []
    try:
        for pair_arg in args.pair:
            pair_specs.append(
                parse_pair_arg(
                    pair_arg,
                    args.source_lang,
                    prompt_cache,
                    template_text,
                    template_label,
                    template_prompt_id,
                    args.prompt_template,
                )
            )
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    source_lang = pair_specs[0].source_lang
    for spec in pair_specs[1:]:
        if spec.source_lang != source_lang:
            print("All language pairs must share the same source language.", file=sys.stderr)
            return 2

    seen_targets: Dict[str, str] = {}
    for spec in pair_specs:
        if spec.target_lang in seen_targets:
            print(
                f"Duplicate target language '{spec.target_lang}' in --pair. "
                f"Already set by {seen_targets[spec.target_lang]}",
                file=sys.stderr,
            )
            return 2
        seen_targets[spec.target_lang] = str(spec.prompt.path)

    override_prompts: Dict[str, Dict[str, PromptFile]] = {}
    try:
        for override_arg in args.model_prompt:
            model_name, lang_key, prompt = parse_model_prompt_override(override_arg, prompt_cache)
            override_prompts.setdefault(model_name, {})[lang_key] = prompt
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    models = parse_models(args.models)
    if not models:
        print("No models provided.", file=sys.stderr)
        return 2

    options_by_model: Dict[str, Dict] = {}
    for model in models:
        model_options, stripped = _strip_sampling_params_if_needed(model, options, args.reasoning_effort)
        options_by_model[model] = model_options
        if stripped:
            print(
                f"Note: removed temperature/top_p/logprobs for {model} with reasoning_effort='{args.reasoning_effort}'.",
                file=sys.stderr,
            )

    line_count = len(source_lines)
    metrics_dir = args.metrics_dir or (args.out_root / "metrics")
    thinking_dir = args.thinking_dir or (args.out_root / "thinking")

    tasks: List[TaskSpec] = []
    tasks_by_model: Dict[str, List[TaskSpec]] = {model: [] for model in models}
    skipped: List[Tuple[str, str, Path]] = []
    task_id = 0
    total_line_units = 0

    for model in models:
        overrides = override_prompts.get(model, {})
        for pair in pair_specs:
            output_lang = sanitize_output_lang(pair.target_lang)
            prompt = overrides.get(output_lang) or overrides.get("*") or pair.prompt
            base_name = build_output_base(line_count, output_lang, model, prompt.prompt_id)
            out_path = args.out_root / f"{base_name}.txt"
            metrics_path = metrics_dir / f"{base_name}.jsonl"
            thinking_path = thinking_dir / f"{base_name}_thinking.jsonl"
            if out_path.exists() and not args.overwrite:
                skipped.append((pair.target_lang, model, out_path))
                continue
            task = TaskSpec(
                task_id=task_id,
                model=model,
                source_lang=pair.source_lang,
                target_lang=output_lang,
                prompt=prompt,
                out_path=out_path,
                metrics_path=metrics_path,
                thinking_path=thinking_path,
            )
            tasks.append(task)
            tasks_by_model[model].append(task)
            total_line_units += line_count
            task_id += 1

    total_possible = len(pair_specs) * len(models)
    skipped_count = total_possible - len(tasks)
    print(f"Found {len(pair_specs)} target(s) and {len(models)} model(s). Total tasks: {len(tasks)}")
    if skipped_count:
        print(f"Skipped {skipped_count} existing output(s).")
        for target_lang, model, out_path in skipped:
            print(f"Skipping existing output for target={target_lang} model={model}: {out_path}")

    if not tasks:
        print("No work to do.", file=sys.stderr)
        return 0

    if tasks:
        max_tasks_per_model = max((len(entries) for entries in tasks_by_model.values()), default=0)
        scheduled_tasks: List[TaskSpec] = []
        for idx in range(max_tasks_per_model):
            for model in models:
                entries = tasks_by_model[model]
                if idx < len(entries):
                    scheduled_tasks.append(entries[idx])
        tasks = scheduled_tasks

    tracker = ProgressTracker(
        total_tasks=len(tasks),
        total_lines=total_line_units,
        enabled=sys.stderr.isatty(),
    )

    def _run(task: TaskSpec) -> TaskResult:
        task_options = options_by_model.get(task.model, options)
        return _translate_task(
            task=task,
            src_lines=source_lines,
            api_base=args.api_base,
            api_key=api_key,
            options=task_options,
            timeout=args.timeout,
            retries=args.retries,
            delay=args.retry_delay,
            keep_multiline=args.keep_multiline,
            max_output_tokens=args.max_output_tokens,
            api_mode=args.api_mode,
            organization=args.organization,
            project=args.project,
            reasoning_effort=args.reasoning_effort,
            reasoning_summary=args.reasoning_summary,
            progress=tracker,
        )

    results: List[TaskResult] = []
    start = time.perf_counter()
    try:
        if args.concurrency > 1:
            import concurrent.futures as futures

            with futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                for result in ex.map(_run, tasks):
                    tracker.log(f"Wrote {result.out_path}")
                    results.append(result)
        else:
            for task in tasks:
                result = _run(task)
                tracker.log(f"Wrote {result.out_path}")
                results.append(result)
    finally:
        tracker.close()

    elapsed = time.perf_counter() - start
    completed_tasks = tracker.completed_tasks
    summary_lines = f"{tracker.completed_lines}/{total_line_units}" if total_line_units else "0/0"
    failed_lines = sum(r.failed_lines for r in results)
    over_limit_lines = sum(r.over_limit_lines for r in results)
    error_lines = sum(r.error_lines for r in results)
    token_estimate_lines = sum(r.token_estimate_lines for r in results)

    print(
        f"Done. Completed {completed_tasks}/{len(tasks)} task(s) "
        f"({summary_lines} lines) in {elapsed:.1f}s."
    )
    if failed_lines:
        print(
            f"Failures: {failed_lines} line(s) ([FAILED]) "
            f"{over_limit_lines} over token cap, {error_lines} request errors."
        )
    if token_estimate_lines:
        print(f"Token count was estimated for {token_estimate_lines} line(s).")

    summaries = summarize_results(results)
    if summaries["model"]:
        print("Model stats (sum of task durations):")
        for model in sorted(summaries["model"]):
            entry = summaries["model"][model]
            elapsed_model = entry["elapsed"]
            lines = entry["lines"]
            rate = lines / elapsed_model if elapsed_model else 0.0
            print(
                f"  {model}: {elapsed_model:.1f}s, {entry['tasks']} target(s), "
                f"{lines} lines, {entry['failed_lines']} failed, {rate:.2f} lines/s"
            )
    if summaries["lang"]:
        print("Target language stats (sum of task durations):")
        for lang in sorted(summaries["lang"]):
            entry = summaries["lang"][lang]
            elapsed_lang = entry["elapsed"]
            lines = entry["lines"]
            rate = lines / elapsed_lang if elapsed_lang else 0.0
            print(
                f"  {lang}: {elapsed_lang:.1f}s, {entry['tasks']} model(s), "
                f"{lines} lines, {entry['failed_lines']} failed, {rate:.2f} lines/s"
            )

    if not args.no_stats:
        stats_path = args.stats_path or (args.out_root / "stats.json")
        payload = {
            "source_file": str(args.source_file),
            "source_lang": source_lang,
            "pairs": [
                {
                    "target_lang": pair.target_lang,
                    "prompt_path": str(pair.prompt.path),
                    "prompt_label": pair.prompt.label,
                    "prompt_id": pair.prompt.prompt_id,
                }
                for pair in pair_specs
            ],
            "model_prompt_overrides": {
                model: {
                    lang_key: {
                        "prompt_path": str(prompt.path),
                        "prompt_label": prompt.label,
                        "prompt_id": prompt.prompt_id,
                    }
                    for lang_key, prompt in overrides.items()
                }
                for model, overrides in override_prompts.items()
            },
            "models": models,
            "max_output_tokens": args.max_output_tokens,
            "options": options,
            "model_options": options_by_model,
            "api_base": args.api_base,
            "api_mode": args.api_mode,
            "reasoning_effort": args.reasoning_effort,
            "reasoning_summary": args.reasoning_summary,
            "total_tasks": len(tasks),
            "skipped_tasks": len(skipped),
            "total_lines": total_line_units,
            "completed_lines": tracker.completed_lines,
            "elapsed_seconds": elapsed,
            "failed_lines": failed_lines,
            "over_limit_lines": over_limit_lines,
            "error_lines": error_lines,
            "token_estimate_lines": token_estimate_lines,
            "model_stats": summaries["model"],
            "lang_stats": summaries["lang"],
            "tasks": [
                {
                    "model": r.model,
                    "target_lang": r.target_lang,
                    "prompt_label": r.prompt_label,
                    "prompt_id": r.prompt_id,
                    "out_path": str(r.out_path),
                    "metrics_path": str(r.metrics_path),
                    "thinking_path": str(r.thinking_path),
                    "elapsed": r.elapsed,
                    "total_lines": r.total_lines,
                    "blank_lines": r.blank_lines,
                    "failed_lines": r.failed_lines,
                    "over_limit_lines": r.over_limit_lines,
                    "error_lines": r.error_lines,
                    "token_estimate_lines": r.token_estimate_lines,
                    "output_tokens_total": r.output_tokens_total,
                    "output_tokens_est_total": r.output_tokens_est_total,
                    "prompt_tokens_total": r.prompt_tokens_total,
                    "attempts_total": r.attempts_total,
                    "request_count": r.request_count,
                    "wall_time_s_total": r.wall_time_s_total,
                    "wall_time_s_max": r.wall_time_s_max,
                }
                for r in results
            ],
        }
        write_stats(stats_path, payload)
        print(f"Wrote stats to {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
