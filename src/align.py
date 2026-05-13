"""Qwen3 forced alignment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


AlignedUnit = dict[str, float | str]


def load_qwen_aligner(model_id: str, device: str = "cuda") -> Any:
    """Load the official Qwen3 forced aligner wrapper."""

    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    return Qwen3ForcedAligner(model_path=model_id, device=device)


def run_qwen_alignment(
    audio_path: str | Path,
    transcript_text: str,
    aligner: Any,
) -> list[AlignedUnit]:
    """Run Qwen forced alignment and normalize its output."""

    result = aligner(audio_path=str(audio_path), text=transcript_text)
    return normalize_qwen_alignment_result(result)


def normalize_qwen_alignment_result(result: Any) -> list[AlignedUnit]:
    """Normalize aligner output to start/end/text dictionaries."""

    if result is None:
        raise ValueError("Qwen aligner returned None")

    if isinstance(result, list):
        normalized = _normalize_list_result(result)
        if normalized:
            return normalized

    if isinstance(result, dict):
        direct = _normalize_list_result([result])
        if direct:
            return direct

        for key in ("segments", "words", "chars", "tokens", "alignment", "result"):
            if key not in result:
                continue
            try:
                return normalize_qwen_alignment_result(result[key])
            except ValueError:
                continue

    raise ValueError(f"Unsupported Qwen alignment result format: {type(result)!r}")


def _normalize_list_result(items: list[Any]) -> list[AlignedUnit]:
    normalized: list[AlignedUnit] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        start = _first_present(item, ("start", "start_time", "begin", "begin_time"))
        end = _first_present(item, ("end", "end_time", "finish", "finish_time"))
        text = _first_present(item, ("text", "word", "char", "token"))

        if start is None or end is None or text is None:
            continue

        text = str(text).strip()
        if not text:
            continue

        normalized.append({"start": float(start), "end": float(end), "text": text})

    return normalized


def _first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None

