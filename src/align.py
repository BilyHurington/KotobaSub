"""Qwen3 forced alignment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


AlignedUnit = dict[str, float | str]


def load_qwen_aligner(model_id: str, device: str = "cuda") -> Any:
    """Load the official Qwen3 forced aligner wrapper."""

    import torch
    from qwen_asr import Qwen3ForcedAligner

    dtype = torch.bfloat16
    if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        dtype = torch.float16

    device_map = "cuda:0" if device == "cuda" else device
    return Qwen3ForcedAligner.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
    )


def run_qwen_alignment(
    audio_path: str | Path,
    transcript_text: str,
    aligner: Any,
) -> list[AlignedUnit]:
    """Run Qwen forced alignment and normalize its output."""

    result = aligner.align(
        audio=str(audio_path),
        text=transcript_text,
        language="Japanese",
    )
    return normalize_qwen_alignment_result(result)


def normalize_qwen_alignment_result(result: Any) -> list[AlignedUnit]:
    """Normalize aligner output to start/end/text dictionaries."""

    if result is None:
        raise ValueError("Qwen aligner returned None")

    if isinstance(result, (list, tuple)):
        flattened = _flatten_items(result)
        normalized = _normalize_list_result(flattened)
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


def _flatten_items(items: list[Any] | tuple[Any, ...]) -> list[Any]:
    flattened: list[Any] = []
    for item in items:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten_items(item))
        else:
            flattened.append(item)
    return flattened


def _first_present(item: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if isinstance(item, dict) and key in item and item[key] is not None:
            return item[key]
        if hasattr(item, key):
            value = getattr(item, key)
            if value is not None:
                return value
    return None
