"""Qwen3 forced alignment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .audio import get_media_duration, slice_audio_16k_mono
from .transcribe import Segment, build_alignment_text


AlignedUnit = dict[str, float | str]


def load_qwen_aligner(model_id: str, device: str = "cuda") -> Any:
    """Load the official Qwen3 forced aligner wrapper."""

    import torch
    try:
        from qwen_asr import Qwen3ForcedAligner
    except KeyError as exc:
        if str(exc).strip("'\"") == "qwen_asr":
            raise RuntimeError(
                "Failed to import qwen_asr because this Colab runtime still has a stale "
                "Qwen3-ASR import path/cache. Restart the Colab runtime, then run the "
                "notebook from the first cell."
            ) from exc
        raise

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


def run_qwen_alignment_chunked(
    audio_path: str | Path,
    whisper_segments: list[Segment],
    aligner: Any,
    work_dir: str | Path,
    max_chunk_seconds: float = 30.0,
    max_chunk_chars: int = 300,
    chunk_padding: float = 1.0,
    fallback_to_whisper: bool = True,
) -> list[AlignedUnit]:
    """Align in segment-bound chunks to avoid Qwen OOM on long media."""

    audio_path = Path(audio_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = get_media_duration(audio_path)

    aligned: list[AlignedUnit] = []
    chunks = build_alignment_chunks(whisper_segments, max_chunk_seconds, max_chunk_chars)

    for index, chunk_segments in enumerate(chunks):
        chunk_start = float(chunk_segments[0]["start"])
        chunk_end = float(chunk_segments[-1]["end"])
        slice_start = max(0.0, chunk_start - chunk_padding)
        slice_end = min(duration, chunk_end + chunk_padding)
        chunk_text = build_alignment_text(chunk_segments)
        chunk_path = work_dir / f"{audio_path.stem}.align_{index:04d}.wav"

        print(
            f"Aligning chunk {index + 1}/{len(chunks)}: "
            f"{chunk_start:.1f}s - {chunk_end:.1f}s, {len(chunk_text)} chars"
        )
        slice_audio_16k_mono(audio_path, chunk_path, slice_start, slice_end)

        try:
            raw_result = aligner.align(
                audio=str(chunk_path),
                text=chunk_text,
                language="Japanese",
            )
            raw_items = extract_qwen_alignment_items(raw_result)
            if has_tail_collapse(raw_items, slice_end - slice_start, chunk_end - slice_start):
                raise ValueError("Qwen alignment collapsed unmatched tail tokens to chunk end")

            chunk_units = normalize_qwen_alignment_result(raw_result)
            if not chunk_units:
                raise ValueError("Qwen alignment produced no usable timestamp units")

            for unit in chunk_units:
                global_start = float(unit["start"]) + slice_start
                global_end = float(unit["end"]) + slice_start
                if global_end <= global_start:
                    continue
                if global_end < chunk_start or global_start > chunk_end:
                    continue

                aligned.append(
                    {
                        "start": max(chunk_start, global_start),
                        "end": min(chunk_end, global_end),
                        "text": str(unit["text"]).strip(),
                    }
                )
        except Exception as exc:
            if not fallback_to_whisper:
                raise
            print(f"Qwen alignment failed for chunk {index + 1}; using Whisper timestamps.")
            print(repr(exc))
            aligned.extend(dict(segment) for segment in chunk_segments)
        finally:
            _empty_cuda_cache()

    return aligned


def build_alignment_chunks(
    segments: list[Segment],
    max_chunk_seconds: float,
    max_chunk_chars: int,
) -> list[list[Segment]]:
    """Group Whisper segments without splitting inside a segment."""

    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    current_chars = 0

    for segment in segments:
        if not str(segment.get("text", "")).strip():
            continue

        segment_chars = len(str(segment["text"]))
        would_duration = (
            float(segment["end"]) - float(current[0]["start"])
            if current
            else float(segment["end"]) - float(segment["start"])
        )
        would_chars = current_chars + segment_chars

        if current and (would_duration > max_chunk_seconds or would_chars > max_chunk_chars):
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(segment)
        current_chars += segment_chars

    if current:
        chunks.append(current)

    return chunks


def normalize_qwen_alignment_result(result: Any) -> list[AlignedUnit]:
    """Normalize aligner output to start/end/text dictionaries."""

    if result is None:
        raise ValueError("Qwen aligner returned None")

    if isinstance(result, (list, tuple)):
        flattened = _flatten_items(result)
        normalized = _normalize_list_result(flattened)
        if normalized:
            return normalized

    if hasattr(result, "items"):
        return normalize_qwen_alignment_result(getattr(result, "items"))

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

        start = float(start)
        end = float(end)
        if end <= start:
            continue

        text = str(text).strip()
        if not text:
            continue

        normalized.append({"start": start, "end": end, "text": text})

    return normalized


def _flatten_items(items: list[Any] | tuple[Any, ...]) -> list[Any]:
    flattened: list[Any] = []
    for item in items:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten_items(item))
        elif hasattr(item, "items"):
            flattened.extend(_flatten_items(getattr(item, "items")))
        else:
            flattened.append(item)
    return flattened


def extract_qwen_alignment_items(result: Any) -> list[Any]:
    """Extract raw alignment items from Qwen result objects."""

    if result is None:
        return []
    if isinstance(result, dict):
        for key in ("items", "segments", "words", "chars", "tokens", "alignment", "result"):
            if key in result:
                return extract_qwen_alignment_items(result[key])
        return [result]
    if isinstance(result, (list, tuple)):
        items: list[Any] = []
        for item in result:
            items.extend(extract_qwen_alignment_items(item))
        return items
    if hasattr(result, "items"):
        return extract_qwen_alignment_items(getattr(result, "items"))
    return [result]


def has_tail_collapse(
    items: list[Any],
    slice_duration: float,
    chunk_end_local: float,
    min_tail_zero_tokens: int = 3,
    end_tolerance: float = 0.05,
) -> bool:
    """Detect Qwen outputs that pin unmatched tail tokens to the audio end."""

    tail_zero_tokens = 0
    for item in reversed(items):
        start = _first_present(item, ("start", "start_time", "begin", "begin_time"))
        end = _first_present(item, ("end", "end_time", "finish", "finish_time"))
        text = _first_present(item, ("text", "word", "char", "token"))

        if start is None or end is None or text is None:
            continue

        start = float(start)
        end = float(end)
        at_slice_end = abs(end - slice_duration) <= end_tolerance
        after_chunk_end = end >= chunk_end_local

        if end <= start and (at_slice_end or after_chunk_end):
            tail_zero_tokens += 1
            continue

        break

    return tail_zero_tokens >= min_tail_zero_tokens


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _first_present(item: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if isinstance(item, dict) and key in item and item[key] is not None:
            return item[key]
        if hasattr(item, key):
            value = getattr(item, key)
            if value is not None:
                return value
    return None
