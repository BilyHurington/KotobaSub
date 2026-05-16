"""Kotoba-Whisper transcription helpers."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .audio import get_media_duration, slice_audio_16k_mono


Segment = dict[str, float | str]


def load_kotoba_model(
    model_id: str | tuple[str, ...],
    compute_types: tuple[str, ...] = ("float16", "int8_float16", "int8"),
    device: str = "cuda",
) -> tuple[Any, str, str]:
    """Load a faster-whisper model with model and compute type fallback."""

    from faster_whisper import WhisperModel

    model_ids = (model_id,) if isinstance(model_id, str) else model_id
    last_error: Exception | None = None
    for candidate_model_id in model_ids:
        for compute_type in compute_types:
            try:
                model = WhisperModel(candidate_model_id, device=device, compute_type=compute_type)
                return model, compute_type, candidate_model_id
            except Exception as exc:
                print(
                    f"Failed to load {candidate_model_id} "
                    f"with compute_type={compute_type}: {exc}"
                )
                last_error = exc

    raise RuntimeError(f"Failed to load any Kotoba-Whisper model: {model_ids}") from last_error


def transcribe_audio(
    model: Any,
    audio_path: str | Path,
    language: str = "ja",
    beam_size: int = 5,
    use_vad: bool = True,
) -> tuple[list[Segment], Any]:
    """Transcribe audio and normalize faster-whisper segments."""

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=use_vad,
        condition_on_previous_text=True,
    )

    segments: list[Segment] = []
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            }
        )

    return segments, info


def transcribe_audio_chunked(
    model: Any,
    audio_path: str | Path,
    work_dir: str | Path,
    language: str = "ja",
    beam_size: int = 5,
    use_vad: bool = False,
    chunk_seconds: float = 30.0,
    chunk_overlap: float = 10.0,
) -> tuple[list[Segment], Any]:
    """Transcribe long audio in fixed core windows with extra context.

    The model sees overlap before and after each core chunk, but only segments
    whose midpoint falls inside the core interval are kept. This gives Whisper
    enough context at chunk boundaries without turning the overlap into
    duplicated transcript output.
    """

    audio_path = Path(audio_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    duration = get_media_duration(audio_path)
    if duration <= chunk_seconds:
        return transcribe_audio(
            model,
            audio_path,
            language=language,
            beam_size=beam_size,
            use_vad=use_vad,
        )

    segments: list[Segment] = []
    info: Any = None
    step = chunk_seconds
    chunk_index = 0
    start = 0.0

    while start < duration:
        core_start = start
        core_end = min(duration, start + chunk_seconds)
        slice_start = max(0.0, core_start - chunk_overlap)
        slice_end = min(duration, core_end + chunk_overlap)
        chunk_path = work_dir / f"{audio_path.stem}.chunk_{chunk_index:04d}.wav"

        print(
            f"Transcribing chunk {chunk_index + 1}: "
            f"{core_start:.1f}s - {core_end:.1f}s core, "
            f"{slice_start:.1f}s - {slice_end:.1f}s context"
        )
        slice_audio_16k_mono(audio_path, chunk_path, slice_start, slice_end)

        chunk_segments, chunk_info = transcribe_audio(
            model,
            chunk_path,
            language=language,
            beam_size=beam_size,
            use_vad=use_vad,
        )
        if info is None:
            info = chunk_info

        for segment in chunk_segments:
            global_start = float(segment["start"]) + slice_start
            global_end = float(segment["end"]) + slice_start
            midpoint = (global_start + global_end) / 2.0
            is_last_chunk = core_end >= duration
            if midpoint < core_start or (midpoint >= core_end and not is_last_chunk):
                continue

            segments.append(
                {
                    "start": max(0.0, global_start),
                    "end": min(duration, global_end),
                    "text": str(segment["text"]).strip(),
                }
            )

        chunk_index += 1
        start += step

    return merge_near_duplicate_segments(segments), info


def merge_near_duplicate_segments(
    segments: list[Segment],
    max_start_delta: float = 4.0,
    min_text_similarity: float = 0.92,
) -> list[Segment]:
    """Remove obvious duplicates created by overlap windows.

    This is intentionally conservative: if two boundary segments are not clearly
    the same utterance, keep both. Repeated subtitles are easier to fix than
    missing speech.
    """

    merged: list[Segment] = []
    for segment in sorted(segments, key=lambda item: (float(item["start"]), float(item["end"]))):
        if not str(segment["text"]).strip():
            continue

        duplicate_index = _find_duplicate_segment(
            merged,
            segment,
            max_start_delta=max_start_delta,
            min_text_similarity=min_text_similarity,
        )
        if duplicate_index is not None:
            existing = merged[duplicate_index]
            existing["start"] = min(float(existing["start"]), float(segment["start"]))
            existing["end"] = max(float(existing["end"]), float(segment["end"]))
            continue

        merged.append(dict(segment))

    return merged


def _find_duplicate_segment(
    candidates: list[Segment],
    segment: Segment,
    max_start_delta: float,
    min_text_similarity: float,
) -> int | None:
    segment_text = str(segment["text"]).strip()
    segment_start = float(segment["start"])
    segment_end = float(segment["end"])

    for index in range(len(candidates) - 1, max(-1, len(candidates) - 8), -1):
        candidate = candidates[index]
        candidate_start = float(candidate["start"])
        candidate_end = float(candidate["end"])

        if abs(candidate_start - segment_start) > max_start_delta:
            continue

        overlap = min(candidate_end, segment_end) - max(candidate_start, segment_start)
        if overlap <= 0:
            continue

        candidate_text = str(candidate["text"]).strip()
        similarity = SequenceMatcher(None, candidate_text, segment_text).ratio()
        if similarity >= min_text_similarity:
            return index

    return None


def build_alignment_text(segments: list[Segment]) -> str:
    """Build plain transcript text for forced alignment."""

    text = " ".join(str(segment["text"]) for segment in segments if segment.get("text"))
    return normalize_for_alignment(text)


def normalize_for_alignment(text: str) -> str:
    """Keep transcript close to the spoken content while normalizing whitespace."""

    return " ".join(text.replace("　", " ").split())
