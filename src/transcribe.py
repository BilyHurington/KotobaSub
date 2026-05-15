"""Kotoba-Whisper transcription helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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


def build_alignment_text(segments: list[Segment]) -> str:
    """Build plain transcript text for forced alignment."""

    text = " ".join(str(segment["text"]) for segment in segments if segment.get("text"))
    return normalize_for_alignment(text)


def normalize_for_alignment(text: str) -> str:
    """Keep transcript close to the spoken content while normalizing whitespace."""

    return " ".join(text.replace("　", " ").split())
