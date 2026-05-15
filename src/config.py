"""Shared defaults for the Colab subtitle workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers and runtime defaults."""

    kotoba_model_id: str = "RoachLin/kotoba-whisper-v2.2-faster"
    kotoba_model_candidates: tuple[str, ...] = (
        "RoachLin/kotoba-whisper-v2.2-faster",
        "kotoba-tech/kotoba-whisper-v2.0-faster",
    )
    qwen_aligner_model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    language: str = "ja"
    beam_size: int = 5
    use_vad: bool = False
    whisper_compute_types: tuple[str, ...] = ("float16", "int8_float16", "int8")
    fallback_to_whisper_timestamps: bool = True
    use_chunked_transcription: bool = True
    transcribe_chunk_seconds: float = 30.0
    transcribe_chunk_overlap: float = 3.0
    max_align_chunk_seconds: float = 30.0
    max_align_chunk_chars: int = 300
    align_chunk_padding: float = 1.0


@dataclass(frozen=True)
class SubtitleConfig:
    """Readable subtitle segmentation defaults."""

    max_subtitle_chars: int = 36
    min_subtitle_duration: float = 1.0
    max_subtitle_duration: float = 6.0
    max_line_chars: int = 18


@dataclass(frozen=True)
class WorkspaceConfig:
    """Colab workspace paths."""

    root: Path = Path("/content/kotoba_qwen_subtitle")

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    def all_dirs(self) -> tuple[Path, Path, Path]:
        return (self.input_dir, self.audio_dir, self.output_dir)


MODEL_CONFIG = ModelConfig()
SUBTITLE_CONFIG = SubtitleConfig()
WORKSPACE_CONFIG = WorkspaceConfig()
