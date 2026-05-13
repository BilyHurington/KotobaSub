"""Audio preprocessing helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_directories(paths: tuple[Path, ...] | list[Path]) -> None:
    """Create directories used by the workflow."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def audio_output_path(input_path: Path, audio_dir: Path) -> Path:
    """Return the canonical extracted WAV path for a media file."""

    return audio_dir / f"{input_path.stem}.16k.mono.wav"


def extract_audio_16k_mono(input_path: str | Path, output_path: str | Path) -> Path:
    """Extract a 16 kHz mono WAV file with ffmpeg."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path

