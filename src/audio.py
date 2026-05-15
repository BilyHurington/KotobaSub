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


def get_media_duration(path: str | Path) -> float:
    """Return media duration in seconds using ffprobe."""

    import json

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])


def slice_audio_16k_mono(
    input_path: str | Path,
    output_path: str | Path,
    start: float,
    end: float,
) -> Path:
    """Slice a 16 kHz mono WAV segment with ffmpeg."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = max(0.0, float(start))
    end = max(start, float(end))
    duration = end - start

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path
