"""Subtitle segmentation and SRT writing helpers."""

from __future__ import annotations

from pathlib import Path

from .config import SUBTITLE_CONFIG, SubtitleConfig


SubtitleSegment = dict[str, float | str]

JP_BREAK_CHARS = "。！？!?、"


def format_srt_time(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""

    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))

    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_subtitles_from_units(
    units: list[SubtitleSegment],
    config: SubtitleConfig = SUBTITLE_CONFIG,
) -> list[SubtitleSegment]:
    """Aggregate aligned units into readable subtitle segments."""

    subtitles: list[SubtitleSegment] = []
    current: list[SubtitleSegment] = []

    for unit in units:
        if not _valid_unit(unit):
            continue

        current.append(unit)
        text = _join_unit_text(current)
        start = float(current[0]["start"])
        end = float(current[-1]["end"])
        duration = end - start

        should_flush = False
        if text[-1:] in JP_BREAK_CHARS and duration >= config.min_subtitle_duration:
            should_flush = True
        if len(text) >= config.max_subtitle_chars:
            should_flush = True
        if duration >= config.max_subtitle_duration:
            should_flush = True

        if should_flush:
            subtitles.append(_make_subtitle(current, config))
            current = []

    if current:
        subtitles.append(_make_subtitle(current, config))

    return merge_tiny_subtitles(subtitles, config)


def merge_tiny_subtitles(
    subtitles: list[SubtitleSegment],
    config: SubtitleConfig = SUBTITLE_CONFIG,
) -> list[SubtitleSegment]:
    """Merge very short subtitle segments into their predecessor when possible."""

    merged: list[SubtitleSegment] = []

    for subtitle in subtitles:
        duration = float(subtitle["end"]) - float(subtitle["start"])
        text = str(subtitle["text"])

        if (
            merged
            and duration < config.min_subtitle_duration
            and len(str(merged[-1]["text"]) + text) <= config.max_subtitle_chars
        ):
            merged[-1]["end"] = subtitle["end"]
            merged[-1]["text"] = str(merged[-1]["text"]) + text
            merged[-1]["text"] = wrap_japanese_lines(
                str(merged[-1]["text"]).replace("\n", ""),
                config.max_line_chars,
            )
            continue

        merged.append(dict(subtitle))

    return merged


def write_srt(segments: list[SubtitleSegment], output_path: str | Path) -> Path:
    """Write subtitle segments to an SRT file."""

    lines: list[str] = []
    index = 1

    for segment in segments:
        if not _valid_unit(segment):
            continue

        text = str(segment["text"]).strip()
        if not text:
            continue

        lines.extend(
            [
                str(index),
                f"{format_srt_time(float(segment['start']))} --> {format_srt_time(float(segment['end']))}",
                text,
                "",
            ]
        )
        index += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def wrap_japanese_lines(text: str, max_line_chars: int = 18) -> str:
    """Wrap Japanese subtitle text into at most two readable lines."""

    text = text.strip()
    if len(text) <= max_line_chars:
        return text

    if len(text) <= max_line_chars * 2:
        split_at = _best_split_index(text, max_line_chars)
        return f"{text[:split_at]}\n{text[split_at:]}"

    return f"{text[:max_line_chars]}\n{text[max_line_chars:max_line_chars * 2]}"


def _make_subtitle(
    units: list[SubtitleSegment],
    config: SubtitleConfig,
) -> SubtitleSegment:
    text = _join_unit_text(units)
    return {
        "start": float(units[0]["start"]),
        "end": float(units[-1]["end"]),
        "text": wrap_japanese_lines(text, config.max_line_chars),
    }


def _join_unit_text(units: list[SubtitleSegment]) -> str:
    return "".join(str(unit["text"]).strip() for unit in units if unit.get("text"))


def _valid_unit(unit: SubtitleSegment) -> bool:
    try:
        return bool(str(unit["text"]).strip()) and float(unit["end"]) > float(unit["start"])
    except (KeyError, TypeError, ValueError):
        return False


def _best_split_index(text: str, target: int) -> int:
    candidates = [idx + 1 for idx, char in enumerate(text) if char in JP_BREAK_CHARS]
    if candidates:
        return min(candidates, key=lambda idx: abs(idx - target))
    return min(len(text), target)

