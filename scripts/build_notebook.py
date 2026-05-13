"""Generate the Colab notebook draft."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notebook_builder import build_notebook, code_cell, markdown_cell, write_notebook


OUTPUT_PATH = ROOT / "notebooks" / "Japanese_ASR_Kotoba_QwenAligner.ipynb"


def main() -> None:
    cells = [
        markdown_cell(
            """
            # Japanese SRT Transcription with Kotoba-Whisper and Qwen Forced Aligner

            This notebook transcribes Japanese video or audio into a Japanese `.srt` subtitle file.

            - ASR: `kotoba-tech/kotoba-whisper-v2.2-faster`
            - Forced alignment: `Qwen/Qwen3-ForcedAligner-0.6B`

            Run this notebook on a Colab GPU runtime.
            """
        ),
        code_cell(
            """
            !nvidia-smi
            !python --version
            """
        ),
        markdown_cell("## Install Dependencies"),
        code_cell(
            """
            !apt-get update -qq
            !apt-get install -y -qq ffmpeg git

            !pip install -q faster-whisper librosa soundfile accelerate qwen-omni-utils
            !pip install -q git+https://github.com/huggingface/transformers

            import os
            from pathlib import Path

            qwen_repo = Path("/content/Qwen3-ASR")
            if not qwen_repo.exists():
                !git clone https://github.com/QwenLM/Qwen3-ASR.git /content/Qwen3-ASR
            """
        ),
        markdown_cell("## Configure Workspace"),
        code_cell(
            """
            from pathlib import Path

            WORK_DIR = Path("/content/kotoba_qwen_subtitle")
            INPUT_DIR = WORK_DIR / "input"
            AUDIO_DIR = WORK_DIR / "audio"
            OUTPUT_DIR = WORK_DIR / "output"

            for path in [INPUT_DIR, AUDIO_DIR, OUTPUT_DIR]:
                path.mkdir(parents=True, exist_ok=True)

            LANGUAGE = "ja"
            USE_VAD = True
            BEAM_SIZE = 5
            WHISPER_COMPUTE_TYPES = ("float16", "int8_float16", "int8")

            KOTOBA_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.2-faster"
            QWEN_ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"

            FALLBACK_TO_WHISPER_TIMESTAMPS = True

            MAX_SUBTITLE_CHARS = 36
            MIN_SUBTITLE_DURATION = 1.0
            MAX_SUBTITLE_DURATION = 6.0
            MAX_LINE_CHARS = 18
            """
        ),
        markdown_cell("## Upload Input Media"),
        code_cell(
            """
            from google.colab import files
            import shutil

            uploaded = files.upload()

            input_paths = []
            for name in uploaded.keys():
                src = Path(name)
                dst = INPUT_DIR / src.name
                shutil.move(str(src), str(dst))
                input_paths.append(dst)

            if not input_paths:
                raise RuntimeError("No input file was uploaded.")

            input_path = input_paths[0]
            print(f"Input: {input_path}")
            """
        ),
        markdown_cell("## Extract Audio"),
        code_cell(
            """
            import subprocess

            audio_path = AUDIO_DIR / f"{input_path.stem}.16k.mono.wav"

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
                str(audio_path),
            ]
            subprocess.run(cmd, check=True)
            print(f"Audio: {audio_path}")
            """
        ),
        markdown_cell("## Transcribe with Kotoba-Whisper"),
        code_cell(
            """
            from faster_whisper import WhisperModel


            def load_kotoba_model(model_id, compute_types, device="cuda"):
                last_error = None
                for compute_type in compute_types:
                    try:
                        model = WhisperModel(
                            model_id,
                            device=device,
                            compute_type=compute_type,
                        )
                        print(f"Loaded {model_id} with compute_type={compute_type}")
                        return model, compute_type
                    except Exception as exc:
                        print(f"Failed compute_type={compute_type}: {exc}")
                        last_error = exc
                raise RuntimeError(f"Failed to load {model_id}") from last_error


            whisper_model, whisper_compute_type = load_kotoba_model(
                KOTOBA_MODEL_ID,
                WHISPER_COMPUTE_TYPES,
            )
            """
        ),
        code_cell(
            """
            segments_iter, info = whisper_model.transcribe(
                str(audio_path),
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=USE_VAD,
                condition_on_previous_text=True,
            )

            whisper_segments = []
            for segment in segments_iter:
                text = segment.text.strip()
                if not text:
                    continue
                whisper_segments.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": text,
                    }
                )

            print(f"Detected language: {info.language} ({info.language_probability:.2f})")
            print(f"Whisper segments: {len(whisper_segments)}")
            whisper_segments[:3]
            """
        ),
        markdown_cell("## Prepare Alignment Text"),
        code_cell(
            """
            def normalize_for_alignment(text):
                return " ".join(text.replace("　", " ").split())


            alignment_text = normalize_for_alignment(
                " ".join(segment["text"] for segment in whisper_segments if segment["text"])
            )

            if not alignment_text:
                raise RuntimeError("Transcription produced no text.")

            print(alignment_text[:1000])
            """
        ),
        markdown_cell("## Align with Qwen3-ForcedAligner"),
        code_cell(
            """
            import sys

            sys.path.append("/content/Qwen3-ASR")

            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

            aligner = Qwen3ForcedAligner(
                model_path=QWEN_ALIGNER_MODEL_ID,
                device="cuda",
            )
            """
        ),
        code_cell(
            """
            def first_present(item, keys):
                for key in keys:
                    if key in item and item[key] is not None:
                        return item[key]
                return None


            def normalize_qwen_alignment_result(result):
                if result is None:
                    raise ValueError("Qwen aligner returned None")

                if isinstance(result, list):
                    normalized = []
                    for item in result:
                        if not isinstance(item, dict):
                            continue
                        start = first_present(item, ("start", "start_time", "begin", "begin_time"))
                        end = first_present(item, ("end", "end_time", "finish", "finish_time"))
                        text = first_present(item, ("text", "word", "char", "token"))

                        if start is None or end is None or text is None:
                            continue

                        text = str(text).strip()
                        if not text:
                            continue

                        normalized.append(
                            {"start": float(start), "end": float(end), "text": text}
                        )

                    if normalized:
                        return normalized

                if isinstance(result, dict):
                    direct = normalize_qwen_alignment_result([result]) if result else []
                    if direct:
                        return direct

                    for key in ("segments", "words", "chars", "tokens", "alignment", "result"):
                        if key not in result:
                            continue
                        try:
                            return normalize_qwen_alignment_result(result[key])
                        except ValueError:
                            pass

                raise ValueError(f"Unsupported Qwen alignment result format: {type(result)!r}")


            def run_qwen_alignment(audio_path, transcript_text, aligner):
                result = aligner(audio_path=str(audio_path), text=transcript_text)
                return normalize_qwen_alignment_result(result)
            """
        ),
        code_cell(
            """
            try:
                aligned_units = run_qwen_alignment(
                    audio_path=audio_path,
                    transcript_text=alignment_text,
                    aligner=aligner,
                )
                used_alignment = "qwen"
            except Exception as exc:
                if not FALLBACK_TO_WHISPER_TIMESTAMPS:
                    raise
                print("Qwen alignment failed. Falling back to Whisper timestamps.")
                print(repr(exc))
                aligned_units = whisper_segments
                used_alignment = "whisper"

            print(f"Alignment source: {used_alignment}")
            print(f"Aligned units: {len(aligned_units)}")
            aligned_units[:5]
            """
        ),
        markdown_cell("## Build SRT Subtitles"),
        code_cell(
            """
            JP_BREAK_CHARS = "。！？!?、"


            def format_srt_time(seconds):
                seconds = max(0.0, float(seconds))
                millis = int(round(seconds * 1000))
                hours = millis // 3_600_000
                millis %= 3_600_000
                minutes = millis // 60_000
                millis %= 60_000
                secs = millis // 1000
                millis %= 1000
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


            def valid_unit(unit):
                try:
                    return bool(str(unit["text"]).strip()) and float(unit["end"]) > float(unit["start"])
                except (KeyError, TypeError, ValueError):
                    return False


            def best_split_index(text, target):
                candidates = [idx + 1 for idx, char in enumerate(text) if char in JP_BREAK_CHARS]
                if candidates:
                    return min(candidates, key=lambda idx: abs(idx - target))
                return min(len(text), target)


            def wrap_japanese_lines(text, max_line_chars=MAX_LINE_CHARS):
                text = text.strip()
                if len(text) <= max_line_chars:
                    return text
                if len(text) <= max_line_chars * 2:
                    split_at = best_split_index(text, max_line_chars)
                    return f"{text[:split_at]}\\n{text[split_at:]}"
                return f"{text[:max_line_chars]}\\n{text[max_line_chars:max_line_chars * 2]}"


            def join_unit_text(units):
                return "".join(str(unit["text"]).strip() for unit in units if unit.get("text"))


            def make_subtitle(units):
                text = join_unit_text(units)
                return {
                    "start": float(units[0]["start"]),
                    "end": float(units[-1]["end"]),
                    "text": wrap_japanese_lines(text),
                }


            def merge_tiny_subtitles(subtitles):
                merged = []
                for subtitle in subtitles:
                    duration = float(subtitle["end"]) - float(subtitle["start"])
                    text = str(subtitle["text"])
                    if (
                        merged
                        and duration < MIN_SUBTITLE_DURATION
                        and len(str(merged[-1]["text"]) + text) <= MAX_SUBTITLE_CHARS
                    ):
                        merged[-1]["end"] = subtitle["end"]
                        merged[-1]["text"] = wrap_japanese_lines(
                            str(merged[-1]["text"]).replace("\\n", "") + text.replace("\\n", "")
                        )
                        continue
                    merged.append(dict(subtitle))
                return merged


            def build_subtitles_from_units(units):
                subtitles = []
                current = []

                for unit in units:
                    if not valid_unit(unit):
                        continue

                    current.append(unit)
                    text = join_unit_text(current)
                    start = float(current[0]["start"])
                    end = float(current[-1]["end"])
                    duration = end - start

                    should_flush = False
                    if text[-1:] in JP_BREAK_CHARS and duration >= MIN_SUBTITLE_DURATION:
                        should_flush = True
                    if len(text) >= MAX_SUBTITLE_CHARS:
                        should_flush = True
                    if duration >= MAX_SUBTITLE_DURATION:
                        should_flush = True

                    if should_flush:
                        subtitles.append(make_subtitle(current))
                        current = []

                if current:
                    subtitles.append(make_subtitle(current))

                return merge_tiny_subtitles(subtitles)


            def write_srt(segments, output_path):
                lines = []
                index = 1

                for segment in segments:
                    if not valid_unit(segment):
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
                output_path.write_text("\\n".join(lines), encoding="utf-8")
                return output_path
            """
        ),
        code_cell(
            """
            subtitle_segments = build_subtitles_from_units(aligned_units)

            srt_path = OUTPUT_DIR / f"{input_path.stem}.ja.srt"
            write_srt(subtitle_segments, srt_path)

            print(f"Subtitle segments: {len(subtitle_segments)}")
            print(f"SRT: {srt_path}")
            """
        ),
        markdown_cell("## Download SRT"),
        code_cell(
            """
            from google.colab import files

            files.download(str(srt_path))
            """
        ),
    ]

    write_notebook(build_notebook(cells), OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

