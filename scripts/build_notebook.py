"""Generate the Colab notebook draft."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notebook_builder import build_notebook, code_cell, markdown_cell, write_notebook


OUTPUT_PATH = ROOT / "notebooks" / "Japanese_ASR_Kotoba_QwenAligner.ipynb"
PROJECT_REPO_URL = "https://github.com/BilyHurington/KotobaSub.git"
PROJECT_REPO_DIR = Path("/content/KotobaSub")
QWEN_REPO_URL = "https://github.com/QwenLM/Qwen3-ASR.git"
QWEN_REPO_DIR = Path("/content/Qwen3-ASR")


def main() -> None:
    cells = [
        markdown_cell(
            """
            # Japanese SRT Transcription with Kotoba-Whisper and Qwen Forced Aligner

            This notebook transcribes Japanese video or audio into a Japanese `.srt` subtitle file.

            - ASR: `kotoba-tech/kotoba-whisper-v2.2-faster`
            - Forced alignment: `Qwen/Qwen3-ForcedAligner-0.6B`

            The notebook clones this repository automatically, so it can be opened from GitHub and run directly in Colab.
            """
        ),
        code_cell(
            """
            !nvidia-smi
            !python --version
            """
        ),
        markdown_cell("## Install System Tools"),
        code_cell(
            """
            !apt-get update -qq
            !apt-get install -y -qq ffmpeg git
            """
        ),
        markdown_cell("## Clone Project Repository"),
        code_cell(
            f"""
            from pathlib import Path
            import sys

            REPO_DIR = Path("/content/KotobaSub")
            if not REPO_DIR.exists():
                !git clone --depth 1 {PROJECT_REPO_URL} /content/KotobaSub

            if str(REPO_DIR) not in sys.path:
                sys.path.insert(0, str(REPO_DIR))

            print(f"Project repo: {{REPO_DIR}}")
            """
        ),
        markdown_cell("## Install Python Dependencies"),
        code_cell(
            """
            !pip install -q -r /content/KotobaSub/requirements-colab.txt
            """
        ),
        markdown_cell("## Configure Workspace and Import Helpers"),
        code_cell(
            """
            from src.config import MODEL_CONFIG, SUBTITLE_CONFIG, WORKSPACE_CONFIG
            from src.audio import ensure_directories, extract_audio_16k_mono, audio_output_path
            from src.transcribe import load_kotoba_model, transcribe_audio, build_alignment_text
            from src.align import load_qwen_aligner, run_qwen_alignment
            from src.subtitles import build_subtitles_from_units, write_srt

            WORK_DIR = WORKSPACE_CONFIG.root
            INPUT_DIR = WORKSPACE_CONFIG.input_dir
            AUDIO_DIR = WORKSPACE_CONFIG.audio_dir
            OUTPUT_DIR = WORKSPACE_CONFIG.output_dir

            ensure_directories(WORKSPACE_CONFIG.all_dirs())

            LANGUAGE = MODEL_CONFIG.language
            USE_VAD = MODEL_CONFIG.use_vad
            BEAM_SIZE = MODEL_CONFIG.beam_size
            WHISPER_COMPUTE_TYPES = MODEL_CONFIG.whisper_compute_types
            KOTOBA_MODEL_ID = MODEL_CONFIG.kotoba_model_id
            QWEN_ALIGNER_MODEL_ID = MODEL_CONFIG.qwen_aligner_model_id
            FALLBACK_TO_WHISPER_TIMESTAMPS = MODEL_CONFIG.fallback_to_whisper_timestamps
            """
        ),
        markdown_cell("## Select Input Media"),
        code_cell(
            """
            import shutil

            # Choose "upload" to upload from your computer, or "drive" to read from Google Drive.
            INPUT_SOURCE = "upload"

            # Used only when INPUT_SOURCE = "drive".
            # Example: "/content/drive/MyDrive/videos/sample.mp4"
            DRIVE_INPUT_PATH = "/content/drive/MyDrive/path/to/your_file.mp4"


            def copy_input_to_workspace(source_path):
                source_path = Path(source_path)
                if not source_path.exists():
                    raise FileNotFoundError(f"Input file does not exist: {source_path}")

                dst = INPUT_DIR / source_path.name
                if source_path.resolve() != dst.resolve():
                    shutil.copy2(source_path, dst)
                return dst


            if INPUT_SOURCE == "upload":
                from google.colab import files

                uploaded = files.upload()
                if not uploaded:
                    raise RuntimeError("No input file was uploaded.")

                first_name = next(iter(uploaded.keys()))
                input_path = copy_input_to_workspace(first_name)

            elif INPUT_SOURCE == "drive":
                from google.colab import drive

                drive.mount("/content/drive")
                input_path = copy_input_to_workspace(DRIVE_INPUT_PATH)

            else:
                raise ValueError('INPUT_SOURCE must be "upload" or "drive".')

            print(f"Input: {input_path}")
            """
        ),
        markdown_cell("## Extract Audio"),
        code_cell(
            """
            audio_path = audio_output_path(input_path, AUDIO_DIR)
            extract_audio_16k_mono(input_path, audio_path)
            print(f"Audio: {audio_path}")
            """
        ),
        markdown_cell("## Transcribe with Kotoba-Whisper"),
        code_cell(
            """
            whisper_model, whisper_compute_type = load_kotoba_model(
                KOTOBA_MODEL_ID,
                WHISPER_COMPUTE_TYPES,
            )

            whisper_segments, info = transcribe_audio(
                whisper_model,
                audio_path,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                use_vad=USE_VAD,
            )

            alignment_text = build_alignment_text(whisper_segments)
            print(f"Detected language: {info.language} ({info.language_probability:.2f})")
            print(f"Whisper compute type: {whisper_compute_type}")
            print(f"Whisper segments: {len(whisper_segments)}")
            print(alignment_text[:1000])
            """
        ),
        markdown_cell("## Clone Qwen3-ASR and Align"),
        code_cell(
            f"""
            from pathlib import Path
            import sys

            QWEN_REPO_DIR = Path("/content/Qwen3-ASR")
            if not QWEN_REPO_DIR.exists():
                !git clone --depth 1 {QWEN_REPO_URL} /content/Qwen3-ASR

            if str(QWEN_REPO_DIR) not in sys.path:
                sys.path.insert(0, str(QWEN_REPO_DIR))

            aligner = load_qwen_aligner(QWEN_ALIGNER_MODEL_ID)
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
            """
        ),
        markdown_cell("## Build SRT Subtitles"),
        code_cell(
            """
            subtitle_segments = build_subtitles_from_units(aligned_units, config=SUBTITLE_CONFIG)

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
