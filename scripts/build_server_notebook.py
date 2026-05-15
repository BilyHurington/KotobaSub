"""Generate a generic Jupyter server notebook."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notebook_builder import build_notebook, code_cell, markdown_cell, write_notebook


OUTPUT_PATH = ROOT / "notebooks" / "Japanese_ASR_Kotoba_QwenAligner_Server.ipynb"


def main() -> None:
    cells = [
        markdown_cell(
            """
            # Japanese SRT Transcription on a Generic Server

            This notebook transcribes Japanese video or audio into a Japanese `.srt` subtitle file on a normal Jupyter server.

            - ASR: `RoachLin/kotoba-whisper-v2.2-faster`
            - ASR fallback: `kotoba-tech/kotoba-whisper-v2.0-faster`
            - Optional forced alignment: `Qwen/Qwen3-ForcedAligner-0.6B`

            It does not use Colab APIs, Google Drive widgets, or browser downloads. Set local paths in the configuration cell.
            """
        ),
        markdown_cell(
            """
            ## PyTorch and System Requirements

            Install `ffmpeg` on the host first:

            ```bash
            sudo apt-get update
            sudo apt-get install -y ffmpeg
            ```

            Install PyTorch according to the server CUDA runtime:

            ```bash
            # CUDA 12.4
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

            # CUDA 12.1
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

            # CPU fallback
            pip install torch torchvision torchaudio
            ```

            Then install project dependencies:

            ```bash
            pip install -r requirements-server.txt
            ```

            A CUDA GPU is strongly recommended for Qwen forced alignment. CPU mode can still generate SRT from Whisper timestamps if alignment is disabled.
            """
        ),
        markdown_cell("## Check Runtime"),
        code_cell(
            """
            import sys
            import subprocess

            print(sys.version)
            try:
                subprocess.run(["nvidia-smi"], check=False)
            except FileNotFoundError:
                print("nvidia-smi not found; CUDA GPU may be unavailable.")

            import torch
            print("torch:", torch.__version__)
            print("cuda available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("cuda device:", torch.cuda.get_device_name(0))
            """
        ),
        markdown_cell("## Import Project Helpers"),
        code_cell(
            """
            from pathlib import Path
            import importlib.metadata as metadata

            from src.config import MODEL_CONFIG, SUBTITLE_CONFIG
            from src.audio import ensure_directories, extract_audio_16k_mono, audio_output_path
            from src.transcribe import (
                load_kotoba_model,
                transcribe_audio,
                transcribe_audio_chunked,
                build_alignment_text,
            )
            from src.align import load_qwen_aligner, run_qwen_alignment_chunked
            from src.subtitles import build_subtitles_from_units, write_srt

            for package_name in ["qwen-asr", "transformers", "faster-whisper"]:
                try:
                    print(f"{package_name}: {metadata.version(package_name)}")
                except metadata.PackageNotFoundError:
                    print(f"{package_name}: not installed")
            """
        ),
        markdown_cell("## Configure Paths and Parameters"),
        code_cell(
            """
            # Change these paths for your server.
            INPUT_PATH = Path("/path/to/input.mp4")
            OUTPUT_PATH = Path("./output.ja.srt")
            WORK_DIR = Path("./kotobasub_work")

            INPUT_DIR = WORK_DIR / "input"
            AUDIO_DIR = WORK_DIR / "audio"
            OUTPUT_DIR = WORK_DIR / "output"
            ensure_directories((INPUT_DIR, AUDIO_DIR, OUTPUT_DIR))

            LANGUAGE = MODEL_CONFIG.language
            USE_VAD = MODEL_CONFIG.use_vad
            BEAM_SIZE = MODEL_CONFIG.beam_size
            WHISPER_COMPUTE_TYPES = MODEL_CONFIG.whisper_compute_types
            KOTOBA_MODEL_CANDIDATES = MODEL_CONFIG.kotoba_model_candidates
            QWEN_ALIGNER_MODEL_ID = MODEL_CONFIG.qwen_aligner_model_id

            USE_CHUNKED_TRANSCRIPTION = True
            TRANSCRIBE_CHUNK_SECONDS = 30.0
            TRANSCRIBE_CHUNK_OVERLAP = 3.0

            USE_QWEN_ALIGNMENT = True
            FALLBACK_TO_WHISPER_TIMESTAMPS = True
            MAX_ALIGN_CHUNK_SECONDS = 30.0
            MAX_ALIGN_CHUNK_CHARS = 300
            ALIGN_CHUNK_PADDING = 1.0

            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            if not INPUT_PATH.exists():
                raise FileNotFoundError(f"INPUT_PATH does not exist: {INPUT_PATH}")

            print(f"Input: {INPUT_PATH}")
            print(f"Output: {OUTPUT_PATH}")
            print(f"Work dir: {WORK_DIR}")
            print(f"Device: {DEVICE}")
            """
        ),
        markdown_cell("## Extract Audio"),
        code_cell(
            """
            audio_path = audio_output_path(INPUT_PATH, AUDIO_DIR)
            extract_audio_16k_mono(INPUT_PATH, audio_path)
            print(f"Audio: {audio_path}")
            """
        ),
        markdown_cell("## Transcribe with Kotoba-Whisper"),
        code_cell(
            """
            whisper_model, whisper_compute_type, loaded_kotoba_model_id = load_kotoba_model(
                KOTOBA_MODEL_CANDIDATES,
                WHISPER_COMPUTE_TYPES,
                device=DEVICE,
            )

            if USE_CHUNKED_TRANSCRIPTION:
                whisper_segments, info = transcribe_audio_chunked(
                    whisper_model,
                    audio_path,
                    work_dir=AUDIO_DIR / "transcribe_chunks",
                    language=LANGUAGE,
                    beam_size=BEAM_SIZE,
                    use_vad=USE_VAD,
                    chunk_seconds=TRANSCRIBE_CHUNK_SECONDS,
                    chunk_overlap=TRANSCRIBE_CHUNK_OVERLAP,
                )
            else:
                whisper_segments, info = transcribe_audio(
                    whisper_model,
                    audio_path,
                    language=LANGUAGE,
                    beam_size=BEAM_SIZE,
                    use_vad=USE_VAD,
                )

            alignment_text = build_alignment_text(whisper_segments)
            print(f"Detected language: {info.language} ({info.language_probability:.2f})")
            print(f"Kotoba model: {loaded_kotoba_model_id}")
            print(f"Whisper compute type: {whisper_compute_type}")
            print(f"Whisper segments: {len(whisper_segments)}")
            print(alignment_text[:1000])
            """
        ),
        markdown_cell("## Optional Qwen Forced Alignment"),
        code_cell(
            """
            if USE_QWEN_ALIGNMENT and DEVICE != "cuda":
                print("CUDA is unavailable; disabling Qwen alignment and using Whisper timestamps.")
                USE_QWEN_ALIGNMENT = False

            if USE_QWEN_ALIGNMENT:
                aligner = load_qwen_aligner(QWEN_ALIGNER_MODEL_ID, device=DEVICE)
                aligned_units = run_qwen_alignment_chunked(
                    audio_path=audio_path,
                    whisper_segments=whisper_segments,
                    aligner=aligner,
                    work_dir=AUDIO_DIR / "align_chunks",
                    max_chunk_seconds=MAX_ALIGN_CHUNK_SECONDS,
                    max_chunk_chars=MAX_ALIGN_CHUNK_CHARS,
                    chunk_padding=ALIGN_CHUNK_PADDING,
                    fallback_to_whisper=FALLBACK_TO_WHISPER_TIMESTAMPS,
                )
                used_alignment = "qwen_chunked"
            else:
                aligned_units = whisper_segments
                used_alignment = "whisper"

            print(f"Alignment source: {used_alignment}")
            print(f"Aligned units: {len(aligned_units)}")
            """
        ),
        markdown_cell("## Write SRT"),
        code_cell(
            """
            subtitle_segments = build_subtitles_from_units(aligned_units, config=SUBTITLE_CONFIG)
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            write_srt(subtitle_segments, OUTPUT_PATH)

            print(f"Subtitle segments: {len(subtitle_segments)}")
            print(f"SRT: {OUTPUT_PATH.resolve()}")
            """
        ),
    ]

    write_notebook(build_notebook(cells, colab=False), OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
