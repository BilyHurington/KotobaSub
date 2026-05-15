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


def main() -> None:
    cells = [
        markdown_cell(
            """
            # Japanese SRT Transcription with Kotoba-Whisper and Qwen Forced Aligner

            This notebook transcribes Japanese video or audio into a Japanese `.srt` subtitle file.

            - ASR: `RoachLin/kotoba-whisper-v2.2-faster`
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
            else:
                !git -C /content/KotobaSub fetch origin main
                !git -C /content/KotobaSub reset --hard origin/main

            if str(REPO_DIR) not in sys.path:
                sys.path.insert(0, str(REPO_DIR))

            print(f"Project repo: {{REPO_DIR}}")
            """
        ),
        markdown_cell("## Install Python Dependencies"),
        code_cell(
            """
            !pip install -q -r /content/KotobaSub/requirements-colab.txt

            import importlib
            import importlib.metadata as metadata
            import sys

            # Older versions of this notebook cloned Qwen3-ASR into /content/Qwen3-ASR.
            # Remove that source checkout from imports so the official qwen-asr package is used.
            sys.path = [path for path in sys.path if path != "/content/Qwen3-ASR"]
            for module_name in list(sys.modules):
                if module_name == "qwen_asr" or module_name.startswith("qwen_asr."):
                    del sys.modules[module_name]
            importlib.invalidate_caches()

            for package_name in ["qwen-asr", "transformers", "faster-whisper"]:
                print(f"{package_name}: {metadata.version(package_name)}")

            try:
                import qwen_asr

                print(f"qwen_asr import path: {qwen_asr.__file__}")
            except Exception as exc:
                raise RuntimeError(
                    "qwen_asr is installed but could not be imported cleanly. "
                    "Restart the Colab runtime, then run the notebook from the first cell."
                ) from exc
            """
        ),
        markdown_cell("## Configure Workspace and Import Helpers"),
        code_cell(
            """
            from src.config import MODEL_CONFIG, SUBTITLE_CONFIG, WORKSPACE_CONFIG
            from src.audio import ensure_directories, extract_audio_16k_mono, audio_output_path
            from src.transcribe import (
                load_kotoba_model,
                transcribe_audio,
                transcribe_audio_chunked,
                build_alignment_text,
            )
            from src.align import load_qwen_aligner, run_qwen_alignment_chunked
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
            KOTOBA_MODEL_CANDIDATES = MODEL_CONFIG.kotoba_model_candidates
            QWEN_ALIGNER_MODEL_ID = MODEL_CONFIG.qwen_aligner_model_id
            FALLBACK_TO_WHISPER_TIMESTAMPS = MODEL_CONFIG.fallback_to_whisper_timestamps
            USE_CHUNKED_TRANSCRIPTION = MODEL_CONFIG.use_chunked_transcription
            TRANSCRIBE_CHUNK_SECONDS = MODEL_CONFIG.transcribe_chunk_seconds
            TRANSCRIBE_CHUNK_OVERLAP = MODEL_CONFIG.transcribe_chunk_overlap
            MAX_ALIGN_CHUNK_SECONDS = MODEL_CONFIG.max_align_chunk_seconds
            MAX_ALIGN_CHUNK_CHARS = MODEL_CONFIG.max_align_chunk_chars
            ALIGN_CHUNK_PADDING = MODEL_CONFIG.align_chunk_padding

            print("Kotoba-Whisper model candidates:")
            for model_id in KOTOBA_MODEL_CANDIDATES:
                print(f"- {model_id}")
            print(f"Chunked transcription: {USE_CHUNKED_TRANSCRIPTION}")
            print(f"Transcription chunk: {TRANSCRIBE_CHUNK_SECONDS}s + {TRANSCRIBE_CHUNK_OVERLAP}s overlap")
            """
        ),
        markdown_cell("## Select Input Media"),
        code_cell(
            """
            # Choose "upload" to upload from your computer, or "drive" to read from Google Drive.
            INPUT_SOURCE = "upload"

            # Used only when INPUT_SOURCE = "drive".
            # Choose "browse" for an interactive Google Drive file tree, or "path" to paste a path.
            DRIVE_PICKER_MODE = "browse"

            # Used when DRIVE_PICKER_MODE = "browse".
            DRIVE_SEARCH_ROOT = "/content/drive/MyDrive"

            # Used when DRIVE_PICKER_MODE = "path".
            DRIVE_INPUT_PATH = "/content/drive/MyDrive/path/to/your_file.mp4"

            SUPPORTED_MEDIA_EXTENSIONS = (
                ".mp4",
                ".mkv",
                ".mov",
                ".avi",
                ".webm",
                ".mp3",
                ".wav",
                ".m4a",
                ".aac",
                ".flac",
                ".ogg",
                ".opus",
            )


            selected_drive_file_path = None


            if INPUT_SOURCE == "drive":
                from google.colab import drive, output
                from IPython.display import display
                import ipywidgets as widgets
                from ipytree import Tree, Node

                drive.mount("/content/drive")
                output.enable_custom_widget_manager()

                if DRIVE_PICKER_MODE == "browse":
                    root_path = Path(DRIVE_SEARCH_ROOT)
                    if not root_path.exists():
                        raise FileNotFoundError(f"DRIVE_SEARCH_ROOT does not exist: {root_path}")

                    status = widgets.HTML("Select a media file from Google Drive, then click Select.")
                    selected_label = widgets.HTML("Selected: none")
                    select_button = widgets.Button(description="Select", button_style="primary")
                    selected_candidate_path = None
                    tree = Tree(stripes=True)
                    root_node = Node(root_path.name, opened=False)
                    root_node.path = str(root_path)
                    root_node.is_file = False
                    tree.add_node(root_node)


                    def is_supported_media(path):
                        return Path(path).suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS


                    def populate_node(node):
                        if getattr(node, "loaded", False):
                            return

                        node.loaded = True
                        try:
                            children = sorted(
                                Path(node.path).iterdir(),
                                key=lambda item: (item.is_file(), item.name.lower()),
                            )
                        except PermissionError:
                            return

                        for child in children:
                            if child.name.startswith("."):
                                continue

                            if child.is_dir():
                                child_node = Node(child.name, opened=False)
                                child_node.path = str(child)
                                child_node.is_file = False
                                node.add_node(child_node)
                                child_node.observe(on_node_opened, "opened")
                            elif child.is_file() and is_supported_media(child):
                                child_node = Node(child.name)
                                child_node.path = str(child)
                                child_node.is_file = True
                                node.add_node(child_node)
                                child_node.observe(on_node_selected, "selected")


                    def on_node_opened(change):
                        if change["new"]:
                            populate_node(change["owner"])


                    def on_node_selected(change):
                        global selected_candidate_path

                        if not change["new"]:
                            return

                        selected_node = change["owner"]
                        if not getattr(selected_node, "is_file", False):
                            return

                        selected_candidate_path = selected_node.path
                        selected_label.value = f"Selected: {selected_candidate_path}"
                        status.value = "File highlighted. Click Select to use it."


                    def on_select(_button):
                        global selected_candidate_path, selected_drive_file_path

                        if not selected_candidate_path:
                            status.value = "No file selected."
                            return

                        selected_drive_file_path = selected_candidate_path
                        selected_label.value = f"Selected: {selected_drive_file_path}"
                        status.value = "Selection saved. Run the next cell."


                    root_node.observe(on_node_opened, "opened")
                    populate_node(root_node)
                    select_button.on_click(on_select)
                    display(status, tree, selected_label, select_button)

                elif DRIVE_PICKER_MODE == "path":
                    selected_drive_file_path = DRIVE_INPUT_PATH
                    print(f"Drive input path: {selected_drive_file_path}")

                else:
                    raise ValueError('DRIVE_PICKER_MODE must be "browse" or "path".')

            elif INPUT_SOURCE == "upload":
                print("Upload mode selected. Run the next cell and choose a local media file.")

            else:
                raise ValueError('INPUT_SOURCE must be "upload" or "drive".')
            """
        ),
        markdown_cell("## Load Selected Input"),
        code_cell(
            """
            import shutil


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
                if not selected_drive_file_path:
                    raise RuntimeError(
                        "No Google Drive file selected. Run the previous cell, select a media file, "
                        "click Select, then run this cell again."
                    )

                input_path = copy_input_to_workspace(selected_drive_file_path)

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
            whisper_model, whisper_compute_type, loaded_kotoba_model_id = load_kotoba_model(
                KOTOBA_MODEL_CANDIDATES,
                WHISPER_COMPUTE_TYPES,
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
        markdown_cell("## Load Qwen3-ForcedAligner"),
        code_cell(
            """
            aligner = load_qwen_aligner(QWEN_ALIGNER_MODEL_ID)
            """
        ),
        code_cell(
            """
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
