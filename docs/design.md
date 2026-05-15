# Design

## Goal

Create a Colab notebook that turns Japanese video or audio into an SRT subtitle file.

## Pipeline

1. Open the notebook through the GitHub-backed Colab link.
2. Install Colab system tools and clone this project repository into `/content/KotobaSub`.
3. Install Python dependencies from `requirements-colab.txt`.
4. Upload a local input media file or select one from Google Drive.
5. Extract audio with `ffmpeg` as 16 kHz mono WAV.
6. Transcribe with `RoachLin/kotoba-whisper-v2.2-faster` through `faster-whisper`.
7. Build a plain Japanese alignment transcript from the ASR output.
8. Clone Qwen3-ASR into `/content/Qwen3-ASR`.
9. Align the transcript to audio with `Qwen/Qwen3-ForcedAligner-0.6B` through the official Qwen3-ASR helper.
10. Normalize alignment output into timestamped units.
11. Aggregate timestamped units into readable subtitle segments.
12. Write and download an SRT file.

## Defaults

- Use faster-whisper VAD during transcription.
- Prefer the v2.2 CTranslate2 conversion at `RoachLin/kotoba-whisper-v2.2-faster`.
- Fall back to the official `kotoba-tech/kotoba-whisper-v2.0-faster` if needed.
- Use CUDA and try `float16`, then `int8_float16`, then `int8`.
- Do not use LLM cleanup in the first implementation.
- Fall back to Kotoba-Whisper timestamps if Qwen alignment fails.
- Default to local upload input, with optional Google Drive browsing through `ipytree`.

## Local Validation Scope

This machine does not have an NVIDIA GPU, so local validation should be limited to:

- Python syntax checks.
- Import checks for modules that do not require heavyweight model dependencies.
- Notebook JSON validation.
- Generated notebook structure review.
