# Design

## Goal

Create a Colab notebook that turns Japanese video or audio into an SRT subtitle file.

## Pipeline

1. Upload or select an input media file.
2. Extract audio with `ffmpeg` as 16 kHz mono WAV.
3. Transcribe with `kotoba-tech/kotoba-whisper-v2.2-faster` through `faster-whisper`.
4. Build a plain Japanese alignment transcript from the ASR output.
5. Align the transcript to audio with `Qwen/Qwen3-ForcedAligner-0.6B` through the official Qwen3-ASR helper.
6. Normalize alignment output into timestamped units.
7. Aggregate timestamped units into readable subtitle segments.
8. Write and download an SRT file.

## Defaults

- Use faster-whisper VAD during transcription.
- Use CUDA and try `float16`, then `int8_float16`, then `int8`.
- Do not use LLM cleanup in the first implementation.
- Fall back to Kotoba-Whisper timestamps if Qwen alignment fails.

## Local Validation Scope

This machine does not have an NVIDIA GPU, so local validation should be limited to:

- Python syntax checks.
- Import checks for modules that do not require heavyweight model dependencies.
- Notebook JSON validation.
- Generated notebook structure review.

