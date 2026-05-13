# Japanese Subtitle Transcription Notebook

Colab-oriented workflow for transcribing Japanese video or audio into Japanese SRT subtitles.

Planned model stack:

- ASR: `kotoba-tech/kotoba-whisper-v2.2-faster`
- Forced alignment: `Qwen/Qwen3-ForcedAligner-0.6B`

The generated notebook is intended to run on Google Colab with a CUDA GPU. Local checks in this repository cover Python syntax, importability, and notebook JSON validity; full model inference requires a GPU runtime.

