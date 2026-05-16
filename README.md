# KotobaSub

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BilyHurington/KotobaSub/blob/main/notebooks/Japanese_ASR_Kotoba_QwenAligner.ipynb)

Colab-oriented workflow for transcribing Japanese video or audio into Japanese SRT subtitles.

## Models

- ASR: `RoachLin/kotoba-whisper-v2.2-faster`
- ASR fallback: `kotoba-tech/kotoba-whisper-v2.0-faster`
- Forced alignment: `Qwen/Qwen3-ForcedAligner-0.6B`

## Quick Start

1. Open the notebook with the Colab badge above.
2. Select a GPU runtime in Colab.
3. Run the cells from top to bottom.
4. Choose an input source in the notebook:
   - `INPUT_SOURCE = "upload"` uploads a file from your computer.
   - `INPUT_SOURCE = "drive"` mounts Google Drive.
     - `DRIVE_PICKER_MODE = "browse"` shows an interactive Drive file tree.
     - `DRIVE_PICKER_MODE = "path"` reads `DRIVE_INPUT_PATH` directly.
5. Download the generated `.ja.srt` file.

The notebook clones this repository inside Colab, installs the required dependencies, transcribes with Kotoba-Whisper, aligns with Qwen3-ForcedAligner, and writes an SRT file. The GitHub repository needs to be public for the Colab link to work for other people without authentication.

Qwen alignment is loaded through the official `qwen-asr` Python package, so the notebook does not clone Qwen3-ASR source code directly.

Long audio is transcribed in 30-second core chunks with 10 seconds of audio context on each side by default. Qwen alignment also runs in smaller core chunks to reduce Colab GPU memory pressure, while sending extra transcript and audio context around each core chunk.

If you previously ran an older version of the notebook in the same Colab runtime, restart the runtime before rerunning. This clears stale `/content/Qwen3-ASR` imports.

Local checks in this repository cover Python syntax, importability, and notebook JSON validity; full model inference requires a CUDA GPU runtime.

## Generic Server Notebook

For a normal Jupyter server, use:

```text
notebooks/Japanese_ASR_Kotoba_QwenAligner_Server.ipynb
```

Install `ffmpeg` on the host, then install PyTorch for your CUDA runtime:

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

The server notebook uses local paths:

```python
INPUT_PATH = Path("/path/to/input.mp4")
OUTPUT_PATH = Path("./output.ja.srt")
```

Qwen forced alignment is intended for CUDA GPU servers. Without CUDA, the server notebook can still produce SRT files from Whisper timestamps by setting `USE_QWEN_ALIGNMENT = False`.
