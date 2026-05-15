# KotobaSub

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BilyHurington/KotobaSub/blob/main/notebooks/Japanese_ASR_Kotoba_QwenAligner.ipynb)

Colab-oriented workflow for transcribing Japanese video or audio into Japanese SRT subtitles.

## Models

- ASR: `kotoba-tech/kotoba-whisper-v2.2-faster`
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

Local checks in this repository cover Python syntax, importability, and notebook JSON validity; full model inference requires a CUDA GPU runtime.
