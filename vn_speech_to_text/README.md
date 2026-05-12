# Vietnamese Speech-to-Text (local)

Local port of [VNSpeech_to_Text.ipynb](https://github.com/Crytics/Text-Analysis/blob/master/VNSpeech_to_Text.ipynb).
Uses `vinai/PhoWhisper-large` via Hugging Face `transformers` and a Gradio UI.

## Setup

You need Python 3.10+ and **FFmpeg** installed and on PATH.
- Install FFmpeg: `winget install Gyan.FFmpeg` (Windows) or `choco install ffmpeg`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU (CUDA) acceleration, install the matching PyTorch build first from
https://pytorch.org/get-started/locally/ before `pip install -r requirements.txt`.

## Run

```powershell
python app.py
```

Open http://127.0.0.1:7860, upload an audio file, click **Transcribe**.

First run will download the PhoWhisper-large weights (~3 GB) into the HF cache.

## Notes
- CPU works but is slow; expect ~real-time-or-slower transcription.
- GPU with `float16` is much faster; ~6 GB VRAM is comfortable for `-large`.
- For lower resource use, swap `MODEL_ID` in `app.py` to `vinai/PhoWhisper-base` or `vinai/PhoWhisper-small`.
