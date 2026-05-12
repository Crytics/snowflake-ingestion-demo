# Vietnamese Speech-to-Text + Translation + Summary (local)

Local pipeline built on top of [VNSpeech_to_Text.ipynb](https://github.com/Crytics/Text-Analysis/blob/master/VNSpeech_to_Text.ipynb):

1. **Transcribe** — `vinai/PhoWhisper-large` (VN audio → VN text)
2. **Translate** — `facebook/nllb-200-distilled-600M` (VN → EN)
3. **Summarize** — `facebook/bart-large-cnn` (EN → short summary)

Steps 2 and 3 are toggleable in the UI.

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

Open http://127.0.0.1:7860, upload an audio file, click **Run**.

## Run with Docker

CPU image (works everywhere):

```powershell
docker compose up --build
```

Then open http://127.0.0.1:7860. The HF model cache is stored in a named volume (`hf-cache`) so subsequent runs skip the ~7 GB download.

For NVIDIA GPU acceleration:
1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.
2. Uncomment the `deploy.resources` block in [`docker-compose.yml`](docker-compose.yml).
3. `docker compose up --build`.

First run will download the PhoWhisper-large weights (~3 GB) into the HF cache.

## Notes
- CPU works but is slow; expect ~real-time-or-slower transcription, plus extra time for translation/summary.
- GPU with `float16` is much faster. Rough VRAM totals when all three models are loaded: ~8–10 GB.
- To shrink the footprint, edit the constants at the top of `app.py`:
  - `ASR_MODEL_ID` → `vinai/PhoWhisper-base` or `-small`
  - `TRANSLATE_MODEL_ID` → already the distilled 600M; `facebook/nllb-200-distilled-1.3B` is higher quality.
  - `SUMMARY_MODEL_ID` → `sshleifer/distilbart-cnn-12-6` for a lighter summarizer.
- Long transcripts are automatically split on sentence boundaries before translation/summary, then re-joined.
- Translation and summary models are lazy-loaded only when their checkbox is enabled.
