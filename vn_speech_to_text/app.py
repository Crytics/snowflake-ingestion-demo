"""Vietnamese speech-to-text + translation + summary, all local.

Pipeline:
  audio  -- PhoWhisper-large -->  VN text
  VN text -- NLLB-200 (vie_Latn -> eng_Latn) -->  EN text
  EN text -- BART-CNN -->  summary

Run: python app.py
"""

import gc
import os
import time
from functools import lru_cache

import gradio as gr
import librosa
import soundfile as sf
import torch
from transformers import pipeline

ASR_MODEL_ID = "vinai/PhoWhisper-large"
TRANSLATE_MODEL_ID = "facebook/nllb-200-distilled-600M"
SUMMARY_MODEL_ID = "facebook/bart-large-cnn"

CHUNK_LENGTH_S = 30
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

VN_LANG = "vie_Latn"
EN_LANG = "eng_Latn"


def _device_str():
    return "cuda" if DEVICE == 0 else "cpu"


@lru_cache(maxsize=1)
def load_asr():
    print(f"[asr] loading {ASR_MODEL_ID} on {_device_str()} ({DTYPE})...")
    p = pipeline(
        task="automatic-speech-recognition",
        model=ASR_MODEL_ID,
        device=DEVICE,
        torch_dtype=DTYPE,
        model_kwargs={"low_cpu_mem_usage": True},
        chunk_length_s=CHUNK_LENGTH_S,
    )
    print("[asr] ready.")
    return p


@lru_cache(maxsize=1)
def load_translator():
    print(f"[mt] loading {TRANSLATE_MODEL_ID} on {_device_str()} ({DTYPE})...")
    p = pipeline(
        task="translation",
        model=TRANSLATE_MODEL_ID,
        device=DEVICE,
        torch_dtype=DTYPE,
        src_lang=VN_LANG,
        tgt_lang=EN_LANG,
        model_kwargs={"low_cpu_mem_usage": True},
    )
    print("[mt] ready.")
    return p


@lru_cache(maxsize=1)
def load_summarizer():
    print(f"[sum] loading {SUMMARY_MODEL_ID} on {_device_str()} ({DTYPE})...")
    p = pipeline(
        task="summarization",
        model=SUMMARY_MODEL_ID,
        device=DEVICE,
        torch_dtype=DTYPE,
        model_kwargs={"low_cpu_mem_usage": True},
    )
    print("[sum] ready.")
    return p


def _free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_audio_duration(path):
    if not path or not os.path.exists(path):
        return 0.0
    try:
        return librosa.get_duration(path=path)
    except Exception:
        try:
            with sf.SoundFile(path) as f:
                return len(f) / f.samplerate
        except Exception:
            return 0.0


def _split_for_model(text, max_chars):
    """Split on sentence boundaries so long text fits in model context."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, buf = [], ""
    for sentence in text.replace("\n", " ").split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        piece = sentence if sentence.endswith(".") else sentence + "."
        if len(buf) + len(piece) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = piece
        else:
            buf = (buf + " " + piece).strip()
    if buf:
        chunks.append(buf.strip())
    return chunks


def translate_vi_to_en(vn_text):
    if not vn_text or not vn_text.strip():
        return ""
    mt = load_translator()
    parts = _split_for_model(vn_text, max_chars=900)
    out = []
    for part in parts:
        res = mt(part, max_length=512)
        out.append(res[0]["translation_text"])
    _free_mem()
    return " ".join(out).strip()


def summarize_en(en_text):
    if not en_text or not en_text.strip():
        return ""
    summarizer = load_summarizer()
    parts = _split_for_model(en_text, max_chars=3000)
    summaries = []
    for part in parts:
        words = len(part.split())
        max_len = min(180, max(60, words // 2))
        min_len = min(40, max(20, words // 6))
        res = summarizer(part, max_length=max_len, min_length=min_len, do_sample=False)
        summaries.append(res[0]["summary_text"])
    _free_mem()
    combined = " ".join(summaries).strip()
    if len(summaries) > 1 and len(combined.split()) > 250:
        res = summarizer(combined, max_length=180, min_length=60, do_sample=False)
        combined = res[0]["summary_text"].strip()
    return combined


def run_pipeline(audio_path, do_translate, do_summarize, progress=gr.Progress()):
    if not audio_path:
        return "", "", "", "No audio file provided."

    duration = get_audio_duration(audio_path)
    timings = []

    progress(0.05, desc="Loading ASR model...")
    asr = load_asr()

    progress(0.15, desc=f"Transcribing ({duration:.1f}s)...")
    t0 = time.time()
    try:
        result = asr(audio_path)
        vn_text = result["text"] if isinstance(result, dict) else str(result)
    except Exception as e:
        return "", "", "", f"Transcription error: {e}"
    finally:
        _free_mem()
    timings.append(f"transcribe {time.time() - t0:.1f}s")

    en_text = ""
    if do_translate or do_summarize:
        progress(0.5, desc="Translating to English...")
        t0 = time.time()
        try:
            en_text = translate_vi_to_en(vn_text)
        except Exception as e:
            return vn_text, "", "", f"Translation error: {e}"
        timings.append(f"translate {time.time() - t0:.1f}s")

    summary = ""
    if do_summarize:
        progress(0.85, desc="Summarizing...")
        t0 = time.time()
        try:
            summary = summarize_en(en_text)
        except Exception as e:
            return vn_text, en_text, "", f"Summary error: {e}"
        timings.append(f"summarize {time.time() - t0:.1f}s")

    progress(1.0, desc="Complete")
    status = f"Done — {', '.join(timings)} (audio: {duration:.1f}s)"
    return vn_text, en_text, summary, status


def prepare_download_file(vn_text, en_text, summary):
    if not any([vn_text, en_text, summary]):
        return None
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcription.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        if vn_text:
            f.write("=== Vietnamese transcription ===\n")
            f.write(vn_text.strip() + "\n\n")
        if en_text:
            f.write("=== English translation ===\n")
            f.write(en_text.strip() + "\n\n")
        if summary:
            f.write("=== Summary ===\n")
            f.write(summary.strip() + "\n")
    return out_path


def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="Vietnamese Speech-to-Text") as demo:
        gr.Markdown(
            "# Vietnamese Speech-to-Text + Translation + Summary\n"
            "Local pipeline: **PhoWhisper-large** → **NLLB-200** (vi→en) → **BART-CNN** summary."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(type="filepath", label="Upload audio (wav/mp3/m4a/...)")
                do_translate = gr.Checkbox(value=True, label="Translate to English (NLLB-200)")
                do_summarize = gr.Checkbox(value=True, label="AI summary (BART-CNN)")
                run_btn = gr.Button("Run", variant="primary")
                status_out = gr.Textbox(label="Status", interactive=False)
                download_btn = gr.Button("Save all as .txt")
                file_out = gr.File(label="Download")
            with gr.Column(scale=2):
                vn_out = gr.Textbox(label="Vietnamese transcription", lines=8, show_copy_button=True)
                en_out = gr.Textbox(label="English translation", lines=8, show_copy_button=True)
                summary_out = gr.Textbox(label="Summary", lines=6, show_copy_button=True)

        run_btn.click(
            run_pipeline,
            inputs=[audio_in, do_translate, do_summarize],
            outputs=[vn_out, en_out, summary_out, status_out],
        )
        download_btn.click(
            prepare_download_file,
            inputs=[vn_out, en_out, summary_out],
            outputs=file_out,
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
