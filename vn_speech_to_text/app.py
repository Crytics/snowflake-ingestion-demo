"""Vietnamese speech-to-text app using PhoWhisper + Gradio.

Local port of https://github.com/Crytics/Text-Analysis/blob/master/VNSpeech_to_Text.ipynb
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

MODEL_ID = "vinai/PhoWhisper-large"
CHUNK_LENGTH_S = 30
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


@lru_cache(maxsize=1)
def load_model():
    print(f"Loading {MODEL_ID} on {'cuda' if DEVICE == 0 else 'cpu'} ({DTYPE})...")
    asr = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_ID,
        device=DEVICE,
        torch_dtype=DTYPE,
        model_kwargs={"low_cpu_mem_usage": True},
        chunk_length_s=CHUNK_LENGTH_S,
    )
    print("Model ready.")
    return asr


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


def transcribe_audio(audio_path, progress=gr.Progress()):
    if not audio_path:
        return "", "No audio file provided."

    progress(0, desc="Loading model...")
    asr = load_model()

    duration = get_audio_duration(audio_path)
    progress(0.1, desc=f"Transcribing ({duration:.1f}s of audio)...")

    start = time.time()
    try:
        result = asr(audio_path)
        text = result["text"] if isinstance(result, dict) else str(result)
    except Exception as e:
        return "", f"Error: {e}"
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start
    status = f"Done in {elapsed:.1f}s (audio: {duration:.1f}s)"
    progress(1.0, desc="Complete")
    return text, status


def prepare_download_file(text):
    if not text:
        return None
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcription.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="Vietnamese Speech-to-Text") as demo:
        gr.Markdown("# Vietnamese Speech-to-Text\nPowered by `vinai/PhoWhisper-large`.")

        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(type="filepath", label="Upload audio (wav/mp3/m4a/...)")
                transcribe_btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                text_out = gr.Textbox(label="Transcription", lines=12, show_copy_button=True)
                status_out = gr.Textbox(label="Status", interactive=False)
                download_btn = gr.Button("Save as .txt")
                file_out = gr.File(label="Download")

        transcribe_btn.click(transcribe_audio, inputs=audio_in, outputs=[text_out, status_out])
        download_btn.click(prepare_download_file, inputs=text_out, outputs=file_out)

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
