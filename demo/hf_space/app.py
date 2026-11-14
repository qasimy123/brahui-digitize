"""Hugging Face Space: Brahui speech/text -> English. ASR = fine-tuned omniASR-CTC-1B, MT = fine-tuned NLLB-200 1.3B.
Runs on ZeroGPU when available (the `spaces` package), otherwise on whatever torch finds (CPU is slow but works)."""
from __future__ import annotations

import os, re, time

try:
    import spaces  # ZeroGPU: must be imported before torch
    GPU = spaces.GPU
except Exception:  # plain CPU/GPU Space
    def GPU(fn=None, **_):
        return fn if fn else (lambda f: f)

import numpy as np
import soundfile as sf
import torch
import gradio as gr

from brahui_text import normalize, romanize

ASR_REPO = os.environ.get("ASR_REPO", "qkhawaja/brahui-asr-ctc1b-s3")
MT_REPO = os.environ.get("MT_REPO", "qkhawaja/brahui-nllb-1.3b-r5")
BRH, ENG = "urd_Arab", "eng_Latn"  # NLLB tag the student was trained with (Brahui in Perso-Arabic script, no own NLLB code)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer, VitsModel, Wav2Vec2ForCTC, WhisperForConditionalGeneration  # noqa: E402

proc = AutoProcessor.from_pretrained(ASR_REPO)
HALF = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
asr = Wav2Vec2ForCTC.from_pretrained(ASR_REPO, dtype=HALF).to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(MT_REPO)
mt = AutoModelForSeq2SeqLM.from_pretrained(MT_REPO, dtype=HALF).to(DEVICE).eval()
EN_ASR_REPO = os.environ.get("EN_ASR_REPO", "openai/whisper-small.en")
wproc = AutoProcessor.from_pretrained(EN_ASR_REPO)
whisper = WhisperForConditionalGeneration.from_pretrained(EN_ASR_REPO, dtype=HALF).to(DEVICE).eval()
# Brahui voice: MMS Urdu VITS fine-tuned on one Common Voice Brahui speaker (scripts/gpu/train_tts.sh). TTS_REPO can point
# back at facebook/mms-tts-urd-script_arabic; that vocab lacks ڷ (/ɬ/), which is then read as ل.
TTS_REPO = os.environ.get("TTS_REPO", "qkhawaja/brahui-tts-vits-lr1e4-300ep")
tts_tok = AutoTokenizer.from_pretrained(TTS_REPO)
tts = VitsModel.from_pretrained(TTS_REPO).to(DEVICE).eval()
TTS_HAS_LATERAL = "ڷ" in tts_tok.get_vocab()
TTS_LABEL = "spoken (Brahui voice, fine-tuned)" if TTS_HAS_LATERAL else "spoken (Urdu voice reading the Brahui)"


def load_audio(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data


def transcribe(audio: np.ndarray) -> str:
    outs = []
    with torch.no_grad():
        for start in range(0, len(audio), 20 * 16000):  # the CTC model degrades past ~20 s of input; chunk
            piece = audio[start:start + 20 * 16000]
            inp = proc(piece, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE, HALF)
            outs.append(proc.batch_decode(torch.argmax(asr(inp).logits, dim=-1))[0].strip())
    return normalize(" ".join(o for o in outs if o))


def translate(text: str, src: str = BRH, tgt: str = ENG) -> str:
    split = r"(?<=[۔؟!])\s+" if src == BRH else r"(?<=[.?!])\s+"
    sents = [x.strip() for x in re.split(split, text) if x.strip()] or [text]
    tok.src_lang = src
    with torch.no_grad():
        enc = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        gen = mt.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), num_beams=4, max_new_tokens=200)
    return " ".join(t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True))


@GPU(duration=30)
def cascade(audio):
    brh = transcribe(audio)
    return brh, (translate(brh) if brh.strip() else "")


@GPU(duration=20)
def translate_only(text, src=BRH, tgt=ENG):
    return translate(text, src, tgt)


def speak_brh(brh: str):
    """(sample_rate, int16 waveform) for a gr.Audio, via the Urdu VITS voice."""
    txt = brh if TTS_HAS_LATERAL else brh.replace("ڷ", "ل")
    with torch.no_grad():
        wav = tts(**tts_tok(txt, return_tensors="pt").to(DEVICE)).waveform[0].float().cpu().numpy()
    return tts.config.sampling_rate, (np.clip(wav, -1, 1) * 32767).astype(np.int16)


def transcribe_en(audio: np.ndarray) -> str:
    outs = []
    with torch.no_grad():
        for start in range(0, len(audio), 30 * 16000):  # Whisper's 30 s window
            feats = wproc(audio[start:start + 30 * 16000], sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, HALF)
            outs.append(wproc.batch_decode(whisper.generate(feats, max_new_tokens=220), skip_special_tokens=True)[0].strip())
    return " ".join(o for o in outs if o)


@GPU(duration=30)
def en_speech_to_brh(audio):
    en = transcribe_en(audio)
    if not en.strip():
        return "", "", None
    brh = normalize(translate(en, ENG, BRH))
    return en, brh, (speak_brh(brh) if brh.strip() else None)


def run_en_speech(path):
    if not path:
        return "", "", "", None, "record or upload a clip first"
    t0 = time.time(); audio = load_audio(path); dur = len(audio) / 16000
    en, brh, aud = en_speech_to_brh(audio); t = time.time() - t0
    if not en.strip():
        return "", "", "", None, f"no speech recognised ({dur:.1f} s clip)"
    return en, brh, romanize(brh), aud, f"{dur:.1f} s audio · ASR+MT+TTS {t:.1f} s · {DEVICE}"


@GPU(duration=20)
def en_to_brh_spoken(en):
    brh = normalize(translate(en, ENG, BRH))
    return brh, (speak_brh(brh) if brh.strip() else None)


def run_en(en):
    en = (en or "").strip()
    if not en:
        return "", "", None, "type some English first"
    t0 = time.time(); brh, audio = en_to_brh_spoken(en)
    return brh, romanize(brh), audio, f"MT+TTS {time.time() - t0:.1f} s · {DEVICE}"


def run_speech(path):
    if not path:
        return "", "", "", "record or upload a clip first"
    t0 = time.time(); audio = load_audio(path); dur = len(audio) / 16000
    brh, en = cascade(audio); t = time.time() - t0
    if not brh.strip():
        return "", "", "", f"no speech recognised ({dur:.1f} s clip)"
    return brh, romanize(brh), en, f"{dur:.1f} s audio · ASR+MT {t:.1f} s · {DEVICE}"


def run_text(brh):
    brh = normalize(brh or "")
    if not brh.strip():
        return "", "", "type some Brahui first"
    t0 = time.time(); en = translate_only(brh)
    return romanize(brh), en, f"MT {time.time() - t0:.1f} s · {DEVICE}"


CSS = """
.brh textarea{font-size:30px !important;line-height:1.8 !important;direction:rtl;text-align:right;
  font-family:"Noto Nastaliq Urdu","Noto Naskh Arabic","Geeza Pro",serif}
.en textarea{font-size:22px !important}
.lat textarea{font-style:italic;color:#8a8f98}
"""
SPEAK_JS = "(t)=>{if(!t)return;speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(t);u.lang='en-US';speechSynthesis.speak(u)}"

with gr.Blocks(css=CSS, title="Brahui → English") as demo:
    gr.Markdown("# Brahui → English\nBrahui ↔ English, spoken or typed. "
                "Under 20 s per clip works best. Models: omniASR-CTC-1B and NLLB-200 1.3B, both fine-tuned for Brahui with no human-annotated parallel data.")
    with gr.Tab("Brahui speech → English"):
        aud = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Brahui speech")
        go = gr.Button("Transcribe + translate", variant="primary")
        brh_o = gr.Textbox(label="Brahui (ASR)", elem_classes="brh", lines=2)
        lat_o = gr.Textbox(label="romanized", elem_classes="lat", lines=1)
        en_o = gr.Textbox(label="English", elem_classes="en", lines=2)
        with gr.Row():
            spk = gr.Button("🔊 Speak the English")
            meta = gr.Markdown()
        go.click(run_speech, aud, [brh_o, lat_o, en_o, meta])
        aud.stop_recording(run_speech, aud, [brh_o, lat_o, en_o, meta])
        spk.click(None, en_o, None, js=SPEAK_JS)
    with gr.Tab("English speech → Brahui"):
        aud4 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="English speech")
        go4 = gr.Button("Transcribe + translate", variant="primary")
        en4 = gr.Textbox(label="English (ASR)", elem_classes="en", lines=2)
        brh4 = gr.Textbox(label="Brahui", elem_classes="brh", lines=2)
        lat4 = gr.Textbox(label="romanized", elem_classes="lat", lines=1)
        out4 = gr.Audio(label=TTS_LABEL, autoplay=True)
        meta4 = gr.Markdown()
        go4.click(run_en_speech, aud4, [en4, brh4, lat4, out4, meta4])
        aud4.stop_recording(run_en_speech, aud4, [en4, brh4, lat4, out4, meta4])
    with gr.Tab("Brahui text → English"):
        brh_i = gr.Textbox(label="Brahui text", elem_classes="brh", lines=3, placeholder="نی اسلام علیکم")
        go2 = gr.Button("Translate", variant="primary")
        lat2 = gr.Textbox(label="romanized", elem_classes="lat", lines=1)
        en2 = gr.Textbox(label="English", elem_classes="en", lines=2)
        with gr.Row():
            spk2 = gr.Button("🔊 Speak the English")
            meta2 = gr.Markdown()
        go2.click(run_text, brh_i, [lat2, en2, meta2])
        brh_i.submit(run_text, brh_i, [lat2, en2, meta2])
        spk2.click(None, en2, None, js=SPEAK_JS)
    with gr.Tab("English → Brahui"):
        en_i = gr.Textbox(label="English text", elem_classes="en", lines=3, placeholder="Where is the nearest hospital?")
        go3 = gr.Button("Translate", variant="primary")
        brh3 = gr.Textbox(label="Brahui", elem_classes="brh", lines=2)
        lat3 = gr.Textbox(label="romanized", elem_classes="lat", lines=1)
        aud3 = gr.Audio(label=TTS_LABEL, autoplay=True)
        meta3 = gr.Markdown()
        go3.click(run_en, en_i, [brh3, lat3, aud3, meta3])
        en_i.submit(run_en, en_i, [brh3, lat3, aud3, meta3])
    gr.Markdown("Brahui is written in Perso-Arabic script with the extra letter ڷ. The ASR was trained on read speech from a few speakers, "
                "so conversational audio, noise and unfamiliar speakers will degrade it. Translations are machine output; verify before relying on them.")

if __name__ == "__main__":
    demo.queue().launch()
