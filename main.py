"""
AI Text Reader — A TTS application using Kokoro-82M (Gradio UI)
"""

import os
import sys
import glob
import site
import shutil

# ── Fix NVIDIA library paths so CUDA/cuDNN loads correctly ───────────
# LD_LIBRARY_PATH must be set BEFORE Python starts, so we re-exec once.
_REEXEC_FLAG = "_AITEXTREADER_REEXECED"


def _candidate_site_packages() -> list[str]:
    """Return existing site-packages paths for the current interpreter."""
    paths = []

    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        user_sp = site.getusersitepackages()
        if isinstance(user_sp, str):
            paths.append(user_sp)
    except Exception:
        pass

    # Keep insertion order while deduplicating.
    out = []
    seen = set()
    for p in paths:
        if p and p not in seen and os.path.isdir(p):
            seen.add(p)
            out.append(p)
    return out


_nvidia_lib_dirs = []
for _sp in _candidate_site_packages():
    _nvidia_lib_dirs.extend(glob.glob(os.path.join(_sp, "nvidia", "*", "lib")))

if _nvidia_lib_dirs and _REEXEC_FLAG not in os.environ:
    _extra = ":".join(_nvidia_lib_dirs)
    os.environ["LD_LIBRARY_PATH"] = _extra + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ[_REEXEC_FLAG] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import tempfile
import numpy as np
import soundfile as sf
import gradio as gr

SAMPLE_RATE = 24_000  # Kokoro native sample rate

# ──────────────────────────────────────────────────────────────────────
# Detect device
# ──────────────────────────────────────────────────────────────────────
def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

DEVICE = _pick_device()
print(f"[AI Text Reader] Using device: {DEVICE}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _check_runtime_dependencies():
    """Validate optional runtime tools and surface clear startup hints."""
    if shutil.which("ffmpeg") is None:
        print("[AI Text Reader] Warning: ffmpeg not found. MP3 compile may fall back to WAV.")

# ──────────────────────────────────────────────────────────────────────
# Voice / language catalogue
# ──────────────────────────────────────────────────────────────────────
LANGUAGES = {
    "American English": "a",
    "British English": "b",
    "Spanish": "e",
    "French": "f",
    "Hindi": "h",
    "Italian": "i",
    "Brazilian Portuguese": "p",
}

VOICES_BY_LANG = {
    "American English": [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "British English": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "Spanish": ["ef_dora", "em_alex", "em_santa"],
    "French": ["ff_siwis"],
    "Hindi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "Italian": ["if_sara", "im_nicola"],
    "Brazilian Portuguese": ["pf_dora", "pm_alex", "pm_santa"],
}

VOICE_LABELS = {
    "af_heart": "Heart (Female)", "af_alloy": "Alloy (Female)",
    "af_aoede": "Aoede (Female)", "af_bella": "Bella (Female)",
    "af_jessica": "Jessica (Female)", "af_kore": "Kore (Female)",
    "af_nicole": "Nicole (Female)", "af_nova": "Nova (Female)",
    "af_river": "River (Female)", "af_sarah": "Sarah (Female)",
    "af_sky": "Sky (Female)", "am_adam": "Adam (Male)",
    "am_echo": "Echo (Male)", "am_eric": "Eric (Male)",
    "am_fenrir": "Fenrir (Male)", "am_liam": "Liam (Male)",
    "am_michael": "Michael (Male)", "am_onyx": "Onyx (Male)",
    "am_puck": "Puck (Male)", "am_santa": "Santa (Male) 🎅",
    "bf_alice": "Alice (Female)", "bf_emma": "Emma (Female)",
    "bf_isabella": "Isabella (Female)", "bf_lily": "Lily (Female)",
    "bm_daniel": "Daniel (Male)", "bm_fable": "Fable (Male)",
    "bm_george": "George (Male)", "bm_lewis": "Lewis (Male)",
    "ef_dora": "Dora (Female)", "em_alex": "Alex (Male)",
    "em_santa": "Santa (Male)", "ff_siwis": "Siwis (Female)",
    "hf_alpha": "Alpha (Female)", "hf_beta": "Beta (Female)",
    "hm_omega": "Omega (Male)", "hm_psi": "Psi (Male)",
    "if_sara": "Sara (Female)", "im_nicola": "Nicola (Male)",
    "pf_dora": "Dora (Female)", "pm_alex": "Alex (Male)",
    "pm_santa": "Santa (Male)",
}


# ──────────────────────────────────────────────────────────────────────
# Pipeline manager (lazy-loaded, cached per language)
# ──────────────────────────────────────────────────────────────────────
_pipeline = None
_pipeline_lang = None


def get_pipeline(lang_code: str):
    global _pipeline, _pipeline_lang
    if _pipeline is None or _pipeline_lang != lang_code:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code=lang_code, device=DEVICE)
        _pipeline_lang = lang_code
    return _pipeline


def wav_to_mp3(wav_path: str, mp3_path: str):
    """Convert WAV → MP3 using pydub + ffmpeg."""
    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise RuntimeError("pydub is not installed; cannot export MP3.") from exc

    seg = AudioSegment.from_wav(wav_path)
    seg.export(mp3_path, format="mp3", bitrate="192k")


# ──────────────────────────────────────────────────────────────────────
# Gradio callbacks
# ──────────────────────────────────────────────────────────────────────
def on_language_change(language):
    """Update voice dropdown when language changes."""
    voices = VOICES_BY_LANG.get(language, [])
    choices = [VOICE_LABELS.get(v, v) for v in voices]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _label_to_id(label: str) -> str:
    """Reverse-lookup: pretty label → voice_id."""
    for vid, lbl in VOICE_LABELS.items():
        if lbl == label:
            return vid
    return label


def _generate_full_audio(text, lang_code, voice_id, speed, volume):
    """Generate all audio chunks, apply volume, return float32 array."""
    pipeline = get_pipeline(lang_code)
    vol = volume / 100.0

    chunks = []
    for _i, (_gs, _ps, audio) in enumerate(
        pipeline(text.strip(), voice=voice_id, speed=speed, split_pattern=r'\n+')
    ):
        chunks.append(audio)

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    full = np.concatenate(chunks)
    # Apply volume and clip
    full = np.clip(full * vol, -1.0, 1.0)
    return full


def _save_wav(audio: np.ndarray, path: str):
    """Save float32 audio as 16-bit WAV so volume scaling is audible."""
    int16 = (audio * 32767).astype(np.int16)
    sf.write(path, int16, SAMPLE_RATE, subtype='PCM_16')


def play_tts(text, language, voice_label, speed, volume):
    """Generate speech, save to temp WAV, return for browser playback."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text first.")

    lang_code = LANGUAGES[language]
    voice_id = _label_to_id(voice_label)

    audio = _generate_full_audio(text, lang_code, voice_id, speed, volume)
    if len(audio) == 0:
        raise gr.Error("No audio was generated. Try different text.")

    # Save to temp WAV and return filepath
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    _save_wav(audio, tmp.name)
    return tmp.name


def compile_tts(text, language, voice_label, speed, volume):
    """Generate ALL speech, save as MP3, return file for download."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text first.")

    lang_code = LANGUAGES[language]
    voice_id = _label_to_id(voice_label)

    audio = _generate_full_audio(text, lang_code, voice_id, speed, volume)
    if len(audio) == 0:
        raise gr.Error("No audio was generated. Try different text.")

    # Save WAV → convert to MP3
    tmp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(tmp_dir, "output.wav")
    mp3_path = os.path.join(tmp_dir, "output.mp3")
    _save_wav(audio, wav_path)

    try:
        wav_to_mp3(wav_path, mp3_path)
        return mp3_path
    except Exception as exc:
        print(f"[AI Text Reader] MP3 conversion failed ({exc}); returning WAV instead.")
        return wav_path


# ──────────────────────────────────────────────────────────────────────
# Build Gradio UI
# ──────────────────────────────────────────────────────────────────────
def build_app():
    default_lang = "American English"
    default_voices = [VOICE_LABELS[v] for v in VOICES_BY_LANG[default_lang]]

    with gr.Blocks(
        title="Kokoro TTS",
    ) as app:
        gr.Markdown(
            f"# Kokoro TTS\n"
            f"*Powered by Kokoro-82M — running on **{DEVICE.upper()}***"
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text to Read",
                    placeholder="Type or paste your text here…",
                    lines=10,
                    max_lines=30,
                )

            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value=default_lang,
                    label="Language",
                )
                voice = gr.Dropdown(
                    choices=default_voices,
                    value=default_voices[0],
                    label="Voice",
                )
                speed = gr.Slider(
                    minimum=0.4, maximum=3.0, step=0.05, value=1.0,
                    label="Speed",
                )
                volume = gr.Slider(
                    minimum=0, maximum=100, step=1, value=80,
                    label="Volume %",
                )

        language.change(
            fn=on_language_change,
            inputs=[language],
            outputs=[voice],
        )

        with gr.Row():
            play_btn = gr.Button("▶  Play", variant="primary", size="lg")
            compile_btn = gr.Button("⚙  Compile to MP3", variant="secondary", size="lg")

        audio_output = gr.Audio(
            label="Audio Player",
            type="filepath",
            autoplay=True,
        )
        file_output = gr.File(label="Download compiled audio")

        play_btn.click(
            fn=play_tts,
            inputs=[text_input, language, voice, speed, volume],
            outputs=[audio_output],
        )
        compile_btn.click(
            fn=compile_tts,
            inputs=[text_input, language, voice, speed, volume],
            outputs=[file_output],
        )

    return app


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _check_runtime_dependencies()
    app = build_app()

    share = _env_bool("AITEXTREADER_SHARE", False)
    inbrowser = _env_bool("AITEXTREADER_INBROWSER", True)
    username = os.getenv("AITEXTREADER_USERNAME")
    password = os.getenv("AITEXTREADER_PASSWORD")
    auth = (username, password) if username and password else None

    app.launch(
        inbrowser=inbrowser,
        share=share,
        auth=auth,
    )
