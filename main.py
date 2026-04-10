"""
AI Text Reader — A TTS application using Kokoro-82M (Gradio UI)
"""

import os
import sys
import glob
import site
import shutil
import gc
import time
import json
import threading

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
_engine_active = False
_activation_in_progress = False
_last_activity_monotonic = 0.0
ENGINE_IDLE_TIMEOUT_SEC = int(os.getenv("AITEXTREADER_IDLE_TIMEOUT_SEC", "300"))
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".aitextreader")
VOICE_USAGE_FILE = os.path.join(STATE_DIR, "voice_usage.json")


def _touch_activity():
    global _last_activity_monotonic
    _last_activity_monotonic = time.monotonic()


def _load_voice_usage() -> dict:
    if not os.path.exists(VOICE_USAGE_FILE):
        return {"used_voices": []}
    try:
        with open(VOICE_USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("used_voices"), list):
                return data
    except Exception:
        pass
    return {"used_voices": []}


def _save_voice_usage(data: dict):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(VOICE_USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def _record_voice_usage(voice_id: str):
    data = _load_voice_usage()
    used = data.get("used_voices", [])
    if voice_id not in used:
        used.append(voice_id)
        data["used_voices"] = sorted(used)
        _save_voice_usage(data)


def _hf_cache_root() -> str:
    env_cache = os.getenv("HF_HUB_CACHE")
    if env_cache:
        return env_cache
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _repo_cache_path(repo_id: str) -> str:
    repo_dir = repo_id.replace("/", "--")
    return os.path.join(_hf_cache_root(), f"models--{repo_dir}")


def _is_repo_cached(repo_id: str) -> bool:
    repo_path = _repo_cache_path(repo_id)
    snapshots_dir = os.path.join(repo_path, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return False

    # Consider cache ready if at least one snapshot has files.
    for _root, _dirs, files in os.walk(snapshots_dir):
        if files:
            return True
    return False


def _emit_progress(progress, value: float, desc: str):
    if callable(progress):
        progress(value, desc=desc)


def _directory_size(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                pass
    return total


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(max(0, num_bytes))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def _ensure_kokoro_assets(progress=None):
    """Download model with visible progress updates."""
    _emit_progress(progress, 0.08, "Checking model cache...")

    repo_id = "hexgrad/Kokoro-82M"
    if _is_repo_cached(repo_id):
        _emit_progress(progress, 0.20, "Model cache found. Skipping download.")
        return

    _emit_progress(progress, 0.20, "Downloading Kokoro model files (this may take 1-3 minutes on first run)...")
    try:
        from huggingface_hub import snapshot_download
        cache_dir = _hf_cache_root()
        print(f"[AI Text Reader] Downloading {repo_id} to {cache_dir}")
        snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
        print(f"[AI Text Reader] Successfully downloaded {repo_id}")
        _emit_progress(progress, 0.65, "Model download complete.")
    except Exception as exc:
        print(f"[AI Text Reader] Download failed: {exc}")
        raise RuntimeError(f"Failed to download Kokoro model: {exc}")


def _ensure_kokoro_cache_ready():
    """Prepare the Kokoro cache before the main UI is shown."""
    repo_id = "hexgrad/Kokoro-82M"
    repo_cache_path = _repo_cache_path(repo_id)
    yield (
        "Starting startup check...",
        gr.update(visible=True),
        gr.update(visible=False),
    )

    if _is_repo_cached(repo_id):
        yield (
            "Kokoro cache already exists. Opening the main interface...",
            gr.update(visible=False),
            gr.update(visible=True),
        )
        return

    yield (
        "Kokoro cache not found. Downloading model files now...",
        gr.update(visible=True),
        gr.update(visible=False),
    )

    download_error: dict[str, object] = {"exc": None}

    def _download_worker():
        try:
            _ensure_kokoro_assets()
        except Exception as exc:
            download_error["exc"] = exc

    worker = threading.Thread(target=_download_worker, daemon=True)
    worker.start()

    estimated_total = int(os.getenv("AITEXTREADER_MODEL_EST_BYTES", str(3 * 1024 * 1024 * 1024)))
    existing_bytes = _directory_size(repo_cache_path)
    if existing_bytes > 0:
        yield (
            (
                "Found partial Kokoro cache. "
                f"Resuming from {_format_size(existing_bytes)} / ~{_format_size(estimated_total)}."
            ),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    start_time = time.monotonic()
    spinner = ["|", "/", "-", "\\"]
    last_size_check = 0.0
    model_bytes = existing_bytes
    while worker.is_alive():
        now = time.monotonic()
        elapsed = now - start_time
        if now - last_size_check >= 3.0:
            model_bytes = _directory_size(repo_cache_path)
            last_size_check = now

        byte_ratio = min(1.0, model_bytes / estimated_total) if estimated_total else 0.0
        approx_pct = int(min(99, max(0, byte_ratio * 100)))
        spin = spinner[int(elapsed * 4) % len(spinner)]
        status = (
            f"{spin} Downloading Kokoro model... {_format_size(model_bytes)} / ~{_format_size(estimated_total)} "
            f"cached (~{approx_pct}%). Elapsed {int(elapsed)}s. Please keep this tab open."
        )
        yield (
            status,
            gr.update(visible=True),
            gr.update(visible=False),
        )
        time.sleep(0.35)

    if download_error["exc"] is not None:
        err = download_error["exc"]
        print(f"[AI Text Reader] Startup download failed: {err}")
        yield (
            f"Startup download failed: {err}. Check terminal logs and refresh to retry.",
            gr.update(visible=True),
            gr.update(visible=False),
        )
        return

    yield (
        "Model download finished. Finalizing startup...",
        gr.update(visible=True),
        gr.update(visible=False),
    )

    yield (
        "Kokoro cache is ready. The main interface is now available.",
        gr.update(visible=False),
        gr.update(visible=True),
    )


def _status_message(active: bool, detail: str = "") -> str:
    timeout_min = max(1, ENGINE_IDLE_TIMEOUT_SEC // 60)
    base = (
        "Engine status: ACTIVE."
        if active
        else f"Engine status: INACTIVE. Click 'Activate Engine' before running TTS."
    )
    timeout_text = f" Auto-deactivates after {timeout_min} minute(s) of inactivity."
    return f"{base}{timeout_text} {detail}".strip()


def _deactivate_engine_internal(detail: str = ""):
    global _pipeline, _pipeline_lang, _engine_active, _last_activity_monotonic
    _pipeline = None
    _pipeline_lang = None
    _engine_active = False
    _last_activity_monotonic = 0.0
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return _status_message(False, detail)


def _inactive_ui_updates(detail: str = ""):
    return (
        _status_message(False, detail),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def _active_ui_updates(detail: str = ""):
    return (
        _status_message(True, detail),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def _expire_if_inactive() -> bool:
    if _activation_in_progress:
        # Don't timeout during activation.
        return False
    if not _engine_active or _last_activity_monotonic <= 0:
        return False
    elapsed = time.monotonic() - _last_activity_monotonic
    if elapsed > ENGINE_IDLE_TIMEOUT_SEC:
        _deactivate_engine_internal("Timed out due to inactivity.")
        return True
    return False


def _ensure_engine_active():
    if _expire_if_inactive() or not _engine_active:
        raise gr.Error(
            _status_message(False, "Please click 'Activate Engine' and wait for ready state.")
        )
    _touch_activity()


def activate_engine(language: str, voice_label: str, progress=gr.Progress(track_tqdm=True)):
    """Load model into memory and enable TTS controls."""
    global _engine_active, _activation_in_progress
    _activation_in_progress = True
    _touch_activity()
    try:
        lang_code = LANGUAGES[language]
        voice_id = _label_to_id(voice_label)
        
        print(f"[AI Text Reader] Activation started for language={language}, voice={voice_id}")

        _ensure_kokoro_assets(progress)
        print(f"[AI Text Reader] Asset download complete")
        _touch_activity()

        progress(0.72, desc="Initializing model in memory...")
        print(f"[AI Text Reader] Loading KPipeline for language {lang_code}...")
        pipeline = get_pipeline(lang_code)
        print(f"[AI Text Reader] KPipeline loaded successfully")
        _touch_activity()

        # Warm up selected voice once so first real request feels responsive.
        progress(0.86, desc=f"Preparing voice '{voice_id}' (downloading voice model if needed)...")
        print(f"[AI Text Reader] Warming up voice {voice_id}...")
        try:
            audio_chunks = list(
                pipeline("Ready.", voice=voice_id, speed=1.0, split_pattern=r'\n+')
            )
            print(f"[AI Text Reader] Voice warmup generated {len(audio_chunks)} audio chunk(s)")
        except Exception as warmup_exc:
            print(f"[AI Text Reader] Voice warmup failed (non-blocking): {warmup_exc}")
        _touch_activity()

        _engine_active = True
        _touch_activity()
        progress(1.0, desc="Engine ready.")
        print(f"[AI Text Reader] Activation complete")
        return _active_ui_updates("Model loaded and ready.")
    except Exception as exc:
        print(f"[AI Text Reader] Activation failed: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        _deactivate_engine_internal("Activation failed.")
        error_msg = f"Activation failed: {exc}"
        print(f"[AI Text Reader] Returning error to UI: {error_msg}")
        return _inactive_ui_updates(error_msg)
    finally:
        _activation_in_progress = False


def begin_activation():
    """Immediate UI feedback while heavy model activation runs."""
    global _activation_in_progress
    print("[AI Text Reader] User clicked Activate Engine")
    _activation_in_progress = True
    _touch_activity()
    return (
        "Engine status: ACTIVATING... Please wait while the model is checked, downloaded if needed, and prepared.",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def deactivate_engine():
    """Unload model from memory and disable TTS controls."""
    _deactivate_engine_internal("Model unloaded.")
    return _inactive_ui_updates("Model unloaded.")


def sync_engine_timeout_state():
    """Background heartbeat that keeps UI in sync with timeout state."""
    if _expire_if_inactive():
        return _inactive_ui_updates("Timed out due to inactivity.")
    if _engine_active:
        return _active_ui_updates()
    return _inactive_ui_updates()


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
    _ensure_engine_active()
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
    _record_voice_usage(voice_id)

    audio = _generate_full_audio(text, lang_code, voice_id, speed, volume)
    if len(audio) == 0:
        raise gr.Error("No audio was generated. Try different text.")

    # Save to temp WAV and return filepath
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    _save_wav(audio, tmp.name)
    _touch_activity()
    return tmp.name, _status_message(True, "Last action: play.")


def compile_tts(text, language, voice_label, speed, volume):
    """Generate ALL speech, save as MP3, return file for download."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text first.")

    lang_code = LANGUAGES[language]
    voice_id = _label_to_id(voice_label)
    _record_voice_usage(voice_id)

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
        _touch_activity()
        return mp3_path, _status_message(True, "Last action: compile MP3.")
    except Exception as exc:
        print(f"[AI Text Reader] MP3 conversion failed ({exc}); returning WAV instead.")
        _touch_activity()
        return wav_path, _status_message(True, "Last action: compile WAV fallback.")


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

        loading_panel = gr.Column(visible=True)
        main_panel = gr.Column(visible=False)

        with loading_panel:
            loading_status = gr.Markdown("## Preparing model cache...\nThe app is checking for Kokoro files and downloading them if needed.")
            gr.Markdown(
                "If this is the first run, the download can take several minutes. "
                "Leave this tab open until the main interface appears."
            )

        with main_panel:
            engine_status = gr.Markdown(_status_message(False))

            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Text to Read",
                        placeholder="Type or paste your text here…",
                        lines=10,
                        max_lines=30,
                        interactive=False,
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
                activate_btn = gr.Button("Activate Engine", variant="primary")
                deactivate_btn = gr.Button("Deactivate Engine", variant="secondary", interactive=False)

            with gr.Row():
                play_btn = gr.Button("▶  Play", variant="primary", size="lg", interactive=False)
                compile_btn = gr.Button("⚙  Compile to MP3", variant="secondary", size="lg", interactive=False)

            audio_output = gr.Audio(
                label="Audio Player",
                type="filepath",
                autoplay=True,
            )
            file_output = gr.File(label="Download compiled audio")

            activate_btn.click(
                fn=begin_activation,
                inputs=[],
                outputs=[engine_status, play_btn, compile_btn, activate_btn, deactivate_btn, text_input],
                queue=False,
            ).then(
                fn=activate_engine,
                inputs=[language, voice],
                outputs=[engine_status, play_btn, compile_btn, activate_btn, deactivate_btn, text_input],
            )
            deactivate_btn.click(
                fn=deactivate_engine,
                inputs=[],
                outputs=[engine_status, play_btn, compile_btn, activate_btn, deactivate_btn, text_input],
            )

            if hasattr(gr, "Timer"):
                heartbeat = gr.Timer(value=5.0, active=True)
                heartbeat.tick(
                    fn=sync_engine_timeout_state,
                    inputs=[],
                    outputs=[engine_status, play_btn, compile_btn, activate_btn, deactivate_btn, text_input],
                    show_progress="hidden",
                )

            play_btn.click(
                fn=play_tts,
                inputs=[text_input, language, voice, speed, volume],
                outputs=[audio_output, engine_status],
            )
            compile_btn.click(
                fn=compile_tts,
                inputs=[text_input, language, voice, speed, volume],
                outputs=[file_output, engine_status],
            )

        app.load(
            fn=_ensure_kokoro_cache_ready,
            inputs=[],
            outputs=[loading_status, loading_panel, main_panel],
            show_progress="hidden",
        )

    app.queue(default_concurrency_limit=1)
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
