# AI Text Reader

A Gradio-based text-to-speech app powered by Kokoro-82M.

## Prerequisites

- Linux/macOS/Windows
- Python 3.10+
- ffmpeg installed and available on PATH (required for MP3 export)

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python main.py
```

The app opens in your browser by default.

## Engine Mode (Recommended UX)

- TTS is disabled at startup.
- Click `Activate Engine` to preload the model into memory.
- During activation, a real progress bar is shown for cache check, download, model init, and voice prep.
- **First run:** Activation may take 2–5 minutes as Kokoro model files (2–3 GB) are downloaded from Hugging Face Hub and cached locally.
- **Subsequent runs:** Activation is much faster (10–30 seconds) when all files are cached.
- While loading, UI shows `ACTIVATING...` and action controls are disabled.
- Once active, `Play` and `Compile to MP3` become available.
- If idle for 5 minutes, the engine auto-deactivates and unloads from memory.
- You can also click `Deactivate Engine` manually any time.
- **Monitor terminal output** to see detailed download and activation logs (useful for troubleshooting).

Timeout behavior details:

- A heartbeat checks engine activity every 5 seconds.
- If idle time exceeds `AITEXTREADER_IDLE_TIMEOUT_SEC`, the engine unloads automatically.
- UI controls are synced to inactive state after timeout.
- **Note:** Activation never times out, even on slow connections or first-run downloads.

## Runtime Configuration

Set these environment variables as needed:

- `AITEXTREADER_SHARE`:
  - `0` or `false` (default): local-only Gradio session
  - `1` or `true`: request a public Gradio share link
- `AITEXTREADER_INBROWSER`:
  - `1` or `true` (default): open browser automatically
  - `0` or `false`: do not auto-open browser
- `AITEXTREADER_IDLE_TIMEOUT_SEC`:
  - default: `300` (5 minutes)
  - set a different inactivity timeout in seconds
- `AITEXTREADER_USERNAME` and `AITEXTREADER_PASSWORD`:
  - if both are set, Gradio auth is enabled

Example:

```bash
AITEXTREADER_SHARE=1 AITEXTREADER_IDLE_TIMEOUT_SEC=600 AITEXTREADER_USERNAME=user AITEXTREADER_PASSWORD=change-me python main.py
```

## Storage & Cache Management

### Why is the cache so large? (~7GB on first run)

- **Kokoro-82M model weights**: ~2–3 GB (core neural network)
- **Hugging Face Hub metadata & indices**: ~1–2 GB (caching manifest, snapshots)
- **PyTorch & CUDA runtime libraries**: ~1–2 GB (dependencies, if GPU used)
- **Partial/resumed downloads**: Small amount if downloads were interrupted

The Kokoro model is downloaded once from Hugging Face Hub and cached locally. Subsequent runs only download the model once.

### Storage Inspection

To see a detailed breakdown of what's taking up space:

```bash
python inspect_cache.py
```

This shows:
- Total cache size by directory
- Recent large downloads
- What percentage is Kokoro vs. other tools

### Model Cache Manager  

To remove or inspect Kokoro model storage:

```bash
python model_manager.py
```

**Important:** All voices (af_heart, am_adam, bf_alice, etc.) are part of the same Kokoro model download and cannot be removed individually. You can either:

- **Keep the full model** (~2–3 GB): Recommended if you use the app regularly. Fast activation on subsequent runs.
- **Remove the full cache**: Use option 4 in the manager. This frees 2–3 GB. You'll need to re-download (~1–5 minutes) on next activation.

Voice usage is tracked in `.aitextreader/voice_usage.json` for reference only.

## Notes

- If `ffmpeg` is not installed, MP3 compile will fall back to WAV output.
- GPU acceleration is used automatically when CUDA is available.
