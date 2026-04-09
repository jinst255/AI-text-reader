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

## Runtime Configuration

Set these environment variables as needed:

- `AITEXTREADER_SHARE`:
  - `0` or `false` (default): local-only Gradio session
  - `1` or `true`: request a public Gradio share link
- `AITEXTREADER_INBROWSER`:
  - `1` or `true` (default): open browser automatically
  - `0` or `false`: do not auto-open browser
- `AITEXTREADER_USERNAME` and `AITEXTREADER_PASSWORD`:
  - if both are set, Gradio auth is enabled

Example:

```bash
AITEXTREADER_SHARE=1 AITEXTREADER_USERNAME=user AITEXTREADER_PASSWORD=change-me python main.py
```

## Notes

- If `ffmpeg` is not installed, MP3 compile will fall back to WAV output.
- GPU acceleration is used automatically when CUDA is available.
