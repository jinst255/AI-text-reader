"""
Terminal model manager for AI Text Reader.

Features:
- Inspect Kokoro cache locations and sizes
- Remove cache artifacts for a specific voice ID
- Remove all detected Kokoro model caches
"""

import glob
import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
STATE_DIR = PROJECT_ROOT / ".aitextreader"
VOICE_USAGE_FILE = STATE_DIR / "voice_usage.json"


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(0, num_bytes))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def load_used_voices() -> list[str]:
    if not VOICE_USAGE_FILE.exists():
        return []
    try:
        data = json.loads(VOICE_USAGE_FILE.read_text(encoding="utf-8"))
        voices = data.get("used_voices", [])
        if isinstance(voices, list):
            return [str(v) for v in voices]
    except Exception:
        pass
    return []


def save_used_voices(voices: list[str]):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"used_voices": sorted(set(voices))}
    VOICE_USAGE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def detect_kokoro_paths() -> list[Path]:
    """Find all Kokoro-related cache paths."""
    paths: set[Path] = set()
    home = Path.home()

    # Hugging Face Hub cache for Kokoro model.
    hf_cache = home / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        # Kokoro is stored as models--hexgrad--Kokoro-82M*
        for p in hf_cache.glob("models--hexgrad--Kokoro-82M*"):
            if p.exists():
                paths.add(p)

    # Also check for any kokoro in torch hub.
    torch_hub = home / ".cache" / "torch" / "hub"
    if torch_hub.exists():
        for p in torch_hub.glob("*kokoro*"):
            if p.exists():
                paths.add(p)

    return sorted(paths)


def find_voice_artifacts(voice_id: str, search_roots: list[Path]) -> list[Path]:
    voice = voice_id.lower().strip()
    if not voice:
        return []

    hits: set[Path] = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue

        # Use glob recursively to find files/folders containing the voice id.
        pattern = f"**/*{voice}*"
        for p_str in glob.glob(str(root / pattern), recursive=True):
            p = Path(p_str)
            if p.exists():
                hits.add(p)

    # Prefer deleting files first. Directories can be removed if empty after file deletion.
    files = sorted([p for p in hits if p.is_file()])
    dirs = sorted([p for p in hits if p.is_dir()], key=lambda p: len(p.parts), reverse=True)
    return files + dirs


def delete_paths(paths: list[Path]) -> tuple[int, int]:
    deleted = 0
    reclaimed = 0

    for p in paths:
        if not p.exists():
            continue

        size_before = path_size(p)
        try:
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            else:
                shutil.rmtree(p, ignore_errors=True)
            deleted += 1
            reclaimed += size_before
        except Exception:
            pass

    return deleted, reclaimed


def remove_empty_dirs(paths: list[Path]):
    for p in sorted(paths, key=lambda x: len(x.parts), reverse=True):
        if p.exists() and p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass


def show_cache_summary(kokoro_paths: list[Path]):
    print("\nDetected Kokoro cache locations:")
    if not kokoro_paths:
        print("- None found.")
        return

    total = 0
    for p in kokoro_paths:
        sz = path_size(p)
        total += sz
        print(f"- {p} ({human_size(sz)})")
    print(f"Total detected size: {human_size(total)}")


def choose_voice(used_voices: list[str]) -> str:
    print("\nVoice selection")
    if used_voices:
        for i, v in enumerate(used_voices, start=1):
            print(f"{i}. {v}")
        print("M. Manual voice id")

        raw = input("Choose a used voice index or M: ").strip().lower()
        if raw == "m":
            return input("Enter voice id (example: af_heart): ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(used_voices):
                return used_voices[idx]
    return input("Enter voice id (example: af_heart): ").strip()


def remove_voice_cache(kokoro_paths: list[Path]):
    print("\n⚠️  Individual voice removal")
    print("═" * 60)
    print("Kokoro model stores ALL voices in a single model download.")
    print("Voices cannot be removed individually without removing the")
    print("entire model (~2-3 GB).\n")
    print("Your options:")
    print("  1. Keep the full model (recommended for active use)")
    print("  2. Use option 4 to remove entire Kokoro cache if not needed")
    print("\nChecking model status...\n")
    
    if not kokoro_paths:
        print("✓ No Kokoro model cache detected. Storage is clear.")
        return
    
    total_size = sum(path_size(p) for p in kokoro_paths)
    print(f"Current Kokoro cache: {human_size(total_size)}")
    for p in kokoro_paths:
        sz = path_size(p)
        print(f"  - {p.name}: {human_size(sz)}")
    print("\nTo free this space, choose option 4 (Remove full cache).")


def remove_full_kokoro(kokoro_paths: list[Path]):
    if not kokoro_paths:
        print("\n✓ No Kokoro cache detected. Nothing to remove.")
        return

    total = sum(path_size(p) for p in kokoro_paths)
    print("\n🔴 REMOVE KOKORO MODEL CACHE")
    print("═" * 60)
    print(f"\nThis will delete ALL Kokoro cache files: {human_size(total)}\n")
    print("Paths to be removed:")
    for p in kokoro_paths:
        sz = path_size(p)
        print(f"  - {p} ({human_size(sz)})")

    confirm = input("\nType 'DELETE' to permanently remove these files: ").strip()
    if confirm != "DELETE":
        print("❌ Cancelled.")
        return

    print("\nDeleting...")
    deleted, reclaimed = delete_paths(kokoro_paths)
    remove_empty_dirs(kokoro_paths)
    print(f"✓ Deleted {deleted} path(s). Freed ~{human_size(reclaimed)}.")
    print("\nYou will need to run 'Activate Engine' again to re-download the model.")


def list_used_voices():
    voices = load_used_voices()
    print("\nUsed voices")
    if not voices:
        print("- None recorded yet.")
        return
    for v in voices:
        print(f"- {v}")


def main():
    print("")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  AI Text Reader - Model Cache Manager                         ║")
    print("║  Manage Kokoro TTS model storage                              ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print("")

    while True:
        kokoro_paths = detect_kokoro_paths()
        total_size = sum(path_size(p) for p in kokoro_paths) if kokoro_paths else 0
        
        print("\n📊 MENU")
        print("─" * 60)
        print(f"Current Kokoro cache: {human_size(total_size)}")
        print("\nOptions:")
        print("  1) Show detailed cache breakdown")
        print("  2) View used voices")
        print("  3) Check individual voice status (informational only)")
        print("  4) Remove full Kokoro model cache")
        print("  5) Exit")
        print()

        choice = input("Choose an option [1-5]: ").strip()
        
        if choice == "1":
            show_cache_summary(kokoro_paths)
        elif choice == "2":
            list_used_voices()
        elif choice == "3":
            used = load_used_voices()
            if not used:
                print("\nNo voices recorded as used yet.")
            else:
                print(f"\nVoices used: {', '.join(used)}")
                print("\n(Note: All voices are part of the same model and cannot be")
                print(" removed individually. Use option 4 to remove the entire model.)")
        elif choice == "4":
            remove_full_kokoro(kokoro_paths)
        elif choice == "5":
            print("\nGoodbye.")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
