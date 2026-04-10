"""
Diagnostic tool to inspect what's taking up space in cache directories.
Shows breakdown of cache usage by type so you understand space consumption.
"""

import os
from pathlib import Path


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(0, num_bytes))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def dir_size(path: Path) -> int:
    """Calculate total size of directory recursively."""
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


def list_subdirs_by_size(path: Path, limit: int = 10) -> list:
    """List top N subdirectories by size."""
    if not path.exists() or not path.is_dir():
        return []
    
    items = []
    try:
        for item in path.iterdir():
            if item.is_dir():
                sz = dir_size(item)
                items.append((item.name, sz, item))
    except Exception:
        pass
    
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:limit]


def main():
    print("AI Text Reader - Cache Inspector")
    print("Shows what's taking up space in your system cache.\n")

    home = Path.home()
    cache_paths = [
        (home / ".cache", ".cache (system cache)"),
        (home / ".cache" / "huggingface" / "hub", "Hugging Face Hub cache"),
        (home / ".cache" / "torch", "PyTorch cache"),
        (home / ".cache" / "pip", "Pip package cache"),
    ]

    for path, label in cache_paths:
        if not path.exists():
            continue

        total_sz = dir_size(path)
        print(f"\n{label}")
        print(f"  Path: {path}")
        print(f"  Total: {human_size(total_sz)}")
        
        items = list_subdirs_by_size(path, limit=8)
        if items:
            print(f"  Top subdirectories:")
            for name, sz, _ in items:
                print(f"    - {name:<50} {human_size(sz):>12}")

    # Specific Kokoro repo info
    print("\n" + "="*70)
    print("Kokoro-82M Specific")
    print("="*70)
    
    hf_cache = home / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        kokoro_dirs = list(hf_cache.glob("models--hexgrad--Kokoro-82M*"))
        if kokoro_dirs:
            print(f"\nFound {len(kokoro_dirs)} Kokoro model snapshots:")
            for d in kokoro_dirs:
                sz = dir_size(d)
                print(f"  {d.name}: {human_size(sz)}")
                
                # List files inside
                try:
                    files = list(d.rglob("*"))
                    file_list = [f for f in files if f.is_file()]
                    if file_list:
                        print(f"    Contains {len(file_list)} files:")
                        for f in sorted(file_list, key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)[:5]:
                            try:
                                sz = f.stat().st_size
                                print(f"      - {f.name}: {human_size(sz)}")
                            except:
                                pass
                except:
                    pass
        else:
            print("\nNo Kokoro model found in Hugging Face cache yet.")
            print("(Will be created on first Activate Engine click)")

    # PyTorch hub
    print("\n" + "="*70)
    print("PyTorch Hub (if model cached there)")
    print("="*70)
    
    torch_hub = home / ".cache" / "torch" / "hub"
    if torch_hub.exists():
        dirs = list(torch_hub.iterdir())
        print(f"\nPyTorch hub cache: {human_size(dir_size(torch_hub))}")
        for d in list_subdirs_by_size(torch_hub, limit=5):
            print(f"  - {d[0]}: {human_size(d[1])}")
    else:
        print("\nPyTorch hub cache not found.")

    print("\n" + "="*70)
    print("Summary: The 7GB is likely a combination of:")
    print("  1. Kokoro-82M model files (core model weights)")
    print("  2. Hugging Face metadata and cache indices")
    print("  3. PyTorch and dependencies")
    print("  4. Potential duplicate/partial downloads if interrupted")
    print("\nKokoro model core is ~2-3 GB; rest is metadata, indices, and tooling.")
    print("="*70)


if __name__ == "__main__":
    main()
