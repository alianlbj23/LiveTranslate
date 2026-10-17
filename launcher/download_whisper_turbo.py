"""Download faster-whisper large-v3-turbo into LiveTranslate's ./models cache.

Usage: python -I download_whisper_turbo.py <project_root>
Prints the resolved snapshot directory on the last line.
"""
import os
import sys

REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"  # same repo faster-whisper maps "large-v3-turbo" to


def main() -> int:
    root = sys.argv[1]
    models_dir = os.path.join(root, "models")
    os.environ["HF_HOME"] = os.path.join(models_dir, "huggingface")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        REPO,
        cache_dir=os.path.join(models_dir, "huggingface", "hub"),
        allow_patterns=["*.bin", "*.json", "*.txt"],
    )
    for name in ("model.bin", "config.json"):
        if not os.path.isfile(os.path.join(path, name)):
            print(f"missing {name} in {path}")
            return 1
    print(os.path.abspath(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
