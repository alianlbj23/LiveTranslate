"""Opens the best-combination benchmark window (bench/window.py). Started by LiveTranslate
(Settings → Home → Maintenance) with pythonw, so no console window appears.

Usage: pythonw -I launcher/bench_window.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from bench.window import main as window_main

    return window_main()


if __name__ == "__main__":
    sys.exit(main())
