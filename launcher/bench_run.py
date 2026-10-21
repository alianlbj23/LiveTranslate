"""Entry point of the best-combination benchmark (run by the benchmark window,
bench/window.py, with --json; the user presses buttons, never types these arguments).

Usage: python -I bench_run.py status | download | start [--quick] | resume | report | apply"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    # Model caches under ./models (TORCH_HOME, HF_HOME ...) before anything imports torch;
    # ASR worker processes inherit these variables
    import model_manager

    model_manager.apply_cache_env()
    from bench.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":  # required: ASR workers are started with multiprocessing spawn
    sys.exit(main())
