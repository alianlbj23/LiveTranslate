"""Where the benchmark keeps its data. LIVETRANSLATE_BENCH_DIR moves bench_clips/ and
bench_results/ elsewhere (tests point it at a temp folder)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    override = os.environ.get("LIVETRANSLATE_BENCH_DIR")
    return Path(override) if override else ROOT


def clips_dir() -> Path:
    return data_root() / "bench_clips"


def results_dir() -> Path:
    return data_root() / "bench_results"


def live_dir() -> Path:
    return clips_dir() / "live"


SETTINGS_FILE = ROOT / "user_settings.json"
CONFIG_FILE = ROOT / "config.yaml"
SELECTED_MODEL_FILE = ROOT / "launcher" / "selected_model.txt"
