"""One QApplication for the whole test session.

Several test modules create a QApplication in a module-scoped fixture. When such a
module finishes, its application object can be garbage-collected and the next module
creates a new one; qfluentwidgets' global `qconfig` does not survive that, so any later
ControlPanel failed with "wrapped C/C++ object of type QConfig has been deleted"
depending on test order. Creating the application here, before any test module is
imported, and keeping a reference makes every `QApplication.instance()` return it."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

# --- Never touch the real user_settings.json (see tests/test_settings_isolation.py) ---
# A ControlPanel created by a test starts a 300 ms auto-save timer; if it fires after a
# test restored its patched _save_settings, it writes the real file. Point the settings
# path at a sandbox for the whole session and deliberately never restore it, so late
# timers land in the sandbox too. The run fails if the real file changed anyway.
import hashlib  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import control_panel as _control_panel  # noqa: E402

_REAL_SETTINGS = _ROOT / "user_settings.json"


def _digest(path: Path):
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else None
    except OSError as e:
        return f"unreadable: {e}"


_REAL_DIGEST = _digest(_REAL_SETTINGS)
_SANDBOX_DIR = Path(tempfile.mkdtemp(prefix="livetranslate_tests_"))
_control_panel.SETTINGS_FILE = _SANDBOX_DIR / "user_settings.json"

import local_models as _local_models  # noqa: E402

# Switching models in the app writes launcher/selected_model.txt; keep tests off it too
_REAL_SELECTED = _ROOT / "launcher" / "selected_model.txt"
_REAL_SELECTED_DIGEST = _digest(_REAL_SELECTED)
_local_models.SELECTED_MODEL_FILE = _SANDBOX_DIR / "selected_model.txt"

# The benchmark (bench/) reads and, when applying a recommendation, writes the same two
# files; its clips / results folders go to the sandbox too
import bench.paths as _bench_paths  # noqa: E402

_bench_paths.SETTINGS_FILE = _SANDBOX_DIR / "user_settings.json"
_bench_paths.SELECTED_MODEL_FILE = _SANDBOX_DIR / "selected_model.txt"
os.environ.setdefault("LIVETRANSLATE_BENCH_DIR", str(_SANDBOX_DIR / "bench"))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_llm_servers(monkeypatch):
    """A benchmark run unloads every local model; a test that reached the user's real
    Ollama / LM Studio would empty their GPU. Real servers are refused and the test fails
    afterwards, even when the code under test caught and logged the refusal."""
    from bench import llm_servers

    reached = []
    real_ollama = llm_servers.Ollama

    class GuardedOllama(real_ollama):
        def __init__(self, root_url, http=None):
            if http is None:
                reached.append(f"Ollama({root_url})")
                raise AssertionError(f"test reached the real Ollama at {root_url}")
            super().__init__(root_url, http=http)

    def refuse(name):
        def call(*args, **kwargs):
            reached.append(name)
            raise AssertionError(f"test called the real llm_servers.{name}")
        return call

    monkeypatch.setattr(llm_servers, "Ollama", GuardedOllama)
    for name in ("lmstudio_unload_all", "start_fast_ollama", "stop_tray_ollama", "stop_ollama"):
        monkeypatch.setattr(llm_servers, name, refuse(name))
    yield
    assert not reached, f"test reached real LLM servers: {reached}; inject fakes (Deps / monkeypatch)"


def pytest_sessionfinish(session, exitstatus):
    if _digest(_REAL_SETTINGS) != _REAL_DIGEST:
        print(f"\nERROR: the test run modified the real {_REAL_SETTINGS}")
        session.exitstatus = 1
    if _digest(_REAL_SELECTED) != _REAL_SELECTED_DIGEST:
        print(f"\nERROR: the test run modified the real {_REAL_SELECTED}")
        session.exitstatus = 1
    shutil.rmtree(_SANDBOX_DIR, ignore_errors=True)
