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


def pytest_sessionfinish(session, exitstatus):
    if _digest(_REAL_SETTINGS) != _REAL_DIGEST:
        print(f"\nERROR: the test run modified the real {_REAL_SETTINGS}")
        session.exitstatus = 1
    shutil.rmtree(_SANDBOX_DIR, ignore_errors=True)
