"""Tests must never write the real user_settings.json.

On 2026-10-19 23:59 a ControlPanel built by a test started its 300 ms auto-save timer;
the timer fired after the test had restored the patched _save_settings and wrote the
test's settings over the user's file. tests/conftest.py now points
control_panel.SETTINGS_FILE at a temporary file for the whole session (never restored,
so late timers land there too) and fails the run if the real file changed."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "user_settings.json"


def _digest(path: Path):
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else None


def test_settings_file_points_at_a_sandbox():
    import control_panel as cp

    assert cp.SETTINGS_FILE.resolve() != REAL.resolve()


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_late_auto_save_lands_in_the_sandbox(qapp, monkeypatch):
    import control_panel as cp
    from PyQt6.QtCore import QEventLoop, QTimer

    if cp.SETTINGS_FILE.resolve() == REAL.resolve():
        pytest.skip("not sandboxed: running this would overwrite the real settings")
    before = _digest(REAL)
    monkeypatch.setattr(cp, "_load_saved_settings", lambda: None)
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    p = cp.ControlPanel(config, saved_settings={"timeout": 33})
    p._timeout_spin.setValue(34)  # starts the real 300 ms auto-save timer
    loop = QEventLoop()
    QTimer.singleShot(700, loop.quit)
    loop.exec()
    assert _digest(REAL) == before, "a test wrote the real user_settings.json"
    assert json.loads(cp.SETTINGS_FILE.read_text(encoding="utf-8"))["timeout"] == 34
