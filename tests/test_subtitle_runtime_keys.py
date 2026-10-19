"""Settings owned by the subtitle window at runtime (lock, position, shown lines) are
changed from menus / dragging, not from the subtitle settings page. A change on the
settings page must not send its stale copy of them back: "固定位置" from the
right-click menu, then any edit on the settings page, used to unlock the window."""
import os
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _panel(monkeypatch):
    import control_panel as cp

    monkeypatch.setattr(cp, "_load_saved_settings", lambda: None)
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    saved = {
        "subtitle_mode": {"enabled": True, "window_x": 10, "window_y": 20, "window_width": 800,
                          "locked": False, "click_through": False},
    }
    p = cp.ControlPanel(config, saved_settings=saved)
    monkeypatch.setattr(p, "_auto_save", lambda: None)  # never touch the real user_settings.json
    return p


def _menu_sets(panel, **kv):
    """What main.py's menu handlers do: update the panel's subtitle_mode in place."""
    settings = panel.get_settings()
    sm = settings.get("subtitle_mode") or {}
    sm.update(kv)
    settings["subtitle_mode"] = sm
    panel._current_settings["subtitle_mode"] = sm


def test_settings_page_edit_keeps_lock_from_menu(qapp, monkeypatch):
    p = _panel(monkeypatch)
    got = []
    p.subtitle_settings_changed.connect(lambda s: got.append(dict(s)))
    _menu_sets(p, locked=True)
    p._subtitle_widget._auto_hide_spin.setValue(p._subtitle_widget._auto_hide_spin.value() + 1)
    p._subtitle_widget._emit_settings()  # what the page's debounce timer does
    assert got and got[-1]["locked"] is True
    assert p._current_settings["subtitle_mode"]["locked"] is True


def test_settings_page_edit_keeps_window_position_and_visibility(qapp, monkeypatch):
    p = _panel(monkeypatch)
    got = []
    p.subtitle_settings_changed.connect(lambda s: got.append(dict(s)))
    _menu_sets(p, window_x=-2400, window_y=880, show_original=False, show_previous=False)
    p._subtitle_widget._emit_settings()
    s = got[-1]
    assert (s["window_x"], s["window_y"]) == (-2400, 880)
    assert s["show_original"] is False and s["show_previous"] is False


def test_settings_page_still_controls_its_own_fields(qapp, monkeypatch):
    """Normal path: values edited on the page (width, click-through) do go through."""
    p = _panel(monkeypatch)
    got = []
    p.subtitle_settings_changed.connect(lambda s: got.append(dict(s)))
    _menu_sets(p, locked=True)
    w = p._subtitle_widget
    w._width_spin.setValue(950)
    w._click_through_check.setChecked(True)
    w._emit_settings()
    s = got[-1]
    assert s["window_width"] == 950 and s["click_through"] is True and s["locked"] is True
