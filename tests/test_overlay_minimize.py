"""Overlay minimize (subtitle_overlay.py): a header button minimizes the window to the
Windows taskbar, and a taskbar click, a left click on the tray icon or the tray menu
brings it back. A Tool window has no taskbar button, so while the Taskbar toggle is off
the window is a normal window only while it is minimized."""
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import subtitle_overlay as so  # noqa: E402

MASK = Qt.WindowType.WindowType_Mask
MIN_HINT = Qt.WindowType.WindowMinimizeButtonHint


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ov(qapp):
    w = so.SubtitleOverlay({})
    w.show()
    qapp.processEvents()
    yield w
    w.close()
    w.deleteLater()
    qapp.processEvents()


def _is_tool(w) -> bool:
    return (w.windowFlags() & MASK) == Qt.WindowType.Tool


def _settle(qapp):
    for _ in range(3):
        qapp.processEvents()


def test_header_has_a_minimize_button(qapp):
    h = so.DragHandle()
    got = []
    h.minimize_clicked.connect(lambda: got.append("min"))
    assert h._minimize_btn.toolTip().strip() and not h._minimize_btn.icon().isNull()
    h._minimize_btn.click()
    assert got == ["min"]


def test_minimize_button_puts_the_window_on_the_taskbar(qapp, ov):
    assert _is_tool(ov)  # no taskbar button by default
    topmost = bool(ov.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    ov._handle._minimize_btn.click()
    _settle(qapp)
    assert ov.isMinimized() and ov.isVisible()
    assert not _is_tool(ov), "a Tool window has no taskbar button to bring it back"
    assert ov.windowFlags() & MIN_HINT, "Windows does not iconify a frameless window without it"
    assert bool(ov.windowFlags() & Qt.WindowType.WindowStaysOnTopHint) == topmost


def test_restored_from_the_taskbar_drops_the_taskbar_button_again(qapp, ov):
    geo = ov.geometry()
    ov.minimize_to_taskbar()
    _settle(qapp)
    ov.showNormal()  # what a click on the taskbar button does
    _settle(qapp)
    assert not ov.isMinimized() and ov.isVisible()
    assert _is_tool(ov) and not ov.windowFlags() & MIN_HINT
    assert ov.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert ov.geometry() == geo


def test_taskbar_toggle_on_keeps_the_taskbar_button_after_restore(qapp, ov):
    ov._handle._taskbar_check.setChecked(True)
    _settle(qapp)
    ov.minimize_to_taskbar()
    _settle(qapp)
    assert ov.isMinimized()
    ov.restore_window()
    _settle(qapp)
    assert not ov.isMinimized() and not _is_tool(ov)


def test_taskbar_toggle_changed_while_minimized_applies_on_restore(qapp, ov):
    ov.minimize_to_taskbar()
    _settle(qapp)
    ov._handle._taskbar_check.setChecked(True)  # e.g. from the tray menu
    _settle(qapp)
    assert ov.isMinimized(), "changing the toggle must not pop the window back up"
    ov.restore_window()
    _settle(qapp)
    assert not _is_tool(ov)
    # and back: turned off while minimized -> Tool again once restored
    ov.minimize_to_taskbar()
    _settle(qapp)
    ov._handle._taskbar_check.setChecked(False)
    _settle(qapp)
    assert ov.isMinimized()
    ov.restore_window()
    _settle(qapp)
    assert _is_tool(ov)


def _tray_click(ov):
    ov._last_tray_toggle = None  # as if the previous click was long ago
    ov.toggle_from_tray()


def test_tray_left_click_toggles(qapp, ov):
    _tray_click(ov)
    _settle(qapp)
    assert ov.isMinimized()
    _tray_click(ov)
    _settle(qapp)
    assert not ov.isMinimized() and ov.isVisible() and _is_tool(ov)


def test_hidden_window_comes_back_from_the_tray(qapp, ov):
    ov.hide()
    _tray_click(ov)
    _settle(qapp)
    assert ov.isVisible() and not ov.isMinimized()
    # hidden while minimized (tray menu "hide"), then shown again: not left minimized
    ov.minimize_to_taskbar()
    _settle(qapp)
    ov.hide()
    ov.restore_window()
    _settle(qapp)
    assert ov.isVisible() and not ov.isMinimized() and _is_tool(ov)


def test_tray_double_click_toggles_once(qapp, ov):
    _tray_click(ov)
    ov.toggle_from_tray()  # second click of a double click (Qt may report it as another Trigger)
    _settle(qapp)
    assert ov.isMinimized()


def test_taskbar_toggled_right_after_restore_wins(qapp, ov):
    """The Tool flag goes back on a zero-delay timer after a restore; a Taskbar toggle in between wins."""
    ov.minimize_to_taskbar()
    _settle(qapp)
    ov.showNormal()
    ov._handle._taskbar_check.setChecked(True)  # before the timer fires
    _settle(qapp)
    assert not _is_tool(ov) and ov._handle._taskbar_check.isChecked()


def test_minimizing_twice_needs_one_restore(qapp, ov):
    ov.minimize_to_taskbar()
    ov.minimize_to_taskbar()
    _settle(qapp)
    ov.restore_window()
    _settle(qapp)
    assert not ov.isMinimized() and _is_tool(ov)


def test_tray_icon_click_and_menu_use_the_overlay_restore():
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "tray.activated.connect(" in main
    assert "ActivationReason.Trigger" in main and "overlay.toggle_from_tray()" in main
    toggle = main[main.index("def on_toggle_overlay"):main.index("overlay_toggle_action.triggered.connect")]
    assert "overlay.restore_window()" in toggle  # a minimized and then hidden window is not shown minimized


@pytest.mark.parametrize("lang", ["zh_TW", "zh", "en"])
def test_texts_for_minimize_and_tray_click(lang):
    text = (ROOT / "i18n" / f"{lang}.yaml").read_text(encoding="utf-8")
    assert re.search(r"^ovl_tip_minimize: \"[^\"]+\"$", text, re.M)
    hint = re.search(r"^hide_tray_hint: \"([^\"]+)\"$", text, re.M).group(1)
    assert not re.search(r"右鍵|右键|Right-click", hint)  # a left click brings it back now
    assert re.search(r"點|单击|Click", hint)


def test_minimized_window_does_not_repaint(qapp, ov):
    """Bug (2026-10-21): the overlay is a layered (translucent) window, and every repaint while
    minimized (the live MIC / RMS / VAD bars) drew it back at its old place on screen."""
    ov.minimize_to_taskbar()
    _settle(qapp)
    assert not ov.updatesEnabled()
    ov.restore_window()
    _settle(qapp)
    assert ov.updatesEnabled()


def test_minimized_from_outside_does_not_repaint_either(qapp, ov):
    """With the Taskbar toggle on, a click on the taskbar button minimizes the window without
    the header button; the paint stop must still apply."""
    ov._handle._taskbar_check.setChecked(True)
    _settle(qapp)
    ov.showMinimized()  # what Windows does on a taskbar click
    _settle(qapp)
    assert not ov.updatesEnabled()
    ov.showNormal()
    _settle(qapp)
    assert ov.updatesEnabled()
