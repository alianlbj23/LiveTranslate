"""Overlay minimize (subtitle_overlay.py): the overlay is a normal taskbar window (no Tool flag,
no Taskbar toggle). A header button minimizes it with the system animation, and a taskbar
click, a left click on the tray icon or the tray menu brings it back."""
import ctypes
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6 import sip  # noqa: E402
from PyQt6.QtCore import QByteArray, Qt  # noqa: E402
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


def _settle(qapp):
    for _ in range(3):
        qapp.processEvents()


def _tray_click(ov):
    ov._last_tray_toggle = None  # as if the previous click was long ago
    ov.toggle_from_tray()


def test_header_has_a_minimize_button(qapp):
    h = so.DragHandle()
    got = []
    h.minimize_clicked.connect(lambda: got.append("min"))
    assert h._minimize_btn.toolTip().strip() and not h._minimize_btn.icon().isNull()
    h._minimize_btn.click()
    assert got == ["min"]


def test_overlay_always_has_a_taskbar_button(qapp, ov):
    """Requested 2026-10-21: the app takes a taskbar slot as soon as it opens, no toggle."""
    assert (ov.windowFlags() & MASK) == Qt.WindowType.Window  # a Tool window has no taskbar button
    assert ov.windowFlags() & MIN_HINT
    assert ov.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert not hasattr(ov._handle, "_taskbar_check")
    assert not hasattr(so.DragHandle, "taskbar_toggled")
    assert not hasattr(so.SubtitleOverlay, "_set_taskbar")


def test_top_most_toggle_keeps_the_taskbar_button(qapp, ov):
    ov._handle._topmost_check.setChecked(False)
    _settle(qapp)
    assert (ov.windowFlags() & MASK) == Qt.WindowType.Window and ov.windowFlags() & MIN_HINT
    assert not ov.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    ov._handle._topmost_check.setChecked(True)
    _settle(qapp)
    assert ov.windowFlags() & Qt.WindowType.WindowStaysOnTopHint


def test_minimize_button_minimizes_and_restore_brings_it_back(qapp, ov):
    geo = ov.geometry()
    flags = ov.windowFlags()
    ov._handle._minimize_btn.click()
    _settle(qapp)
    assert ov.isMinimized() and ov.isVisible()
    ov.showNormal()  # what a click on the taskbar button does
    _settle(qapp)
    assert not ov.isMinimized() and ov.isVisible()
    assert ov.windowFlags() == flags and ov.geometry() == geo


def test_minimized_window_does_not_repaint(qapp, ov):
    """Bug (2026-10-21): the overlay is a layered (translucent) window, and every repaint while
    minimized (the live MIC / RMS / VAD bars) drew it back at its old place on screen."""
    ov.minimize_to_taskbar()
    _settle(qapp)
    assert not ov.updatesEnabled()
    ov.restore_window()
    _settle(qapp)
    assert ov.updatesEnabled()


def test_minimized_from_the_taskbar_does_not_repaint_either(qapp, ov):
    ov.showMinimized()  # what Windows does on a click on the taskbar button
    _settle(qapp)
    assert not ov.updatesEnabled()
    ov.showNormal()
    _settle(qapp)
    assert ov.updatesEnabled()


def test_tray_left_click_toggles(qapp, ov):
    _tray_click(ov)
    _settle(qapp)
    assert ov.isMinimized()
    _tray_click(ov)
    _settle(qapp)
    assert not ov.isMinimized() and ov.isVisible()


def test_tray_double_click_toggles_once(qapp, ov):
    _tray_click(ov)
    ov.toggle_from_tray()  # second click of a double click (Qt may report it as another Trigger)
    _settle(qapp)
    assert ov.isMinimized()


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
    assert ov.isVisible() and not ov.isMinimized() and ov.updatesEnabled()


def test_minimizing_twice_needs_one_restore(qapp, ov):
    ov.minimize_to_taskbar()
    ov.minimize_to_taskbar()
    _settle(qapp)
    ov.restore_window()
    _settle(qapp)
    assert not ov.isMinimized() and ov.updatesEnabled()


class _MSG(ctypes.Structure):
    _fields_ = [("hWnd", ctypes.c_void_p), ("message", ctypes.c_uint), ("wParam", ctypes.c_size_t),
                ("lParam", ctypes.c_ssize_t), ("time", ctypes.c_uint), ("pt", ctypes.c_long * 2)]


def _native(ov, message, wparam):
    m = _MSG(None, message, wparam, 0, 0)
    return ov.nativeEvent(QByteArray(b"windows_generic_MSG"), sip.voidptr(ctypes.addressof(m)))


def test_caption_added_for_the_animation_takes_no_room(qapp, ov):
    """Windows animates minimize / restore only for windows with a caption; the caption style
    is added natively and WM_NCCALCSIZE makes the whole window client area."""
    assert tuple(_native(ov, 0x0083, 1)) == (True, 0)  # WM_NCCALCSIZE
    assert tuple(_native(ov, 0x0083, 0))[0] is False
    assert tuple(_native(ov, 0x0200, 0))[0] is False  # WM_MOUSEMOVE: Qt handles it


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 window styles")
def test_animation_styles_are_only_set_on_a_real_window(qapp, ov):
    # offscreen platform: winId() is not an HWND, nothing may be written to it
    calls = []
    real = so._user32

    class Spy:
        def __getattr__(self, name):
            calls.append(name)
            return getattr(real, name)

    so._user32 = Spy()
    try:
        ov._add_window_animation()
    finally:
        so._user32 = real
    assert calls == []


def test_tray_icon_click_and_menu_use_the_overlay_restore():
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "tray.activated.connect(" in main
    assert "ActivationReason.Trigger" in main and "overlay.toggle_from_tray()" in main
    toggle = main[main.index("def on_toggle_overlay"):main.index("overlay_toggle_action.triggered.connect")]
    assert "overlay.restore_window()" in toggle  # a minimized and then hidden window is not shown minimized
    assert "taskbar_action" not in main and "_taskbar_check" not in main


@pytest.mark.parametrize("lang", ["zh_TW", "zh", "en"])
def test_texts_for_minimize_and_tray_click(lang):
    text = (ROOT / "i18n" / f"{lang}.yaml").read_text(encoding="utf-8")
    assert re.search(r"^ovl_tip_minimize: \"[^\"]+\"$", text, re.M)
    hint = re.search(r"^hide_tray_hint: \"([^\"]+)\"$", text, re.M).group(1)
    assert not re.search(r"右鍵|右键|Right-click", hint)  # a left click brings it back now
    assert re.search(r"點|单击|Click", hint)
