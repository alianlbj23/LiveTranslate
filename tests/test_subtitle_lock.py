"""Subtitle window: separate "lock position" and "click-through" options, reachable from
the overlay's 字幕 dropdown and the window's right-click menu."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from subtitle_overlay import DragHandle  # noqa: E402
from subtitle_window import SubtitleWindow  # noqa: E402

L = Qt.MouseButton.LeftButton
NB = Qt.MouseButton.NoButton


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _ev(t, lx, ly, gx, gy, btn, btns):
    return QMouseEvent(t, QPointF(lx, ly), QPointF(gx, gy), btn, btns, Qt.KeyboardModifier.NoModifier)


def _drag(w, dx):
    w.mousePressEvent(_ev(QEvent.Type.MouseButtonPress, 100, 10, w.x() + 100, w.y() + 10, L, L))
    w.mouseMoveEvent(_ev(QEvent.Type.MouseMove, 100 + dx, 10, w.x() + 100 + dx, w.y() + 10, NB, L))
    w.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease, 100 + dx, 10, w.x() + 100 + dx, w.y() + 10, L, NB))


def _resize(w, dx):
    gx = w.x() + w.width() - 5
    w.mousePressEvent(_ev(QEvent.Type.MouseButtonPress, w.width() - 5, 10, gx, w.y() + 10, L, L))
    w.mouseMoveEvent(_ev(QEvent.Type.MouseMove, w.width() - 5 + dx, 10, gx + dx, w.y() + 10, NB, L))
    w.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease, w.width() - 5 + dx, 10, gx + dx, w.y() + 10, L, NB))


def _window(qapp, **extra):
    cfg = {"window_width": 600, "window_x": 100, "window_y": 100, "click_through": False}
    cfg.update(extra)
    w = SubtitleWindow(cfg)
    w.show()
    qapp.processEvents()
    return w


# ---------------------------------------------------------------------------
# SubtitleWindow: lock position
# ---------------------------------------------------------------------------

def test_locked_window_ignores_drag_and_resize(qapp):
    w = _window(qapp)
    w.set_locked(True)
    assert w.is_locked() is True
    assert w.is_click_through() is False  # lock is independent of click-through
    x0, width0 = w.x(), w.width()
    _drag(w, 50)
    _resize(w, -200)
    assert (w.x(), w.width()) == (x0, width0)


def test_unlocked_window_moves_and_resizes(qapp):
    w = _window(qapp)
    w.set_locked(False)
    x0, width0 = w.x(), w.width()
    _drag(w, 50)
    _resize(w, -200)
    assert w.x() == x0 + 50
    assert w.width() == width0 - 200


def test_locked_from_settings_and_apply_settings(qapp):
    w = _window(qapp, locked=True)
    assert w.is_locked() is True
    w.apply_settings({"window_width": 600, "locked": False})
    assert w.is_locked() is False


def test_double_click_does_not_lock_when_position_locked(qapp):
    w = _window(qapp)
    w.set_locked(True)
    w.mouseDoubleClickEvent(_ev(QEvent.Type.MouseButtonDblClick, 100, 10, 200, 110, L, L))
    assert w.is_click_through() is False


def test_context_menu_actions_emit_signals(qapp):
    w = _window(qapp)
    got = []
    w.lock_requested.connect(lambda v: got.append(("lock", v)))
    w.click_through_requested.connect(lambda v: got.append(("ct", v)))
    w.settings_requested.connect(lambda: got.append(("settings", None)))
    menu = w._build_context_menu()
    actions = {a.text(): a for a in menu.actions() if a.text()}
    lock_act = next(a for t, a in actions.items() if "固定" in t or "Lock" in t)
    ct_act = next(a for t, a in actions.items() if "穿透" in t or "lick-through" in t)
    set_act = next(a for t, a in actions.items() if "設定" in t or "设置" in t or "ettings" in t)
    assert lock_act.isCheckable() and lock_act.isChecked() is False
    lock_act.trigger()
    ct_act.trigger()
    set_act.trigger()
    assert ("lock", True) in got
    assert ("ct", True) in got
    assert ("settings", None) in got


# ---------------------------------------------------------------------------
# Overlay header: 字幕 dropdown
# ---------------------------------------------------------------------------

def test_drag_handle_subtitle_menu_signals_and_sync(qapp):
    h = DragHandle()
    got = []
    h.subtitle_lock_toggled.connect(lambda v: got.append(("lock", v)))
    h.subtitle_click_through_toggled.connect(lambda v: got.append(("ct", v)))
    h.subtitle_settings_requested.connect(lambda: got.append(("settings", None)))
    h._subtitle_lock_action.trigger()
    h._subtitle_ct_action.trigger()
    h._subtitle_settings_action.trigger()
    assert got == [("lock", True), ("ct", True), ("settings", None)]
    # external sync must not re-emit
    got.clear()
    h.set_subtitle_lock_state(False, False)
    assert got == []
    assert h._subtitle_lock_action.isChecked() is False
    assert h._subtitle_ct_action.isChecked() is False
