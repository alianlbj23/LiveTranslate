"""Subtitle window can be stretched up and down: drag its top or bottom edge (or the
bottom-right corner) to set a box height. The box never clips text: if the text needs
more room the window grows, and the text stays vertically centred in a taller box.
Locked windows do not resize."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from subtitle_window import SubtitleWindow  # noqa: E402

L = Qt.MouseButton.LeftButton
NB = Qt.MouseButton.NoButton


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _ev(t, local, glob, btn, btns):
    return QMouseEvent(t, QPointF(*local), QPointF(*glob), btn, btns, Qt.KeyboardModifier.NoModifier)


def _drag(widget, local_xy, dx, dy):
    """Press at local_xy, move by (dx, dy), release — all on `widget`."""
    lx, ly = local_xy
    g = widget.mapToGlobal(widget.rect().topLeft())
    gx, gy = g.x() + lx, g.y() + ly
    widget.mousePressEvent(_ev(QEvent.Type.MouseButtonPress, (lx, ly), (gx, gy), L, L))
    widget.mouseMoveEvent(_ev(QEvent.Type.MouseMove, (lx + dx, ly + dy), (gx + dx, gy + dy), NB, L))
    widget.mouseReleaseEvent(_ev(QEvent.Type.MouseButtonRelease, (lx + dx, ly + dy), (gx + dx, gy + dy), L, NB))


# ---------------------------------------------------------------------------
# subtitle window
# ---------------------------------------------------------------------------

def _subwin(qapp, **extra):
    cfg = {"window_width": 600, "window_x": 200, "window_y": 300, "click_through": False, "min_display_ms": 0}
    cfg.update(extra)
    w = SubtitleWindow(cfg)
    w.show()
    qapp.processEvents()
    w._insert_sentence("こんにちは", {"zh": "你好"})
    w._fit_height_snap()
    qapp.processEvents()
    return w


def test_subtitle_top_and_bottom_edges_are_vertical_resize_zones(qapp):
    w = _subwin(qapp)
    mid = w.width() // 2
    assert w._edge_zone(QPointF(mid, 2).toPoint()) == "top"
    assert w._edge_zone(QPointF(mid, w.height() - 2).toPoint()) == "bottom"
    assert w._edge_zone(QPointF(mid, w.height() // 2).toPoint()) is None
    w._update_cursor(QPointF(mid, w.height() - 2).toPoint())
    assert w.cursor().shape() == Qt.CursorShape.SizeVerCursor


def test_subtitle_drag_bottom_edge_makes_box_taller(qapp):
    w = _subwin(qapp)
    y0, h0 = w.y(), w.height()
    moved = []
    w.position_changed.connect(lambda: moved.append(1))
    _drag(w, (w.width() // 2, h0 - 2), 0, 80)
    assert w.height() == h0 + 80
    assert w.y() == y0, "dragging the bottom edge keeps the top in place"
    assert w.box_height() == h0 + 80
    assert moved, "release must trigger saving"


def test_subtitle_drag_top_edge_keeps_bottom_in_place(qapp):
    w = _subwin(qapp)
    bottom0, h0 = w.y() + w.height(), w.height()
    _drag(w, (w.width() // 2, 2), 0, -50)
    assert w.height() == h0 + 50
    assert w.y() + w.height() == bottom0


def test_subtitle_corner_resizes_width_and_height(qapp):
    w = _subwin(qapp)
    w0, h0 = w.width(), w.height()
    _drag(w, (w.width() - 3, w.height() - 3), 40, 30)
    assert (w.width(), w.height()) == (w0 + 40, h0 + 30)


def test_subtitle_box_never_clips_text(qapp):
    w = _subwin(qapp)
    content_h = w.height()
    _drag(w, (w.width() // 2, w.height() - 2), 0, -(content_h - 5))  # try to squash it
    assert w.height() == content_h


def test_subtitle_text_is_centred_in_a_taller_box(qapp):
    w = _subwin(qapp)
    _drag(w, (w.width() // 2, w.height() - 2), 0, 120)
    h_box = w.height()
    w._insert_sentence("さようなら", {"zh": "再見"})
    qapp.processEvents()
    assert w.height() == h_box, "a new sentence must not shrink the user's box"
    tops = [tw.geometry().top() for tw in w._text_widgets]
    bottoms = [tw.geometry().bottom() for tw in w._text_widgets]
    text_mid = (min(tops) + max(bottoms)) / 2
    assert abs(text_mid - h_box / 2) <= 8


def test_subtitle_locked_window_does_not_resize_vertically(qapp):
    w = _subwin(qapp)
    w.set_locked(True)
    h0 = w.height()
    _drag(w, (w.width() // 2, h0 - 2), 0, 80)
    assert w.height() == h0


def test_subtitle_box_height_from_settings_and_bad_values(qapp):
    w = _subwin(qapp, window_height=220)
    assert w.box_height() == 220 and w.height() == 220
    for bad in ("abc", -40, None, [3], True):
        w2 = _subwin(qapp, window_height=bad)
        assert w2.box_height() == 0, bad  # 0 = fit to text
    w.apply_settings({"window_width": 600, "window_height": 300})
    qapp.processEvents()
    assert w.height() == 300
