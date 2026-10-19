"""The LiveTranslate overlay can be stretched up and down: invisible handles along the
top and bottom edges of the card (in the shadow margin) resize the window vertically."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


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
# overlay (LiveTranslate window)
# ---------------------------------------------------------------------------

def _overlay(qapp):
    import subtitle_overlay as so

    ov = so.SubtitleOverlay({})
    ov.setGeometry(300, 200, 620, 480)
    ov.show()
    qapp.processEvents()
    return ov


def test_overlay_has_top_and_bottom_resize_handles(qapp):
    ov = _overlay(qapp)
    card = ov._container.geometry()
    for handle, edge in ((ov._top_resize, Qt.Edge.TopEdge), (ov._bottom_resize, Qt.Edge.BottomEdge)):
        assert handle.cursor().shape() == Qt.CursorShape.SizeVerCursor
        assert handle.edge() == edge
        g = handle.geometry()
        assert g.width() >= card.width() - 40
    assert ov._top_resize.geometry().top() == 0
    assert ov._bottom_resize.geometry().bottom() == ov.height() - 1


def test_overlay_drag_bottom_handle_grows_downwards(qapp):
    ov = _overlay(qapp)
    y0, h0 = ov.y(), ov.height()
    hd = ov._bottom_resize
    _drag(hd, (hd.width() // 2, hd.height() // 2), 0, 100)
    assert ov.height() == h0 + 100 and ov.y() == y0


def test_overlay_drag_top_handle_grows_upwards(qapp):
    ov = _overlay(qapp)
    bottom0, h0 = ov.y() + ov.height(), ov.height()
    hd = ov._top_resize
    _drag(hd, (hd.width() // 2, hd.height() // 2), 0, -60)
    assert ov.height() == h0 + 60 and ov.y() + ov.height() == bottom0


def test_overlay_cannot_shrink_below_minimum_height(qapp):
    ov = _overlay(qapp)
    hd = ov._bottom_resize
    _drag(hd, (hd.width() // 2, hd.height() // 2), 0, -2000)
    assert ov.height() == ov.minimumHeight()


def test_overlay_handles_follow_the_window_size(qapp):
    ov = _overlay(qapp)
    ov.resize(700, 560)
    qapp.processEvents()
    assert ov._bottom_resize.geometry().bottom() == ov.height() - 1
    assert ov._top_resize.geometry().width() >= ov._container.width() - 40
