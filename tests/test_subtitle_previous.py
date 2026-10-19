"""Subtitle window: option to hide the previous sentence and show only the newest one
(the sentence being spoken, or the last finalized one when nobody is speaking)."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from subtitle_overlay import DragHandle  # noqa: E402
from subtitle_window import SubtitleWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(qapp, **extra):
    cfg = {"window_width": 600, "window_x": 100, "window_y": 100, "click_through": False, "min_display_ms": 0}
    cfg.update(extra)
    w = SubtitleWindow(cfg)
    w.show()
    qapp.processEvents()
    return w


def _pump(ms=30):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _texts(w):
    return [tw._text for tw in w._text_widgets]


def _live(w, original, translation=None, seq=1):
    w.update_live(original, translation, complete=translation is not None, seq=seq)
    _pump()


def _find(actions, *needles):
    return next(a for a in actions if a.text() and any(n in a.text() for n in needles))


# ---------------------------------------------------------------------------
# SubtitleWindow
# ---------------------------------------------------------------------------

def test_default_shows_previous_sentence_next_to_live_preview(qapp):
    w = _window(qapp)
    assert w.show_previous() is True
    w._insert_sentence("甲です", {"zh": "甲"})
    _live(w, "乙です", "乙")
    assert _texts(w) == ["甲です | 乙です", "甲 | 乙"]


def test_hide_previous_shows_only_the_sentence_being_spoken(qapp):
    w = _window(qapp)
    w._insert_sentence("甲です", {"zh": "甲"})
    assert w.set_show_previous(False) is False
    _live(w, "乙です", "乙")
    assert _texts(w) == ["乙です", "乙"]


def test_hide_previous_without_live_shows_last_final_sentence(qapp):
    w = _window(qapp, sentences=3)
    w.set_show_previous(False)
    for o, tl in (("甲です", "甲"), ("乙です", "乙"), ("丙です", "丙")):
        w._insert_sentence(o, {"zh": tl})
    assert _texts(w) == ["丙です", "丙"]


def test_late_final_of_older_sentence_does_not_push_back_current_one(qapp):
    """Sentence A's final translation lands while B is already being previewed."""
    w = _window(qapp)
    w.set_show_previous(False)
    _live(w, "乙です", "乙")
    w._insert_sentence("甲です", {"zh": "甲"})
    assert _texts(w) == ["乙です", "乙"]


def test_toggle_back_restores_previous_sentence(qapp):
    w = _window(qapp)
    w._insert_sentence("甲です", {"zh": "甲"})
    w.set_show_previous(False)
    _live(w, "乙です", "乙")
    assert w.set_show_previous(True) is True
    assert _texts(w) == ["甲です | 乙です", "甲 | 乙"]


def test_setting_from_config_and_apply_settings(qapp):
    w = _window(qapp, show_previous=False)
    assert w.show_previous() is False
    w.apply_settings({"window_width": 500, "show_previous": True})
    assert w.show_previous() is True


def test_invalid_config_values_fall_back_to_default(qapp):
    """Error path: user_settings.json is external input; non-bool values must not
    silently flip the option (a string "false" is truthy)."""
    assert _window(qapp, show_previous="false").show_previous() is False
    assert _window(qapp, show_previous="true").show_previous() is True
    assert _window(qapp, show_previous=None).show_previous() is True
    assert _window(qapp, show_previous=[1]).show_previous() is True
    w = _window(qapp)
    assert w.set_show_previous("maybe") is True  # unknown value: keep current state
    assert w.set_show_previous(0) is False


def test_context_menu_toggle_emits_request(qapp):
    w = _window(qapp)
    got = []
    w.show_previous_requested.connect(got.append)
    act = _find(w._build_context_menu().actions(), "前一句", "上一句", "previous")
    assert act.isCheckable() and act.isChecked()
    act.trigger()
    assert got == [False]


# ---------------------------------------------------------------------------
# Overlay header: 字幕 dropdown
# ---------------------------------------------------------------------------

def test_drag_handle_show_previous_signal_and_sync(qapp):
    h = DragHandle()
    got = []
    h.subtitle_show_previous_toggled.connect(got.append)
    assert h._subtitle_show_prev_action.isChecked() is True
    h._subtitle_show_prev_action.trigger()
    assert got == [False]
    h.set_subtitle_show_previous(True)  # external sync must not re-emit
    assert got == [False]
    assert h._subtitle_show_prev_action.isChecked() is True
