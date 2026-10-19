"""Subtitle window: choose freely whether the source line, the translation line or both
are shown — from the window's right-click menu and the overlay's 字幕 dropdown."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from subtitle_overlay import DragHandle  # noqa: E402
from subtitle_window import SubtitleWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(qapp, **extra):
    cfg = {"window_width": 600, "window_x": 100, "window_y": 100, "click_through": False}
    cfg.update(extra)
    w = SubtitleWindow(cfg)
    w.show()
    qapp.processEvents()
    return w


def _texts(w):
    return [tw._text for tw in w._text_widgets]


def _find(actions, *needles):
    return next(a for a in actions if a.text() and any(n in a.text() for n in needles))


# ---------------------------------------------------------------------------
# SubtitleWindow
# ---------------------------------------------------------------------------

def test_default_shows_original_and_translation(qapp):
    w = _window(qapp)
    assert w.line_visibility() == (True, True)
    w._insert_sentence("こんにちは", {"zh": "你好"})
    assert _texts(w) == ["こんにちは", "你好"]


def test_hide_one_line_removes_its_widget_and_keeps_text(qapp):
    w = _window(qapp)
    w._insert_sentence("こんにちは", {"zh": "你好"})
    assert w.set_line_visibility(False, True) == (False, True)
    assert _texts(w) == ["你好"]
    assert w.set_line_visibility(True, False) == (True, False)
    assert _texts(w) == ["こんにちは"]
    assert w.set_line_visibility(True, True) == (True, True)
    assert _texts(w) == ["こんにちは", "你好"]


def test_cannot_hide_both_lines(qapp):
    """Error path: hiding everything would leave an empty window; the request is refused."""
    w = _window(qapp)
    w.set_line_visibility(False, True)
    assert w.set_line_visibility(False, False) == (False, True)
    assert w.line_visibility() == (False, True)
    assert len(w._text_widgets) == 1
    # non-bool input is coerced, never raises
    assert w.set_line_visibility("yes", None) == (True, False)


def test_visibility_survives_apply_settings_and_initial_config(qapp):
    w = _window(qapp, show_original=False)
    assert w.line_visibility() == (False, True)
    w.apply_settings({"window_width": 500, "show_original": True, "show_translation": False})
    assert w.line_visibility() == (True, False)
    assert len(w._text_widgets) == 1


def test_context_menu_has_checked_toggles_that_emit_pair(qapp):
    w = _window(qapp)
    got = []
    w.line_visibility_requested.connect(lambda o, t: got.append((o, t)))
    actions = w._build_context_menu().actions()
    orig = _find(actions, "原文", "original")
    trans = _find(actions, "譯文", "译文", "translation")
    assert orig.isCheckable() and orig.isChecked() and trans.isCheckable() and trans.isChecked()
    orig.trigger()
    assert got == [(False, True)]
    w.set_line_visibility(False, True)
    trans = _find(w._build_context_menu().actions(), "譯文", "译文", "translation")
    trans.trigger()  # asks to hide the last visible line; the window decides, not the menu
    assert got[-1] == (False, False)


# ---------------------------------------------------------------------------
# Overlay header: 字幕 dropdown
# ---------------------------------------------------------------------------

def test_drag_handle_line_visibility_signals_and_sync(qapp):
    h = DragHandle()
    got = []
    h.subtitle_line_visibility_toggled.connect(lambda o, t: got.append((o, t)))
    assert h._subtitle_show_orig_action.isChecked() and h._subtitle_show_trans_action.isChecked()
    h._subtitle_show_orig_action.trigger()
    assert got == [(False, True)]
    h.set_subtitle_line_visibility(True, False)  # external sync must not re-emit
    assert got == [(False, True)]
    assert h._subtitle_show_orig_action.isChecked() is True
    assert h._subtitle_show_trans_action.isChecked() is False
    h._subtitle_show_trans_action.trigger()
    assert got[-1] == (True, True)


def test_toggle_while_auto_hidden_keeps_lines_hidden_until_next_sentence(qapp):
    w = _window(qapp)
    w._insert_sentence("こんにちは", {"zh": "你好"})
    w._on_auto_hide_timeout()
    assert w._is_hidden_by_timeout
    w.set_line_visibility(False, True)
    assert w._is_hidden_by_timeout
    assert all(tw._content_opacity_val == 0.0 for tw in w._text_widgets)
    w._insert_sentence("さようなら", {"zh": "再見"})
    assert not w._is_hidden_by_timeout
    assert _texts(w)[-1].endswith("再見")


def test_translation_only_with_multiple_languages_uses_line_language(qapp):
    w = _window(qapp)
    w.set_line_visibility(False, True)
    w._insert_sentence("こんにちは", {"en": "hello", "zh": "你好"})
    assert _texts(w) == ["你好"]


def test_settings_widget_round_trip_keeps_visibility_flags(qapp):
    """The settings page must not silently reset the flags when it re-emits its settings."""
    from subtitle_settings import SubtitleSettingsWidget

    widget = SubtitleSettingsWidget({"show_original": False, "show_translation": True})
    s = widget.get_settings()
    assert s["show_original"] is False and s["show_translation"] is True
    widget.update_settings({"show_original": True, "show_translation": False})
    s = widget.get_settings()
    assert s["show_original"] is True and s["show_translation"] is False
