"""Overlay redesign (subtitle_overlay.py): rounded translucent card with a soft shadow,
Fluent icon buttons with tooltips, Fluent combo boxes, a collapsible status section,
and quieter message typography. Behaviour (signals, main.py-facing attributes) must
stay the same."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication, QCheckBox, QPushButton  # noqa: E402
from qfluentwidgets import ComboBox, TransparentToggleToolButton, TransparentToolButton  # noqa: E402

import subtitle_overlay as so  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _overlay(qapp):
    ov = so.SubtitleOverlay({})
    ov.show()
    qapp.processEvents()
    return ov


# ---------------------------------------------------------------------------
# header: icon buttons, tooltips, toggles, combos
# ---------------------------------------------------------------------------

def test_header_uses_fluent_icon_buttons_with_tooltips(qapp):
    h = so.DragHandle()
    assert not h.findChildren(QCheckBox), "check boxes should be icon toggle buttons"
    plain = [b for b in h.findChildren(QPushButton) if type(b) is QPushButton]
    assert not plain, "text buttons should be Fluent icon buttons"
    buttons = h.findChildren(TransparentToolButton) + h.findChildren(TransparentToggleToolButton)
    assert len(buttons) >= 11
    for b in buttons:
        assert b.toolTip().strip(), f"{b.objectName() or b} has no tooltip"
        assert not b.icon().isNull()
        assert not b.text(), "icon-only buttons"


def test_toggle_buttons_keep_main_py_api(qapp):
    """main.py syncs the tray menu through _ct_check / _topmost_check / _auto_scroll.
    The Taskbar toggle was removed on request (2026-10-21): the overlay is always on the taskbar."""
    h = so.DragHandle()
    got = []
    h.click_through_toggled.connect(lambda v: got.append(("ct", v)))
    h.topmost_toggled.connect(lambda v: got.append(("top", v)))
    h.auto_scroll_toggled.connect(lambda v: got.append(("scroll", v)))
    for name in ("_ct_check", "_topmost_check", "_auto_scroll"):
        assert isinstance(getattr(h, name), TransparentToggleToolButton), name
    assert not hasattr(h, "_taskbar_check")
    assert h._ct_check.isChecked() is False and h._topmost_check.isChecked() is True
    assert h.auto_scroll is True
    h._ct_check.setChecked(True)
    h._topmost_check.click()
    h._auto_scroll.click()
    assert got == [("ct", True), ("top", False), ("scroll", False)]
    assert h.auto_scroll is False


def test_action_buttons_emit_the_same_signals(qapp):
    h = so.DragHandle()
    got = []
    for sig in ("subtitle_clicked", "clear_clicked", "settings_clicked", "quit_clicked",
                "start_clicked", "stop_clicked"):
        getattr(h, sig).connect(lambda *_a, s=sig: got.append(s))
    h._subtitle_btn.click()
    h._clear_btn.click()
    h._settings_btn.click()
    h._quit_btn.click()
    h.set_running(False)
    h._start_stop_btn.click()
    h.set_running(True)
    h._start_stop_btn.click()
    assert got == ["subtitle_clicked", "clear_clicked", "settings_clicked", "quit_clicked",
                   "start_clicked", "stop_clicked"]


def test_header_has_no_hide_button(qapp):
    """The "hide this window" header button was removed on request (2026-10-19);
    the tray menu still shows / hides the window."""
    h = so.DragHandle()
    assert not hasattr(h, "_hide_btn")
    assert not hasattr(so.DragHandle, "hide_clicked")
    assert not hasattr(so.SubtitleOverlay, "hide_requested")


def test_running_state_changes_icon_tooltip(qapp):
    h = so.DragHandle()
    h.set_running(False)
    paused_tip = h._start_stop_btn.toolTip()
    h.set_running(True)
    assert h._start_stop_btn.toolTip() != paused_tip


def test_subtitle_button_reflects_window_state(qapp):
    h = so.DragHandle()
    h.set_subtitle_checked(True)
    assert h._subtitle_btn.isChecked() is True
    h.set_subtitle_checked(False)
    assert h._subtitle_btn.isChecked() is False


def test_mode_button_toggles_and_compact_hides_second_row(qapp):
    h = so.DragHandle()
    h.show()
    modes = []
    h.mode_changed.connect(modes.append)
    full_tip = h._mode_btn.toolTip()
    h._mode_btn.click()
    assert modes == ["compact"]
    assert h._row2_widget.isHidden()
    assert h._mode_btn.toolTip() != full_tip
    h._mode_btn.click()
    assert modes == ["compact", "full"] and not h._row2_widget.isHidden()


def test_combos_are_fluent_and_keep_data(qapp):
    h = so.DragHandle()
    for c in (h._model_combo, h._source_lang, h._target_lang):
        assert isinstance(c, ComboBox)
    got = []
    h.model_changed.connect(lambda i: got.append(("model", i)))
    h.source_language_changed.connect(lambda c: got.append(("src", c)))
    h.target_language_changed.connect(lambda c: got.append(("tgt", c)))
    h.set_models([{"name": "Ollama qwen3.5:4b"}, {"model": "qwen3.5:9b"}], 1)
    assert [h._model_combo.itemText(i) for i in range(h._model_combo.count())] == ["Ollama qwen3.5:4b", "qwen3.5:9b"]
    assert h._model_combo.currentIndex() == 1
    h.set_source_language("ja")
    h.set_target_language("zh")
    assert got == [], "programmatic sync must not emit"
    assert h._source_lang.currentData() == "ja" and h._target_lang.currentData() == "zh"
    h._model_combo.setCurrentIndex(0)
    h._target_lang.setCurrentIndex(h._target_lang.findData("en"))
    assert ("model", 0) in got and ("tgt", "en") in got


def test_combo_sync_with_unknown_values_is_ignored(qapp):
    """Error path: unknown codes / empty model lists never raise or change the selection."""
    h = so.DragHandle()
    h.set_target_language("zh")
    h.set_target_language("xx-not-a-language")
    assert h._target_lang.currentData() == "zh"
    h.set_source_language(None)
    h.set_models([], 3)
    assert h._model_combo.count() == 0
    h.set_models([{"name": "only"}], 5)  # out-of-range index
    assert h._model_combo.count() == 1


def test_subtitle_dropdown_actions_still_exist(qapp):
    h = so.DragHandle()
    for name in ("_subtitle_lock_action", "_subtitle_ct_action", "_subtitle_show_orig_action",
                 "_subtitle_show_trans_action", "_subtitle_show_prev_action", "_subtitle_settings_action"):
        assert getattr(h, name) is not None
    assert isinstance(h._subtitle_menu_btn, TransparentToolButton)


# ---------------------------------------------------------------------------
# card, status section, messages
# ---------------------------------------------------------------------------

def test_default_style_is_rounded_translucent_dark_card(qapp):
    s = so.DEFAULT_STYLE
    assert s["border_radius"] == 12
    assert 0.83 <= s["bg_opacity"] / 255 <= 0.87
    ov = _overlay(qapp)
    css = ov._container.styleSheet()
    assert "#overlayCard" in css and "border-radius: 12px" in css
    assert ov._handle.styleSheet().lstrip().startswith("#overlayHeader"), "header style must not cascade into Fluent widgets"
    ov.grab()  # paints the shadow; must not raise


def test_status_section_collapsed_by_default_and_toggles(qapp):
    ov = _overlay(qapp)
    assert ov._status.is_expanded() is False
    assert not ov._monitor.isVisibleTo(ov)
    ov._status._toggle_btn.click()
    assert ov._status.is_expanded() is True and ov._monitor.isVisibleTo(ov)
    ov._status._toggle_btn.click()
    assert not ov._monitor.isVisibleTo(ov)


def test_compact_mode_hides_status_section(qapp):
    ov = _overlay(qapp)
    ov._status.set_expanded(True)
    ov._handle._mode_btn.click()
    assert not ov._status.isVisibleTo(ov)
    ov._handle._mode_btn.click()
    assert ov._status.isVisibleTo(ov) and ov._status.is_expanded() is True


def test_message_typography(qapp):
    so.ChatMessage._compact_mode = False
    so.ChatMessage._current_style = dict(so.DEFAULT_STYLE)
    m = so.ChatMessage(1, "22:04:53", "えぇ!", "ja", 168)
    s = so.DEFAULT_STYLE
    header = m._header_label.text()
    assert s["original_color"] in header and "ASR 168ms" in header
    m.set_translation("欸！", 632)
    trans = m._trans_label.text()
    assert "font-weight:600" in trans and s["translation_color"] in trans
    tag = trans[trans.index("TL 632ms") - 120:trans.index("TL 632ms")]
    assert "font-size:8pt" in tag and s["timestamp_color"] in tag
    assert m._header_label.font().pointSize() < m._trans_label.font().pointSize()


def test_bad_style_values_fall_back_instead_of_crashing(qapp):
    """Error path: the style comes from user_settings.json."""
    ov = _overlay(qapp)
    ov.apply_style({"bg_color": "not-a-colour", "bg_opacity": "x", "border_radius": -5,
                    "window_opacity": 500, "original_font_size": None, "translation_color": 12})
    css = ov._container.styleSheet()
    assert "border-radius: 12px" in css or "border-radius: 0px" in css
    assert 0.2 <= ov.windowOpacity() <= 1.0
    ov._on_add_message(1, "00:00:00", "テスト", "ja", 100)
    ov._on_update_translation(1, "測試", 50)
