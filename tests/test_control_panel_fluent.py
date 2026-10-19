"""ControlPanel redesign (PyQt6-Fluent-Widgets): navigation pages, home page,
preserved settings/signal surface used by main.py, theme and font."""
import os
import re
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


def _config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _saved():
    return {
        "hub": "hf",
        "ui_lang": "zh_TW",
        "audio_device": None,
        "mic_device": None,
        "asr_engine": "whisper",
        "whisper_model_size": "medium",
        "funasr_model": "sensevoice-small",
        "asr_language": "ja",
        "asr_device": "cpu",
        "whisper_pad_seconds": 0.5,
        "sensevoice_pad_seconds": 0.5,
        "incremental_asr": True,
        "interim_interval": 0.5,
        "live_preview": True,
        "vad_mode": "silero",
        "vad_threshold": 0.3,
        "energy_threshold": 0.02,
        "min_speech_duration": 0.6,
        "max_speech_duration": 5.0,
        "silence_mode": "fixed",
        "silence_duration": 0.5,
        "target_language": "zh",
        "system_prompt": "翻譯成{target_lang}：{source_lang}",
        "timeout": 7,
        "auto_save_transcript": False,
        "models": [{"name": "Ollama qwen", "api_base": "http://127.0.0.1:11435/v1", "api_key": "ollama", "model": "qwen3.5:9b", "proxy": "none"}],
        "active_model": 0,
        "style": {
            "preset": "custom", "bg_color": "#101010", "bg_opacity": 200, "header_color": "#202030", "header_opacity": 180,
            "border_radius": 6, "original_font_family": "Microsoft JhengHei", "translation_font_family": "Microsoft JhengHei",
            "original_font_size": 12, "translation_font_size": 16, "original_color": "#cccccc", "translation_color": "#ffffff",
            "timestamp_color": "#888899", "window_opacity": 90,
        },
        "subtitle_mode": {"sentences": 2, "window_width": 900, "min_display_ms": 300, "click_through": False},
    }


def _panel(qapp, saved="default"):
    from control_panel import ControlPanel

    return ControlPanel(_config(), saved_settings=_saved() if saved == "default" else saved)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def test_is_fluent_window_with_navigation_pages(qapp):
    from qfluentwidgets import FluentWindow

    p = _panel(qapp)
    assert isinstance(p, FluentWindow)
    for name in ("home_interface", "asr_interface", "translation_interface", "appearance_interface",
                 "benchmark_interface", "cache_interface", "changelog_interface"):
        page = getattr(p, name)
        assert page.objectName() == name
        assert p.stackedWidget.indexOf(page) >= 0, f"{name} not added to the window"
    assert p.stackedWidget.currentWidget() is p.home_interface


def test_show_page_switches(qapp):
    p = _panel(qapp)
    p.show_page("subtitle")
    assert p.stackedWidget.currentWidget() is p.appearance_interface
    p.show_page("cache")
    assert p.stackedWidget.currentWidget() is p.cache_interface
    p.show_page("does-not-exist")  # error path: ignored, no exception
    assert p.stackedWidget.currentWidget() is p.cache_interface


def test_main_facing_surface_preserved(qapp):
    p = _panel(qapp)
    for sig in ("settings_changed", "model_changed", "models_list_changed", "subtitle_settings_changed", "reset_positions",
                "start_requested", "stop_requested"):
        assert hasattr(p, sig), sig
    assert p._asr_lang.findData("ja") >= 0 and p._asr_lang.currentData() == "ja"
    assert hasattr(p._subtitle_widget, "_width_spin") and hasattr(p._subtitle_widget, "_click_through_check")
    assert isinstance(p.get_settings(), dict)
    assert p.get_active_model()["model"] == "qwen3.5:9b"
    p._refresh_model_list()
    assert p._model_list.count() == 1


def test_apply_settings_round_trip(qapp):
    p = _panel(qapp)
    got = []
    p.settings_changed.connect(got.append)
    p._apply_settings()
    s = got[-1]
    expect = _saved()
    for key in ("asr_engine", "asr_language", "asr_device", "hub", "vad_mode", "min_speech_duration", "max_speech_duration",
                "silence_mode", "silence_duration", "incremental_asr", "interim_interval", "live_preview", "timeout",
                "auto_save_transcript", "whisper_pad_seconds", "sensevoice_pad_seconds", "system_prompt", "whisper_model_size"):
        assert s[key] == expect[key], key
    # opacities travel through a percent spin box (0-255 -> % -> 0-255), same rounding as upstream
    for key, val in expect["style"].items():
        if key in ("bg_opacity", "header_opacity"):
            assert abs(s["style"][key] - val) <= 2, key
        elif key in ("original_font_family", "translation_font_family"):
            # the offscreen font database may not have the requested family; the value must mirror the widget
            combo = p._orig_font_combo if key.startswith("original") else p._trans_font_combo
            assert s["style"][key] == combo.currentFont().family(), key
        else:
            assert s["style"][key] == val, key
    assert s["audio_device"] is None and s["mic_device"] is None


def test_defaults_without_saved_settings(qapp, monkeypatch):
    import control_panel as cp

    monkeypatch.setattr(cp, "_load_saved_settings", lambda: None)  # ignore the real user_settings.json
    p = _panel(qapp, saved=None)
    assert p.get_active_model()["model"] == _config()["translation"]["model"]
    p._apply_settings()  # must not raise


# ---------------------------------------------------------------------------
# home page
# ---------------------------------------------------------------------------

def test_home_toggle_emits_start_then_stop(qapp):
    p = _panel(qapp)
    got = []
    p.start_requested.connect(lambda: got.append("start"))
    p.stop_requested.connect(lambda: got.append("stop"))
    p.update_home_status(running=False)
    p._home_toggle_btn.click()
    p.update_home_status(running=True)
    p._home_toggle_btn.click()
    assert got == ["start", "stop"]


def test_home_status_text_and_bad_values(qapp):
    p = _panel(qapp)
    p.update_home_status(asr_engine="Whisper large-v3-turbo", model_name="qwen3.5:9b", asr_ms=123.4, tl_ms=45.6)
    assert "Whisper large-v3-turbo" in p._home_engine_card.contentLabel.text()
    assert "qwen3.5:9b" in p._home_model_card.contentLabel.text()
    assert "123" in p._home_asr_card.contentLabel.text()
    assert "46" in p._home_tl_card.contentLabel.text() or "45" in p._home_tl_card.contentLabel.text()
    # error path: None / NaN / negative never raise and show a placeholder
    p.update_home_status(asr_ms=None, tl_ms=float("nan"))
    p.update_home_status(asr_ms=-5)
    assert "ms" not in p._home_asr_card.contentLabel.text() or "—" in p._home_asr_card.contentLabel.text()


# ---------------------------------------------------------------------------
# theme / font / stylesheets
# ---------------------------------------------------------------------------

def test_dark_theme_and_ui_font(qapp):
    from qfluentwidgets import Theme, qconfig

    p = _panel(qapp)
    assert qconfig.theme == Theme.DARK
    assert p._whisper_status.font().family() == "Microsoft JhengHei UI"
    assert p._home_toggle_btn.font().family() == "Microsoft JhengHei UI"


def test_no_hand_written_stylesheet_colors():
    src = (ROOT / "control_panel.py").read_text(encoding="utf-8")
    assert "setStyleSheet(" not in src
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", src.replace("#!", "")), "hex colour literal left in control_panel.py"
