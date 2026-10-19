"""「回復原廠設定」: every user-tunable value goes back to what the one-click installer
writes (factory_defaults.py, shared with launcher/write_settings.py). Kept: the
translation model list / Ollama connection, the ASR model, audio devices and the
download hub. Window positions are dropped so the windows return to their defaults."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import factory_defaults as fd  # noqa: E402

CUSTOM_MODELS = [{"name": "Ollama qwen3.5:9b", "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama",
                  "model": "qwen3.5:9b", "proxy": "none", "context_turns": 5}]


def _tweaked():
    """A settings dict that differs from the factory values almost everywhere."""
    return {
        "models": copy.deepcopy(CUSTOM_MODELS), "active_model": 0,
        "asr_engine": "whisper", "whisper_model_size": r"D:\models\turbo", "asr_device": "cuda:0",
        "funasr_model": "sensevoice-small", "audio_device": None, "mic_device": None,
        "hub": "hf", "download_proxy": "system",
        "ui_lang": "zh_TW", "asr_language": "ja",
        "vad_mode": "energy", "vad_threshold": 0.55, "energy_threshold": 0.05,
        "min_speech_duration": 1.4, "max_speech_duration": 9.0, "silence_mode": "auto",
        "silence_duration": 1.2, "vad_min_speech_density": 0.1,
        "incremental_asr": False, "interim_interval": 1.5, "live_preview": False,
        "whisper_pad_seconds": 1.0, "sensevoice_pad_seconds": 1.0,
        "timeout": 30, "auto_save_transcript": False, "target_language": "zh",
        "system_prompt": "自訂提示詞 {source_lang} {target_lang}",
        "style": {**fd.STYLE, "border_radius": 4, "bg_opacity": 255, "translation_font_size": 22},
        "subtitle_mode": {"enabled": True, "sentences": 1, "window_x": -2000, "window_y": 900,
                          "window_width": 1300, "window_height": 220, "locked": True, "show_previous": False},
        "overlay_x": 10, "overlay_y": 20, "overlay_w": 700, "overlay_h": 600,
    }


# ---------------------------------------------------------------------------
# factory_defaults
# ---------------------------------------------------------------------------

def test_factory_reset_restores_tunables_and_keeps_connections():
    cur = _tweaked()
    cur["audio_device"] = "Speakers (X)"  # a chosen output device is kept
    new = fd.factory_reset(cur)
    defaults = fd.tunable_defaults()
    for key, value in defaults.items():
        if key == "subtitle_mode":
            continue
        assert new[key] == value, key
    for key in fd.KEEP_ON_RESET:
        if key in cur:
            assert new[key] == cur[key], key
    assert new["subtitle_mode"] == {**defaults["subtitle_mode"], "enabled": True}
    for gone in ("overlay_x", "overlay_y", "overlay_w", "overlay_h"):
        assert gone not in new
    new["models"][0]["model"] = "changed"
    assert cur["models"][0]["model"] == "qwen3.5:9b", "the result must not share objects with the input"


def test_factory_reset_tolerates_garbage_input():
    """Error path: a damaged settings file must still reset cleanly."""
    for bad in (None, "x", [], {"models": "not-a-list", "subtitle_mode": "nope", "audio_device": 3}):
        new = fd.factory_reset(bad)
        assert new["timeout"] == fd.tunable_defaults()["timeout"]
        assert isinstance(new["subtitle_mode"], dict)


def test_installer_writes_the_same_factory_values(tmp_path):
    """launcher/write_settings.py uses factory_defaults; existing keys it does not own survive."""
    (tmp_path / "user_settings.json").write_text(json.dumps({"overlay_x": 42}), encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", str(ROOT / "launcher" / "write_settings.py"), str(tmp_path),
                        r"D:\models\turbo", "http://127.0.0.1:11434/v1", "qwen3.5:9b"],
                       capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    data = json.loads((tmp_path / "user_settings.json").read_text(encoding="utf-8"))
    for key, value in fd.tunable_defaults().items():
        assert data[key] == value, key
    assert data["overlay_x"] == 42
    assert data["models"][0]["model"] == "qwen3.5:9b" and data["whisper_model_size"] == r"D:\models\turbo"


# ---------------------------------------------------------------------------
# control panel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(qapp, monkeypatch):
    import control_panel as cp

    saved_files = []
    monkeypatch.setattr(cp, "_load_saved_settings", lambda: None)
    monkeypatch.setattr(cp, "_save_settings", lambda s: saved_files.append(copy.deepcopy(s)))
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    p = cp.ControlPanel(config, saved_settings=_tweaked())
    monkeypatch.setattr(p, "_auto_save", lambda *a: None)
    p._saved_files = saved_files
    return p


def test_restore_factory_settings_updates_widgets_settings_and_windows(panel):
    p = panel
    got, subs, resets = [], [], []
    p.settings_changed.connect(got.append)
    p.subtitle_settings_changed.connect(lambda s: subs.append(dict(s)))
    p.reset_positions.connect(lambda: resets.append(1))
    p.restore_factory_settings()
    d = fd.tunable_defaults()
    # widgets
    assert p._vad_threshold_slider.value() == round(d["vad_threshold"] * 100)
    assert p._vad_mode.currentIndex() == 0  # silero
    assert p._silence_mode.currentIndex() == 1 and p._silence_duration.isEnabled()
    assert p._silence_duration.value() == d["silence_duration"]
    assert p._min_density_spin.value() == 25
    assert p._interim_interval_spin.value() == d["interim_interval"] and p._live_preview_cb.isChecked()
    assert p._timeout_spin.value() == d["timeout"]
    assert p._prompt_edit.toPlainText() == d["system_prompt"]
    assert p._border_radius.value() == d["style"]["border_radius"]
    # emitted + saved settings
    s = got[-1]
    for key in ("vad_mode", "vad_threshold", "silence_mode", "silence_duration", "vad_min_speech_density",
                "interim_interval", "live_preview", "timeout", "system_prompt", "auto_save_transcript"):
        assert s[key] == d[key], key
    assert s["style"]["border_radius"] == d["style"]["border_radius"]
    assert s["models"] == CUSTOM_MODELS and s["audio_device"] is None  # system default output kept
    assert s["whisper_model_size"] == r"D:\models\turbo"
    assert "overlay_x" not in s
    assert p._saved_files and p._saved_files[-1]["timeout"] == d["timeout"]
    # windows
    assert subs and subs[-1]["sentences"] == d["subtitle_mode"]["sentences"]
    assert subs[-1].get("locked") is not True and subs[-1].get("window_height", 0) == 0
    assert resets == [1]


def test_reset_button_asks_first(panel, monkeypatch):
    import control_panel as cp

    calls = []
    monkeypatch.setattr(panel, "restore_factory_settings", lambda: calls.append(1))

    class _No:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return False

    class _Yes(_No):
        def exec(self):
            return True

    monkeypatch.setattr(cp, "MessageBox", _No)
    panel._factory_reset_btn.click()
    assert calls == []
    monkeypatch.setattr(cp, "MessageBox", _Yes)
    panel._factory_reset_btn.click()
    assert calls == [1]
