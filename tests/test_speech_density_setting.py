"""The minimum speech density (share of a segment's 32 ms chunks that Silero judges to be
speech) used to be hard-coded at 25 %. It is now the setting vad_min_speech_density,
editable on the control panel's voice detection page; lowering it keeps short replies
such as 「え？」 that were being dropped."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vad_processor import VADProcessor  # noqa: E402


def _vad():
    vad = VADProcessor.__new__(VADProcessor)  # no Silero model needed for the density check
    vad.mode = "silero"
    vad.threshold = 0.3
    vad.sample_rate = 16000
    vad._chunk_duration = 0.032
    vad._silence_mode = "fixed"
    vad._fixed_silence_dur = 0.5
    vad._silence_limit = 16
    vad.min_speech_samples = 0
    vad.max_speech_samples = 10 ** 9
    vad.energy_threshold = 0.02
    vad.min_speech_density = 0.25
    return vad


def _segment_with_density(vad, voiced: int, total: int):
    """Load a finished segment whose chunks are `voiced` speech out of `total`."""
    vad._reset()
    vad._confidence_history = [0.9] * voiced + [0.1] * (total - voiced)
    vad._speech_buffer = [np.zeros(512, np.float32) for _ in range(total)]
    vad._speech_samples = 512 * total


def test_vad_constructor_default_is_25_percent():
    vad = VADProcessor(threshold=0.3, min_speech_duration=0.6, max_speech_duration=5.0)
    assert vad.min_speech_density == 0.25


def test_short_reply_dropped_at_default_kept_at_15_percent():
    vad = _vad()
    _segment_with_density(vad, voiced=2, total=10)  # 20 %: e.g. a quiet 「え？」
    assert vad._flush_segment() is None
    vad.update_settings({"vad_min_speech_density": 0.15})
    _segment_with_density(vad, voiced=2, total=10)
    seg = vad._flush_segment()
    assert seg is not None and len(seg) == 512 * 10


def test_zero_disables_the_density_filter():
    vad = _vad()
    vad.update_settings({"vad_min_speech_density": 0})
    _segment_with_density(vad, voiced=0, total=10)
    assert vad._flush_segment() is not None


@pytest.mark.parametrize("bad", ["abc", -0.1, 1.5, True, None, [0.2]])
def test_invalid_values_are_ignored(bad, caplog):
    """Error path: user_settings.json is external input."""
    vad = _vad()
    vad.update_settings({"vad_min_speech_density": 0.15})
    with caplog.at_level("WARNING", logger="LiveTranslate.VAD"):
        vad.update_settings({"vad_min_speech_density": bad})
    assert vad.min_speech_density == 0.15
    assert "vad_min_speech_density" in caplog.text


# ---------------------------------------------------------------------------
# control panel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _panel(monkeypatch, saved):
    import control_panel as cp

    monkeypatch.setattr(cp, "_load_saved_settings", lambda: None)
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    p = cp.ControlPanel(config, saved_settings=saved)
    monkeypatch.setattr(p, "_auto_save", lambda *a: None)
    return p


def test_panel_spin_loads_saved_value_and_emits_changes(qapp, monkeypatch):
    p = _panel(monkeypatch, {"vad_min_speech_density": 0.15})
    assert p._min_density_spin.value() == 15
    got = []
    p.settings_changed.connect(got.append)
    p._min_density_spin.setValue(10)
    p._apply_settings()
    assert got[-1]["vad_min_speech_density"] == 0.1


def test_panel_defaults_to_25_and_ignores_bad_saved_value(qapp, monkeypatch):
    assert _panel(monkeypatch, {})._min_density_spin.value() == 25
    assert _panel(monkeypatch, {"vad_min_speech_density": "x"})._min_density_spin.value() == 25
    assert _panel(monkeypatch, {"vad_min_speech_density": 3.0})._min_density_spin.value() == 25
