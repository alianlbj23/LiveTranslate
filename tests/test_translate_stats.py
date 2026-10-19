"""After a translation the statistics go to the overlay and the latency to the home
page, independently: a missing overlay must not turn a good translation into a
"Translate error"."""
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module  # noqa: E402


class _Translator:
    last_usage = (10, 5)

    def translate_iter(self, text, lang):
        yield "你好"


class _Overlay:
    def __init__(self):
        self.stats = []
        self.translations = []

    def update_streaming(self, msg_id, partial):
        pass

    def update_translation(self, msg_id, text, ms):
        self.translations.append(text)

    def update_stats(self, *args):
        self.stats.append(args)


class _Panel:
    def __init__(self):
        self.home = []

    def update_home_status(self, **kw):
        self.home.append(kw)


class _Transcript:
    def __init__(self):
        self.written = []
        self.no_translation = []

    def write_translation(self, msg_id, text):
        self.written.append(text)

    def finalize_no_translation(self, msg_id):
        self.no_translation.append(msg_id)


def _app(panel, overlay):
    app = main_module.LiveTranslateApp.__new__(main_module.LiveTranslateApp)
    app._translator = _Translator()
    app._overlay = overlay
    app._panel = panel
    app._subwin = None
    app._transcript = _Transcript()
    app._translate_count = 0
    app._total_prompt_tokens = 0
    app._total_completion_tokens = 0
    app._input_price = 0.0
    app._output_price = 0.0
    app._asr_count = 1
    app._target_language = "zh"
    return app


def test_stats_reach_overlay_without_panel():
    overlay = _Overlay()
    app = _app(panel=None, overlay=overlay)
    app._translate_async(1, "こんにちは", "ja")
    assert len(overlay.stats) == 1
    assert overlay.translations == ["你好"]


def test_panel_without_overlay_is_not_a_translation_error(caplog):
    panel = _Panel()
    app = _app(panel=panel, overlay=None)
    with caplog.at_level(logging.WARNING, logger="LiveTranslate"):
        app._translate_async(1, "こんにちは", "ja")
    assert "Translate error" not in caplog.text
    assert app._transcript.written == ["你好"]
    assert app._transcript.no_translation == []
    assert panel.home and "tl_ms" in panel.home[-1]
