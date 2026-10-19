"""Regression tests for the live-preview path (code-review findings on commit 25a2373).

SubtitleWindow tests run Qt offscreen; LiveTranslateApp tests build the object with
__new__ and stub the collaborators, so no audio device, ASR worker or Ollama is needed.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from subtitle_window import SubtitleWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _pump(ms=30):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _window(qapp):
    w = SubtitleWindow({"window_width": 800, "sentences": 2, "min_display_ms": 0})
    w.show()
    _pump()
    return w


# ---------------------------------------------------------------------------
# SubtitleWindow
# ---------------------------------------------------------------------------

def test_late_partial_after_final_is_dropped(qapp):
    """Finding 3: a streamed partial of the generation that was just finalized must not
    re-show the finished sentence as a live line."""
    w = _window(qapp)
    w.update_live("今日はいい天気ですね", seq=5)
    w.update_live("今日はいい天気ですね", "今天天氣", complete=False, seq=5)
    _pump()
    w.update_text("今日はいい天気ですね", {"zh": "今天天氣真好呢"})
    _pump()
    assert w._live is None
    # late partial from the finalized generation (main bumped its seq before emitting the final)
    w.update_live("今日はいい天気ですね", "今天天氣真", complete=False, seq=5)
    _pump()
    assert w._live is None, "late partial re-showed the finalized sentence"
    # a genuinely new generation is still accepted
    w.update_live("次の文です", seq=6)
    _pump()
    assert w._live == ("次の文です", "")


def test_shorter_newer_generation_complete_replaces_preview(qapp):
    """Finding 7: when ASR settles on a shorter text and its complete translation arrives
    with a newer seq, both text and translation must replace the stale longer preview."""
    w = _window(qapp)
    w.update_live("ABCDEF", seq=1)
    w.update_live("ABCDEF", "翻譯到一", complete=False, seq=1)
    _pump()
    w.update_live("ABCD", seq=2)
    _pump()
    assert w._live[0] == "ABCDEF"  # longer text kept while the shorter one is only a guess
    w.update_live("ABCD", "完整翻譯", complete=True, seq=2)
    _pump()
    assert w._live == ("ABCD", "完整翻譯")


def test_shrink_without_complete_keeps_longer_text(qapp):
    w = _window(qapp)
    w.update_live("ABCDEF", "翻譯", complete=True, seq=1)
    _pump()
    w.update_live("ABCD", "翻", complete=False, seq=2)
    _pump()
    assert w._live == ("ABCDEF", "翻譯")


def test_final_with_different_leading_char_clears_preview(qapp):
    """Finding 4: a preview whose first characters were re-recognized differently still
    belongs to the finalized sentence when the tails match."""
    w = _window(qapp)
    w.update_live("は天気がいいです", seq=1)
    _pump()
    w.update_text("今日が天気がいいですね", {"zh": "今天天氣很好呢"})
    _pump()
    assert w._live is None


def test_short_final_does_not_clear_unrelated_preview(qapp):
    w = _window(qapp)
    w.update_live("えっ、マジで今の見た", seq=1)
    _pump()
    w.update_text("はい。", {"zh": "是。"})
    _pump()
    assert w._live is not None


# ---------------------------------------------------------------------------
# LiveTranslateApp (stubbed)
# ---------------------------------------------------------------------------

class _Subwin:
    def __init__(self):
        self.live_calls = []
        self.text_calls = []
        self.cleared = 0

    def isVisible(self):
        return True

    def get_target_languages(self):
        return {"zh"}

    def update_live(self, original, translation=None, complete=False, seq=0):
        self.live_calls.append((original, translation, complete, seq))

    def update_text(self, original, translations):
        self.text_calls.append((original, translations))

    def clear_live(self):
        self.cleared += 1


class _Executor:
    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args, **kwargs):
        self.submitted.append((fn, args, kwargs))


class _Transcript:
    def write_original(self, *a, **k):
        pass

    def write_translation(self, *a, **k):
        pass

    def finalize_no_translation(self, *a, **k):
        pass

    def close(self):
        pass


class _Vad:
    def __init__(self, audio):
        self._audio = audio
        self.trimmed = []
        self._is_speaking = False
        self._speech_samples = len(audio)

    def peek_buffer(self):
        return self._audio, len(self._audio) / 16000

    def trim_front(self, n):
        self.trimmed.append(n)

    def flush(self):
        return None

    def force_flush(self):
        return None


class _Audio:
    def stop(self):
        pass


def _app(asr_text="", live_text="", seq=5):
    import threading

    import main as main_module

    app = main_module.LiveTranslateApp.__new__(main_module.LiveTranslateApp)
    app._config = {"audio": {"sample_rate": 16000, "chunk_duration": 0.032}}
    app._panel = None
    app._overlay = None
    app._subwin = _Subwin()
    app._transcript = _Transcript()
    app._tl_executor = _Executor()
    app._live_executor = _Executor()
    app._translator = object()
    app._target_language = "zh"
    app._asr_count = 0
    app._msg_id = 0
    app._last_original = ""
    app._last_msg_id = 0
    app._live_preview = True
    app._live_seq = seq
    app._live_last_text = live_text
    app._live_lang = "ja"
    app._live_dirty = bool(live_text)
    app._live_inflight = False
    app._interim_interval = 0.5
    app._interim_pending = ""
    app._interim_active = False
    app._interim_committed_tail = ""
    app._vad_lock = threading.Lock()
    app._vad = _Vad(np.zeros(16000 * 3, dtype=np.float32))
    app._run_asr = lambda audio, kind, **kw: ({"text": asr_text, "language": "ja"}, 10.0)
    return app


def test_process_segment_text_keeps_unrelated_preview():
    """Finding 1: finalizing sentence A must not cancel the preview of sentence B."""
    app = _app(live_text="明日も晴れるといいな")
    app._process_segment_text("今日はいい天気ですね。", "ja")
    assert app._live_seq == 5
    assert app._live_last_text == "明日も晴れるといいな"


def test_process_segment_text_invalidates_matching_preview():
    app = _app(live_text="今日はいい天気ですね")
    app._process_segment_text("今日はいい天気ですね。", "ja")
    assert app._live_seq == 6
    assert app._live_last_text == ""
    assert app._live_dirty is False


def test_process_segment_invalidates_matching_preview():
    """Finding 2: the non-interim final path must invalidate the preview too."""
    app = _app(asr_text="今日はいい天気ですね", live_text="今日はいい天気ですね")
    app._process_segment(np.zeros(16000 * 2, dtype=np.float32))
    assert app._live_seq == 6
    assert app._live_last_text == ""


def test_interim_preview_survives_commit():
    """Finding 1 (ordering): after an interim pass commits the complete sentences, the
    trailing sentence must remain the active preview with a pending translation."""
    app = _app(asr_text="今日はいい天気ですね。明日も晴れるといいな")
    committed = app._do_interim_asr()
    assert committed is True
    assert app._live_last_text == "明日も晴れるといいな"
    assert app._live_dirty is True
    assert app._subwin.live_calls[-1][0] == "明日も晴れるといいな"
    assert app._live_executor.submitted, "live translation worker was not started"


def test_stop_shuts_down_live_executor():
    """Finding 5: stop() must shut down the live worker so quitting cannot block on an
    in-flight provisional translation."""
    import threading

    import main as main_module

    app = _app()
    app._running = True
    app._audio = _Audio()
    app._capture_thread = None
    app._asr_thread = None
    app._asr_queue = __import__("queue").Queue()
    app._asr_ready = False
    app._last_interim_samples = 0
    app._last_interim_check_time = 0.0
    app._tl_executor = ThreadPoolExecutor(max_workers=1)
    app._live_executor = ThreadPoolExecutor(max_workers=1)
    app._mem_periodic_timer = None
    app._mem_baseline_mb = 0.0
    app._mem_asr_call_count = 0
    app._mem_snapshot = lambda: {"rss": 0.0, "gpu_alloc": 0.0, "gpu_reserved": 0.0}
    app._shutdown_asr_worker = lambda: None
    app._asr_lock = threading.RLock()
    app.stop()
    with pytest.raises(RuntimeError):
        app._live_executor.submit(lambda: None)
    assert isinstance(main_module.LiveTranslateApp, type)
