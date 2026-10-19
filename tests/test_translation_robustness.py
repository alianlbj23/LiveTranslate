"""Translation must fail fast instead of hanging when the LLM server crawls (VRAM
pressure): hard wall-clock deadline on streams, no SDK retries for local servers,
and a visible warning when translations pile up."""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import translator as translator_module  # noqa: E402
from translator import Translator, make_openai_client  # noqa: E402


# ---------------------------------------------------------------------------
# helpers: a fake OpenAI client whose stream trickles one chunk then stalls
# ---------------------------------------------------------------------------

class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = None


class _StallingStream:
    """Yields one chunk immediately, then blocks until closed (like a GPU that is
    swapping: the connection stays open but nothing arrives)."""

    def __init__(self, stall_seconds):
        self._stall = stall_seconds
        self._closed = threading.Event()
        self._n = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._n += 1
        if self._n == 1:
            return _Chunk("今")
        if self._n > 2:
            raise StopIteration
        # block until close() or the stall elapses, polling like a socket read would
        if self._closed.wait(self._stall):
            raise StopIteration
        return _Chunk("天")

    def close(self):
        self._closed.set()


class _Completions:
    def __init__(self, stall):
        self._stall = stall

    def create(self, **kwargs):
        if kwargs.get("stream"):
            return _StallingStream(self._stall)
        raise AssertionError("non-streaming path not expected in this test")


class _Chat:
    def __init__(self, stall):
        self.completions = _Completions(stall)


class _FakeClient:
    def __init__(self, stall=30.0):
        self.chat = _Chat(stall)


def _translator(monkeypatch, timeout, stall=30.0):
    monkeypatch.setattr(translator_module, "make_openai_client", lambda *a, **k: _FakeClient(stall))
    return Translator(api_base="http://127.0.0.1:11435/v1", api_key="ollama", model="m",
                      streaming=True, timeout=timeout, thinking_style="off")


# ---------------------------------------------------------------------------
# translator
# ---------------------------------------------------------------------------

def test_local_client_disables_sdk_retries():
    c = make_openai_client("http://127.0.0.1:11435/v1", "ollama", "none", timeout=5)
    assert c.max_retries == 0


def test_remote_client_keeps_default_retries():
    c = make_openai_client("https://api.deepseek.com/v1", "k", "none", timeout=5)
    assert c.max_retries >= 1


def test_streaming_translate_raises_timeout_without_waiting_for_next_chunk(monkeypatch):
    tr = _translator(monkeypatch, timeout=1.0, stall=30.0)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        tr.translate("今日はいい天気ですね", "ja")
    assert time.monotonic() - t0 < 3.0, "translate() waited for the stalled chunk instead of cutting at the deadline"


def test_translate_iter_raises_timeout_without_waiting(monkeypatch):
    tr = _translator(monkeypatch, timeout=1.0, stall=30.0)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        for _ in tr.translate_iter("今日はいい天気ですね", "ja"):
            pass
    assert time.monotonic() - t0 < 3.0


def test_fast_stream_is_not_cut(monkeypatch):
    tr = _translator(monkeypatch, timeout=5.0, stall=0.05)
    assert tr.translate("x", "ja") == "今天"


# ---------------------------------------------------------------------------
# main: backlog warning
# ---------------------------------------------------------------------------

class _Subwin:
    def isVisible(self):
        return False

    def get_target_languages(self):
        return set()

    def update_live(self, *a, **k):
        pass

    def clear_live(self):
        pass


class _Executor:
    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args, **kwargs):
        self.submitted.append(fn)


class _Transcript:
    def write_original(self, *a, **k):
        pass

    def finalize_no_translation(self, *a, **k):
        pass


def _stub_app():
    import main as main_module

    app = main_module.LiveTranslateApp.__new__(main_module.LiveTranslateApp)
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
    app._live_preview = False
    app._live_seq = 0
    app._live_last_text = ""
    app._live_lang = "ja"
    app._live_dirty = False
    app._live_inflight = False
    app._tl_inflight = 0
    app._tl_backlog_threshold = 3
    app._tl_backlog_last_warn = 0.0
    app._tl_backlog_callback = None
    return app


def test_backlog_warning_fires_once_per_window():
    app = _stub_app()
    warned = []
    app.set_translation_backlog_callback(lambda n: warned.append(n))
    for i in range(5):
        app._process_segment_text(f"文{i}です", "ja")
    assert app._tl_inflight == 5
    assert len(warned) == 1 and warned[0] >= 3, "backlog warning should fire exactly once within the rate-limit window"


def test_inflight_counter_returns_to_zero_after_translation(monkeypatch):
    app = _stub_app()
    app._translate_async = lambda *a, **k: None  # the real body needs a translator; the wrapper is what we test
    app._process_segment_text("文です", "ja")
    assert app._tl_inflight == 1
    fn = app._tl_executor.submitted[-1]
    fn(1, "文です", "ja", None)
    assert app._tl_inflight == 0
