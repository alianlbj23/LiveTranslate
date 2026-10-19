"""Abandoned translation streams must be closed right away.

The live preview stops reading a stream as soon as newer text arrives. An OpenAI
`Stream` that is dropped unread sits in a reference cycle and is only closed later
by the cyclic garbage collector. When that collection happens while the same thread
holds httpcore's connection-pool lock, closing the response tries to take the lock
again and the thread deadlocks; every other translation thread then blocks on the
pool forever (no error, no timeout). This is the 2026-10-19 20:23 "translations
silently stop" stall."""
import gc
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator import Translator  # noqa: E402


# ---------------------------------------------------------------------------
# unit level: fake client, no network
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


class _TrackedStream:
    def __init__(self, n=20):
        self._n = n
        self.closed = False

    def __iter__(self):
        for i in range(self._n):
            if self.closed:
                return
            yield _Chunk("字")

    def close(self):
        self.closed = True


class _Completions:
    def __init__(self):
        self.streams = []

    def create(self, **kwargs):
        s = _TrackedStream()
        self.streams.append(s)
        return s


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions()


def _translator(client):
    tr = Translator(api_base="http://127.0.0.1:9/v1", api_key="x", model="m",
                    target_language="zh", streaming=True, timeout=10)
    tr._client = client
    return tr


def test_breaking_out_of_translate_iter_closes_the_stream_immediately():
    client = _FakeClient()
    tr = _translator(client)
    gc_was_enabled = gc.isenabled()
    gc.disable()  # prove it does not depend on the garbage collector
    try:
        for _partial in tr.translate_iter("テスト", "ja"):
            break
        assert client.chat.completions.streams[0].closed is True
    finally:
        if gc_was_enabled:
            gc.enable()


def test_completed_translate_iter_closes_the_stream():
    client = _FakeClient()
    tr = _translator(client)
    out = None
    for out in tr.translate_iter("テスト", "ja"):
        pass
    assert out == "字" * 20
    assert client.chat.completions.streams[0].closed is True


def test_translate_iter_closes_the_stream_when_post_processing_raises(monkeypatch):
    """Error path: an exception while handling chunks still releases the connection."""
    client = _FakeClient()
    tr = _translator(client)

    def boom(_text):
        raise RuntimeError("post-processing failed")

    monkeypatch.setattr(tr, "_post", boom)
    with pytest.raises(RuntimeError):
        for _ in tr.translate_iter("テスト", "ja"):
            pass
    assert client.chat.completions.streams[0].closed is True


def test_stream_close_error_is_logged_not_raised(caplog):
    """Error path: a failing close() must not mask the translation result."""
    client = _FakeClient()
    tr = _translator(client)
    real_create = client.chat.completions.create

    def create(**kw):
        s = real_create(**kw)

        def bad_close():
            raise OSError("socket already gone")

        s.close = bad_close
        return s

    client.chat.completions.create = create
    out = None
    with caplog.at_level("DEBUG", logger="LiveTranslate.TL"):
        for out in tr.translate_iter("テスト", "ja"):
            pass
    assert out == "字" * 20
    assert "socket already gone" in caplog.text


def test_blocking_translate_closes_the_stream():
    """translate() (used for extra target languages) goes through _translate_streaming."""
    client = _FakeClient()
    tr = _translator(client)
    assert tr.translate("テスト", "ja") == "字" * 20
    assert client.chat.completions.streams[0].closed is True


def test_blocking_translate_closes_the_stream_on_error(monkeypatch):
    client = _FakeClient()
    tr = _translator(client)

    def boom(_text):
        raise RuntimeError("post-processing failed")

    monkeypatch.setattr(tr, "_post", boom)
    with pytest.raises(RuntimeError):
        tr.translate("テスト", "ja")
    assert client.chat.completions.streams[0].closed is True


def test_stream_options_fallback_stream_is_closed():
    """Servers that reject stream_options get a second request; that stream is closed too."""
    client = _FakeClient()
    real_create = client.chat.completions.create

    def create(**kw):
        if "stream_options" in kw:
            raise ValueError("stream_options not supported")
        return real_create(**kw)

    client.chat.completions.create = create
    tr = _translator(client)
    for _partial in tr.translate_iter("テスト", "ja"):
        break
    assert len(client.chat.completions.streams) == 1
    assert client.chat.completions.streams[0].closed is True


# ---------------------------------------------------------------------------
# reproduction: real OpenAI client + local SSE server, in a child process so a
# deadlock is killed by the timeout instead of hanging the test run
# ---------------------------------------------------------------------------

def test_abandoned_streams_do_not_deadlock_the_pool():
    """Deterministic reproduction of the 20:23 stall: the child runs the garbage
    collector exactly where it happened, inside httpcore's pool lock
    (ConnectionPool.handle_request -> _assign_requests_to_connections). Any
    abandoned-but-unclosed stream is finalized there and its response.close()
    re-enters the same non-reentrant lock."""
    root = str(Path(__file__).resolve().parents[1])
    script = Path(__file__).with_name("_stream_deadlock_child.py")
    try:
        proc = subprocess.run([sys.executable, "-X", "utf8", str(script), root],
                              capture_output=True, text=True, encoding="utf-8", timeout=30)
    except subprocess.TimeoutExpired:
        pytest.fail("translation deadlocked on the HTTP connection pool (child killed after 30 s)")
    assert proc.returncode == 0, proc.stdout + proc.stderr
