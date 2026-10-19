"""benchmark.run_benchmark must close its streams even when a chunk breaks the loop,
otherwise the abandoned stream can deadlock the connection pool during GC."""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark  # noqa: E402


class _Msg:
    content = "翻譯"


class _RespChoice:
    message = _Msg()


class _Resp:
    choices = [_RespChoice()]


class _BadChunk:
    choices = []  # chunk.choices[0] -> IndexError inside the stream loop


class _Stream:
    def __init__(self):
        self.closed = False

    def __iter__(self):
        yield _BadChunk()

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _Completions:
    def __init__(self):
        self.streams = []

    def create(self, **kw):
        if kw.get("stream"):
            s = _Stream()
            self.streams.append(s)
            return s
        return _Resp()


class _Client:
    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions()


def test_benchmark_closes_stream_when_loop_raises(monkeypatch):
    client = _Client()
    monkeypatch.setattr(benchmark, "make_openai_client", lambda *a, **k: client)
    done = threading.Event()
    lines = []

    def cb(line):
        lines.append(line)
        if line == "__DONE__":
            done.set()

    benchmark.run_benchmark([{"name": "fake", "api_base": "http://x/v1", "api_key": "k", "model": "m"}],
                            "ja", "zh", 5, "翻譯", cb)
    assert done.wait(10), lines
    assert client.chat.completions.streams, "benchmark never streamed"
    assert all(s.closed for s in client.chat.completions.streams)
    assert not any("FAIL" in ln for ln in lines), lines  # fell back to non-streaming and finished
