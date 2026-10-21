"""ollama_server.py: the local Ollama the app starts itself (it used to be the launcher's
console script). Every server, process and HTTP call is faked."""
import json
import subprocess
from pathlib import Path

import httpx
import pytest

import ollama_server as osrv


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)


def _lines(*objs) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


# ---------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------

def _pull_server(body: bytes, status: int = 200):
    def handler(request):
        assert request.url.path == "/api/pull"
        assert json.loads(request.content) == {"model": "qwen3.5:4b", "stream": True}
        return httpx.Response(status, content=body)
    return osrv.Ollama("http://127.0.0.1:11434", http=_client(handler))


def test_pull_reports_progress_over_all_layers():
    server = _pull_server(_lines(
        {"status": "pulling manifest"},
        {"status": "pulling a", "digest": "a", "total": 1000, "completed": 500},
        {"status": "pulling b", "digest": "b", "total": 3000, "completed": 0},
        {"status": "pulling a", "digest": "a", "total": 1000, "completed": 1000},
        {"status": "pulling b", "digest": "b", "total": 3000, "completed": 3000},
        {"status": "verifying sha256 digest"},
        {"status": "success"},
    ))
    seen = []
    server.pull("qwen3.5:4b", on_progress=lambda d, t, s: seen.append((d, t, s)))
    assert seen[1] == (500, 1000, "pulling a")
    assert seen[2] == (500, 4000, "pulling b")
    assert seen[-1] == (4000, 4000, "success")


def test_pull_error_from_ollama_is_raised():
    server = _pull_server(_lines({"status": "pulling manifest"}, {"error": "pull model manifest: file does not exist"}))
    with pytest.raises(osrv.PullError, match="does not exist"):
        server.pull("qwen3.5:4b")


def test_pull_http_error_and_missing_success_are_errors():
    with pytest.raises(osrv.PullError, match="HTTP 500"):
        _pull_server(b"boom", status=500).pull("qwen3.5:4b")
    with pytest.raises(osrv.PullError, match="without reporting success"):
        _pull_server(_lines({"status": "pulling manifest"})).pull("qwen3.5:4b")


def test_pull_connection_failure_is_a_pull_error():
    def handler(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(osrv.PullError, match="ConnectError"):
        osrv.Ollama("http://127.0.0.1:11434", http=_client(handler)).pull("qwen3.5:4b")


def test_pull_can_be_cancelled_and_skips_bad_lines(caplog):
    server = _pull_server(b"not json\n" + _lines({"status": "pulling a", "digest": "a", "total": 10, "completed": 1},
                                                 {"status": "success"}))
    calls = []
    with pytest.raises(osrv.PullCancelled):
        server.pull("qwen3.5:4b", on_progress=lambda *a: calls.append(a), cancelled=lambda: len(calls) >= 1)
    assert "not JSON" in caplog.text


def test_loaded_vram_sums_size_vram(caplog):
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "a", "size_vram": 2 * 2**30},
                                                    {"name": "b", "size_vram": "x"}, {"name": "c"}]})
    server = osrv.Ollama("http://127.0.0.1:11434", http=_client(handler))
    assert server.loaded_vram_bytes() == 2 * 2**30
    assert "non-numeric size_vram" in caplog.text


# ---------------------------------------------------------------------------
# which port / which server (the old launcher's rules)
# ---------------------------------------------------------------------------

class _World:
    def __init__(self, alive=(), models=None, owners=None, ours=None):
        self.alive = set(alive)          # ports that answer
        self.models = models or {}       # port -> installed model names
        self.owners = owners or {}       # port -> (name, pid)
        self.ours = ours                 # pid in launcher/ollama.pid
        self.started, self.tray_stopped = [], 0


def _install(monkeypatch, world: _World):
    class FakeOllama:
        def __init__(self, root, http=None):
            self.root = root
            self.port = int(root.rsplit(":", 1)[1])

        def alive(self):
            return self.port in world.alive

        def installed(self):
            return set(world.models.get(self.port, ()))

    def start(port, log_dir, parent_pid=None):
        world.started.append(port)
        world.alive.add(port)
        return 4242

    def stop_tray():
        world.tray_stopped += 1

    monkeypatch.setattr(osrv, "Ollama", FakeOllama)
    monkeypatch.setattr(osrv, "port_owner", lambda port: world.owners.get(port, ("unknown", 0)))
    monkeypatch.setattr(osrv, "pid_file_server", lambda: world.ours)
    monkeypatch.setattr(osrv, "start_fast_ollama", start)
    monkeypatch.setattr(osrv, "stop_tray_ollama", stop_tray)


def test_nothing_running_starts_ollama_on_the_default_port(monkeypatch, tmp_path):
    w = _World()
    _install(monkeypatch, w)
    said = []
    assert osrv.ensure_server("qwen3.5:9b", tmp_path, say=said.append) == 11434
    assert w.started == [11434] and w.tray_stopped == 0 and said


def test_our_ollama_is_used_as_it_is(monkeypatch, tmp_path):
    w = _World(alive={11434}, owners={11434: ("ollama.exe", 77)}, ours=77)
    _install(monkeypatch, w)
    assert osrv.ensure_server("qwen3.5:9b", tmp_path) == 11434
    assert w.started == [] and w.tray_stopped == 0


def test_tray_ollama_is_replaced_by_one_with_the_fast_settings(monkeypatch, tmp_path):
    w = _World(alive={11434}, owners={11434: ("ollama.exe", 55)}, ours=None)
    _install(monkeypatch, w)
    assert osrv.ensure_server("qwen3.5:9b", tmp_path) == 11434
    assert w.tray_stopped == 1 and w.started == [11434]


def test_wsl_ollama_with_the_model_is_used(monkeypatch, tmp_path):
    w = _World(alive={11434}, owners={11434: ("wslrelay.exe", 9)}, models={11434: {"qwen3.5:9b"}})
    _install(monkeypatch, w)
    assert osrv.ensure_server("qwen3.5:9b", tmp_path) == 11434
    assert w.started == [] and w.tray_stopped == 0


def test_wsl_ollama_without_the_model_moves_to_the_spare_port(monkeypatch, tmp_path):
    w = _World(alive={11434}, owners={11434: ("wslrelay.exe", 9)})
    _install(monkeypatch, w)
    assert osrv.ensure_server("qwen3.5:9b", tmp_path) == 11435
    assert w.started == [11435]


def test_spare_port_already_ours_is_reused(monkeypatch, tmp_path):
    w = _World(alive={11434, 11435}, owners={11434: ("wslrelay.exe", 9), 11435: ("ollama.exe", 77)}, ours=77)
    _install(monkeypatch, w)
    assert osrv.ensure_server("qwen3.5:9b", tmp_path) == 11435
    assert w.started == [] and w.tray_stopped == 0


def test_start_failure_is_raised(monkeypatch, tmp_path):
    w = _World()
    _install(monkeypatch, w)

    def fail(port, log_dir, parent_pid=None):
        raise RuntimeError("找不到 Windows 版 Ollama")
    monkeypatch.setattr(osrv, "start_fast_ollama", fail)
    with pytest.raises(RuntimeError, match="找不到"):
        osrv.ensure_server("qwen3.5:9b", tmp_path)


# ---------------------------------------------------------------------------
# releasing the GPU
# ---------------------------------------------------------------------------

def test_release_unloads_and_stops_only_our_ollama(monkeypatch):
    unloaded, stopped = [], []

    class FakeOllama:
        def __init__(self, root, http=None):
            self.root = root

        def unload_all(self, timeout=30.0):
            unloaded.append(self.root)
            return True

    monkeypatch.setattr(osrv, "Ollama", FakeOllama)
    monkeypatch.setattr(osrv, "stop_ollama", stopped.append)
    monkeypatch.setattr(osrv, "pid_file_server", lambda: 77)
    osrv.release(11435)
    assert unloaded == ["http://127.0.0.1:11435"] and stopped == [77]
    monkeypatch.setattr(osrv, "pid_file_server", lambda: None)  # someone else's Ollama
    osrv.release(11434)
    assert unloaded[-1] == "http://127.0.0.1:11434" and stopped == [77]


def test_release_survives_an_ollama_that_does_not_answer(monkeypatch, caplog):
    class DeadOllama:
        def __init__(self, root, http=None):
            pass

        def unload_all(self, timeout=30.0):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(osrv, "Ollama", DeadOllama)
    monkeypatch.setattr(osrv, "pid_file_server", lambda: None)
    osrv.release(11434)
    assert "Could not unload" in caplog.text


# ---------------------------------------------------------------------------
# starting `ollama serve`
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, exits=False):
        self.pid = 4242
        self.returncode = 1 if exits else None
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True


def _start_env(monkeypatch, tmp_path, real, proc, alive=True):
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(osrv, "OLLAMA_EXE", exe)
    monkeypatch.setattr(osrv, "PID_FILE", tmp_path / "ollama.pid")
    monkeypatch.setattr(osrv.time, "sleep", lambda s: None)
    popen = []
    monkeypatch.setattr(osrv.subprocess, "Popen", lambda args, **kw: popen.append((args, kw)) or proc)
    watchdog = []
    monkeypatch.setattr(osrv, "start_watchdog", lambda *a: watchdog.append(a))

    class FakeOllama:
        def __init__(self, root, http=None):
            pass

        def alive(self):
            return alive

    monkeypatch.setattr(osrv, "Ollama", FakeOllama)
    return popen, watchdog


def test_serve_starts_without_a_window_with_the_fast_settings(monkeypatch, tmp_path, real_ollama_server):
    popen, watchdog = _start_env(monkeypatch, tmp_path, real_ollama_server, _FakeProc())
    assert real_ollama_server["start_fast_ollama"](11435, tmp_path / "logs", parent_pid=99) == 4242
    args, kw = popen[0]
    assert args[1:] == ["serve"]
    assert kw["creationflags"] & getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert kw["env"]["OLLAMA_HOST"] == "127.0.0.1:11435" and kw["env"]["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert (tmp_path / "ollama.pid").read_text(encoding="ascii") == "4242"
    assert watchdog == [(11435, tmp_path / "logs", 99)]


def test_serve_that_exits_at_once_or_never_answers_fails(monkeypatch, tmp_path, real_ollama_server):
    _start_env(monkeypatch, tmp_path, real_ollama_server, _FakeProc(exits=True))
    with pytest.raises(RuntimeError, match="立刻結束"):
        real_ollama_server["start_fast_ollama"](11434, tmp_path)
    proc = _FakeProc()
    _start_env(monkeypatch, tmp_path, real_ollama_server, proc, alive=False)
    with pytest.raises(RuntimeError, match="30 秒"):
        real_ollama_server["start_fast_ollama"](11434, tmp_path)
    assert proc.killed


def test_serve_without_ollama_installed_fails(monkeypatch, tmp_path, real_ollama_server):
    monkeypatch.setattr(osrv, "OLLAMA_EXE", Path(tmp_path / "missing.exe"))
    with pytest.raises(RuntimeError, match="找不到"):
        real_ollama_server["start_fast_ollama"](11434, tmp_path)


def test_watchdog_runs_without_a_window(monkeypatch, tmp_path, real_ollama_server):
    monkeypatch.setattr(osrv, "WATCHDOG", tmp_path / "gpu_watchdog.ps1")
    (tmp_path / "gpu_watchdog.ps1").write_text("", encoding="utf-8")
    popen = []
    monkeypatch.setattr(osrv.subprocess, "Popen", lambda args, **kw: popen.append((args, kw)))
    real_ollama_server["start_watchdog"](11434, tmp_path, 1234)
    args, kw = popen[0]
    assert "-WindowStyle" in args and "Hidden" in args and "1234" in args
    assert kw["creationflags"] & getattr(subprocess, "CREATE_NO_WINDOW", 0)


def test_port_owner_reads_the_listening_process(monkeypatch):
    import psutil

    class Conn:
        def __init__(self, port, status, pid):
            self.laddr = type("A", (), {"port": port})()
            self.status = status
            self.pid = pid

    class Proc:
        def __init__(self, pid):
            self.pid = pid

        def name(self):
            return "Ollama.exe"

    monkeypatch.setattr(psutil, "net_connections", lambda kind: [Conn(80, psutil.CONN_LISTEN, 1),
                                                                 Conn(11434, psutil.CONN_LISTEN, 77)])
    monkeypatch.setattr(psutil, "Process", Proc)
    assert osrv.port_owner(11434) == ("ollama.exe", 77)
    assert osrv.port_owner(11435) == ("unknown", 0)

    def denied(kind):
        raise psutil.AccessDenied()
    monkeypatch.setattr(psutil, "net_connections", denied)
    assert osrv.port_owner(11434) == ("unknown", 0)
