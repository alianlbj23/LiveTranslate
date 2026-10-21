"""Local Ollama for the translation models (no Qt): start it with the fast settings, pick the
port, download models with progress and release the GPU.

Used by main.py (quit), local_model_prep (start / downloads), the settings and bench/
(bench/llm_servers re-exports it). The port rules are the ones the old console
launcher used: 11434, or 11435 when another Ollama (usually the
one inside WSL) holds 11434 without the model; a tray Ollama without the fast settings is
replaced."""
import json
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

log = logging.getLogger("LiveTranslate.Ollama")

ROOT = Path(__file__).resolve().parent
OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
PID_FILE = ROOT / "launcher" / "ollama.pid"
WATCHDOG = ROOT / "launcher" / "gpu_watchdog.ps1"
DEFAULT_PORT = 11434
ALT_PORT = 11435


class PullError(RuntimeError):
    """Ollama could not download a model (message from Ollama or the connection)."""


class PullCancelled(Exception):
    """The user cancelled a model download."""


def fast_env(port: int) -> dict:
    """Environment for `ollama serve`: models stay loaded, flash attention, 8-bit KV cache,
    two slots (final translation and live preview do not queue), short context, and one
    model at a time (switching unloads the previous model)."""
    return {
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_NUM_PARALLEL": "2",
        "OLLAMA_CONTEXT_LENGTH": "2048",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }


def root_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


class Ollama:
    def __init__(self, root: str, http=None):
        import httpx

        self.root = root.rstrip("/")
        self._http = http or httpx.Client(timeout=httpx.Timeout(30.0, connect=3.0), trust_env=False)

    def version(self) -> str | None:
        import httpx

        try:
            resp = self._http.get(f"{self.root}/api/version", timeout=3.0)
            if resp.status_code != 200:
                return None
            return str(resp.json().get("version", "")) or "?"
        except (httpx.HTTPError, ValueError) as e:
            log.debug(f"Ollama not answering at {self.root}: {e}")
            return None

    def alive(self) -> bool:
        return self.version() is not None

    def _models(self, path: str) -> list:
        """Model entries from /api/tags or /api/ps; an HTTP error raises (it must not read
        as "no models")."""
        resp = self._http.get(f"{self.root}{path}", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("models", []), list):
            raise ValueError(f"unexpected answer from {self.root}{path}: {str(data)[:120]}")
        return [m for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]

    def installed(self) -> set:
        return {m["name"] for m in self._models("/api/tags")}

    def loaded(self) -> list:
        return [m["name"] for m in self._models("/api/ps")]

    def loaded_vram_bytes(self) -> int:
        """GPU memory the loaded models hold (/api/ps size_vram)."""
        total = 0
        for m in self._models("/api/ps"):
            try:
                total += int(m.get("size_vram") or 0)
            except (TypeError, ValueError):
                log.warning(f"Ollama reported a non-numeric size_vram for {m.get('name')}: {m.get('size_vram')!r}")
        return total

    def unload(self, model: str):
        resp = self._http.post(f"{self.root}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30.0)
        if resp.status_code != 200:
            log.warning(f"Ollama could not unload {model}: HTTP {resp.status_code} {resp.text[:200]}")

    def unload_all(self, timeout: float = 30.0, sleep=time.sleep) -> bool:
        """True when /api/ps is empty within `timeout`."""
        deadline = time.monotonic() + timeout
        while True:
            models = self.loaded()
            if not models:
                return True
            if time.monotonic() > deadline:
                log.warning(f"Ollama still has {models} loaded after {timeout:.0f}s")
                return False
            for m in models:
                self.unload(m)
            sleep(1.0)

    def load(self, model: str, timeout: float = 300.0) -> float:
        """Load `model` (keep_alive 30m); returns the seconds it took."""
        t0 = time.perf_counter()
        resp = self._http.post(f"{self.root}/api/generate", json={"model": model, "keep_alive": "30m"},
                               timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama could not load {model}: HTTP {resp.status_code} {resp.text[:200]}")
        return time.perf_counter() - t0

    def pull(self, model: str, on_progress=None, cancelled=None, stall_timeout: float = 120.0):
        """Download `model`. on_progress(done_bytes, total_bytes, status) after every status
        line; cancelled() is asked between lines. Raises PullError / PullCancelled. A
        connection that sends nothing for `stall_timeout` seconds fails."""
        import httpx

        totals, done = {}, {}
        try:
            with self._http.stream("POST", f"{self.root}/api/pull", json={"model": model, "stream": True},
                                   timeout=httpx.Timeout(stall_timeout, connect=5.0)) as resp:
                if resp.status_code != 200:
                    raise PullError(f"HTTP {resp.status_code}: {resp.read()[:200].decode('utf-8', 'replace')}")
                for line in resp.iter_lines():
                    if cancelled is not None and cancelled():
                        raise PullCancelled()
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        log.warning(f"Ollama pull sent a line that is not JSON: {line[:200]!r}")
                        continue
                    if data.get("error"):
                        raise PullError(str(data["error"]))
                    digest = data.get("digest")
                    if digest and data.get("total"):
                        try:
                            totals[digest] = int(data["total"])
                            done[digest] = int(data.get("completed") or 0)
                        except (TypeError, ValueError):
                            log.warning(f"Ollama pull sent non-numeric sizes: {data!r}")
                    if on_progress is not None:
                        on_progress(sum(done.values()), sum(totals.values()), str(data.get("status", "")))
                    if data.get("status") == "success":
                        return
        except httpx.HTTPError as e:
            raise PullError(f"{type(e).__name__}: {e}") from e
        raise PullError("Ollama ended the download without reporting success")


def pid_file_server() -> int | None:
    """PID of the Ollama this app (or the old launcher) started with the fast settings."""
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    import psutil

    try:
        return pid if psutil.Process(pid).name().lower().startswith("ollama") else None
    except psutil.Error:
        return None


def port_owner(port: int) -> tuple:
    """(process name, pid) listening on 127.0.0.1:port, ("unknown", 0) when it cannot be told."""
    import psutil

    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                return psutil.Process(c.pid).name().lower(), c.pid
    except (psutil.Error, OSError) as e:
        log.warning(f"Cannot tell which program holds port {port}: {e}")
    return "unknown", 0


def start_fast_ollama(port: int, log_dir: Path, parent_pid: int | None = None) -> int:
    """Start `ollama serve` with the fast settings (no window, own process group) and a
    watchdog that cleans up after `parent_pid` (default: this process) ends in any way.
    Writes launcher/ollama.pid. Returns the PID."""
    if not OLLAMA_EXE.exists():
        raise RuntimeError(f"找不到 Windows 版 Ollama：{OLLAMA_EXE}")
    env = dict(os.environ)
    env.update(fast_env(port))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    serve_log = log_dir / "ollama_serve.log"
    with open(serve_log, "ab") as out:  # the child keeps its own handle
        proc = subprocess.Popen([str(OLLAMA_EXE), "serve"], env=env, creationflags=flags, stdout=out, stderr=out)
    PID_FILE.write_text(str(proc.pid), encoding="ascii")
    server = Ollama(root_url(port))
    for _ in range(30):
        time.sleep(1.0)
        if proc.poll() is not None:
            raise RuntimeError(f"Ollama 啟動後立刻結束（結束代碼 {proc.returncode}），原因見 {serve_log}")
        if server.alive():
            start_watchdog(port, log_dir, parent_pid or os.getpid())
            return proc.pid
    proc.kill()
    raise RuntimeError(f"Ollama 在 30 秒內沒有回應，原因見 {serve_log}")


def start_watchdog(port: int, log_dir: Path, parent_pid: int):
    """launcher/gpu_watchdog.ps1 (no window) unloads the models and stops the Ollama we
    started once `parent_pid` ends in any way (closed, crashed)."""
    if not WATCHDOG.exists():
        log.warning("gpu_watchdog.ps1 not found; Ollama will not be cleaned up after a crash")
        return
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
                      "-File", str(WATCHDOG), "-ParentPid", str(parent_pid), "-Port", str(port),
                      "-PidFile", str(PID_FILE), "-LogFile", str(Path(log_dir) / "gpu_watchdog.log")],
                     creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_tray_ollama():
    """Stop the tray app and the serve it started (it has none of the fast settings)."""
    import psutil

    for p in psutil.process_iter(["name"]):
        name = (p.info.get("name") or "").lower()
        if name in ("ollama app.exe", "ollama.exe"):
            try:
                p.kill()
            except psutil.Error as e:
                log.warning(f"Could not stop {name} (pid {p.pid}): {e}")
    time.sleep(2.0)


def stop_ollama(pid: int):
    """End our `ollama serve` and its runner processes (a runner left behind would keep the
    model in graphics memory)."""
    import psutil

    try:
        p = psutil.Process(pid)
        if p.name().lower().startswith("ollama"):
            for child in p.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error as e:
                    log.warning(f"Could not stop Ollama runner {child.pid}: {e}")
            p.kill()
    except psutil.Error as e:
        log.info(f"Ollama pid {pid} already gone: {e}")
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning(f"Could not remove {PID_FILE}: {e}")


def _has_model(server: Ollama, model: str) -> bool:
    import httpx

    try:
        return model in server.installed()
    except (httpx.HTTPError, ValueError) as e:
        log.warning(f"Cannot list the models of {server.root}: {e}")
        return False


def ensure_server(model: str, log_dir: Path, say=None) -> int:
    """Make sure an Ollama with the fast settings answers and return its port. `say(text)`
    reports each decision (shown to the user while starting). Raises RuntimeError when
    Ollama cannot be started."""
    say = say or log.info
    ours = pid_file_server()

    def restart(port: int, why: str) -> int:
        say(why)
        stop_tray_ollama()
        start_fast_ollama(port, log_dir)
        return port

    server = Ollama(root_url(DEFAULT_PORT))
    if not server.alive():
        say(f"啟動 Ollama（連接埠 {DEFAULT_PORT}）")
        start_fast_ollama(DEFAULT_PORT, log_dir)
        return DEFAULT_PORT
    name, pid = port_owner(DEFAULT_PORT)
    if name.startswith("ollama"):
        if ours and pid == ours:
            say(f"Ollama 已在連接埠 {DEFAULT_PORT} 執行")
            return DEFAULT_PORT
        return restart(DEFAULT_PORT, "系統匣的 Ollama 沒有快速設定，重新啟動")
    if _has_model(server, model):
        say(f"連接埠 {DEFAULT_PORT} 的 Ollama 不是 Windows 版（{name}），但有 {model}，直接使用")
        return DEFAULT_PORT
    say(f"連接埠 {DEFAULT_PORT} 被 {name} 佔用（通常是 WSL 內的 Ollama），改用 {ALT_PORT}")
    alt = Ollama(root_url(ALT_PORT))
    if alt.alive():
        _name2, pid2 = port_owner(ALT_PORT)
        if ours and pid2 == ours:
            say(f"Ollama 已在連接埠 {ALT_PORT} 執行")
            return ALT_PORT
        return restart(ALT_PORT, f"連接埠 {ALT_PORT} 的 Ollama 沒有快速設定，重新啟動")
    start_fast_ollama(ALT_PORT, log_dir)
    return ALT_PORT


def release_ours(timeout: float = 15.0) -> bool:
    """Unload every model of the Ollama this app started (whatever model is active now),
    then stop it with its runners. False when we did not start one (an Ollama someone else
    started is left alone). Safe to call more than once."""
    import httpx

    pid = pid_file_server()
    if not pid:
        return False
    for port in (DEFAULT_PORT, ALT_PORT):
        if port_owner(port)[1] != pid:
            continue
        try:
            Ollama(root_url(port)).unload_all(timeout=timeout)
        except (httpx.HTTPError, ValueError) as e:
            log.warning(f"Could not unload the models on port {port}: {e}")
    stop_ollama(pid)
    log.info("Ollama stopped (it is started again on the next launch)")
    return True
