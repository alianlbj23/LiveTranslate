"""Translation model servers: classify a model entry, and load / unload local models so
each one is measured alone (Ollama) or at least waited out (LM Studio).

Ollama is started with the same fast settings as launcher/start_jp_subtitles.ps1, so the
numbers match real use. main.py's release_gpu / _warm_local_model cannot be imported
(main pulls in PyQt), so the same /api/ps + keep_alive calls are made here."""
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from bench import paths

log = logging.getLogger("LiveTranslate.Bench.LLM")

OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
PID_FILE = paths.ROOT / "launcher" / "ollama.pid"
WATCHDOG = paths.ROOT / "launcher" / "gpu_watchdog.ps1"


def fast_env(port: int) -> dict:
    """Same values as Start-WindowsOllama in launcher/start_jp_subtitles.ps1 (a test keeps
    them in sync)."""
    return {
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_NUM_PARALLEL": "2",
        "OLLAMA_CONTEXT_LENGTH": "2048",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }


def model_kind(entry: dict) -> str:
    """"ollama", "lmstudio", "local" (another localhost server) or "cloud"."""
    import local_models

    base = str(entry.get("api_base", ""))
    if local_models.is_local_ollama(entry):
        return "ollama"
    if ":1234" in base:
        return "lmstudio"
    host = base.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return "local"
    return "cloud"


def fingerprint(entry: dict) -> str:
    """Stable id of a model entry without its API key."""
    return f"{entry.get('name', '')}|{entry.get('model', '')}|{entry.get('api_base', '')}"


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def active_ollama_port() -> int:
    """Same rule as the launcher: 11435 when something listens there, else 11434."""
    return 11435 if _port_open(11435) else 11434


class Ollama:
    def __init__(self, root_url: str, http=None):
        import httpx

        self.root = root_url.rstrip("/")
        self._http = http or httpx.Client(timeout=httpx.Timeout(30.0, connect=3.0), trust_env=False)

    def alive(self) -> bool:
        import httpx

        try:
            return self._http.get(f"{self.root}/api/version", timeout=3.0).status_code == 200
        except httpx.HTTPError as e:
            log.debug(f"Ollama not answering at {self.root}: {e}")
            return False

    def installed(self) -> set:
        data = self._http.get(f"{self.root}/api/tags", timeout=5.0).json()
        return {m.get("name") for m in data.get("models", []) if m.get("name")}

    def loaded(self) -> list:
        data = self._http.get(f"{self.root}/api/ps", timeout=5.0).json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]

    def unload(self, model: str):
        self._http.post(f"{self.root}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30.0)

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


def pid_file_server() -> int | None:
    """PID of the Ollama the launcher (or this tool) started with the fast settings."""
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    import psutil

    try:
        return pid if psutil.Process(pid).name().lower().startswith("ollama") else None
    except psutil.Error:
        return None


def start_fast_ollama(port: int, log_dir: Path) -> int:
    """Start `ollama serve` with the fast settings (hidden, own process group so Ctrl+C
    in the menu does not kill it); writes launcher/ollama.pid. Returns the PID."""
    if not OLLAMA_EXE.exists():
        raise RuntimeError(f"找不到 Windows 版 Ollama：{OLLAMA_EXE}")
    env = dict(os.environ)
    env.update(fast_env(port))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen([str(OLLAMA_EXE), "serve"], env=env, creationflags=flags,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    PID_FILE.write_text(str(proc.pid), encoding="ascii")
    server = Ollama(f"http://127.0.0.1:{port}")
    for _ in range(30):
        time.sleep(1.0)
        if server.alive():
            _start_watchdog(port, log_dir)
            return proc.pid
    raise RuntimeError("Ollama 在 30 秒內沒有回應")


def _start_watchdog(port: int, log_dir: Path):
    """launcher/gpu_watchdog.ps1 unloads the models and stops the Ollama we started when
    this process ends in any way (closed window, crash)."""
    if not WATCHDOG.exists():
        log.warning("gpu_watchdog.ps1 not found; Ollama will not be cleaned up after a crash")
        return
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
                      "-File", str(WATCHDOG), "-ParentPid", str(os.getpid()), "-Port", str(port),
                      "-PidFile", str(PID_FILE), "-LogFile", str(log_dir / "gpu_watchdog.log")],
                     creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_tray_ollama():
    """Stop the tray app and the serve it started (what the launcher's Stop-WindowsOllama
    does); only called after the user agreed and LiveTranslate is closed."""
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
    import psutil

    try:
        p = psutil.Process(pid)
        if p.name().lower().startswith("ollama"):
            p.kill()
    except psutil.Error as e:
        log.info(f"Ollama pid {pid} already gone: {e}")
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def lmstudio_unload_all() -> bool:
    """`lms unload --all` when the LM Studio CLI is installed; False when it is not."""
    lms = shutil.which("lms")
    if not lms:
        return False
    try:
        subprocess.run([lms, "unload", "--all"], capture_output=True, timeout=60, check=False)
        return True
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning(f"lms unload failed: {e}")
        return False
