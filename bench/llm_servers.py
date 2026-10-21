"""Translation model servers: classify a model entry, and load / unload local models so
each one is measured alone (Ollama) or at least waited out (LM Studio).

The Ollama part lives in ollama_server.py (shared with the app): when the benchmark starts
(or, with the user's consent, restarts) Ollama itself, it uses the same fast settings as
LiveTranslate, so the numbers match real use; an Ollama the user started from the tray
keeps its own settings. The names are re-exported here so the benchmark and its tests
keep calling llm_servers.<name>."""
import logging
import shutil
import subprocess

import ollama_server
from ollama_server import (  # noqa: F401  (re-exported)
    OLLAMA_EXE,
    PID_FILE,
    WATCHDOG,
    Ollama,
    fast_env,
    pid_file_server,
    stop_ollama,
    stop_tray_ollama,
)

log = logging.getLogger("LiveTranslate.Bench.LLM")
_port_open = ollama_server.port_open


def start_fast_ollama(port: int, log_dir):
    return ollama_server.start_fast_ollama(port, log_dir)


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


def lmstudio_unload_all() -> bool:
    """`lms unload --all` when the LM Studio CLI is installed. False (logged) when it is
    not installed or failed: the next measurement then waits for VRAM instead."""
    lms = shutil.which("lms")
    if not lms:
        log.info("LM Studio has no CLI (lms); waiting for VRAM instead of unloading")
        return False
    try:
        done = subprocess.run([lms, "unload", "--all"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning(f"lms unload failed: {e}")
        return False
    if done.returncode != 0:
        log.warning(f"lms unload --all exited with {done.returncode}: {(done.stderr or done.stdout or '').strip()[:200]}")
        return False
    return True
