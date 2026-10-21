"""Starting LiveTranslate's own programs without a console window: LiveTranslate itself
(from the benchmark window) and the best-combination benchmark window (from the settings).
No Qt."""
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("LiveTranslate.Launch")

ROOT = Path(__file__).resolve().parent


def pythonw(root: Path = ROOT) -> str:
    """The venv's pythonw.exe (no console window), else this interpreter."""
    p = Path(root) / ".venv" / "Scripts" / "pythonw.exe"
    return str(p) if p.exists() else sys.executable


def _start(args, root: Path, popen) -> bool:
    try:
        popen(args, cwd=str(root), close_fds=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as e:
        log.error(f"Cannot start {args[1:]}: {e}")
        return False
    return True


def start_livetranslate(root: Path = ROOT, popen=subprocess.Popen) -> bool:
    return _start([pythonw(root), "main.py"], root, popen)


def open_bench_window(root: Path = ROOT, popen=subprocess.Popen) -> bool:
    return _start([pythonw(root), "-I", str(Path(root) / "launcher" / "bench_window.py")], root, popen)
