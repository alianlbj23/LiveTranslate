"""LiveTranslate.exe (launcher/LiveTranslate.cs, built by launcher/build_exe.ps1): the
double-click entry without a console window. Built once per session into a temporary
folder and run against stand-in installations (whoami.exe plays pythonw.exe)."""
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
pytestmark = pytest.mark.skipif(sys.platform != "win32" or not CSC.exists(), reason="Windows C# compiler")


@pytest.fixture(scope="module")
def exe(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("exe") / "LiveTranslate.exe"
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(ROOT / "launcher" / "build_exe.ps1"), "-Out", str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0 and out.exists(), r.stdout + r.stderr
    return out


def _root(tmp_path, exe, installed=False) -> Path:
    shutil.copy(exe, tmp_path / "LiveTranslate.exe")
    if installed:
        scripts = tmp_path / ".venv" / "Scripts"
        scripts.mkdir(parents=True)
        shutil.copy(r"C:\Windows\System32\whoami.exe", scripts / "pythonw.exe")
        (tmp_path / ".venv" / ".livetranslate-ready").write_text("x", encoding="ascii")
        (tmp_path / "user_settings.json").write_text("{}", encoding="utf-8")
    return tmp_path


def _setup_script(root: Path, body: str):
    (root / "launcher").mkdir(exist_ok=True)
    (root / "launcher" / "setup.ps1").write_text("\ufeff" + body, encoding="utf-8")


def test_exe_is_a_windows_program_without_a_console(exe):
    data = exe.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[pe:pe + 4] == b"PE\0\0"
    subsystem = struct.unpack_from("<H", data, pe + 24 + 68)[0]
    assert subsystem == 2  # IMAGE_SUBSYSTEM_WINDOWS_GUI (3 would be a console program)


def test_probe_tells_an_incomplete_installation(tmp_path, exe):
    root = _root(tmp_path, exe)
    assert subprocess.run([str(root / "LiveTranslate.exe"), "--probe"], timeout=30).returncode == 2
    _root(tmp_path, exe, installed=True)
    assert subprocess.run([str(root / "LiveTranslate.exe"), "--probe"], timeout=30).returncode == 0
    (root / "user_settings.json").unlink()  # settings never written: not installed
    assert subprocess.run([str(root / "LiveTranslate.exe"), "--probe"], timeout=30).returncode == 2


def test_installed_app_starts_at_once(tmp_path, exe):
    root = _root(tmp_path, exe, installed=True)
    t0 = time.monotonic()
    assert subprocess.run([str(root / "LiveTranslate.exe")], timeout=30).returncode == 0
    assert time.monotonic() - t0 < 15


def test_install_runs_the_setup_and_then_starts_the_app(tmp_path, exe):
    root = _root(tmp_path, exe)
    _setup_script(root, (
        "Write-Host '[12:00:00] 建立 .venv 並安裝套件...'\n"
        "New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '..\\.venv\\Scripts') | Out-Null\n"
        "Copy-Item 'C:\\Windows\\System32\\whoami.exe' (Join-Path $PSScriptRoot '..\\.venv\\Scripts\\pythonw.exe')\n"
        "Set-Content (Join-Path $PSScriptRoot '..\\.venv\\.livetranslate-ready') 'x'\n"
        "Set-Content (Join-Path $PSScriptRoot '..\\user_settings.json') '{}'\n"
        "exit 0\n"))
    p = subprocess.run([str(root / "LiveTranslate.exe"), "--install"], timeout=90)
    assert p.returncode == 0  # setup succeeded, the app was started, the window closed itself
    assert (root / "user_settings.json").exists()


def test_failed_setup_keeps_the_window_open(tmp_path, exe):
    root = _root(tmp_path, exe)
    _setup_script(root, "Set-Content (Join-Path $PSScriptRoot 'ran.txt') 'x'\nWrite-Host '  [錯誤] PyTorch 安裝失敗'\nexit 3\n")
    p = subprocess.Popen([str(root / "LiveTranslate.exe"), "--install"])
    try:
        end = time.monotonic() + 60
        while not (root / "launcher" / "ran.txt").exists() and time.monotonic() < end:
            time.sleep(0.2)
        time.sleep(2)
        assert (root / "launcher" / "ran.txt").exists()
        assert p.poll() is None, "after a failure the window waits for the user (retry / close)"
        assert not (root / ".venv").exists()
    finally:
        p.kill()
