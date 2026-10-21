"""launcher/setup.ps1, the one installer (LiveTranslate.exe runs it hidden, the debug entry in
a console). Review findings of 2026-10-21."""
import ctypes
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = (ROOT / "launcher" / "setup.ps1").read_text(encoding="utf-8-sig")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows installer")
def test_second_installation_at_the_same_time_stops_at_once(tmp_path):
    """Two installations writing the same .venv (exe window + debug entry, or two exe
    windows) can break it; the second one stops before doing anything.

    Runs a copy in a folder with a space in its name: should the lock ever be missing, the
    copy stops at the path check instead of installing anything."""
    root = tmp_path / "copy with space"
    (root / "launcher").mkdir(parents=True)
    for name in ("setup.ps1", "vram.ps1"):
        shutil.copy(ROOT / "launcher" / name, root / "launcher" / name)
    k = ctypes.windll.kernel32
    k.CreateMutexW.restype = ctypes.c_void_p
    handle = k.CreateMutexW(None, True, "Global\\LiveTranslate-Setup")
    assert handle
    try:
        t0 = time.monotonic()
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                            "[Console]::OutputEncoding = [Text.Encoding]::UTF8; & '" + str(root / "launcher" / "setup.ps1")
                            + "' -Yes -SkipModels; exit $LASTEXITCODE"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        assert r.returncode == 1 and "另一個安裝" in r.stdout, r.stdout
        assert "檢查環境" not in r.stdout  # stopped before the first step
        assert time.monotonic() - t0 < 60
    finally:
        k.ReleaseMutex(ctypes.c_void_p(handle))
        k.CloseHandle(ctypes.c_void_p(handle))


def test_driver_cuda_version_is_read_from_old_and_new_nvidia_smi():
    """Driver 610.88 prints "CUDA UMD Version: 13.3" instead of "CUDA Version: 12.8"; the old
    pattern read 0 and the installer told an RTX 50 owner to update a current driver."""
    line = next(ln for ln in SETUP.splitlines() if "$smi -match" in ln)
    pattern = re.search(r"-match '([^']+)'", line).group(1)
    old = "| NVIDIA-SMI 581.57   Driver Version: 581.57   CUDA Version: 13.0     |"
    new = "| NVIDIA-SMI 610.88   KMD Version: 610.88   CUDA UMD Version: 13.3     |"
    assert re.search(pattern, old).group(1) == "13.0"
    assert re.search(pattern, new).group(1) == "13.3"


def test_no_empty_catch_blocks():
    assert not re.search(r"catch\s*\{\s*\}", SETUP)


def test_ready_marker_is_written_last():
    """Review: written before the speech model and the settings, a failed reinstall still
    looked installed."""
    marker = SETUP.rindex("Set-Content -LiteralPath $Ready")
    assert marker > SETUP.index("# ---------- 4.") and marker > SETUP.index("# ---------- 5.")
    assert SETUP.index("Remove-Item -LiteralPath $Ready") < SETUP.index("pip check")


def test_temp_folder_is_restored_after_it_is_removed():
    cleanup = SETUP.index("Remove-Item -Recurse -Force $env:PIP_CACHE_DIR, $env:TMP")
    after = SETUP[cleanup:SETUP.index("# ---------- 4.")]
    assert "$env:TMP = $OldTmp" in after and "$env:TEMP = $OldTemp" in after


def test_install_path_with_brackets_is_refused():
    check = SETUP[SETUP.index("# ---------- 0."):SETUP.index("$drive = ")]
    assert "[\\[\\]]" in check
