"""Install-time graphics card choice (launcher/vram.ps1, dot-sourced by setup.ps1): the
defaults a fresh install picks for the card. The decision comes from hardware_profile.py
through launcher/hardware_advice.py; these tests run the PowerShell side against it. The
start-time memory check moved into the app (tests/test_local_model_prep.py)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VRAM = ROOT / "launcher" / "vram.ps1"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SAKURA = "hf.co/SakuraLLM/Sakura-GalTransl-7B-v3:IQ4_XS"

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="PowerShell launcher")
needs_venv = pytest.mark.skipif(not PYTHON.exists(), reason="the project venv runs the advice helper")


def _ps(command: str) -> subprocess.CompletedProcess:
    script = ("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $ErrorActionPreference = 'Stop'; "
              f". '{VRAM}'; {command}")
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)


def _json(r):
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@windows_only
@needs_venv
@pytest.mark.parametrize("total_mb, asr, model", [
    (12227, "whisper-turbo", "qwen3.5:9b"),
    (10240, "whisper-turbo", SAKURA),
    (8192, "sensevoice", SAKURA),
])
def test_install_choice_follows_the_card(total_mb, asr, model):
    c = _json(_ps(f"Get-InstallChoice -Python '{PYTHON}' -Root '{ROOT}' -TotalMB {total_mb} | ConvertTo-Json -Compress"))
    assert (c["asr_choice"], c["model"]) == (asr, model)
    assert c["label"] and c["size_gb"] > 0 and c["message"]


@windows_only
def test_install_choice_without_a_working_helper_keeps_the_old_defaults(tmp_path):
    r = _ps(f"Get-InstallChoice -Python '{tmp_path / 'missing.exe'}' -Root '{ROOT}' -TotalMB 8192 | ConvertTo-Json -Compress")
    c = _json(r)
    assert (c["asr_choice"], c["model"], c["level"]) == ("whisper-turbo", "qwen3.5:9b", "unknown")


def test_installer_picks_the_defaults_for_the_card():
    setup = (ROOT / "launcher" / "setup.ps1").read_text(encoding="utf-8-sig")
    step4 = setup[setup.index("# ---------- 4."):]
    assert "Get-InstallChoice" in step4
    # the chosen model becomes the active one in user_settings.json; the app downloads it
    assert re.search(r"write_settings\.py[^\n]*\$choice\.model[^\n]*\$choice\.asr_choice", step4)
    assert "selected_model.txt" not in setup and "model.ps1" not in setup
    turbo = step4.index("download_whisper_turbo.py")
    # Whisper turbo is downloaded only when chosen, or when the existing settings already use it
    assert "whisper-turbo" in step4[:turbo] and "whisper:turbo" in step4[:turbo]
    assert "LiveTranslate.bat" not in setup  # the entries are LiveTranslate.exe and the debug .bat
    step5 = setup[setup.index("# ---------- 5."):]
    assert "build_exe.ps1" in step5  # a git checkout gets LiveTranslate.exe here


@windows_only
def test_advice_helper_that_hangs_is_given_up(tmp_path):
    """Review finding: the launcher waited forever when reading the card hung."""
    (tmp_path / "launcher").mkdir()
    (tmp_path / "launcher" / "hardware_advice.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    r = _ps("$t = [Diagnostics.Stopwatch]::StartNew(); "
            f"$a = Invoke-HardwareAdvice -Python '{sys.executable}' -Root '{tmp_path}' -Arguments @('launcher') -TimeoutMs 1500; "
            "@{ none = ($null -eq $a); ms = $t.ElapsedMilliseconds } | ConvertTo-Json -Compress")
    res = _json(r)
    assert res["none"] is True and res["ms"] < 10000
