"""Graphics card checks in the launcher scripts (launcher/vram.ps1): the defaults a fresh
install picks for the card (setup.ps1) and the free-memory check before subtitles start
(start_jp_subtitles.ps1). The decisions come from hardware_profile.py through
launcher/hardware_advice.py; these tests run the PowerShell side against it."""
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


@windows_only
@needs_venv
def test_start_check_uses_the_pair_in_use(tmp_path):
    settings = tmp_path / "user_settings.json"
    settings.write_text(json.dumps({"asr_engine": "funasr", "funasr_model": "sensevoice-small"}), encoding="utf-8")
    need = _json(_ps(f"Get-StartupVramNeed -Python '{PYTHON}' -Root '{ROOT}' -Settings '{settings}' "
                     "-Model 'qwen3.5:4b' | ConvertTo-Json -Compress"))
    assert need["free_need_mb"] == round((0.7 + 3.7 + 0.5) * 1024)
    assert need["asr_label"] == "SenseVoice Small" and need["model_label"] == "Qwen3.5 4B" and need["gpu_index"] == 0
    r = _ps(f"Get-StartupVramNeed -Python '{tmp_path / 'missing.exe'}' -Root '{ROOT}' -Settings '{settings}' -Model 'x'")
    assert r.returncode == 0 and r.stdout.strip() == ""


@windows_only
def test_low_free_memory_names_the_pair_and_points_to_the_recommendation():
    r = _ps("Test-LowVram -FreeMB 4096 -ThresholdMB 9000 -Pair 'Whisper large-v3-turbo＋Qwen3.5 9B'")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert out.strip().splitlines()[-1] == "True"
    assert "4.0 GB" in out and "8.8 GB" in out and "Qwen3.5 9B" in out and "【推薦】" in out
    assert "按 3" not in out  # the old advice pointed at the wrong menu number
    assert _ps("Test-LowVram -FreeMB 9500 -ThresholdMB 9000").stdout.strip().splitlines()[-1] == "False"
    assert _ps("Test-LowVram -FreeMB -1").stdout.strip().splitlines()[-1] == "False"  # unknown: no warning


def test_installer_picks_the_defaults_for_the_card():
    setup = (ROOT / "launcher" / "setup.ps1").read_text(encoding="utf-8-sig")
    step4 = setup[setup.index("# ---------- 4."):]
    assert "Get-InstallChoice" in step4
    assert re.search(r"write_settings\.py[^\n]*\$choice\.model[^\n]*\$choice\.asr_choice", step4)
    assert "Set-LauncherModel" in step4  # the start script pulls the chosen model, not always 9B
    turbo = step4.index("download_whisper_turbo.py")
    # Whisper turbo is downloaded only when chosen, or when the existing settings already use it
    assert "whisper-turbo" in step4[:turbo] and "whisper:turbo" in step4[:turbo]
    assert "按 3" not in setup and "3) 切換" not in setup


def test_start_script_checks_free_memory_for_the_pair_in_use():
    start = (ROOT / "launcher" / "start_jp_subtitles.ps1").read_text(encoding="utf-8-sig")
    assert "Get-StartupVramNeed" in start and "-Pair" in start
    assert "約 6 GB" not in start  # the download size comes from the catalog
    # measured once Ollama is ready; models it already holds are swapped out, so they count as free
    check = start.index("Get-FreeVramMB -Index")
    assert start.index("# ---- Step 2:") < check < start.index("# ---- Step 4:")
    assert "Get-OllamaLoadedVramMB" in start[check:start.index("# ---- Step 4:")]


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


@windows_only
def test_ollama_that_does_not_answer_counts_as_nothing_loaded():
    r = _ps("Get-OllamaLoadedVramMB -Port 1")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "0"
