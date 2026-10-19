"""LiveTranslate.bat opens a small window (launcher/gui.ps1) instead of the text menu:
a big Start button (Enter), the translation model list from local_models.json with
download state / size, and check / guide / exit buttons. The decision logic lives in
plain functions so it can be tested without clicking a window."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "launcher" / "gui.ps1"
SAKURA = "hf.co/SakuraLLM/Sakura-GalTransl-7B-v3:IQ4_XS"

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="PowerShell / WinForms launcher")


def _ps(command: str) -> subprocess.CompletedProcess:
    script = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $ErrorActionPreference = 'Stop'; "
        f". '{GUI}'; {command}"
    )
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA", "-Command", script],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)


def _json(r):
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _fake_models_dir(tmp_path):
    lib = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5"
    lib.mkdir(parents=True)
    (lib / "9b").write_text("{}", encoding="utf-8")
    return tmp_path


@windows_only
def test_catalog_is_read_from_local_models_json():
    rows = _json(_ps(f"Get-LauncherCatalog -Root '{ROOT}' | ConvertTo-Json -Compress"))
    assert [r["model"] for r in rows][:2] == ["qwen3.5:9b", "qwen3.5:4b"]
    assert SAKURA in [r["model"] for r in rows]


@windows_only
def test_download_state_comes_from_ollama_manifests(tmp_path):
    d = _fake_models_dir(tmp_path)
    sak = tmp_path / "manifests" / "hf.co" / "SakuraLLM" / "Sakura-GalTransl-7B-v3"
    r = _json(_ps(
        f"@((Test-OllamaModelDownloaded -Model 'qwen3.5:9b' -ModelsDir '{d}'), "
        f"(Test-OllamaModelDownloaded -Model 'qwen3.5:4b' -ModelsDir '{d}'), "
        f"(Test-OllamaModelDownloaded -Model '{SAKURA}' -ModelsDir '{d}')) | ConvertTo-Json -Compress"))
    assert r == [True, False, False]
    sak.mkdir(parents=True)
    (sak / "IQ4_XS").write_text("{}", encoding="utf-8")
    assert _ps(f"Test-OllamaModelDownloaded -Model '{SAKURA}' -ModelsDir '{d}'").stdout.strip() == "True"


@windows_only
def test_choice_text_shows_state_and_size():
    r = _ps(
        "$e = [pscustomobject]@{ model='x'; label='Sakura GalTransl 7B'; size_gb=4.3; note='日翻中專用' }; "
        "Write-Output (Format-ModelChoice -Entry $e -Downloaded $false -Current $false); "
        "Write-Output (Format-ModelChoice -Entry $e -Downloaded $true -Current $true)")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert "Sakura GalTransl 7B" in lines[0] and "4.3 GB" in lines[0] and "需下載" in lines[0]
    assert "已下載" in lines[1] and "使用中" in lines[1]


@windows_only
@pytest.mark.parametrize("selected, current, downloaded, confirm, expected", [
    ("qwen3.5:9b", "qwen3.5:9b", True, True, "none"),      # already in use
    ("qwen3.5:4b", "qwen3.5:9b", True, False, "switch"),    # downloaded: no question
    (SAKURA, "qwen3.5:9b", False, True, "switch"),          # needs download: user agreed
    (SAKURA, "qwen3.5:9b", False, False, "none"),           # needs download: user declined
])
def test_switch_decision(selected, current, downloaded, confirm, expected):
    asked = "$true" if confirm else "$false"
    dl = "$true" if downloaded else "$false"
    r = _ps(
        f"$e = [pscustomobject]@{{ model='{selected}'; label='L'; size_gb=4.3; note='' }}; "
        f"Resolve-SwitchRequest -Entry $e -CurrentModel '{current}' -Downloaded {dl} "
        f"-Confirm {{ param($text) {asked} }}")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == expected


@windows_only
def test_window_builds_with_models_and_default_button(tmp_path):
    d = _fake_models_dir(tmp_path)
    r = _json(_ps(
        f"$ui = New-LauncherForm -Catalog (Get-LauncherCatalog -Root '{ROOT}') -CurrentModel 'qwen3.5:9b' -ModelsDir '{d}'; "
        "@{ items = $ui.ModelBox.Items.Count; selected = $ui.ModelBox.SelectedIndex; "
        "accept = $ui.Form.AcceptButton.Name; cancel = $ui.Form.CancelButton.Name; "
        "buttons = @($ui.Form.Controls.Find('btnVerify', $true).Count, $ui.Form.Controls.Find('btnGuide', $true).Count, "
        "$ui.Form.Controls.Find('btnSwitch', $true).Count) } | ConvertTo-Json -Compress"))
    assert r["items"] >= 3 and r["selected"] == 0
    assert r["accept"] == "btnStart" and r["cancel"] == "btnExit"
    assert r["buttons"] == [1, 1, 1]


def test_menu_uses_the_window_with_a_text_fallback():
    menu = (ROOT / "launcher" / "menu.ps1").read_text(encoding="utf-8-sig")
    assert "Show-LauncherWindow" in menu
    assert re.search(r"catch\s*\{[^}]*Read-Host|Show-TextMenu", menu, re.S), "a text menu must remain as fallback"
