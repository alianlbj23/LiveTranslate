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


def _ps_script(tmp_path, body: str) -> subprocess.CompletedProcess:
    """Run like LiveTranslate.bat does: gui.ps1 dot-sourced into a *script* file scope
    (-File), not -Command. Button handlers run inside the WinForms message loop, where
    closures and script-scope functions behave differently than in a console command."""
    script = tmp_path / "click_test.ps1"
    script.write_text(
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
        "$ErrorActionPreference = 'Stop'\n"
        f". '{GUI}'\n" + body, encoding="utf-8-sig")
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA", "-File", str(script)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)


_CLICK_FN = r"""
Add-Type -AssemblyName System.Windows.Forms
$global:asked = $null
function Click-On([string]$button, [string]$select, $answer = $null) {
    $ui = New-LauncherForm -Catalog (Get-LauncherCatalog -Root '__ROOT__') -CurrentModel 'qwen3.5:9b' -ModelsDir '__MODELS__'
    $f = $ui.Form
    if ($null -ne $answer) { $f.Tag.Confirm = { param($text) $global:asked = $text; $answer }.GetNewClosure() }
    $f.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
    $f.Location = New-Object System.Drawing.Point(-5000, -5000)
    $f.ShowInTaskbar = $false
    $f.Show()
    [System.Windows.Forms.Application]::DoEvents()
    if ($select) {
        $i = 0
        foreach ($item in $ui.ModelBox.Items) { if ($item -like "*$select*") { $ui.ModelBox.SelectedIndex = $i }; $i++ }
    }
    $f.Controls.Find($button, $true)[0].PerformClick()
    [System.Windows.Forms.Application]::DoEvents()
    $out = @{ action = $ui.Result.Action; model = $ui.Result.Model; open = $f.Visible; selected = [string]$ui.ModelBox.SelectedItem }
    $f.Dispose()
    return $out
}
"""

_CLICK = _CLICK_FN + r"""
$r = [ordered]@{}
foreach ($b in 'btnStart', 'btnVerify', 'btnGuide', 'btnExit') { $r[$b] = Click-On $b $null }
$r['switch4b'] = Click-On 'btnSwitch' 'Qwen3.5 4B'
$r['switchSame'] = Click-On 'btnSwitch' 'Qwen3.5 9B'
$r['sakuraYes'] = Click-On 'btnSwitch' 'Sakura' $true
$r['sakuraNo'] = Click-On 'btnSwitch' 'Sakura' $false
$r | ConvertTo-Json -Compress -Depth 4
"""


@windows_only
def test_every_button_works_when_clicked_like_the_real_launcher(tmp_path):
    """Regression: 2026-10-20 clicking 檢查環境 / 使用說明 / 離開 raised
    "The property 'Action' cannot be found on this object" (closure lost $result)."""
    models = tmp_path / "models"
    lib = models / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5"
    lib.mkdir(parents=True)
    for tag in ("9b", "4b"):
        (lib / tag).write_text("{}", encoding="utf-8")
    body = _CLICK.replace("__ROOT__", str(ROOT)).replace("__MODELS__", str(models))
    r = _ps_script(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["btnStart"]["action"] == "start" and res["btnStart"]["open"] is False
    assert res["btnVerify"]["action"] == "verify"
    assert res["btnGuide"]["action"] == "guide"
    assert res["btnExit"]["action"] == "exit"
    assert {k: res["switch4b"][k] for k in ("action", "model", "open")} == {"action": "switch", "model": "qwen3.5:4b", "open": False}
    # already in use: nothing to do, no blocking message box, window stays open
    assert res["switchSame"]["action"] == "exit" and res["switchSame"]["open"] is True
    # not downloaded: asks first (size in the question), switches only on "yes"
    assert {k: res["sakuraYes"][k] for k in ("action", "model", "open")} == {"action": "switch", "model": SAKURA, "open": False}
    assert res["sakuraNo"]["action"] == "exit" and res["sakuraNo"]["open"] is True


_START_CASES = r"""
$r = [ordered]@{}
$r['same'] = Click-On 'btnStart' 'Qwen3.5 9B'
$r['other'] = Click-On 'btnStart' 'Qwen3.5 4B'
$global:asked = $null
$r['sakuraYes'] = Click-On 'btnStart' 'Sakura' $true
$r['askedYes'] = $global:asked
$r['sakuraNo'] = Click-On 'btnStart' 'Sakura' $false
$r | ConvertTo-Json -Compress -Depth 4
"""


@windows_only
def test_start_uses_the_model_selected_in_the_list(tmp_path):
    """Regression: 2026-10-20 picking Sakura in the list and pressing 啟動字幕 started
    with the previous model (qwen3.5:9b); only 切換 applied the list. Start must use
    what is selected: switch (download after asking) first, then start."""
    models = tmp_path / "models"
    lib = models / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5"
    lib.mkdir(parents=True)
    for tag in ("9b", "4b"):
        (lib / tag).write_text("{}", encoding="utf-8")
    body = (_CLICK_FN + _START_CASES).replace("__ROOT__", str(ROOT)).replace("__MODELS__", str(models))
    r = _ps_script(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    pick = lambda d: {k: d[k] for k in ("action", "model", "open")}  # noqa: E731
    # the model already in use: plain start, nothing to switch
    assert pick(res["same"]) == {"action": "start", "model": None, "open": False}
    # another downloaded model: start with it, no question
    assert pick(res["other"]) == {"action": "start", "model": "qwen3.5:4b", "open": False}
    # not downloaded: asks with the size, then starts with it
    assert pick(res["sakuraYes"]) == {"action": "start", "model": SAKURA, "open": False}
    assert "4.3 GB" in (res["askedYes"] or "")
    # declined: nothing starts, the window stays and the list goes back to the current model
    assert res["sakuraNo"]["action"] == "exit" and res["sakuraNo"]["open"] is True
    assert "Qwen3.5 9B" in res["sakuraNo"]["selected"]


def test_menu_switches_before_starting_and_stops_on_failure():
    """menu.ps1: a start request that carries a model switches first; when the switch
    fails (download error) it must not start with the old model silently."""
    menu = (ROOT / "launcher" / "menu.ps1").read_text(encoding="utf-8-sig")
    start = re.search(r'"start"\s*\{(.*?)\n\s*\}\s*\n\s*"verify"', menu, re.S)
    assert start, "start branch not found"
    body = start.group(1)
    assert "$r.Model" in body and "Switch-Model" in body
    assert body.index("Switch-Model") < body.index("$StartScript")
    fn = re.search(r"function Switch-Model \{(.*?)\n\}", menu, re.S).group(1)
    assert "return $false" in fn and "return $true" in fn
    # native output must not leak into the return value
    assert re.search(r"& \$Ollama pull \$Model \| Out-Host", fn)
    assert re.search(r"& \$Python [^\n]*\| Out-Host", fn)
    # a failed settings update is a failed switch (checked before the choice is saved)
    helper = fn.index("& $Python")
    assert fn.find("$LASTEXITCODE", helper) != -1
    assert fn.index("$LASTEXITCODE", helper) < fn.index("Set-LauncherModel")
    # an already downloaded model is not pulled again (works offline)
    assert fn.find("Test-OllamaModelDownloaded") != -1
    assert fn.index("Test-OllamaModelDownloaded") < fn.index("& $Ollama pull")
    assert not re.search(r"catch\s*\{\s*\}", menu), "empty catch blocks hide errors"


@windows_only
def test_current_model_outside_the_list_is_not_switched_by_enter(tmp_path):
    """A custom model (not in local_models.json) is in use: nothing is preselected, and
    pressing 啟動字幕 keeps it instead of switching to the first catalog entry."""
    models = _fake_models_dir(tmp_path)
    body = (
        "Add-Type -AssemblyName System.Windows.Forms\n"
        f"$ui = New-LauncherForm -Catalog (Get-LauncherCatalog -Root '{ROOT}') -CurrentModel 'my-custom:7b' -ModelsDir '{models}'\n"
        "$f = $ui.Form; $f.StartPosition = 'Manual'; $f.Location = New-Object System.Drawing.Point(-5000, -5000)\n"
        "$f.ShowInTaskbar = $false; $f.Show(); [System.Windows.Forms.Application]::DoEvents()\n"
        "$before = $ui.ModelBox.SelectedIndex\n"
        "$f.Controls.Find('btnStart', $true)[0].PerformClick(); [System.Windows.Forms.Application]::DoEvents()\n"
        "@{ before = $before; action = $ui.Result.Action; model = $ui.Result.Model; open = $f.Visible } | ConvertTo-Json -Compress\n"
        "$f.Dispose()\n")
    r = _ps_script(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res == {"before": -1, "action": "start", "model": None, "open": False}
