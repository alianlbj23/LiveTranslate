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
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

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


# ---------------------------------------------------------------- graphics card advice

_ADVICE_8GB = {
    "gpu": {"name": "NVIDIA GeForce RTX 4060", "total_gb": 8.0, "free_gb": 7.0},
    "asr": {"key": "whisper:turbo", "label": "Whisper large-v3-turbo", "vram_gb": 2.5},
    "models": [
        {"model": "qwen3.5:9b", "label": "Qwen3.5 9B", "vram_gb": 6.3, "rank": 1, "total_need_gb": 10.3, "level": "too_big"},
        {"model": "qwen3.5:4b", "label": "Qwen3.5 4B", "vram_gb": 3.7, "rank": 3, "total_need_gb": 7.7, "level": "tight"},
        {"model": SAKURA, "label": "Sakura GalTransl 7B", "vram_gb": 4.2, "rank": 2, "total_need_gb": 8.2, "level": "too_big"},
    ],
    "recommended": "qwen3.5:4b",
}


@windows_only
def test_choice_text_marks_what_the_card_can_run():
    r = _ps(
        "$e = [pscustomobject]@{ model='x'; label='Qwen3.5 9B'; size_gb=6.6; note='預設' }; "
        "Write-Output (Format-ModelChoice -Entry $e -Downloaded $true -Current $false -Level 'ok' -Recommended $true); "
        "Write-Output (Format-ModelChoice -Entry $e -Downloaded $true -Current $false -Level 'tight'); "
        "Write-Output (Format-ModelChoice -Entry $e -Downloaded $true -Current $false -Level 'too_big'); "
        "Write-Output (Format-ModelChoice -Entry $e -Downloaded $true -Current $false -Level 'unknown')")
    assert r.returncode == 0, r.stderr
    rec, tight, big, unknown = r.stdout.strip().splitlines()
    assert "【推薦】" in rec and "【接近上限】" in tight and "【顯示卡可能不夠】" in big
    assert "【" not in unknown


_ADVICE_FN = r"""
Add-Type -AssemblyName System.Windows.Forms
$global:asked = @()
$advice = '__ADVICE__' | ConvertFrom-Json
function Open-Form($adv) {
    $ui = New-LauncherForm -Catalog (Get-LauncherCatalog -Root '__ROOT__') -CurrentModel 'qwen3.5:4b' -ModelsDir '__MODELS__' -Advice $adv
    $f = $ui.Form; $f.StartPosition = 'Manual'; $f.Location = New-Object System.Drawing.Point(-5000, -5000)
    $f.ShowInTaskbar = $false; $f.Show(); [System.Windows.Forms.Application]::DoEvents()
    return $ui
}
function Start-With([string]$select, $answer) {
    $ui = Open-Form $advice
    $f = $ui.Form
    $f.Tag.Confirm = { param($text) $global:asked += $text; $answer }.GetNewClosure()
    $i = 0
    foreach ($item in $ui.ModelBox.Items) { if ($item -like "*$select*") { $ui.ModelBox.SelectedIndex = $i }; $i++ }
    $f.Controls.Find('btnStart', $true)[0].PerformClick(); [System.Windows.Forms.Application]::DoEvents()
    $out = @{ action = $ui.Result.Action; model = $ui.Result.Model; open = $f.Visible; selected = [string]$ui.ModelBox.SelectedItem }
    $f.Dispose()
    return $out
}
"""


def _advice_script(tmp_path, body, advice=_ADVICE_8GB):
    models = _fake_models_dir(tmp_path)
    lib = models / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5"
    (lib / "4b").write_text("{}", encoding="utf-8")
    script = (_ADVICE_FN + body).replace("__ROOT__", str(ROOT)).replace("__MODELS__", str(models)) \
        .replace("__ADVICE__", json.dumps(advice, ensure_ascii=False).replace("'", "''"))
    r = _ps_script(tmp_path, script)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@windows_only
def test_window_shows_the_card_and_marks_each_model(tmp_path):
    res = _advice_script(tmp_path, r"""
$ui = Open-Form $advice
$items = @($ui.ModelBox.Items | ForEach-Object { [string]$_ })
$hw = $ui.Form.Controls.Find('lblHardware', $true)[0].Text
$none = Open-Form ('{"gpu": null, "asr": {"label": "SenseVoice Small"}, "models": [], "recommended": null}' | ConvertFrom-Json)
$noneText = $none.Form.Controls.Find('lblHardware', $true)[0].Text
$plain = Open-Form $null
@{ items = $items; hw = $hw; none = $noneText; plainItems = @($plain.ModelBox.Items | ForEach-Object { [string]$_ });
   plainHw = $plain.Form.Controls.Find('lblHardware', $true)[0].Visible } | ConvertTo-Json -Compress
""")
    items = {next(m for m in ("Qwen3.5 9B", "Qwen3.5 4B", "Sakura") if m in t): t for t in res["items"]}
    assert "【顯示卡可能不夠】" in items["Qwen3.5 9B"] and "【顯示卡可能不夠】" in items["Sakura"]
    assert "【推薦】" in items["Qwen3.5 4B"]
    assert "RTX 4060" in res["hw"] and "8.0 GB" in res["hw"] and "Qwen3.5 4B" in res["hw"] and "Whisper" in res["hw"]
    assert "讀不到" in res["none"]
    # no advice (helper failed): the window works as before, without marks
    assert not any("【" in t for t in res["plainItems"]) and res["plainHw"] is False


@windows_only
def test_model_the_card_cannot_hold_needs_confirmation(tmp_path):
    """Picking a model marked 顯示卡可能不夠 asks first (with the numbers); declining keeps
    the current model, agreeing goes on as usual (download question included)."""
    res = _advice_script(tmp_path, r"""
$r = [ordered]@{}
$global:asked = @()
$r['no'] = Start-With 'Qwen3.5 9B' $false
$r['askedNo'] = $global:asked
$global:asked = @()
$r['yes'] = Start-With 'Qwen3.5 9B' $true
$r['askedYes'] = $global:asked
$global:asked = @()
$r['sakura'] = Start-With 'Sakura' $true
$r['askedSakura'] = $global:asked
$r | ConvertTo-Json -Compress -Depth 4
""")
    assert res["no"]["action"] == "exit" and res["no"]["open"] is True and "Qwen3.5 4B" in res["no"]["selected"]
    assert len(res["askedNo"]) == 1 and "8.0 GB" in res["askedNo"][0] and "10.3 GB" in res["askedNo"][0]
    assert "Qwen3.5 4B" in res["askedNo"][0]  # points to the recommended model
    assert {k: res["yes"][k] for k in ("action", "model", "open")} == {"action": "start", "model": "qwen3.5:9b", "open": False}
    assert len(res["askedYes"]) == 1  # downloaded: only the card question
    # too big and not downloaded: the card question, then the download question
    assert res["sakura"]["model"] == SAKURA and len(res["askedSakura"]) == 2
    assert "4.3 GB" in res["askedSakura"][1]


@windows_only
@pytest.mark.skipif(not PYTHON.exists(), reason="the project venv runs the advice helper")
def test_advice_comes_from_the_python_helper(tmp_path):
    settings = tmp_path / "user_settings.json"
    settings.write_text(json.dumps({"asr_engine": "funasr", "funasr_model": "sensevoice-small"}), encoding="utf-8")
    r = _ps(f"$a = Get-HardwareAdvice -Root '{ROOT}' -Python '{PYTHON}' -Settings '{settings}' -TotalMB 8192; "
            "@{ rec = $a.recommended; total = $a.gpu.total_gb; asr = $a.asr.key } | ConvertTo-Json -Compress")
    res = _json(r)
    assert res == {"rec": SAKURA, "total": 8.0, "asr": "funasr:sensevoice-small"}
    # no Python (not installed yet) or a failing helper: no advice, no exception
    r = _ps(f"$a = Get-HardwareAdvice -Root '{ROOT}' -Python '{tmp_path / 'missing.exe'}'; [string]($null -eq $a)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "True"


def test_launcher_and_text_menu_use_the_card_advice():
    gui = GUI.read_text(encoding="utf-8-sig")
    menu = (ROOT / "launcher" / "menu.ps1").read_text(encoding="utf-8-sig")
    show = re.search(r"function Show-LauncherWindow \{(.*?)\n\}", gui, re.S).group(1)
    assert "Get-HardwareAdvice" in show and "-Advice" in show
    assert "Show-LauncherWindow -Root $Root -Python $Python" in menu
    text = re.search(r"function Show-TextMenu \{(.*?)\n\}", menu, re.S).group(1)
    assert "Get-HardwareAdvice" in text and "Resolve-HardwareRequest" in text
