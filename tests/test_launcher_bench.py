"""Launcher entry for the best-combination benchmark: the 最佳組合評測 button, the
benchmark menu (launcher/bench.ps1) and the Python entry point."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from test_launcher_gui import _CLICK_FN, _ps_script

ROOT = Path(__file__).resolve().parents[1]
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="PowerShell / WinForms launcher")


@windows_only
def test_bench_button_returns_the_bench_action(tmp_path):
    models = tmp_path / "models"
    lib = models / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5"
    lib.mkdir(parents=True)
    (lib / "9b").write_text("{}", encoding="utf-8")
    body = (_CLICK_FN + r"""
$r = Click-On 'btnBench' $null
$ui = New-LauncherForm -Catalog (Get-LauncherCatalog -Root '__ROOT__') -CurrentModel 'qwen3.5:9b' -ModelsDir '__MODELS__'
$row = @('btnVerify', 'btnBench', 'btnGuide', 'btnExit') | ForEach-Object {
    $b = $ui.Form.Controls.Find($_, $true)[0]; @{ name = $_; x = $b.Left; right = $b.Right; text = $b.Text } }
$ui.Form.Dispose()
@{ click = $r; row = $row } | ConvertTo-Json -Compress -Depth 4
""").replace("__ROOT__", str(ROOT)).replace("__MODELS__", str(models))
    r = _ps_script(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["click"]["action"] == "bench" and res["click"]["open"] is False
    row = res["row"]
    assert [b["text"] for b in row] == ["檢查環境", "最佳組合評測", "使用說明", "離開"]
    assert all(row[i]["right"] < row[i + 1]["x"] for i in range(3))  # no overlap
    assert row[-1]["right"] <= 540


@windows_only
def test_bench_menu_numbers_map_to_actions(tmp_path):
    script = tmp_path / "map.ps1"
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f". '{ROOT / 'launcher' / 'bench.ps1'}'\n"
        "@('1','2','3','4','5','0','9','',' 2 ') | ForEach-Object { [string](Get-BenchAction -Choice $_) } | ConvertTo-Json -Compress\n",
        encoding="utf-8-sig")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == \
        ["download", "start", "resume", "report", "apply", "back", "", "", "start"]


def test_menu_wires_the_bench_menu():
    menu = (ROOT / "launcher" / "menu.ps1").read_text(encoding="utf-8-sig")
    assert re.search(r'\. \(Join-Path \$Root "launcher\\bench\.ps1"\)', menu)
    assert re.search(r'\. \(Join-Path \$Root "launcher\\process\.ps1"\)', menu)
    assert re.search(r'"bench"\s*\{\s*Show-BenchMenu -Root \$Root -Python \$Python\s*\}', menu)
    assert "$benchNo" in menu and 'Action = "bench"' in menu  # text-menu fallback too


def test_bench_menu_script_runs_python_on_the_console_and_has_no_empty_catch():
    bench = (ROOT / "launcher" / "bench.ps1").read_text(encoding="utf-8-sig")
    assert "Start-TrackedProcess" in bench and "Wait-TrackedProcess" in bench
    assert "| Out-Host" not in bench and "| Out-String" not in bench  # prompts must not be buffered
    assert not re.search(r"catch\s*\{\s*\}", bench)
    for item in ("下載測試資料", "開始評測", "從上次中斷的地方繼續", "查看報告", "套用推薦組合"):
        assert item in bench


def _entry(*args, env_dir=None):
    import os

    env = dict(os.environ)
    if env_dir:
        env["LIVETRANSLATE_BENCH_DIR"] = str(env_dir)
    return subprocess.run([sys.executable, "-I", str(ROOT / "launcher" / "bench_run.py"), *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, env=env)


def test_entry_point_rejects_unknown_actions():
    r = _entry("nonsense")
    assert r.returncode == 2 and "status" in r.stdout


def test_entry_point_status_shows_the_data_state(tmp_path):
    r = _entry("status", env_dir=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "測試資料" in r.stdout and "動畫 0/200" in r.stdout and "自己的錄音：0 檔" in r.stdout


def test_bench_data_folders_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "bench_clips/" in ignore and "bench_results/" in ignore
