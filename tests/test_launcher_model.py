"""The translation model chosen in the menu lives in a git-ignored file
(launcher/selected_model.txt). Switching models must never rewrite a tracked
script: a dirty working tree makes launcher/update.ps1 skip every auto-update."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launcher"
MODEL_PS1 = LAUNCHER / "model.ps1"
DEFAULT = "qwen3.5:9b"

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="PowerShell launcher scripts")


def _read(name: str) -> str:
    return (LAUNCHER / name).read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# static: no tracked file is used as model storage
# ---------------------------------------------------------------------------

def test_menu_does_not_rewrite_the_start_script():
    menu = _read("menu.ps1")
    assert "WriteAllText($StartScript" not in menu
    assert not re.search(r"Set-Content[^\n]*\$StartScript", menu)
    assert "Set-LauncherModel" in menu


def test_start_script_reads_model_from_selection_file():
    start = _read("start_jp_subtitles.ps1")
    assert not re.search(r'^\s*\$Model\s*=\s*"', start, re.M), "model name hard-coded in a tracked script"
    assert "Get-LauncherModel" in start


def test_verify_uses_the_selected_model():
    verify = _read("verify.ps1")
    assert '"qwen3.5:9b"' not in verify
    assert "Get-LauncherModel" in verify


def test_selection_file_is_git_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "launcher/selected_model.txt" in ignored


# ---------------------------------------------------------------------------
# functional: launcher/model.ps1 through PowerShell
# ---------------------------------------------------------------------------

def _ps(command: str) -> subprocess.CompletedProcess:
    script = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        f". '{MODEL_PS1}'; {command}"
    )
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


def _get(path: Path) -> subprocess.CompletedProcess:
    return _ps(f"Write-Output (Get-LauncherModel -Path '{path}')")


@windows_only
def test_missing_file_gives_default(tmp_path):
    r = _get(tmp_path / "selected_model.txt")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == DEFAULT


@windows_only
def test_set_then_get_round_trip(tmp_path):
    f = tmp_path / "selected_model.txt"
    r = _ps(f"Write-Output (Set-LauncherModel -Path '{f}' -Model 'qwen3.5:4b')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "True"
    assert f.read_bytes() == b"qwen3.5:4b", "UTF-8 without BOM, no trailing newline"
    assert _get(f).stdout.strip().splitlines()[-1] == "qwen3.5:4b"


@windows_only
def test_bom_and_whitespace_are_tolerated(tmp_path):
    f = tmp_path / "selected_model.txt"
    f.write_bytes(b"\xef\xbb\xbf  hf.co/unsloth/Qwen3.5-9B-MTP-GGUF:Q4_K_M \r\n")
    assert _get(f).stdout.strip().splitlines()[-1] == "hf.co/unsloth/Qwen3.5-9B-MTP-GGUF:Q4_K_M"


@windows_only
@pytest.mark.parametrize("content", ['qwen"; Remove-Item C:\\', "", "   ", "a" * 200, "-rf", "模型"])
def test_invalid_file_content_falls_back_to_default_with_warning(tmp_path, content):
    """Error path: the file is user-editable; anything that is not a plain model tag
    must not reach `ollama pull` or the JSON request bodies."""
    f = tmp_path / "selected_model.txt"
    f.write_text(content, encoding="utf-8")
    r = _get(f)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[-1] == DEFAULT
    if content.strip():
        assert any("selected_model.txt" in ln for ln in lines[:-1]), "a warning should name the file"


@windows_only
def test_set_rejects_invalid_name_and_keeps_previous_choice(tmp_path):
    f = tmp_path / "selected_model.txt"
    f.write_bytes(b"qwen3.5:4b")
    r = _ps(f"Write-Output (Set-LauncherModel -Path '{f}' -Model 'x\"; calc')")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "False"
    assert f.read_bytes() == b"qwen3.5:4b"
