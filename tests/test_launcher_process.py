"""The launcher must read LiveTranslate's real exit code. On Windows PowerShell 5.1 a
Process object from `Start-Process -PassThru` reports ExitCode as $null after exit
unless its Handle was touched right after starting, so a normal close (exit 0) was
reported as "LiveTranslate 異常結束（exit code ）" and the window waited for Enter."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launcher"
PROCESS_PS1 = LAUNCHER / "process.ps1"

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="PowerShell launcher scripts")


def _ps(command: str) -> subprocess.CompletedProcess:
    script = f"[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; . '{PROCESS_PS1}'; {command}"
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


def _exit_code_of(tmp_path, code: int) -> str:
    child = tmp_path / f"exit_{code}.py"
    child.write_text(f"import sys\nsys.exit({code})\n", encoding="utf-8")
    r = _ps(
        f"$p = Start-TrackedProcess -FilePath '{sys.executable}' -ArgumentList '{child}' -WorkingDirectory '{tmp_path}'; "
        "Write-Output ('CODE=[' + (Wait-TrackedProcess -Process $p) + ']')"
    )
    assert r.returncode == 0, r.stderr
    m = re.search(r"CODE=\[(.*?)\]", r.stdout)
    assert m, r.stdout + r.stderr
    return m.group(1)


@windows_only
def test_normal_exit_reports_zero(tmp_path):
    assert _exit_code_of(tmp_path, 0) == "0"


@windows_only
def test_failure_exit_code_is_reported(tmp_path):
    assert _exit_code_of(tmp_path, 3) == "3"


def test_start_script_uses_tracked_process_helpers():
    start = (LAUNCHER / "start_jp_subtitles.ps1").read_text(encoding="utf-8-sig")
    assert "Start-TrackedProcess" in start and "Wait-TrackedProcess" in start
    assert "$app.ExitCode" not in start


def test_unknown_exit_code_does_not_claim_a_crash():
    """Error path: if the code still cannot be read, say so instead of reporting a crash
    and blocking on Read-Host."""
    start = (LAUNCHER / "start_jp_subtitles.ps1").read_text(encoding="utf-8-sig")
    null_branch = start.index("$null -eq $code")
    crash_branch = start.index("異常結束")
    assert null_branch < crash_branch
