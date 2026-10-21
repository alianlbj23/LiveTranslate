"""An interrupted installation must never look installed: LiveTranslate.exe and the debug
entry start the app only with the venv, the ready marker and user_settings.json; the
installer and the update window write the marker only after `pip check` passed."""
from pathlib import Path

READY_MARKER = ".venv\\.livetranslate-ready"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8-sig").lower()


def test_exe_starts_only_a_complete_installation():
    src = _read("launcher/LiveTranslate.cs")
    check = src[src.index("public static bool isinstalled"):src.index("public static bool startapp")]
    assert "file.exists(pythonw(root))" in check and ".livetranslate-ready" in check and "user_settings.json" in check
    helper = src[src.index("public static string pythonw"):src.index("public static bool isinstalled")]
    assert r".venv\scripts\pythonw.exe" in helper


def test_debug_entry_starts_only_a_complete_installation():
    bat = _read("LiveTranslate（除錯）.bat")
    run = bat.index(":run")
    for need in (".venv\\scripts\\python.exe", READY_MARKER, "user_settings.json"):
        assert f'if not exist "{need}" goto setup' in bat[:run]
    assert "launcher\\setup.ps1" in bat[bat.index(":setup"):run]


def test_installer_marks_ready_only_after_pip_check():
    setup = _read("launcher/setup.ps1")
    assert setup.index("remove-item -literalpath $ready") < setup.index("pip check")
    assert setup.rindex("set-content -literalpath $ready") > setup.rindex("pip check")


def test_update_window_marks_ready_only_after_its_last_step():
    window = _read("launcher/update_window.ps1")
    assert "remove-item -literalpath $ready" in window
    steps = window[window.index('"steps" {'):window.index('"launch" {')]
    assert steps.index("$script:stepindex -lt $steps.count") < steps.index("set-content -literalpath $ready")
