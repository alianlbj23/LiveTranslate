"""Update check / update (app_update.py + launcher/update_window.ps1), which replaced the
launcher's console update step. git and processes are faked; the update window runs for
real with stand-in programs for python.exe / pythonw.exe."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import app_update as au

ROOT = Path(__file__).resolve().parents[1]
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows update window")


def _done(out="", code=0, err=""):
    return subprocess.CompletedProcess([], code, stdout=out, stderr=err)


def _repo(tmp_path):
    (tmp_path / ".git").mkdir()
    return tmp_path


def _git(answers):
    """answers: {first git argument: CompletedProcess or exception}"""
    calls = []

    def git(args, root, timeout):
        calls.append(args)
        key = args[0] if args[0] != "-c" else "fetch"
        a = answers[key]
        if isinstance(a, BaseException):
            raise a
        return a
    git.calls = calls
    return git


def _answers(**over):
    a = {"rev-parse": _done("main\n"), "ls-remote": _done("abc\trefs/heads/main\n"), "fetch": _done(),
         "rev-list": _done("3\n"), "status": _done("")}
    a.update(over)
    return a


def test_release_zip_and_missing_git(tmp_path):
    assert au.check(tmp_path).state == au.NOT_GIT
    assert au.check(_repo(tmp_path), git=_git({"rev-parse": FileNotFoundError()})).state == au.NO_GIT_EXE


def test_newer_commits_on_github_can_be_updated(tmp_path):
    git = _git(_answers())
    s = au.check(_repo(tmp_path), git=git)
    assert (s.state, s.branch, s.behind, s.can_update) == (au.BEHIND, "main", 3, True)
    fetch = next(c for c in git.calls if "fetch" in c)
    assert "http.lowSpeedLimit=1000" in fetch  # a stalled connection gives up instead of hanging


@pytest.mark.parametrize("over, state", [
    ({"rev-list": _done("0\n")}, au.UP_TO_DATE),
    ({"status": _done(" M main.py\n")}, au.DIRTY),
    ({"rev-parse": _done("HEAD\n")}, au.DETACHED),
    ({"ls-remote": _done(code=2)}, au.LOCAL_BRANCH),          # the 2026-10-21 "cannot reach GitHub" report
    ({"ls-remote": _done(code=128, err="Could not resolve host: github.com")}, au.OFFLINE),
    ({"fetch": _done(code=1, err="fatal: early EOF")}, au.OFFLINE),
    ({"fetch": subprocess.TimeoutExpired(["git", "fetch"], 120)}, au.OFFLINE),
])
def test_every_outcome_has_its_own_state(tmp_path, over, state):
    s = au.check(_repo(tmp_path), git=_git(_answers(**over)))
    assert s.state == state and s.can_update is (state == au.BEHIND)


def test_a_branch_only_on_this_computer_is_named(tmp_path):
    s = au.check(_repo(tmp_path), git=_git(_answers(**{"rev-parse": _done("feature/gui-only\n"),
                                                       "ls-remote": _done(code=2)})))
    assert (s.state, s.branch) == (au.LOCAL_BRANCH, "feature/gui-only")


def test_pull_reports_changed_files_and_errors(tmp_path):
    def git(args, root, timeout):
        return {"rev-parse": _done("abc\n"), "pull": _done(), "diff": _done("main.py\nrequirements.txt\n")}[args[0]]
    changed, err = au.pull("main", tmp_path, git=git)
    assert changed == ["main.py", "requirements.txt"] and err is None and au.packages_changed(changed)
    assert not au.packages_changed(["main.py", "docs/requirements.txt.md"])

    def failing(args, root, timeout):
        return {"rev-parse": _done("abc\n"), "pull": _done(code=1, err="fatal: Not possible to fast-forward")}[args[0]]
    assert au.pull("main", tmp_path, git=failing) == ([], "fatal: Not possible to fast-forward")


def test_git_never_opens_a_window_or_waits_for_a_password(monkeypatch, tmp_path):
    seen = {}

    def run(args, **kw):
        seen.update(kw)
        seen["args"] = args
        return _done("main\n")
    monkeypatch.setattr(au.subprocess, "run", run)
    au._git(["rev-parse", "HEAD"], tmp_path, 5)
    assert seen["env"]["GIT_TERMINAL_PROMPT"] == "0" and seen["timeout"] == 5
    assert seen["creationflags"] & getattr(subprocess, "CREATE_NO_WINDOW", 0)


def test_restart_window_is_started_hidden(monkeypatch, tmp_path):
    launched = []
    assert au.launch_restart(42, packages=True, root=tmp_path, popen=lambda a, **kw: launched.append((a, kw)))
    args, kw = launched[0]
    assert args[args.index("-ParentPid") + 1] == "42" and "-Packages" in args and "Hidden" in args
    assert kw["creationflags"] & getattr(subprocess, "CREATE_NO_WINDOW", 0)
    au.launch_restart(42, packages=False, root=tmp_path, popen=lambda a, **kw: launched.append((a, kw)))
    assert "-Packages" not in launched[1][0]

    def broken(a, **kw):
        raise OSError("no powershell")
    assert au.launch_restart(42, packages=False, root=tmp_path, popen=broken) is False
    monkeypatch.setattr(au, "UPDATE_WINDOW", tmp_path / "missing.ps1")
    assert au.launch_restart(42, packages=False, root=tmp_path, popen=broken) is False


# ---------------------------------------------------------------------------
# launcher/update_window.ps1, for real (stand-in programs, temporary folder)
# ---------------------------------------------------------------------------

def _fake_root(tmp_path, python_exe: str):
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    system32 = Path(r"C:\Windows\System32")
    shutil.copy(system32 / python_exe, scripts / "python.exe")
    shutil.copy(system32 / "whoami.exe", scripts / "pythonw.exe")  # "starting LiveTranslate" does nothing harmful
    return tmp_path


def _window(root, *extra):
    return subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA", "-File",
                             str(ROOT / "launcher" / "update_window.ps1"), "-ParentPid", "0", "-Root", str(root), *extra],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _log(root) -> str:
    logs = sorted((root / "logs").glob("update_*.log"))
    return logs[-1].read_text(encoding="utf-8-sig") if logs else ""


@windows_only
def test_window_restarts_the_app_and_closes_itself(tmp_path):
    root = _fake_root(tmp_path, "where.exe")
    p = _window(root)
    try:
        p.wait(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, p.stderr.read().decode("utf-8", "replace")
    assert "啟動 LiveTranslate" in _log(root)


@windows_only
def test_failed_package_update_stays_open_and_keeps_the_app_closed(tmp_path):
    root = _fake_root(tmp_path, "where.exe")  # "python -m pip ..." fails
    (root / ".venv" / ".livetranslate-ready").write_text("x", encoding="ascii")
    p = _window(root, "-Packages")
    try:
        end = time.monotonic() + 60
        while "[失敗]" not in _log(root) and time.monotonic() < end:
            time.sleep(0.2)
        log = _log(root)
        assert "[失敗]" in log and "更新套件" in log
        assert p.poll() is None, "the window waits for the user after a failure"
        assert "啟動 LiveTranslate" not in log
        assert not (root / ".venv" / ".livetranslate-ready").exists()  # the next start knows packages are not ready
    finally:
        p.kill()


# ---------------------------------------------------------------------------
# Qt side: the Maintenance card and "update and restart"
# ---------------------------------------------------------------------------

@pytest.fixture
def zh_tw():
    import i18n

    before = i18n.get_lang()
    i18n.set_lang("zh_TW")
    yield
    i18n.set_lang(before)


def _wait(cond, timeout=5.0):
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    end = time.monotonic() + timeout
    while not cond():
        app.processEvents()
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.01)


def _controller(status, pulled=(["main.py"], None), launch_ok=True, confirm=True):
    import app_update_ui

    calls = {"show": [], "notify": [], "launch": [], "quit": 0, "confirm": []}

    def check():
        if isinstance(status, BaseException):
            raise status
        return status

    def launch(pid, packages):
        calls["launch"].append((pid, packages))
        return launch_ok

    def quit_app():
        calls["quit"] += 1
    c = app_update_ui.UpdateController(
        show_status=lambda text, can, busy: calls["show"].append((text, can, busy)),
        notify=calls["notify"].append,
        confirm=lambda title, text: calls["confirm"].append(text) or confirm,
        quit_app=quit_app, check=check, pull=lambda branch: pulled, launch=launch, pid=4242)
    return c, calls


def test_quiet_check_after_start_notifies_only_about_new_versions(zh_tw):
    c, calls = _controller(au.UpdateStatus(au.BEHIND, "main", 3))
    c.check(quiet=True)
    _wait(lambda: c.status is not None)
    assert calls["show"][0] == ("檢查中…", False, True)
    assert calls["show"][-1] == ("GitHub 上有 3 個新版本（main）", True, False)
    assert calls["notify"] == ["GitHub 上有 3 個新版本，可以到 設定 → 首頁 → 維護 更新。"]
    quiet_local, calls2 = _controller(au.UpdateStatus(au.LOCAL_BRANCH, "feature/gui-only"))
    quiet_local.check(quiet=True)
    _wait(lambda: quiet_local.status is not None)
    assert calls2["notify"] == [] and "只在這台電腦上" in calls2["show"][-1][0]


def test_a_crashing_check_is_shown_not_left_spinning(zh_tw):
    c, calls = _controller(RuntimeError("boom"))
    c.check()
    _wait(lambda: c.status is not None)
    assert calls["show"][-1] == ("連不到 GitHub，稍後再試", False, False)


@pytest.mark.parametrize("changed, packages", [(["main.py"], False), (["requirements.txt"], True)])
def test_update_and_restart(zh_tw, changed, packages):
    c, calls = _controller(au.UpdateStatus(au.BEHIND, "main", 2), pulled=(changed, None))
    c.check()
    _wait(lambda: c.status is not None)
    assert c.apply()
    _wait(lambda: calls["quit"])
    assert calls["launch"] == [(4242, packages)] and calls["quit"] == 1
    assert "2 個新版本" in calls["confirm"][0]


def test_update_declined_failed_or_without_restart(zh_tw):
    c, calls = _controller(au.UpdateStatus(au.BEHIND, "main", 2), confirm=False)
    c.check()
    _wait(lambda: c.status is not None)
    assert c.apply() is False and calls["launch"] == []
    c, calls = _controller(au.UpdateStatus(au.BEHIND, "main", 2), pulled=([], "fatal: Not possible to fast-forward"))
    c.check()
    _wait(lambda: c.status is not None)
    c.apply()
    _wait(lambda: "更新失敗" in calls["show"][-1][0])
    assert calls["quit"] == 0 and "fast-forward" in calls["show"][-1][0]
    c, calls = _controller(au.UpdateStatus(au.BEHIND, "main", 2), launch_ok=False)
    c.check()
    _wait(lambda: c.status is not None)
    c.apply()
    _wait(lambda: "無法自動重新啟動" in calls["show"][-1][0])
    assert calls["quit"] == 0


def test_nothing_to_apply_when_up_to_date(zh_tw):
    c, calls = _controller(au.UpdateStatus(au.UP_TO_DATE, "main"))
    c.check()
    _wait(lambda: c.status is not None)
    assert c.apply() is False and calls["confirm"] == []


def test_maintenance_card_buttons(zh_tw, monkeypatch):
    import copy

    import yaml
    from PyQt6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    import control_panel as cp

    monkeypatch.setattr(cp, "_load_saved_settings", lambda: None)
    monkeypatch.setattr(cp, "_save_settings", lambda s: copy.deepcopy(s))
    p = cp.ControlPanel(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")), saved_settings={})
    monkeypatch.setattr(p, "_auto_save", lambda *a: None)
    got = []
    p.update_check_requested.connect(lambda: got.append("check"))
    p.update_apply_requested.connect(lambda: got.append("apply"))
    assert p._update_card.contentLabel.text() == "尚未檢查" and p._update_apply_btn.isHidden()
    p._update_check_btn.click()
    p.set_update_status("GitHub 上有 1 個新版本（main）", True, False)
    assert not p._update_apply_btn.isHidden() and p._update_apply_btn.isEnabled()
    p._update_apply_btn.click()
    assert got == ["check", "apply"]
    p.set_update_status("下載新版本中…", False, True)
    assert p._update_apply_btn.isHidden() and not p._update_check_btn.isEnabled()


def test_main_checks_quietly_after_start():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "QTimer.singleShot(15000, lambda: updater.check(quiet=True))" in src
    assert "quit_app=on_quit" in src and "panel.update_apply_requested.connect(updater.apply)" in src
