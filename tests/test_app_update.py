"""Update check / update (app_update.py + launcher/update_window.ps1), which replaced the
launcher's console update step. git and processes are faked; the update window runs for
real with stand-in programs for python.exe / pythonw.exe."""
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import app_update as au

ROOT = Path(__file__).resolve().parents[1]
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows update window")



@pytest.fixture(autouse=True)
def _collect_widgets_now():
    """Parentless Fluent widgets (dialogs, groups) left in reference cycles are otherwise
    collected at some later allocation. When that happens while qfluentwidgets' setTheme()
    walks its style-sheet registry (every ControlPanel calls it), the widget's `destroyed` →
    deregister changes that dict mid-iteration: "dictionary changed size during iteration"
    in an unrelated test. Collect them at the end of each test instead."""
    yield
    gc.collect()

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


class _WindowProc:
    """Fake update window: writes its -StartedFile (or not), may have ended."""

    def __init__(self, args, start=True, code=None):
        self.args, self.returncode = args, code
        if start:
            Path(args[args.index("-StartedFile") + 1]).write_text("1", encoding="ascii")

    def poll(self):
        return self.returncode


def test_restart_window_is_started_hidden(monkeypatch, tmp_path):
    launched = []

    def popen(a, **kw):
        launched.append((a, kw))
        return _WindowProc(a)
    assert au.launch_restart(42, True, root=tmp_path, popen=popen)
    args, kw = launched[0]
    assert args[args.index("-ParentPid") + 1] == "42" and "-Packages" in args and "Hidden" in args
    assert kw["creationflags"] & getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert au.launch_restart(42, False, root=tmp_path, popen=popen)
    assert "-Packages" not in launched[1][0]
    assert not list((tmp_path / ".update").glob("window_started_*"))  # cleaned up

    def broken(a, **kw):
        raise OSError("no powershell")
    assert au.launch_restart(42, False, root=tmp_path, popen=broken) is False
    monkeypatch.setattr(au, "UPDATE_WINDOW", tmp_path / "missing.ps1")
    assert au.launch_restart(42, False, root=tmp_path, popen=broken) is False


def test_restart_window_that_never_comes_up_keeps_livetranslate_open(tmp_path, caplog):
    """Review: a window blocked by policy used to leave the user with nothing at all."""
    assert au.launch_restart(42, False, root=tmp_path, popen=lambda a, **kw: _WindowProc(a, start=False, code=1),
                             sleep=lambda s: None) is False
    assert "ended at once" in caplog.text
    assert au.launch_restart(42, False, root=tmp_path, popen=lambda a, **kw: _WindowProc(a, start=False),
                             started_timeout=0.3, sleep=lambda s: time.sleep(0.05)) is False
    assert "did not come up" in caplog.text


# ---------------------------------------------------------------------------
# launcher/update_window.ps1, for real (stand-in programs, temporary folder)
# ---------------------------------------------------------------------------

def _fake_root(tmp_path, python_exe: str, pythonw: Path):
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    system32 = Path(r"C:\Windows\System32")
    shutil.copy(system32 / python_exe, scripts / "python.exe")
    shutil.copy(pythonw, scripts / "pythonw.exe")  # "starting LiveTranslate" runs the stand-in
    return tmp_path


def _window(root, *extra, env=None):
    return subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA", "-File",
                             str(ROOT / "launcher" / "update_window.ps1"), "-ParentPid", "0", "-Root", str(root), *extra],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)


def _log(root) -> str:
    logs = sorted((root / "logs").glob("update_*.log"))
    return logs[-1].read_text(encoding="utf-8-sig") if logs else ""


@windows_only
def test_window_restarts_the_app_and_closes_itself(tmp_path, standin_exe):
    root = _fake_root(tmp_path, "where.exe", standin_exe)
    p = _window(root)
    try:
        p.wait(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, p.stderr.read().decode("utf-8", "replace")
    assert "啟動 LiveTranslate" in _log(root)


@windows_only
def test_failed_package_update_stays_open_and_keeps_the_app_closed(tmp_path, standin_exe):
    root = _fake_root(tmp_path, "where.exe", standin_exe)  # "python -m pip ..." fails
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


def _controller(status, pulled=(["main.py"], None), launch_ok=True, confirm=True, downloaded=None):
    import app_update_ui

    calls = {"show": [], "notify": [], "launch": [], "quit": 0, "confirm": []}

    def check():
        if isinstance(status, BaseException):
            raise status
        return status

    def launch(pid, packages, release_zip=None):
        calls["launch"].append((pid, packages) if release_zip is None else (pid, packages, release_zip))
        return launch_ok

    def quit_app():
        calls["quit"] += 1
    c = app_update_ui.UpdateController(
        show_status=lambda text, can, busy: calls["show"].append((text, can, busy)),
        notify=calls["notify"].append,
        confirm=lambda title, text: calls["confirm"].append(text) or confirm,
        quit_app=quit_app, check=check, pull=lambda branch: pulled, launch=launch, pid=4242,
        download=lambda st: downloaded)
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


def test_update_window_cleans_up_when_closed_mid_step():
    """Review: closing the window with X left pip changing the venv."""
    src = (ROOT / "launcher" / "update_window.ps1").read_text(encoding="utf-8-sig")
    closing = src[src.index("$form.Add_FormClosing("):src.index("# ---------- work")]
    assert "Stop-StepTree" in closing and "$e.Cancel = $true" in closing  # asks first, then ends the tree
    tree = src[src.index("function Stop-StepTree"):src.index("$btnStart.Add_Click(")]
    assert "taskkill.exe" in tree and '"/T"' in tree


def test_update_window_pip_steps_have_a_time_limit_and_stay_off_the_system_drive():
    src = (ROOT / "launcher" / "update_window.ps1").read_text(encoding="utf-8-sig")
    assert "$StepTimeoutMinutes" in src and "StepStarted" in src
    assert 'EnvironmentVariables["PIP_CACHE_DIR"]' in src and 'EnvironmentVariables["TMP"]' in src
    assert "Remove-Item -Recurse -Force -LiteralPath $PipCache, $PipTemp" in src


def test_no_second_livetranslate_while_the_first_is_still_closing():
    src = (ROOT / "launcher" / "update_window.ps1").read_text(encoding="utf-8-sig")
    wait = src[src.index('"wait" {'):src.index('"steps" {')]
    assert "Set-Failed" in wait and "-NoStart" in wait



# ---------------------------------------------------------------------------
# release packages (no .git): VERSION.json + the repository's latest GitHub release
# ---------------------------------------------------------------------------

import json  # noqa: E402
import zipfile  # noqa: E402

import httpx  # noqa: E402

REPO = "alianlbj23/LiveTranslate"
ZIP_URL = f"https://github.com/{REPO}/releases/download/v2026.10.22/LiveTranslate-portable-v2026.10.22.zip"


def _release_root(tmp_path, version="v2026.10.21", repo=REPO):
    (tmp_path / "VERSION.json").write_text(json.dumps({"version": version, "repo": repo}), encoding="utf-8")
    return tmp_path


def _github(latest="v2026.10.22", status=200, assets=None):
    if assets is None:
        assets = [{"name": "LiveTranslate-portable-v2026.10.22.zip", "browser_download_url": ZIP_URL}]
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if status != 200:
            return httpx.Response(status, json={"message": "x"})
        return httpx.Response(200, json={"tag_name": latest, "assets": assets})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    client.seen = seen
    return client


def test_version_file_is_read_and_checked(tmp_path):
    assert au.read_version(_release_root(tmp_path)) == {"version": "v2026.10.21", "repo": REPO}
    (tmp_path / "VERSION.json").write_text("{broken", encoding="utf-8")
    assert au.read_version(tmp_path) is None
    _release_root(tmp_path, repo="../../evil")
    assert au.read_version(tmp_path) is None
    (tmp_path / "VERSION.json").unlink()
    assert au.read_version(tmp_path) is None


def test_versions_compare_as_numbers():
    assert au.is_newer("v2026.10.22", "v2026.10.21") and au.is_newer("v2026.11.1", "v2026.10.21")
    assert not au.is_newer("v2026.10.21", "v2026.10.21") and not au.is_newer("v2026.9.30", "v2026.10.21")
    assert not au.is_newer("beta-2", "beta-1")  # other schemes are never offered (no downgrade by name)
    assert not au.is_newer("v2026.10.22", "dev-build")


def test_release_package_finds_a_newer_release(tmp_path):
    http = _github()
    s = au.check(_release_root(tmp_path), http=http)
    assert (s.state, s.version, s.latest, s.asset_url, s.can_update) == (au.RELEASE_NEWER, "v2026.10.21",
                                                                        "v2026.10.22", ZIP_URL, True)
    assert http.seen == [f"https://api.github.com/repos/{REPO}/releases/latest"]


@pytest.mark.parametrize("http, state", [
    (_github(latest="v2026.10.21"), au.RELEASE_UP_TO_DATE),
    (_github(status=404), au.NO_RELEASE),             # nothing published yet / no such repository
    (_github(status=500), au.OFFLINE),
    (_github(assets=[{"name": "LiveTranslate-portable-x.zip",
                      "browser_download_url": "https://evil.example.com/x.zip"}]), au.OFFLINE),
])
def test_release_check_outcomes(tmp_path, http, state):
    s = au.check(_release_root(tmp_path), http=http)
    assert s.state == state and not s.can_update and s.version == "v2026.10.21"


def test_release_check_without_network(tmp_path):
    def handler(request):
        raise httpx.ConnectError("offline")
    s = au.check(_release_root(tmp_path), http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert s.state == au.OFFLINE and "ConnectError" in s.detail


def test_neither_git_nor_release_offers_nothing(tmp_path):
    assert au.check(tmp_path).state == au.NOT_GIT


def _release_zip(path: Path, files=None) -> bytes:
    files = files if files is not None else {
        "LiveTranslate/main.py": "print('new')\n", "LiveTranslate/LiveTranslate.exe": "exe",
        "LiveTranslate/VERSION.json": json.dumps({"version": "v2026.10.22", "repo": REPO}),
        "LiveTranslate/requirements.txt": "numpy\n"}
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return path.read_bytes()


def _download_client(body: bytes, status=200, length=None):
    def handler(request):
        headers = {"Content-Length": str(length if length is not None else len(body))}
        return httpx.Response(status, content=body, headers=headers)
    return httpx.Client(transport=httpx.MockTransport(handler))


NEWER = au.UpdateStatus(au.RELEASE_NEWER, version="v2026.10.21", latest="v2026.10.22", asset_url=ZIP_URL)


def test_release_download_is_checked(tmp_path):
    body = _release_zip(tmp_path / "src.zip")
    seen = []
    path, err = au.download_release(NEWER, tmp_path, http=_download_client(body),
                                    on_progress=lambda d, t: seen.append((d, t)))
    assert err is None and path == tmp_path / ".update" / "LiveTranslate-update.zip" and path.read_bytes() == body
    assert seen and seen[-1] == (len(body), len(body))


@pytest.mark.parametrize("files, problem", [
    ({"LiveTranslate/main.py": "x", "LiveTranslate/VERSION.json": "{}"}, "LiveTranslate.exe is missing"),
    ({"LiveTranslate/main.py": "x", "LiveTranslate/LiveTranslate.exe": "x", "LiveTranslate/VERSION.json": "{}",
      "LiveTranslate/../../evil.py": "x"}, "unexpected path"),
    ({"Other/main.py": "x"}, "unexpected path"),
])
def test_incomplete_or_unsafe_release_zip_is_refused(tmp_path, files, problem):
    body = _release_zip(tmp_path / "src.zip", files)
    path, err = au.download_release(NEWER, tmp_path, http=_download_client(body))
    assert path is None and problem in err


def test_release_download_problems(tmp_path):
    bad_url = au.UpdateStatus(au.RELEASE_NEWER, latest="v9", asset_url="https://evil.example.com/x.zip")
    assert au.download_release(bad_url, tmp_path)[1].startswith("refusing")
    assert au.download_release(NEWER, tmp_path, http=_download_client(b"", status=404))[1] == "HTTP 404"
    assert "too big" in au.download_release(NEWER, tmp_path, http=_download_client(b"x", length=10**12))[1]
    assert "not a zip" in au.download_release(NEWER, tmp_path, http=_download_client(b"garbage"))[1]


def test_packages_are_reinstalled_only_when_requirements_changed(tmp_path):
    z = tmp_path / "r.zip"
    _release_zip(z)
    (tmp_path / "requirements.txt").write_bytes(b"numpy\r\n")  # same text, other line ends
    assert au.release_packages_changed(z, tmp_path) is False
    (tmp_path / "requirements.txt").write_text("numpy\ntorch\n", encoding="utf-8")
    assert au.release_packages_changed(z, tmp_path) is True


def test_restart_window_gets_the_release_zip(tmp_path):
    launched = []
    au.launch_restart(7, False, root=tmp_path, popen=lambda a, **kw: launched.append(a) or _WindowProc(a),
                      release_zip=tmp_path / "u.zip")
    args = launched[0]
    assert args[args.index("-Zip") + 1] == str(tmp_path / "u.zip") and "-Packages" not in args


def test_release_update_downloads_and_hands_over_to_the_window(zh_tw, tmp_path):
    z = tmp_path / "u.zip"
    _release_zip(z)
    c, calls = _controller(NEWER, downloaded=(z, None))
    c.check(quiet=True)
    _wait(lambda: c.status is not None)
    assert calls["show"][-1] == ("有新版本 v2026.10.22（目前 v2026.10.21）", True, False)
    assert calls["notify"] == ["有新版本 v2026.10.22，可以到 設定 → 首頁 → 維護 更新。"]
    assert c.apply()
    _wait(lambda: calls["quit"])
    assert calls["launch"] == [(4242, au.release_packages_changed(z), z)]
    assert "v2026.10.22" in calls["confirm"][0] and "保留" in calls["confirm"][0]


def test_release_download_failure_is_shown_and_nothing_restarts(zh_tw):
    c, calls = _controller(NEWER, downloaded=(None, "HTTP 404"))
    c.check()
    _wait(lambda: c.status is not None)
    c.apply()
    _wait(lambda: "更新失敗" in calls["show"][-1][0])
    assert calls["quit"] == 0 and calls["launch"] == [] and "HTTP 404" in calls["show"][-1][0]


@windows_only
def test_window_installs_a_release_and_keeps_the_users_files(tmp_path, standin_exe):
    root = _fake_root(tmp_path, "where.exe", standin_exe)
    (root / ".venv" / ".livetranslate-ready").write_text("x", encoding="ascii")
    (root / "config.yaml").write_text("mine: 1\n", encoding="utf-8")
    (root / "main.py").write_text("print('old')\n", encoding="utf-8")
    (root / "user_settings.json").write_text('{"mine": true}', encoding="utf-8")
    (root / ".update").mkdir()
    z = root / ".update" / "LiveTranslate-update.zip"
    _release_zip(z, {"LiveTranslate/main.py": "print('new')\n", "LiveTranslate/LiveTranslate.exe": "exe",
                     "LiveTranslate/VERSION.json": json.dumps({"version": "v2026.10.22", "repo": REPO}),
                     "LiveTranslate/bench/new_module.py": "x = 1\n", "LiveTranslate/config.yaml": "new: 1\n"})
    p = _window(root, "-Zip", str(z))
    try:
        p.wait(timeout=90)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, p.stderr.read().decode("utf-8", "replace")
    assert (root / "main.py").read_text(encoding="utf-8") == "print('new')\n"
    assert (root / "bench" / "new_module.py").exists() and json.loads((root / "VERSION.json").read_text())["version"] == "v2026.10.22"
    assert (root / "user_settings.json").read_text(encoding="utf-8") == '{"mine": true}'
    assert not (root / ".update").exists()
    assert (root / ".venv" / ".livetranslate-ready").exists()  # written again after the copy
    assert (root / "config.yaml").read_text(encoding="utf-8") == "new: 1\n"
    backups = list(root.glob("config.backup_*.yaml"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == "mine: 1\n"
    log = _log(root)
    assert "新版本已安裝" in log and "啟動 LiveTranslate" in log


@windows_only
def test_window_refuses_an_incomplete_release(tmp_path, standin_exe):
    root = _fake_root(tmp_path, "where.exe", standin_exe)
    (root / "main.py").write_text("print('old')\n", encoding="utf-8")
    z = root / "broken.zip"
    _release_zip(z, {"LiveTranslate/readme.txt": "x"})
    p = _window(root, "-Zip", str(z))
    try:
        end = time.monotonic() + 60
        while "[失敗]" not in _log(root) and time.monotonic() < end:
            time.sleep(0.2)
        assert "[失敗]" in _log(root) and "不完整" in _log(root)
        assert (root / "main.py").read_text(encoding="utf-8") == "print('old')\n"
        assert p.poll() is None  # waits for the user; the current version can still be started
    finally:
        p.kill()


def test_release_build_writes_the_version_file():
    src = (ROOT / "build_release.ps1").read_text(encoding="utf-8-sig")
    block = src[src.index("# ── 3b."):src.index("# ── 4.")]
    assert "VERSION.json" in block and "version = $Tag" in block and "repo = $Repo" in block
    assert "GITHUB_REPOSITORY" in block


# ---------------------------------------------------------------------------
# review of the release update (2026-10-21)
# ---------------------------------------------------------------------------

def test_repository_name_case_does_not_matter(tmp_path):
    http = _github(assets=[{"name": "LiveTranslate-portable-v2026.10.22.zip",
                            "browser_download_url": ZIP_URL.replace("alianlbj23", "AlianLBJ23")}])
    assert au.check(_release_root(tmp_path), http=http).state == au.RELEASE_NEWER


@pytest.mark.parametrize("name", ["LiveTranslate/.. /.. /evil.py", "LiveTranslate/. ./x.py", "LiveTranslate//x.py"])
def test_paths_windows_would_resolve_outside_are_refused(tmp_path, name):
    files = {"LiveTranslate/main.py": "x", "LiveTranslate/LiveTranslate.exe": "x",
             "LiveTranslate/VERSION.json": "{}", name: "x"}
    body = _release_zip(tmp_path / "src.zip", files)
    path, err = au.download_release(NEWER, tmp_path, http=_download_client(body))
    assert path is None and "unexpected path" in err


def test_zip_that_unpacks_too_big_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(au, "MAX_UNPACKED_BYTES", 10)
    body = _release_zip(tmp_path / "src.zip")
    path, err = au.download_release(NEWER, tmp_path, http=_download_client(body))
    assert path is None and "unpacks" in err


def test_redirect_to_another_host_is_refused_and_nothing_is_left(tmp_path):
    def handler(request):
        if request.url.host == "github.com":
            return httpx.Response(302, headers={"Location": "https://evil.example.com/x.zip"})
        return httpx.Response(200, content=b"x")
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    path, err = au.download_release(NEWER, tmp_path, http=client)
    assert path is None and "evil.example.com" in err
    assert not (tmp_path / ".update" / "LiveTranslate-update.zip").exists()


def test_github_cdn_redirect_is_accepted(tmp_path):
    body = _release_zip(tmp_path / "src.zip")

    def handler(request):
        if request.url.host == "github.com":
            return httpx.Response(302, headers={"Location": "https://release-assets.githubusercontent.com/a.zip"})
        return httpx.Response(200, content=body, headers={"Content-Length": str(len(body))})
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    path, err = au.download_release(NEWER, tmp_path, http=client)
    assert err is None and path.exists()


def test_failed_download_leaves_no_file(tmp_path):
    path, err = au.download_release(NEWER, tmp_path, http=_download_client(b"garbage"))
    assert path is None and not (tmp_path / ".update" / "LiveTranslate-update.zip").exists()


def test_git_pull_without_the_current_commit_is_an_error(tmp_path):
    def git(args, root, timeout):
        return {"rev-parse": _done("", code=128, err="fatal: not a git repository")}[args[0]]
    changed, err = au.pull("main", tmp_path, git=git)
    assert changed == [] and "not a git repository" in err


def test_release_downloaded_but_window_not_started_says_so(zh_tw, tmp_path):
    z = tmp_path / "u.zip"
    _release_zip(z)
    c, calls = _controller(NEWER, downloaded=(z, None), launch_ok=False)
    c.check()
    _wait(lambda: c.status is not None)
    c.apply()
    _wait(lambda: "尚未安裝" in calls["show"][-1][0] or "還沒安裝" in calls["show"][-1][0])
    assert calls["quit"] == 0


@windows_only
def test_window_reports_a_livetranslate_that_ends_at_once(tmp_path, standin_exe):
    import os

    root = _fake_root(tmp_path, "where.exe", standin_exe)
    p = _window(root, env=dict(os.environ, STANDIN_CODE="3"))
    try:
        end = time.monotonic() + 60
        while "馬上結束" not in _log(root) and time.monotonic() < end:
            time.sleep(0.2)
        assert "馬上結束" in _log(root) and "3" in _log(root)
        assert p.poll() is None  # stays open with the reason
    finally:
        p.kill()


def test_update_window_reports_that_it_runs_and_guards_a_half_copy():
    src = (ROOT / "launcher" / "update_window.ps1").read_text(encoding="utf-8-sig")
    head = src[:src.index("# ---------- window ----------")]
    assert "if ($StartedFile) { Set-Content -LiteralPath $StartedFile" in head
    install = src[src.index("function Install-Release"):src.index("$btnStart.Add_Click(")]
    assert "[System.IO.Compression.ZipFile]::ExtractToDirectory" in install and "Expand-Archive" not in install
    assert install.index("Remove-Item -LiteralPath $Ready") < install.index("robocopy.exe")
    assert "Set-Failed $msg -NoStart" in install
