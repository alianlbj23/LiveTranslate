"""The best-combination benchmark window (bench/window.py, bench/gui_protocol.py) that
replaced launcher/bench.ps1's console menu, and how the app opens it."""
import gc
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.gui_protocol import JsonConsole, ProgressStream  # noqa: E402



@pytest.fixture(autouse=True)
def _collect_widgets_now():
    """Parentless Fluent widgets (dialogs, groups) left in reference cycles are otherwise
    collected at some later allocation. When that happens while qfluentwidgets' setTheme()
    walks its style-sheet registry (every ControlPanel calls it), the widget's `destroyed` →
    deregister changes that dict mid-iteration: "dictionary changed size during iteration"
    in an unrelated test. Collect them at the end of each test instead."""
    yield
    gc.collect()

def _console(*answers):
    out = io.StringIO()
    inp = io.StringIO("".join(json.dumps(a) + "\n" for a in answers))
    return JsonConsole(out, inp), out


def _sent(out):
    return [json.loads(x) for x in out.getvalue().splitlines()]


# ---------------------------------------------------------------------------
# protocol (child side)
# ---------------------------------------------------------------------------

def test_say_and_questions_are_json_lines():
    con, out = _console({"text": " 0 "}, {"yes": True}, {})
    con.say("a", 1)
    assert con.ask("關閉後按 Enter 重新檢查，輸入 0 取消：") == "0"
    assert con.confirm("開始？", False) is True
    con.pause("按 Enter 繼續")
    sent = _sent(out)
    assert sent[0] == {"t": "say", "lines": ["a", "1"]}
    assert sent[1]["t"] == "ask" and sent[2] == {"t": "confirm", "q": "開始？", "default": False}
    assert sent[3] == {"t": "pause", "message": "按 Enter 繼續"}


def test_a_window_that_went_away_cancels():
    con, _out = _console()
    assert con.ask("x") == "0" and con.confirm("x", True) is False
    assert con.choose_one("t", [("a", "A")]) is None
    assert con.choose_many("t", [("a", "A", True, "")], set()) is None
    garbage = JsonConsole(io.StringIO(), io.StringIO("not json\n"))
    assert garbage.confirm("x", True) is False


def test_choose_one_maps_the_index_back_to_the_key():
    run = object()
    con, out = _console({"index": 1}, {"index": None}, {"index": 7})
    assert con.choose_one("選擇一次評測：", [("x", "X"), (run, "R")], 0) is run
    assert con.choose_one("t", [("x", "X")]) is None
    assert con.choose_one("t", [("x", "X"), ("y", "Y")], default=1, allow_cancel=False) == "y"
    assert _sent(out)[0] == {"t": "choose_one", "title": "選擇一次評測：", "labels": ["X", "R"], "default": 0,
                             "cancel": True}


def test_choose_many_checks_minimum_and_disabled_items():
    opts = [("a", "A", True, ""), ("b", "B", False, "沒有下載"), ("c", "C", True, "")]
    con, out = _console({"indexes": [1]}, {"indexes": [0, 2]})
    assert con.choose_many("翻譯模型", opts, {"a", "b"}, minimum=1) == ["a", "c"]
    sent = _sent(out)
    assert sent[0]["items"][0] == {"label": "A", "enabled": True, "note": "", "checked": True}
    assert sent[0]["items"][1]["checked"] is False  # a disabled item cannot stay ticked
    assert sent[1]["t"] == "say" and "至少要勾選" in sent[1]["lines"][0]


def test_progress_rewrites_become_progress_messages():
    got = []
    s = ProgressStream(got.append)
    s.write("\r  3/10 句")
    s.flush()
    s.write("\r  4/10 句")
    s.flush()
    s.write("\r完成" + " " * 5 + "\n")
    print("請手動打開：x", file=s)
    assert got == [{"t": "progress", "text": "3/10 句"}, {"t": "progress", "text": "4/10 句"},
                   {"t": "progress", "text": ""}, {"t": "say", "lines": ["完成     "]},
                   {"t": "say", "lines": ["請手動打開：x"]}]


def test_cli_json_mode_sends_the_status(monkeypatch):
    from bench import cli

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["status", "--json"]) == 0
    sent = [json.loads(x) for x in out.getvalue().splitlines()]
    assert sent and sent[0]["t"] == "say" and "測試資料" in sent[0]["lines"][0]


# ---------------------------------------------------------------------------
# window (GUI side)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(qapp, cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond():
        qapp.processEvents()
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.01)


class FakeProc:
    def __init__(self, messages, code=0, stderr=()):
        self.stdout = iter([json.dumps(m, ensure_ascii=False) + "\n" for m in messages])
        self.stderr = iter(list(stderr))
        self.stdin = io.StringIO()
        self.code = code
        self.pid = 999

    def wait(self):
        return self.code


class FakeDialogs:
    def __init__(self, **answers):
        self.answers = answers
        self.asked = []

    def __getattr__(self, name):
        def answer(*args):
            self.asked.append((name, args))
            return self.answers.get(name)
        return answer


def _window(qapp, script, dialogs=None, **kw):
    from bench.window import BenchWindow

    spawned = []

    def spawn(command):
        spawned.append(command)
        msgs, code = script.get(command, ([{"t": "say", "lines": [" 測試資料：動畫 0/200"]}], 0))
        p = FakeProc(msgs, code, kw.get("stderr", ()))
        spawned_procs.append(p)
        return p
    spawned_procs = []
    killed = []
    w = BenchWindow(spawn=spawn, dialogs=dialogs or FakeDialogs(), start_app=lambda: spawned.append("app"),
                    kill=killed.append)
    return w, spawned, spawned_procs, killed


def test_status_fills_the_header(qapp):
    w, spawned, _procs, _k = _window(qapp, {})
    w.run("status")
    _wait(qapp, lambda: w.proc is None and w.status.text())
    assert w.status.text() == "測試資料：動畫 0/200" and spawned == ["status"]
    assert all(b.isEnabled() for b in w.buttons.values())


def test_questions_are_answered_with_dialogs_and_status_is_refreshed(qapp):
    script = {"start": ([
        {"t": "say", "lines": ["", "要跑哪一種？"]},
        {"t": "choose_one", "title": "要跑哪一種？", "labels": ["完整", "快速"], "default": 0, "cancel": True},
        {"t": "choose_many", "title": "翻譯模型", "items": [{"label": "A", "enabled": True, "note": "", "checked": True}],
         "minimum": 1},
        {"t": "confirm", "q": "開始評測？", "default": True},
        {"t": "progress", "text": "[階段 1] 第 3/10 句"},
    ], 0)}
    dialogs = FakeDialogs(choose_one=1, choose_many=[0], confirm=True)
    w, spawned, procs, _k = _window(qapp, script, dialogs)
    w.run("start")
    _wait(qapp, lambda: spawned[-1:] == ["status"] and w.proc is None)
    answers = [json.loads(x) for x in procs[0].stdin.getvalue().splitlines()]
    assert answers == [{"index": 1}, {"indexes": [0]}, {"yes": True}]
    assert "要跑哪一種？" in w.output.toPlainText() and w.progress.text() == ""


def test_warnings_and_failures_are_shown(qapp):
    w, spawned, _p, _k = _window(qapp, {"apply": ([{"t": "say", "lines": ["x"]}], 1)},
                                 stderr=["[WARNING] Ollama could not unload"])
    w.run("apply")
    _wait(qapp, lambda: spawned[-1:] == ["status"] and w.proc is None)
    text = w.output.toPlainText()
    assert "⚠ [WARNING] Ollama could not unload" in text and "⚠ （結束代碼 1）" in text


def test_stop_ends_the_child_only_after_confirming(qapp):
    from bench.window import BenchWindow

    killed = []
    blocker = type("P", (), {"stdin": io.StringIO(), "pid": 1})()
    w = BenchWindow(spawn=lambda c: blocker, dialogs=FakeDialogs(confirm=False), kill=killed.append)
    w.proc, w._command = blocker, "start"
    assert w.stop() is False and killed == []
    w.dialogs = FakeDialogs(confirm=True)
    assert w.stop() is True and killed == [blocker]


def test_closing_during_a_run_asks_first(qapp):
    from PyQt6.QtGui import QCloseEvent

    from bench.window import BenchWindow

    killed = []
    w = BenchWindow(spawn=lambda c: None, dialogs=FakeDialogs(confirm=False), kill=killed.append)
    w.proc, w._command = object(), "start"
    e = QCloseEvent()
    w.closeEvent(e)
    assert not e.isAccepted() and killed == []
    w.proc, w._command = object(), "status"  # a status refresh is simply ended
    e = QCloseEvent()
    w.closeEvent(e)
    assert e.isAccepted() and len(killed) == 1


def test_open_livetranslate_button(qapp):
    w, spawned, _p, _k = _window(qapp, {})
    w.open_app_btn.click()
    assert spawned == ["app"]


def test_choose_many_dialog_needs_the_minimum(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialogButtonBox

    from bench.window import ChooseDialog

    d = ChooseDialog("t", [{"label": "A", "enabled": True}, {"label": "B", "enabled": False, "note": "沒有下載"}],
                     many=True, minimum=1)
    ok = d.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()
    assert not (d.list.item(1).flags() & Qt.ItemFlag.ItemIsEnabled)
    d.list.item(0).setCheckState(Qt.CheckState.Checked)
    assert ok.isEnabled() and d.checked() == [0]


@pytest.mark.skipif(not (ROOT / ".venv" / "Scripts" / "python.exe").exists(), reason="needs the project venv")
def test_real_child_reports_its_status(qapp):
    """End to end: the window starts launcher/bench_run.py status --json without a console
    (the sandboxed LIVETRANSLATE_BENCH_DIR from conftest is inherited)."""
    from bench.window import BenchWindow

    w = BenchWindow(dialogs=FakeDialogs())
    assert w.run("status")
    _wait(qapp, lambda: w.proc is None, timeout=90)
    assert "測試資料" in w.status.text(), w.output.toPlainText()


# ---------------------------------------------------------------------------
# opening it from LiveTranslate
# ---------------------------------------------------------------------------

def test_launch_helpers_start_without_a_console(tmp_path):
    import app_launch

    started = []
    assert app_launch.open_bench_window(tmp_path, popen=lambda a, **kw: started.append((a, kw)))
    assert app_launch.start_livetranslate(tmp_path, popen=lambda a, **kw: started.append((a, kw)))
    (bench_args, kw), (app_args, _kw) = started
    assert bench_args[1:] == ["-I", str(tmp_path / "launcher" / "bench_window.py")] and app_args[1:] == ["main.py"]
    assert kw["creationflags"] & getattr(subprocess, "CREATE_NO_WINDOW", 0) and kw["cwd"] == str(tmp_path)

    def broken(a, **kw):
        raise OSError("no python")
    assert app_launch.start_livetranslate(tmp_path, popen=broken) is False


def test_settings_open_the_benchmark_and_quit():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    bench = src[src.index("    def _open_bench():"):src.index("    panel.bench_requested.connect(_open_bench)")]
    assert bench.index("app_launch.open_bench_window()") < bench.index("on_quit()")
    assert 't("label_bench")' in (ROOT / "control_panel.py").read_text(encoding="utf-8")


def test_json_that_is_not_an_object_is_shown_as_text(qapp):
    """Review: a line like `1` or `null` raised in the slot and aborted the window."""
    w, spawned, _p, _k = _window(qapp, {"download": ([1, None, "x", {"t": "say", "lines": ["ok"]}], 0)})
    w.run("download")
    _wait(qapp, lambda: spawned[-1:] == ["status"] and w.proc is None)
    text = w.output.toPlainText()
    assert "1" in text and "null" in text and "ok" in text


def test_stopping_is_not_reported_as_an_error(qapp):
    from bench.window import BenchWindow

    killed = []
    w = BenchWindow(spawn=lambda c: FakeProc([{"t": "say", "lines": ["x"]}]),  # the status refresh afterwards
                    dialogs=FakeDialogs(confirm=True), kill=killed.append)
    w.proc, w._command = type("P", (), {"stdin": io.StringIO(), "pid": 1})(), "start"
    assert w.stop()
    w._on_finished(1)  # TerminateProcess exit code
    _wait(qapp, lambda: w.proc is None)
    assert "結束代碼" not in w.output.toPlainText() and "已停止" in w.output.toPlainText()


def test_window_stays_when_livetranslate_cannot_start(qapp):
    from bench.window import BenchWindow

    w = BenchWindow(spawn=lambda c: None, dialogs=FakeDialogs(), start_app=lambda: False)
    closed = []
    w.close = lambda: closed.append(True)
    w._open_app()
    assert closed == [] and "無法開啟 LiveTranslate" in w.output.toPlainText()


def test_window_start_failures_are_logged_and_shown():
    src = (ROOT / "launcher" / "bench_window.py").read_text(encoding="utf-8")
    assert "except Exception" in src and "logging" in src and "MessageBox" in src
