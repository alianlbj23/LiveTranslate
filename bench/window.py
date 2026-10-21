"""The best-combination benchmark window (replaces launcher/bench.ps1's console menu).

Each button runs one benchmark command in a child Python without a window
(launcher/bench_run.py <command> --json); its questions arrive as JSON lines
(bench/gui_protocol.py) and are asked here with dialogs. "Stop" ends the child and its ASR
workers, like closing the old console window; the run continues later from where it
stopped. LiveTranslate is closed while this window is used (main.py quits after opening
it); "Open LiveTranslate" starts it again."""
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CaptionLabel, PrimaryPushButton, PushButton, Theme, TitleLabel, setTheme

log = logging.getLogger("LiveTranslate.Bench.Window")

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    ("download", "下載測試資料"),
    ("start", "開始評測"),
    ("resume", "從上次中斷的地方繼續"),
    ("report", "查看報告"),
    ("apply", "套用推薦組合"),
]


def child_python() -> str:
    """python.exe of the venv (a console program: run without a window, its output piped)."""
    p = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(p) if p.exists() else sys.executable


def spawn_child(command: str) -> subprocess.Popen:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.Popen(
        [child_python(), "-I", str(ROOT / "launcher" / "bench_run.py"), command, "--json"],
        cwd=str(ROOT), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def kill_tree(proc) -> None:
    """End the child and the ASR workers it started (what closing the console did)."""
    import psutil

    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.Error as e:
                log.warning(f"Could not stop benchmark process {child.pid}: {e}")
    except psutil.Error as e:
        log.info(f"Benchmark process already gone: {e}")
    try:
        proc.kill()
    except OSError as e:
        log.info(f"Benchmark process already gone: {e}")


class ChooseDialog(QDialog):
    """choose_one (single) / choose_many (checkboxes, at least `minimum`)."""

    def __init__(self, title: str, items: list, many: bool, minimum: int = 1, default: int = 0,
                 cancel: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("最佳組合評測")
        self.resize(640, 420)
        self._many = many
        self._minimum = minimum
        lay = QVBoxLayout(self)
        heading = BodyLabel(title, self)
        heading.setWordWrap(True)
        lay.addWidget(heading)
        self.list = QListWidget(self)
        for i, it in enumerate(items):
            text = it["label"] + (f"  — {it['note']}" if it.get("note") else "")
            item = QListWidgetItem(text)
            if many:
                flags = Qt.ItemFlag.ItemIsUserCheckable
                if it.get("enabled", True):
                    flags |= Qt.ItemFlag.ItemIsEnabled
                item.setFlags(flags)
                item.setCheckState(Qt.CheckState.Checked if it.get("checked") else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        if not many and items:
            self.list.setCurrentRow(default if 0 <= default < len(items) else 0)
            self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        lay.addWidget(self.list, 1)
        self.hint = CaptionLabel(f"至少勾選 {minimum} 項" if many else "", self)
        lay.addWidget(self.hint)
        buttons = QDialogButtonBox.StandardButton.Ok
        if cancel:
            buttons |= QDialogButtonBox.StandardButton.Cancel
        self.buttons = QDialogButtonBox(buttons, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        if many:
            self.list.itemChanged.connect(self._update_ok)
            self._update_ok()

    def checked(self) -> list:
        return [i for i in range(self.list.count()) if self.list.item(i).checkState() == Qt.CheckState.Checked]

    def _update_ok(self, *_a):
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(len(self.checked()) >= self._minimum)


class Dialogs:
    """The questions as Qt dialogs (replaced by fakes in tests)."""

    def __init__(self, parent):
        self.parent = parent

    def ask(self, prompt: str) -> str:
        if "Enter" in prompt and "0" in prompt:  # "close LiveTranslate, then press Enter to check again / 0 cancels"
            box = QMessageBox(QMessageBox.Icon.Information, "最佳組合評測", prompt.replace("按 Enter", "按「重新檢查」")
                              .replace("輸入 0 取消", "或取消"), parent=self.parent)
            again = box.addButton("重新檢查", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            return "" if box.clickedButton() is again else "0"
        text, ok = QInputDialog.getText(self.parent, "最佳組合評測", prompt)
        return text if ok else "0"

    def confirm(self, question: str, default: bool) -> bool:
        buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        box = QMessageBox(QMessageBox.Icon.Question, "最佳組合評測", question, buttons, self.parent)
        box.setDefaultButton(QMessageBox.StandardButton.Yes if default else QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def pause(self, message: str):
        QMessageBox.information(self.parent, "最佳組合評測", message)

    def choose_one(self, title, labels, default, cancel):
        d = ChooseDialog(title, [{"label": x} for x in labels], many=False, default=default, cancel=cancel,
                         parent=self.parent)
        return d.list.currentRow() if d.exec() else None

    def choose_many(self, title, items, minimum):
        d = ChooseDialog(title, items, many=True, minimum=minimum, parent=self.parent)
        return d.checked() if d.exec() else None


class _Pipe(QObject):
    message = pyqtSignal(object)
    error_line = pyqtSignal(str)
    finished = pyqtSignal(int)


class BenchWindow(QWidget):
    def __init__(self, spawn=None, dialogs=None, start_app=None, kill=None):
        super().__init__()
        self.setWindowTitle("LiveTranslate 最佳組合評測")
        self.resize(820, 600)
        # Same dark Fluent look as LiveTranslate's settings, whatever the system theme
        setTheme(Theme.DARK, save=False)
        self.setObjectName("benchWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("#benchWindow { background-color: #202020; }"
                           "QPlainTextEdit { background-color: #2b2b2b; color: #e6e6e6; border: 1px solid #3a3a3a; }")
        self._spawn = spawn or spawn_child
        self._kill = kill or kill_tree
        self._start_app = start_app or self._start_livetranslate
        self.dialogs = dialogs or Dialogs(self)
        self.proc = None
        self._command = None
        self._pipe = _Pipe()
        self._pipe.message.connect(self._on_message)
        self._pipe.error_line.connect(lambda line: self._append(line, warn=True))
        self._pipe.finished.connect(self._on_finished)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.addWidget(TitleLabel("最佳組合評測", self))
        intro = CaptionLabel("用測試資料比較語音辨識、語音偵測與翻譯模型的組合，找出這台電腦最適合的設定。"
                             "評測期間 LiveTranslate 需要關閉。", self)
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.status = BodyLabel("", self)
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        row = QHBoxLayout()
        self.buttons = {}
        for cmd, label in COMMANDS:
            b = PrimaryPushButton(label, self) if cmd == "start" else PushButton(label, self)
            b.clicked.connect(lambda _c=False, c=cmd: self.run(c))
            row.addWidget(b)
            self.buttons[cmd] = b
        row.addStretch(1)
        lay.addLayout(row)

        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        lay.addWidget(self.output, 1)
        self.progress = CaptionLabel("", self)
        lay.addWidget(self.progress)

        bottom = QHBoxLayout()
        self.stop_btn = PushButton("停止", self)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        bottom.addWidget(self.stop_btn)
        bottom.addStretch(1)
        self.open_app_btn = PushButton("開啟 LiveTranslate", self)
        self.open_app_btn.clicked.connect(self._open_app)
        bottom.addWidget(self.open_app_btn)
        close = PushButton("關閉", self)
        close.clicked.connect(self.close)
        bottom.addWidget(close)
        lay.addLayout(bottom)

    # -- running a command --

    def run(self, command: str) -> bool:
        if self.proc is not None:
            return False
        self._command = command
        if command != "status":
            self._append(f"── {dict(COMMANDS).get(command, command)} ──")
        try:
            self.proc = self._spawn(command)
        except OSError as e:
            self._append(f"無法啟動評測程式：{e}", warn=True)
            self.proc = None
            return False
        self._set_running(True)
        proc = self.proc
        threading.Thread(target=self._read_stdout, args=(proc,), name="BenchOut", daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(proc,), name="BenchErr", daemon=True).start()
        return True

    def _read_stdout(self, proc):
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._pipe.message.emit(json.loads(line))
            except ValueError:
                self._pipe.message.emit({"t": "say", "lines": [line]})
        self._pipe.finished.emit(proc.wait())

    def _read_stderr(self, proc):
        for line in proc.stderr:
            if line.strip():
                self._pipe.error_line.emit(line.rstrip())

    def _answer(self, obj: dict):
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except OSError as e:  # the child ended while the question was open
            log.info(f"Benchmark ended before the answer: {e}")

    def _on_message(self, msg: dict):
        kind = msg.get("t")
        if kind == "say":
            lines = msg.get("lines", [])
            if self._command == "status":
                self.status.setText("\n".join(str(x).strip() for x in lines if str(x).strip()))
            else:
                for line in lines:
                    self._append(str(line))
        elif kind == "progress":
            self.progress.setText(str(msg.get("text", "")))
        elif kind == "ask":
            self._answer({"text": self.dialogs.ask(str(msg.get("prompt", "")))})
        elif kind == "confirm":
            self._answer({"yes": self.dialogs.confirm(str(msg.get("q", "")), bool(msg.get("default", True)))})
        elif kind == "pause":
            self.dialogs.pause(str(msg.get("message", "")))
            self._answer({})
        elif kind == "choose_one":
            self._answer({"index": self.dialogs.choose_one(msg.get("title", ""), msg.get("labels", []),
                                                           int(msg.get("default", 0)), bool(msg.get("cancel", True)))})
        elif kind == "choose_many":
            self._answer({"indexes": self.dialogs.choose_many(msg.get("title", ""), msg.get("items", []),
                                                              int(msg.get("minimum", 1)))})
        else:
            log.warning(f"Unknown benchmark message: {msg!r}")

    def _on_finished(self, code: int):
        command = self._command
        self.proc = None
        self._command = None
        self.progress.setText("")
        self._set_running(False)
        if command != "status":
            if code not in (0, None):
                self._append(f"（結束代碼 {code}）", warn=code != 130)
            self.run("status")  # the header shows what changed

    def _set_running(self, running: bool):
        for b in self.buttons.values():
            b.setEnabled(not running)
        self.stop_btn.setEnabled(running and self._command != "status")
        self.open_app_btn.setEnabled(not running)

    def _append(self, text: str, warn: bool = False):
        self.output.appendPlainText(("⚠ " if warn else "") + text)

    # -- stop / close / LiveTranslate --

    def stop(self) -> bool:
        if self.proc is None:
            return False
        if not self.dialogs.confirm("要停止目前的評測嗎？已完成的部分會保留，之後可以從中斷的地方繼續。", False):
            return False
        self._append("已停止。", warn=True)
        self._kill(self.proc)
        return True

    def closeEvent(self, event):
        if self.proc is not None and self._command != "status" and not self.stop():
            event.ignore()
            return
        if self.proc is not None:
            self._kill(self.proc)
        super().closeEvent(event)

    def _open_app(self):
        if self.proc is not None:
            return
        self._start_app()
        self.close()

    @staticmethod
    def _start_livetranslate():
        import app_launch

        app_launch.start_livetranslate(ROOT)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    w = BenchWindow()
    w.show()
    w.run("status")
    return app.exec()
