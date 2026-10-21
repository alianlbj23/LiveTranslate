"""Dialog of the environment check (env_check.py): the checks run on a thread and each
result is added as it arrives; "copy" puts the plain text on the clipboard."""
import threading

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    FluentIcon as FIF,
    IconWidget,
    IndeterminateProgressBar,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    StrongBodyLabel,
)

import env_check
from i18n import t

_ICONS = {env_check.OK: FIF.ACCEPT_MEDIUM, env_check.WARN: FIF.INFO, env_check.FAIL: FIF.CLOSE}


class _ResultRow(QWidget):
    def __init__(self, result: env_check.Check, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        icon = IconWidget(_ICONS.get(result.status, FIF.INFO), self)
        icon.setFixedSize(16, 16)
        row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(StrongBodyLabel(result.title, self))
        detail = CaptionLabel(result.detail, self)
        detail.setWordWrap(True)
        detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        col.addWidget(detail)
        row.addLayout(col, 1)


class EnvCheckDialog(QDialog):
    _result = pyqtSignal(object)
    _finished = pyqtSignal()

    def __init__(self, model_entry, parent=None, run_all=None):
        super().__init__(parent)
        self.setWindowTitle(t("env_title"))
        self.resize(640, 520)
        self._run_all = run_all or env_check.run_all
        self.results = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        self.status = BodyLabel(t("env_running"), self)
        lay.addWidget(self.status)
        self.busy = IndeterminateProgressBar(self)
        lay.addWidget(self.busy)
        self._list = QWidget()
        self._rows = QVBoxLayout(self._list)
        self._rows.setContentsMargins(0, 0, 8, 0)
        self._rows.addStretch(1)
        scroll = ScrollArea(self)
        scroll.setWidget(self._list)
        scroll.setWidgetResizable(True)
        scroll.enableTransparentBackground()
        lay.addWidget(scroll, 1)
        buttons = QHBoxLayout()
        self.copy_btn = PushButton(t("env_copy"), self)
        self.copy_btn.setIcon(FIF.COPY)
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._copy)
        buttons.addWidget(self.copy_btn)
        buttons.addStretch(1)
        close = PrimaryPushButton(t("env_close"), self)
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self._result.connect(self._add)
        self._finished.connect(self._done)
        threading.Thread(target=self._work, args=(model_entry,), name="EnvCheck", daemon=True).start()

    def _work(self, model_entry):
        try:
            self._run_all(model_entry, self._result.emit)
        finally:
            self._finished.emit()

    def _add(self, result):
        self.results.append(result)
        self._rows.insertWidget(self._rows.count() - 1, _ResultRow(result, self._list))

    def _done(self):
        self.busy.stop()
        self.busy.hide()
        self.status.setText(t("env_done"))
        self.copy_btn.setEnabled(True)

    def _copy(self):
        QApplication.clipboard().setText(env_check.as_text(self.results))
        self.status.setText(t("env_copied"))
