"""Qt side of local_model_prep: runs prepare() on a thread when LiveTranslate starts, asks
before downloading a translation model, shows the download progress and hands the result
(model entries, warnings) to the app on the GUI thread. This replaces the launcher's
console window."""
import logging
import threading

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QMessageBox, QVBoxLayout
from qfluentwidgets import CaptionLabel, ProgressBar, PushButton, StrongBodyLabel

import local_model_prep
from i18n import t

log = logging.getLogger("LiveTranslate.LocalModel")


def _gb(n) -> str:
    return f"{n / 2**30:.1f} GB"


class ModelPullDialog(QDialog):
    """Progress of one model download; closing it or Cancel asks to stop."""

    cancel_requested = pyqtSignal()

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("lm_ask_title"))
        # The app's Fluent theme is dark; a plain dialog would follow the system's light palette
        self.setObjectName("modelPullDialog")
        self.setStyleSheet("#modelPullDialog { background-color: #202020; }")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(10)
        self.title = StrongBodyLabel(t("lm_pull_heading").format(label=label), self)
        lay.addWidget(self.title)
        self.bar = ProgressBar(self)
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.status = CaptionLabel(t("lm_pull_connecting"), self)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel_btn = PushButton(t("lm_pull_cancel"), self)
        self.cancel_btn.clicked.connect(self._cancel)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)
        self._cancelled = False

    def set_progress(self, done, total, status: str):
        if total:
            self.bar.setValue(int(min(done, total) * 1000 / total))
            self.status.setText(f"{_gb(done)} / {_gb(total)}")
        elif status:
            self.status.setText(status)

    def _cancel(self):
        if self._cancelled:
            return
        self._cancelled = True
        self.cancel_btn.setEnabled(False)
        self.status.setText(t("lm_pull_cancelling"))
        self.cancel_requested.emit()

    def closeEvent(self, event):
        self._cancel()
        super().closeEvent(event)


def ask_download(label: str, size_gb, parent=None) -> bool:
    size = t("lm_ask_size").format(gb=size_gb) if isinstance(size_gb, (int, float)) else ""
    box = QMessageBox(QMessageBox.Icon.Question, t("lm_ask_title"), t("lm_ask_text").format(label=label, size=size),
                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, parent)
    box.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
    return box.exec() == QMessageBox.StandardButton.Yes


class LocalModelStarter(QObject):
    """Prepares a local translation model (local_model_prep) on a thread. get_settings()
    returns the user settings as they are now; apply_settings(settings) and warm(model_entry)
    run on the GUI thread after a successful preparation; notify(title, text, error) shows
    warnings and errors. measure(settings) re-measures free graphics memory for later runs
    (the first run uses free_gb_at_start, measured before the speech model loaded)."""

    finished = pyqtSignal(object)   # PrepResult, after it was applied
    _ask_requested = pyqtSignal(str, str, object)
    _progress = pyqtSignal(object, object, str)
    _done = pyqtSignal(object, object, bool)

    def __init__(self, apply_settings, warm, notify, log_dir, get_settings, free_gb_at_start=None,
                 prepare=None, ask=None, measure=None, parent=None):
        super().__init__(parent)
        self._apply_settings = apply_settings
        self._warm = warm
        self._notify = notify
        self._log_dir = log_dir
        self._get_settings = get_settings
        self._free_gb = free_gb_at_start
        self._first_run = True
        self._prepare = prepare or local_model_prep.prepare
        self._ask_fn = ask or ask_download
        self._measure = measure or local_model_prep.measure_free_gb
        self._busy = threading.Lock()
        self._answer_ready = threading.Event()
        self._answer = False
        self._cancel_event = threading.Event()
        self._user_request = False
        self._dialog = None
        self._ask_requested.connect(self._on_ask)
        self._progress.connect(self._on_progress)
        self._done.connect(self._on_done)

    def start(self, settings: dict, model_entry: dict, user_request: bool = False) -> bool:
        """False when a preparation is already running. user_request=True: the settings'
        "use" / "download and use" button: no download question, and the model becomes the
        active one even if another model was picked while it downloaded."""
        if not self._busy.acquire(blocking=False):
            log.info("Local model preparation already running")
            return False
        self._cancel_event.clear()
        self._user_request = user_request
        first, self._first_run = self._first_run, False
        threading.Thread(target=self._run, args=(dict(settings), dict(model_entry), first, user_request),
                         name="LocalModelPrep", daemon=True).start()
        return True

    def _run(self, settings, entry, first, user_request):
        try:
            # later runs: the speech model is loaded and other programs may have taken memory
            free = self._free_gb if first else self._measure(settings)
            result = self._prepare(
                settings, entry, self._log_dir, ask_download=self._ask_blocking,
                on_progress=lambda d, t, s: self._progress.emit(d, t, s),
                cancelled=self._cancel_event.is_set, free_gb_at_start=free, asr_loaded=not first,
            )
        except Exception as e:  # report every failure instead of leaving translation silently broken
            log.exception("Local model preparation failed")
            result = local_model_prep.PrepResult(error=t("lm_err_crash").format(e=e))
        self._done.emit(result, entry, user_request)

    def _ask_blocking(self, model: str, label: str, size_gb) -> bool:
        self._answer_ready.clear()
        self._ask_requested.emit(model, label, size_gb)
        self._answer_ready.wait()
        return self._answer

    @pyqtSlot(str, str, object)
    def _on_ask(self, model, label, size_gb):
        self._answer = False
        try:
            self._answer = True if self._user_request else bool(self._ask_fn(label, size_gb))
            if self._answer:
                self._dialog = ModelPullDialog(label)
                self._dialog.cancel_requested.connect(self._cancel_event.set)
                self._dialog.show()
        except Exception:  # the worker must never wait forever: a failed question counts as "no"
            log.exception(f"Could not ask about downloading {model}")
            self._answer = False
        finally:
            self._answer_ready.set()

    @pyqtSlot(object, object, str)
    def _on_progress(self, done, total, status):
        if self._dialog is not None:
            self._dialog.set_progress(done, total, status)

    @pyqtSlot(object, object, bool)
    def _on_done(self, result, entry, user_request):
        try:
            if self._dialog is not None:
                self._dialog.cancel_requested.disconnect()
                self._dialog.close()
                self._dialog.deleteLater()
                self._dialog = None
            if result.error:
                log.warning(f"Translation model not ready: {result.error}")
                self._notify(t("lm_title_model"), result.error, True)
                return
            # the settings as they are now: a download can take minutes
            new = local_model_prep.settings_after(self._get_settings(), result, activate=user_request)
            if new is not None:
                self._apply_settings(new)  # re-creates the translator, which warms it up
            else:
                self._warm(entry)  # Ollama may have just started: the first warm-up could not reach it
            if result.low_vram:
                log.warning(result.low_vram)
                self._notify(t("lm_title_low_vram"), result.low_vram, False)
        finally:
            self._busy.release()
            self.finished.emit(result)
