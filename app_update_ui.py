"""Qt side of app_update: checks on a thread and shows the state on the control panel's
Maintenance card; a quiet check after start notifies once when GitHub has new versions.
"Update and restart" pulls, starts launcher/update_window.ps1 (packages, then restart)
and quits LiveTranslate."""
import logging
import os
import threading

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

import app_update
from i18n import t

log = logging.getLogger("LiveTranslate.Update")


def status_text(s: app_update.UpdateStatus) -> str:
    return {
        app_update.NOT_GIT: t("update_not_git"),
        app_update.NO_GIT_EXE: t("update_no_git"),
        app_update.DETACHED: t("update_detached"),
        app_update.LOCAL_BRANCH: t("update_local_branch").format(branch=s.branch),
        app_update.OFFLINE: t("update_offline"),
        app_update.UP_TO_DATE: t("update_up_to_date").format(branch=s.branch),
        app_update.BEHIND: t("update_behind").format(n=s.behind, branch=s.branch),
        app_update.DIRTY: t("update_dirty").format(n=s.behind),
    }.get(s.state, s.state)


class UpdateController(QObject):
    """show_status(text, can_update, busy) updates the card; notify(text) shows a tray
    message; confirm(title, text) -> bool asks before updating; quit_app() closes
    LiveTranslate the normal way (GPU released, settings saved)."""

    _checked = pyqtSignal(object, bool)
    _pulled = pyqtSignal(object, object)

    def __init__(self, show_status, notify, confirm, quit_app, check=None, pull=None, launch=None,
                 pid=None, parent=None):
        super().__init__(parent)
        self._show = show_status
        self._notify = notify
        self._confirm = confirm
        self._quit = quit_app
        self._check_fn = check or app_update.check
        self._pull_fn = pull or app_update.pull
        self._launch_fn = launch or app_update.launch_restart
        self._pid = pid or os.getpid()
        self.status = None
        self._busy = False
        self._checked.connect(self._on_checked)
        self._pulled.connect(self._on_pulled)

    def check(self, quiet: bool = False) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._show(t("update_checking"), False, True)

        def work():
            try:
                status = self._check_fn()
            except Exception as e:  # never leave the card spinning; reported on the card and logged
                log.exception("Update check failed")
                status = app_update.UpdateStatus(app_update.OFFLINE, detail=str(e))
            self._checked.emit(status, quiet)
        threading.Thread(target=work, name="UpdateCheck", daemon=True).start()
        return True

    @pyqtSlot(object, bool)
    def _on_checked(self, status, quiet):
        self._busy = False
        self.status = status
        if status.detail:
            log.info(f"Update check: {status.state} ({status.detail})")
        self._show(status_text(status), status.can_update, False)
        if quiet and status.can_update:
            self._notify(t("update_tray").format(n=status.behind))

    def apply(self) -> bool:
        s = self.status
        if self._busy or s is None or not s.can_update:
            return False
        if not self._confirm(t("update_confirm_title"), t("update_confirm_text").format(n=s.behind)):
            return False
        self._busy = True
        self._show(t("update_pulling"), False, True)

        def work():
            try:
                changed, error = self._pull_fn(s.branch)
            except Exception as e:  # shown on the card and logged
                log.exception("Update failed")
                changed, error = [], str(e)
            self._pulled.emit(changed, error)
        threading.Thread(target=work, name="UpdatePull", daemon=True).start()
        return True

    @pyqtSlot(object, object)
    def _on_pulled(self, changed, error):
        self._busy = False
        if error:
            log.error(f"Update failed: {error}")
            self._show(t("update_failed").format(e=error), False, False)
            return
        if self._launch_fn(self._pid, app_update.packages_changed(changed)):
            log.info("Restarting for the update")
            self._quit()
        else:
            self._show(t("update_restart_failed"), False, False)
