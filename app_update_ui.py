"""Qt side of app_update: checks on a thread and shows the state on the control panel's
Maintenance card; a quiet check after start notifies once when GitHub has something newer.
"Update and restart" pulls (git checkout) or downloads the release zip (release package),
starts launcher/update_window.ps1 (copy the release, packages, then restart) and quits
LiveTranslate. Nothing is updated without the user's click."""
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
        app_update.RELEASE_UP_TO_DATE: t("update_release_up_to_date").format(version=s.version),
        app_update.RELEASE_NEWER: t("update_release_newer").format(latest=s.latest, version=s.version),
        app_update.NO_RELEASE: t("update_no_release").format(version=s.version),
    }.get(s.state, s.state)


class UpdateController(QObject):
    """show_status(text, can_update, busy) updates the card; notify(text) shows a tray
    message; confirm(title, text) -> bool asks before updating; quit_app() closes
    LiveTranslate the normal way (GPU released, settings saved)."""

    _checked = pyqtSignal(object, bool)
    _handed_over = pyqtSignal(bool, object)   # the update window runs, error text

    def __init__(self, show_status, notify, confirm, quit_app, check=None, pull=None, launch=None,
                 download=None, pid=None, parent=None):
        super().__init__(parent)
        self._show = show_status
        self._notify = notify
        self._confirm = confirm
        self._quit = quit_app
        self._check_fn = check or app_update.check
        self._pull_fn = pull or app_update.pull
        self._launch_fn = launch or app_update.launch_restart
        self._download_fn = download or app_update.download_release
        self._pid = pid or os.getpid()
        self.status = None
        self._busy = False
        self._checked.connect(self._on_checked)
        self._handed_over.connect(self._on_handed_over)

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
            if status.state == app_update.RELEASE_NEWER:
                self._notify(t("update_release_tray").format(latest=status.latest))
            else:
                self._notify(t("update_tray").format(n=status.behind))

    def apply(self) -> bool:
        s = self.status
        if self._busy or s is None or not s.can_update:
            return False
        # busy while asking too: the quiet check after start must not run during the question
        self._busy = True
        release = s.state == app_update.RELEASE_NEWER
        text = (t("update_release_confirm").format(latest=s.latest) if release
                else t("update_confirm_text").format(n=s.behind))
        if not self._confirm(t("update_confirm_title"), text):
            self._busy = False
            return False
        self._show(t("update_pulling"), False, True)

        def work():
            # fetch (git pull / release download), then hand over to the update window; the
            # launch waits until the window reports that it runs, so it stays off the GUI thread
            try:
                if release:
                    release_zip, error = self._download_fn(s)
                    if error:
                        self._handed_over.emit(False, t("update_failed").format(e=error))
                        return
                    packages = app_update.release_packages_changed(release_zip)
                else:
                    release_zip = None
                    changed, error = self._pull_fn(s.branch)
                    if error:
                        self._handed_over.emit(False, t("update_failed").format(e=error))
                        return
                    packages = app_update.packages_changed(changed)
                if self._launch_fn(self._pid, packages, release_zip=release_zip):
                    self._handed_over.emit(True, None)
                else:  # a release is downloaded but not installed; a git pull is already done
                    self._handed_over.emit(False, t("update_release_not_installed") if release
                                           else t("update_restart_failed"))
            except Exception as e:  # shown on the card and logged
                log.exception("Update failed")
                self._handed_over.emit(False, t("update_failed").format(e=e))
        threading.Thread(target=work, name="UpdateFetch", daemon=True).start()
        return True

    @pyqtSlot(bool, object)
    def _on_handed_over(self, ok, message):
        self._busy = False
        if not ok:
            log.error(f"Update not finished: {message}")
            self._show(message, False, False)
            return
        log.info("Restarting for the update")
        self._quit()
