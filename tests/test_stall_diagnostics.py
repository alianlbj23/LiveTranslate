"""When translation workers get stuck without errors, the app must (1) log where they
are stuck, (2) still quit within seconds so the launcher can release the GPU, and
(3) never call the tray icon from a worker thread."""
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import main as main_module  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _stub_app():
    app = main_module.LiveTranslateApp.__new__(main_module.LiveTranslateApp)
    app._tl_inflight = 0
    app._tl_backlog_threshold = 3
    app._tl_backlog_last_warn = 0.0
    app._tl_backlog_callback = None
    app._abandoned_threads = False
    return app


def _blocked_until(event, name):
    def _block_here():
        event.wait()

    th = threading.Thread(target=_block_here, name=name, daemon=True)
    th.start()
    return th


# ---------------------------------------------------------------------------
# thread dump
# ---------------------------------------------------------------------------

def test_backlog_warning_dumps_worker_stacks(caplog):
    app = _stub_app()
    release = threading.Event()
    _blocked_until(release, "FakeTranslationWorker")
    try:
        app._tl_inflight = 5
        with caplog.at_level(logging.WARNING, logger="LiveTranslate"):
            app._check_translation_backlog()
        text = caplog.text
        assert "Translation backlog" in text
        assert "Thread dump" in text
        assert "FakeTranslationWorker" in text
        assert "_block_here" in text, "the blocked frame must be visible in the dump"
    finally:
        release.set()


def test_thread_dump_never_raises(monkeypatch, caplog):
    """Error path: a broken frame table must not crash the pipeline thread."""
    app = _stub_app()
    monkeypatch.setattr(sys, "_current_frames", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level(logging.WARNING, logger="LiveTranslate"):
        app._dump_worker_threads("test")
    assert "thread dump failed" in caplog.text


# ---------------------------------------------------------------------------
# bounded executor shutdown
# ---------------------------------------------------------------------------

def test_shutdown_abandons_stuck_worker_within_timeout(caplog):
    app = _stub_app()
    release = threading.Event()
    ex = ThreadPoolExecutor(max_workers=1)
    ex.submit(release.wait)
    time.sleep(0.1)
    try:
        t0 = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="LiveTranslate"):
            ok = app._shutdown_executor(ex, "translation", timeout=1.0)
        elapsed = time.monotonic() - t0
        assert ok is False
        assert elapsed < 3.0, f"shutdown must not wait for the stuck worker (took {elapsed:.1f}s)"
        assert app._abandoned_threads is True
        assert "translation executor: 1 worker(s) still running" in caplog.text
        assert "Thread dump (translation executor shutdown)" in caplog.text
    finally:
        release.set()


def test_shutdown_waits_for_fast_worker_without_warning(caplog):
    app = _stub_app()
    ex = ThreadPoolExecutor(max_workers=1)
    done = []
    ex.submit(lambda: (time.sleep(0.2), done.append(1)))
    with caplog.at_level(logging.WARNING, logger="LiveTranslate"):
        ok = app._shutdown_executor(ex, "translation", timeout=5.0)
    assert ok is True and done == [1]
    assert app._abandoned_threads is False
    assert "still running" not in caplog.text


def test_shutdown_cancels_queued_tasks():
    """Queued (not yet started) translations are dropped instead of being run during quit."""
    app = _stub_app()
    release = threading.Event()
    ex = ThreadPoolExecutor(max_workers=1)
    ex.submit(release.wait)
    ran = []
    queued = ex.submit(lambda: ran.append(1))
    time.sleep(0.1)
    try:
        app._shutdown_executor(ex, "translation", timeout=0.5)
        assert queued.cancelled()
        assert ran == []
    finally:
        release.set()


# ---------------------------------------------------------------------------
# tray balloon marshalled to the GUI thread
# ---------------------------------------------------------------------------

class _FakeTray:
    def __init__(self):
        self.calls = []

    def showMessage(self, title, text, *rest):
        self.calls.append((threading.get_ident(), title, text))


def test_tray_notifier_shows_balloon_on_gui_thread(qapp):
    tray = _FakeTray()
    notifier = main_module._TrayNotifier(tray)
    gui_thread = threading.get_ident()
    done = threading.Event()

    def worker():
        notifier.notify("LiveTranslate", "排隊 7 句")
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    assert done.wait(5)
    assert tray.calls == [], "the balloon must not be shown synchronously on the worker thread"
    loop = QEventLoop()
    QTimer.singleShot(200, loop.quit)
    loop.exec()
    assert len(tray.calls) == 1
    ident, title, text = tray.calls[0]
    assert ident == gui_thread
    assert (title, text) == ("LiveTranslate", "排隊 7 句")


def test_tray_notifier_coerces_non_string_payload(qapp):
    tray = _FakeTray()
    notifier = main_module._TrayNotifier(tray)
    notifier.notify(None, 12)  # error path: never raises, shows something readable
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)
    loop.exec()
    assert tray.calls and tray.calls[0][1:] == ("None", "12")
