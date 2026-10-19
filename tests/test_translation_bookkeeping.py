"""The in-flight translation counter (backlog warning) is incremented on the ASR
thread and decremented on the translation workers; it must stay exact."""
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module  # noqa: E402


# ---------------------------------------------------------------------------
# in-flight counter: increments on the ASR thread, decrements on 8 worker threads
# ---------------------------------------------------------------------------

class _InlineExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


def test_inflight_counter_is_exact_under_concurrency():
    app = main_module.LiveTranslateApp.__new__(main_module.LiveTranslateApp)
    app._tl_inflight = 0
    app._tl_backlog_threshold = 10 ** 9
    app._tl_backlog_last_warn = 0.0
    app._tl_backlog_callback = None
    app._tl_executor = _InlineExecutor()
    app._translate_async = lambda *a, **k: None

    def work():
        for _ in range(10000):
            app._submit_translation(1, "x", "ja")

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads constantly, like a busy app at its worst
    try:
        threads = [threading.Thread(target=work) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        sys.setswitchinterval(old)
    assert app._tl_inflight == 0, f"lost updates: counter ended at {app._tl_inflight}"
