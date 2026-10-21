"""The app prepares the local translation model itself (local_model_prep / local_model_ui):
Ollama, a missing model (asked, then downloaded with progress), the model entries' port and
a free graphics memory check. This used to be the launcher's console window."""
import copy
import os
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import local_model_prep as prep  # noqa: E402
import local_models  # noqa: E402
import ollama_server  # noqa: E402

BASE = "http://127.0.0.1:11434/v1"


def _settings(model="qwen3.5:9b", base=BASE):
    return {"asr_engine": "whisper", "whisper_model_size": "large-v3-turbo",
            "models": [local_models.catalog_entry(model, base)], "active_model": 0}


class _Server:
    """Fake ollama_server.Ollama: installed models, pull outcome."""

    installed_models = {"qwen3.5:9b"}
    pull_outcome = None   # None = success, or an exception to raise
    pulled = []
    list_error = None

    def __init__(self, root, http=None):
        self.root = root

    def installed(self):
        if _Server.list_error:
            raise _Server.list_error
        return set(_Server.installed_models)

    def pull(self, model, on_progress=None, cancelled=None):
        _Server.pulled.append(model)
        if on_progress:
            on_progress(5, 10, "pulling")
        if _Server.pull_outcome:
            raise _Server.pull_outcome


@pytest.fixture
def world(monkeypatch):
    _Server.installed_models = {"qwen3.5:9b"}
    _Server.pull_outcome = None
    _Server.pulled = []
    _Server.list_error = None
    state = {"port": 11434}

    def ensure(model, log_dir, say=None):
        if isinstance(state["port"], Exception):
            raise state["port"]
        return state["port"]

    monkeypatch.setattr(ollama_server, "ensure_server", ensure)
    monkeypatch.setattr(ollama_server, "Ollama", _Server)
    return state


def _prepare(settings, model, asked=None, answer=True, **kw):
    asked = asked if asked is not None else []

    def ask(m, label, size):
        asked.append((m, label, size))
        return answer
    entry = next(e for e in settings["models"] if e["model"] == model)
    return prep.prepare(settings, entry, Path("logs"), ask_download=ask, **kw)


def test_ready_model_on_the_same_port_changes_nothing(world):
    r = _prepare(_settings(), "qwen3.5:9b")
    assert r.error is None and r.port == 11434 and r.settings is None and r.low_vram is None


def test_ollama_on_the_spare_port_repoints_the_entries(world):
    world["port"] = 11435
    r = _prepare(_settings(), "qwen3.5:9b")
    assert r.error is None and r.port == 11435
    assert r.settings["models"][r.settings["active_model"]]["api_base"] == "http://127.0.0.1:11435/v1"


def test_missing_model_is_asked_for_and_downloaded(world):
    _Server.installed_models = set()
    asked, progress = [], []
    r = _prepare(_settings("qwen3.5:4b"), "qwen3.5:4b", asked=asked, on_progress=lambda *a: progress.append(a))
    assert asked and asked[0][0] == "qwen3.5:4b" and asked[0][1] == "Qwen3.5 4B" and asked[0][2] > 0
    assert _Server.pulled == ["qwen3.5:4b"] and progress == [(5, 10, "pulling")]
    assert r.error is None


def test_declined_download_is_reported_and_nothing_is_pulled(world):
    _Server.installed_models = set()
    r = _prepare(_settings(), "qwen3.5:9b", answer=False)
    assert "還沒下載" in r.error and _Server.pulled == []


@pytest.mark.parametrize("outcome, text", [
    (ollama_server.PullError("disk full"), "下載失敗：disk full"),
    (ollama_server.PullCancelled(), "已取消下載"),
])
def test_failed_or_cancelled_download_is_reported(world, outcome, text):
    _Server.installed_models = set()
    _Server.pull_outcome = outcome
    r = _prepare(_settings(), "qwen3.5:9b")
    assert text in r.error


def test_ollama_that_cannot_start_is_reported(world):
    world["port"] = RuntimeError("找不到 Windows 版 Ollama")
    r = _prepare(_settings(), "qwen3.5:9b")
    assert r.error.startswith("無法啟動 Ollama") and "找不到" in r.error


def test_unknown_model_list_downloads_nothing_and_removes_nothing(world, caplog):
    _Server.list_error = httpx.ConnectError("refused")
    s = _settings()
    s["models"].append(local_models.catalog_entry("qwen3.5:4b", BASE))
    r = _prepare(s, "qwen3.5:9b")
    assert r.error is None and _Server.pulled == []
    kept = (r.settings or s)["models"]
    assert {m["model"] for m in kept} == {"qwen3.5:9b", "qwen3.5:4b"}
    assert "Cannot list the models" in caplog.text


@pytest.mark.parametrize("free, warned", [(2.0, True), (40.0, False), (None, False)])
def test_free_graphics_memory_is_checked_for_the_pair(world, free, warned):
    r = _prepare(_settings(), "qwen3.5:9b", free_gb_at_start=free)
    assert (r.low_vram is not None) == warned
    if warned:
        assert "Qwen3.5 9B" in r.low_vram and "2.0 GB" in r.low_vram and "【推薦】" in r.low_vram


def test_free_memory_counts_what_ollama_holds(monkeypatch, caplog):
    import hardware_profile

    monkeypatch.setattr(hardware_profile, "detect_gpu", lambda i: {"name": "x", "total_gb": 12, "free_gb": 4.0})
    monkeypatch.setattr(ollama_server, "port_open", lambda p: p == 11434)

    class Holding:
        def __init__(self, root, http=None):
            pass

        def loaded_vram_bytes(self):
            return 2 * 2**30

    monkeypatch.setattr(ollama_server, "Ollama", Holding)
    assert prep.measure_free_gb({}) == pytest.approx(6.0)

    class Broken(Holding):
        def loaded_vram_bytes(self):
            raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(ollama_server, "Ollama", Broken)
    assert prep.measure_free_gb({}) == pytest.approx(4.0)
    assert "Cannot read the models Ollama holds" in caplog.text
    monkeypatch.setattr(hardware_profile, "detect_gpu", lambda i: None)
    assert prep.measure_free_gb({}) is None


# ---------------------------------------------------------------------------
# Qt side
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(qapp, cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        qapp.processEvents()
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.01)


def _starter(qapp, prepare_fn, answer=True):
    import local_model_ui

    calls = {"apply": [], "warm": [], "notify": [], "asked": []}

    def ask(label, size):
        calls["asked"].append(label)
        return answer
    s = local_model_ui.LocalModelStarter(
        apply_settings=calls["apply"].append, warm=calls["warm"].append,
        notify=lambda title, text, error: calls["notify"].append((title, text, error)),
        log_dir=Path("logs"), free_gb_at_start=3.0, prepare=prepare_fn, ask=ask,
    )
    return s, calls


def test_starter_applies_new_settings_on_the_gui_thread(qapp):
    new = {"models": [], "active_model": 0}
    s, calls = _starter(qapp, lambda *a, **k: prep.PrepResult(port=11435, settings=new))
    assert s.start({}, {"model": "qwen3.5:9b"})
    _wait(qapp, lambda: calls["apply"])
    assert calls["apply"] == [new] and calls["warm"] == [] and calls["notify"] == []


def test_starter_warms_again_when_nothing_changed_and_warns_on_low_memory(qapp):
    s, calls = _starter(qapp, lambda *a, **k: prep.PrepResult(port=11434, low_vram="不夠"))
    s.start({}, {"model": "qwen3.5:9b"})
    _wait(qapp, lambda: calls["warm"])
    assert calls["warm"] == [{"model": "qwen3.5:9b"}]
    assert calls["notify"] == [("顯示記憶體不足", "不夠", False)]


def test_starter_reports_errors_and_crashes(qapp):
    s, calls = _starter(qapp, lambda *a, **k: prep.PrepResult(error="還沒下載"))
    s.start({}, {"model": "m"})
    _wait(qapp, lambda: calls["notify"])
    assert calls["notify"] == [("翻譯模型", "還沒下載", True)] and calls["apply"] == calls["warm"] == []

    def boom(*a, **k):
        raise KeyError("models")
    s2, calls2 = _starter(qapp, boom)
    s2.start({}, {"model": "m"})
    _wait(qapp, lambda: calls2["notify"])
    assert calls2["notify"][0][2] is True and "發生錯誤" in calls2["notify"][0][1]


def test_starter_asks_on_the_gui_thread_and_shows_download_progress(qapp):
    seen = {}

    def prepare_fn(settings, entry, log_dir, *, ask_download, on_progress, cancelled, free_gb_at_start):
        seen["answer"] = ask_download("qwen3.5:4b", "Qwen3.5 4B", 3.4)
        on_progress(2**30, 4 * 2**30, "pulling")
        time.sleep(0.2)  # let the GUI thread show the progress
        seen["cancelled"] = cancelled()
        return prep.PrepResult(port=11434)
    s, calls = _starter(qapp, prepare_fn)
    s.start({}, {"model": "qwen3.5:4b"})
    _wait(qapp, lambda: s._dialog is not None and s._dialog.bar.value() > 0)
    assert s._dialog.bar.value() == 250 and "1.0 GB / 4.0 GB" in s._dialog.status.text()
    _wait(qapp, lambda: calls["warm"])
    assert seen == {"answer": True, "cancelled": False} and calls["asked"] == ["Qwen3.5 4B"]
    assert s._dialog is None


def test_cancel_in_the_progress_window_reaches_the_download(qapp):
    seen = {}

    def prepare_fn(settings, entry, log_dir, *, ask_download, on_progress, cancelled, free_gb_at_start):
        ask_download("m", "M", None)
        end = time.monotonic() + 5
        while not cancelled() and time.monotonic() < end:
            time.sleep(0.01)
        seen["cancelled"] = cancelled()
        return prep.PrepResult(error="已取消下載 M")
    s, calls = _starter(qapp, prepare_fn)
    s.start({}, {"model": "m"})
    _wait(qapp, lambda: s._dialog is not None)
    s._dialog.cancel_btn.click()
    _wait(qapp, lambda: calls["notify"])
    assert seen["cancelled"] is True


def test_only_one_preparation_runs_at_a_time(qapp):
    import threading

    release = threading.Event()

    def slow(*a, **k):
        release.wait(5)
        return prep.PrepResult(port=11434)
    s, calls = _starter(qapp, slow)
    assert s.start({}, {"model": "m"}) is True
    assert s.start({}, {"model": "m"}) is False
    release.set()
    _wait(qapp, lambda: calls["warm"])
    assert s.start({}, {"model": "m"}) is True
    _wait(qapp, lambda: len(calls["warm"]) == 2)


# ---------------------------------------------------------------------------
# control panel
# ---------------------------------------------------------------------------

def test_panel_takes_the_prepared_model_entries(qapp, monkeypatch):
    import control_panel as cp

    saved = []
    monkeypatch.setattr(cp, "_load_saved_settings", lambda: None)
    monkeypatch.setattr(cp, "_save_settings", lambda s: saved.append(copy.deepcopy(s)))
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    p = cp.ControlPanel(config, saved_settings=_settings())
    monkeypatch.setattr(p, "_auto_save", lambda *a: None)
    lists, actives = [], []
    p.models_list_changed.connect(lambda models, idx: lists.append((models, idx)))
    p.model_changed.connect(actives.append)
    new = _settings(base="http://127.0.0.1:11435/v1")
    p.replace_local_models(new)
    assert p.get_active_model()["api_base"] == "http://127.0.0.1:11435/v1"
    assert lists and lists[-1][1] == 0 and actives[-1]["api_base"].endswith(":11435/v1")
    assert saved[-1]["models"][0]["api_base"].endswith(":11435/v1")


# ---------------------------------------------------------------------------
# main.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("active", [None, {"name": "cloud", "api_base": "https://api.example.com/v1", "model": "x"}])
def test_quitting_stops_the_ollama_started_for_the_session(monkeypatch, active):
    """Also after switching to a cloud model: the old launcher stopped it after the app."""
    import main

    stopped = []
    monkeypatch.setattr(ollama_server, "pid_file_server", lambda: 77)
    monkeypatch.setattr(ollama_server, "stop_ollama", stopped.append)
    app = main.LiveTranslateApp.__new__(main.LiveTranslateApp)
    app._panel = type("P", (), {"get_active_model": lambda self: active})()
    app.release_gpu()
    app.release_gpu()  # safe twice
    assert stopped == [77]


def test_startup_hands_a_local_model_to_the_starter():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    init = src[src.index("    def _deferred_init():"):src.index("    QTimer.singleShot(100, _deferred_init)")]
    assert "local_models.is_local_ollama(active_model)" in init and "local_starter.start(" in init
    # measured before the speech model starts loading (_deferred_init starts it)
    assert src.index("local_model_prep.measure_free_gb(") < src.index("QTimer.singleShot(100, _deferred_init)")
