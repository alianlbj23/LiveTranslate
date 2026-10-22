"""The app prepares the local translation model itself (local_model_prep / local_model_ui):
Ollama, a missing model (asked, then downloaded with progress), the model entries' port and
a free graphics memory check. This used to be the launcher's console window."""
import gc
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



@pytest.fixture(autouse=True)
def _collect_widgets_now():
    """Parentless Fluent widgets (dialogs, groups) left in reference cycles are otherwise
    collected at some later allocation. When that happens while qfluentwidgets' setTheme()
    walks its style-sheet registry (every ControlPanel calls it), the widget's `destroyed` →
    deregister changes that dict mid-iteration: "dictionary changed size during iteration"
    in an unrelated test. Collect them at the end of each test instead."""
    yield
    gc.collect()

@pytest.fixture(autouse=True)
def _zh_tw():
    """The messages are checked in Traditional Chinese whatever the machine's language."""
    import i18n

    before = i18n.get_lang()
    i18n.set_lang("zh_TW")
    yield
    i18n.set_lang(before)


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
    s = _settings()
    r = _prepare(s, "qwen3.5:9b")
    assert r.error is None and r.port == 11434 and r.low_vram is None and r.installed == {"qwen3.5:9b"}
    assert prep.settings_after(s, r, activate=False) is None


def test_ollama_on_the_spare_port_repoints_the_entries(world):
    world["port"] = 11435
    s = _settings()
    r = _prepare(s, "qwen3.5:9b")
    assert r.error is None and r.port == 11435
    new = prep.settings_after(s, r, activate=False)
    assert new["models"][new["active_model"]]["api_base"] == "http://127.0.0.1:11435/v1"


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
    kept = (prep.settings_after(s, r, activate=False) or s)["models"]
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


def _starter(qapp, prepare_fn, answer=True, settings=None):
    import local_model_ui

    calls = {"apply": [], "warm": [], "notify": [], "asked": []}

    def ask(label, size):
        calls["asked"].append(label)
        return answer
    s = local_model_ui.LocalModelStarter(
        apply_settings=calls["apply"].append, warm=calls["warm"].append,
        notify=lambda title, text, error: calls["notify"].append((title, text, error)),
        log_dir=Path("logs"), get_settings=lambda: settings or _settings(), free_gb_at_start=3.0,
        prepare=prepare_fn, ask=ask, measure=lambda settings: 3.0,  # later runs: never the real card / Ollama
    )
    return s, calls


def test_starter_applies_new_settings_on_the_gui_thread(qapp):
    s, calls = _starter(qapp, lambda *a, **k: prep.PrepResult(port=11435, model="qwen3.5:9b",
                                                              installed={"qwen3.5:9b"}))
    assert s.start({}, {"model": "qwen3.5:9b"})
    _wait(qapp, lambda: calls["apply"])
    applied = calls["apply"][0]
    assert applied["models"][applied["active_model"]]["api_base"] == "http://127.0.0.1:11435/v1"
    assert calls["warm"] == [] and calls["notify"] == []


def test_starter_warms_again_when_nothing_changed_and_warns_on_low_memory(qapp):
    s, calls = _starter(qapp, lambda *a, **k: prep.PrepResult(port=11434, model="qwen3.5:9b", installed={"qwen3.5:9b"}, low_vram="不夠"))
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

    def prepare_fn(settings, entry, log_dir, *, ask_download, on_progress, cancelled, free_gb_at_start, asr_loaded):
        seen["answer"] = ask_download("qwen3.5:4b", "Qwen3.5 4B", 3.4)
        on_progress(2**30, 4 * 2**30, "pulling")
        time.sleep(0.2)  # let the GUI thread show the progress
        seen["cancelled"] = cancelled()
        return prep.PrepResult(port=11434, model="qwen3.5:9b", installed={"qwen3.5:9b"})
    s, calls = _starter(qapp, prepare_fn)
    s.start({}, {"model": "qwen3.5:4b"})
    _wait(qapp, lambda: s._dialog is not None and s._dialog.bar.value() > 0)
    assert s._dialog.bar.value() == 250 and "1.0 GB / 4.0 GB" in s._dialog.status.text()
    _wait(qapp, lambda: calls["warm"])
    assert seen == {"answer": True, "cancelled": False} and calls["asked"] == ["Qwen3.5 4B"]
    assert s._dialog is None


def test_cancel_in_the_progress_window_reaches_the_download(qapp):
    seen = {}

    def prepare_fn(settings, entry, log_dir, *, ask_download, on_progress, cancelled, free_gb_at_start, asr_loaded):
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
        return prep.PrepResult(port=11434, model="qwen3.5:9b", installed={"qwen3.5:9b"})
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


# ---------------------------------------------------------------------------
# review findings (2026-10-21)
# ---------------------------------------------------------------------------

def test_remote_ollama_is_not_this_computers():
    """An Ollama on another computer (api_key "ollama" is its usual placeholder) must not be
    started, downloaded into, or re-pointed at 127.0.0.1."""
    remote = {"name": "LAN", "api_base": "http://192.168.1.20:11434/v1", "api_key": "ollama", "model": "qwen3.5:9b"}
    assert not local_models.is_local_ollama(remote)
    assert local_models.is_local_ollama({"api_base": "http://localhost:11435/v1", "api_key": "x", "model": "m"})
    assert local_models.is_local_ollama({"api_base": "http://127.0.0.1:8080/v1", "api_key": "ollama", "model": "m"})
    s = {"models": [remote, local_models.catalog_entry("qwen3.5:4b", BASE)], "active_model": 1}
    new = local_models.sync_local_models(s, "http://127.0.0.1:11435/v1", "qwen3.5:4b", {"qwen3.5:4b"})
    assert new["models"][0] == remote


def _result(port=11435, model="qwen3.5:9b", installed=frozenset({"qwen3.5:9b"})):
    return prep.PrepResult(port=port, model=model, installed=set(installed))


def test_result_is_applied_to_the_settings_as_they_are_now():
    """Review: a download can take minutes; edits made meanwhile must survive."""
    now = _settings()
    now["models"].append({"name": "Cloud", "api_base": "https://api.example.com/v1", "api_key": "k", "model": "gpt"})
    now["active_model"] = 1  # the user switched to the cloud model during the download
    new = prep.settings_after(now, _result(), activate=False)
    assert [m["name"] for m in new["models"]][:2] == ["Qwen3.5 9B", "Cloud"]
    assert new["models"][new["active_model"]]["name"] == "Cloud"
    assert new["models"][0]["api_base"] == "http://127.0.0.1:11435/v1"
    picked = prep.settings_after(now, _result(), activate=True)  # "download and use" in the settings
    assert picked["models"][picked["active_model"]]["model"] == "qwen3.5:9b"
    assert prep.settings_after(_settings(), _result(port=11434), activate=False) is None  # nothing changed


def test_model_list_is_asked_again_once(world):
    calls = {"n": 0}
    real = _Server.installed

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("busy right after start")
        return real(self)
    _Server.installed = flaky
    try:
        _Server.installed_models = set()
        asked = []
        r = _prepare(_settings("qwen3.5:4b"), "qwen3.5:4b", asked=asked)
        assert calls["n"] == 2 and asked and r.installed == {"qwen3.5:4b"}
    finally:
        _Server.installed = real


def test_speech_model_already_loaded_is_not_counted_twice(world):
    """Review: a later switch measures free memory with the speech model loaded."""
    free = 6.0  # 9B needs 6.3 GB + 0.5 margin; Whisper turbo (2.5 GB) is already in use
    at_start = _prepare(_settings(), "qwen3.5:9b", free_gb_at_start=free)
    later = _prepare(_settings(), "qwen3.5:9b", free_gb_at_start=free, asr_loaded=True)
    assert at_start.low_vram is not None and later.low_vram is not None
    roomy = _prepare(_settings(), "qwen3.5:9b", free_gb_at_start=7.0, asr_loaded=True)
    assert roomy.low_vram is None


def test_ollama_answer_that_is_not_an_object_is_handled(monkeypatch, caplog):
    import hardware_profile

    def handler(request):
        return httpx.Response(200, json=[1, 2, 3])
    server = ollama_server.Ollama("http://127.0.0.1:11434", http=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ValueError):
        server.loaded_vram_bytes()
    monkeypatch.setattr(hardware_profile, "detect_gpu", lambda i: {"name": "x", "total_gb": 12, "free_gb": 4.0})
    monkeypatch.setattr(ollama_server, "port_open", lambda p: p == 11434)
    monkeypatch.setattr(ollama_server, "Ollama", lambda root, http=None: server)
    assert prep.measure_free_gb({}) == pytest.approx(4.0)


def test_question_that_crashes_counts_as_no(qapp, caplog):
    import local_model_ui

    seen = {}

    def prepare_fn(settings, entry, log_dir, *, ask_download, on_progress, cancelled, free_gb_at_start, asr_loaded):
        seen["answer"] = ask_download("m", "M", 1.0)
        return prep.PrepResult(error="還沒下載")

    def broken_ask(label, size):
        raise RuntimeError("dialog failed")
    notes = []
    s = local_model_ui.LocalModelStarter(apply_settings=lambda x: None, warm=lambda e: None,
                                         notify=lambda *a: notes.append(a), log_dir=Path("logs"),
                                         get_settings=dict, prepare=prepare_fn, ask=broken_ask,
                                         measure=lambda settings: 3.0)
    s.start({}, {"model": "m"})
    _wait(qapp, lambda: notes)
    assert seen == {"answer": False} and "dialog failed" in caplog.text
    assert s.start({}, {"model": "m"})  # not stuck "busy"


def test_later_switches_measure_free_memory_again(qapp):
    import local_model_ui

    calls = []

    def prepare_fn(settings, entry, log_dir, *, ask_download, on_progress, cancelled, free_gb_at_start, asr_loaded):
        calls.append((free_gb_at_start, asr_loaded))
        return prep.PrepResult(port=11434, model="m", installed={"m"})
    done = []
    s = local_model_ui.LocalModelStarter(apply_settings=lambda x: None, warm=lambda e: None, notify=lambda *a: None,
                                         log_dir=Path("logs"), get_settings=dict, free_gb_at_start=5.0,
                                         prepare=prepare_fn, measure=lambda settings: 2.5)
    s.finished.connect(done.append)
    s.start({}, {"model": "m"})
    _wait(qapp, lambda: len(done) == 1)
    s.start({}, {"model": "m"}, user_request=True)
    _wait(qapp, lambda: len(done) == 2)
    assert calls == [(5.0, False), (2.5, True)]


def test_quitting_unloads_before_ending_our_ollama_and_its_runners(monkeypatch, real_ollama_server, tmp_path):
    """Review: with a cloud model active the loaded local model was not unloaded before the
    serve was killed, and Ollama's runner child process could keep the memory."""
    import psutil

    order = []

    class Fake:
        def __init__(self, root, http=None):
            self.root = root

        def unload_all(self, timeout=30.0):
            order.append(("unload", self.root))
            return True

    class Proc:
        def __init__(self, pid):
            self.pid = pid

        def name(self):
            return "ollama.exe" if self.pid == 77 else "ollama_llama_server.exe"

        def children(self, recursive=False):
            return [Proc(78)] if self.pid == 77 else []

        def kill(self):
            order.append(("kill", self.pid))

    monkeypatch.setattr(ollama_server, "Ollama", Fake)
    monkeypatch.setattr(ollama_server, "pid_file_server", lambda: 77)
    monkeypatch.setattr(ollama_server, "port_owner", lambda port: ("ollama.exe", 77) if port == 11435 else ("x", 5))
    monkeypatch.setattr(psutil, "Process", Proc)
    monkeypatch.setattr(ollama_server, "stop_ollama", real_ollama_server["stop_ollama"])
    monkeypatch.setattr(ollama_server, "PID_FILE", tmp_path / "ollama.pid")
    assert ollama_server.release_ours() is True
    assert order == [("unload", "http://127.0.0.1:11435"), ("kill", 78), ("kill", 77)]
    monkeypatch.setattr(ollama_server, "pid_file_server", lambda: None)
    assert ollama_server.release_ours() is False  # someone else's Ollama: left alone


def test_settings_button_passes_a_full_entry():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    use = src[src.index("    def _use_local_model(model: str):"):src.index("    panel.local_model_requested.connect(_use_local_model)")]
    assert "local_models.catalog_entry(" in use and "user_request=True" in use
    quit_src = src[src.index("    def release_gpu(self):"):src.index("    def _unload_local_models(self):")]
    assert "ollama_server.release_ours()" in quit_src
