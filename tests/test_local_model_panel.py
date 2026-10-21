"""Settings → Translation → 本機翻譯模型 (local_model_panel.py): the catalog with download
state and graphics card advice, replacing the launcher's model list."""
import gc
import os
import sys
import time
from pathlib import Path

import httpx
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import local_model_panel as lmp  # noqa: E402
import ollama_server  # noqa: E402

SAKURA = "hf.co/SakuraLLM/Sakura-GalTransl-7B-v3:IQ4_XS"



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
    import i18n

    before = i18n.get_lang()
    i18n.set_lang("zh_TW")
    yield
    i18n.set_lang(before)


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _advice(levels=("ok", "ok", "ok"), recommended="qwen3.5:9b", gpu=True):
    models = [
        {"model": "qwen3.5:9b", "label": "Qwen3.5 9B", "vram_gb": 6.3, "rank": 1, "total_need_gb": 10.3,
         "size_gb": 6.6, "note": "", "level": levels[0]},
        {"model": SAKURA, "label": "Sakura GalTransl 7B", "vram_gb": 4.2, "rank": 2, "total_need_gb": 8.2,
         "size_gb": 4.3, "note": "日文特化", "level": levels[1]},
        {"model": "qwen3.5:4b", "label": "Qwen3.5 4B", "vram_gb": 3.7, "rank": 3, "total_need_gb": 7.7,
         "size_gb": 3.4, "note": "", "level": levels[2]},
    ]
    return {"gpu": {"name": "NVIDIA GeForce RTX 3060", "total_gb": 8.0, "free_gb": 6.0} if gpu else None,
            "asr": {"key": "whisper:turbo", "label": "Whisper large-v3-turbo", "vram_gb": 2.5},
            "models": models, "recommended": recommended}


def _settings(active="qwen3.5:4b"):
    return {"models": [{"name": "x", "api_base": "http://127.0.0.1:11435/v1", "api_key": "ollama", "model": active}],
            "active_model": 0}


def test_summary_line_like_the_launcher():
    assert lmp.summary_text(_advice()) == "RTX 3060（8.0 GB）｜辨識：Whisper large-v3-turbo｜推薦：Qwen3.5 9B"
    assert "放不下任何本機翻譯模型" in lmp.summary_text(_advice(recommended=None))
    assert "讀不到 NVIDIA" in lmp.summary_text(_advice(gpu=False))


def test_tags_and_the_too_big_question():
    a = _advice(levels=("too_big", "tight", "ok"), recommended="qwen3.5:4b")
    big, tight, ok = a["models"]
    assert lmp.tag_for(big, a["recommended"]) == "【顯示卡可能不夠】"
    assert lmp.tag_for(tight, a["recommended"]) == "【接近上限】"
    assert lmp.tag_for(ok, a["recommended"]) == "【推薦】"
    text = lmp.too_big_text(big, a)
    assert "8.0 GB" in text and "10.3 GB" in text and "「Qwen3.5 4B」" in text
    assert lmp.too_big_text(ok, a) is None
    assert lmp.too_big_text(big, _advice(levels=("too_big", "ok", "ok"), gpu=False)) is None


def test_port_comes_from_the_local_model_entries():
    assert lmp.ollama_port(_settings()) == 11435
    assert lmp.ollama_port({"models": [{"api_base": "https://api.example.com/v1", "model": "x"}]}) == 11434


def test_installed_list_is_unknown_without_ollama(monkeypatch, caplog):
    monkeypatch.setattr(ollama_server, "port_open", lambda port: False)
    assert lmp.installed_models(_settings()) is None
    monkeypatch.setattr(ollama_server, "port_open", lambda port: True)

    class Broken:
        def __init__(self, root, http=None):
            pass

        def installed(self):
            raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(ollama_server, "Ollama", Broken)
    assert lmp.installed_models(_settings()) is None
    assert "Cannot list the models" in caplog.text


def _group(qapp, advice=None, installed=frozenset({"qwen3.5:4b", "qwen3.5:9b"}), confirm=True, settings=None):
    calls = {"installed": 0, "confirm": [], "use": []}

    def installed_fn(s):
        calls["installed"] += 1
        return None if installed is None else set(installed)

    def confirm_fn(title, text):
        calls["confirm"].append(title)
        return confirm
    g = lmp.LocalModelsGroup(lambda: settings or _settings(), advice=lambda s: advice or _advice(),
                             installed=installed_fn, confirm=confirm_fn)
    g.use_requested.connect(calls["use"].append)
    return g, calls


def _wait(qapp, cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        qapp.processEvents()
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.01)


def test_building_the_window_does_not_ask_ollama(qapp):
    g, calls = _group(qapp)
    qapp.processEvents()
    assert calls["installed"] == 0
    assert g.card("qwen3.5:9b").button.text() == "使用"  # download state unknown yet
    g.refresh()
    _wait(qapp, lambda: calls["installed"] == 1 and g._installed is not None)


def test_rows_show_tags_download_state_and_the_model_in_use(qapp):
    g, calls = _group(qapp)
    g.refresh()
    _wait(qapp, lambda: g._installed is not None)
    nine, sakura, four = g.card("qwen3.5:9b"), g.card(SAKURA), g.card("qwen3.5:4b")
    assert nine.titleLabel.text() == "Qwen3.5 9B 【推薦】"
    assert "已下載" in nine.contentLabel.text() and nine.button.text() == "使用" and nine.button.isEnabled()
    assert sakura.button.text() == "下載並使用" and "未下載" in sakura.contentLabel.text()
    assert "日文特化" in sakura.contentLabel.text() and "4.3 GB" in sakura.contentLabel.text()
    assert four.button.text() == "使用中" and not four.button.isEnabled()


def test_using_a_model_that_fits_needs_no_question(qapp):
    g, calls = _group(qapp)
    g.card("qwen3.5:9b").button.click()
    assert calls["use"] == ["qwen3.5:9b"] and calls["confirm"] == []


@pytest.mark.parametrize("answer, used", [(False, []), (True, ["qwen3.5:9b"])])
def test_a_model_the_card_may_not_fit_is_confirmed_first(qapp, answer, used):
    g, calls = _group(qapp, advice=_advice(levels=("too_big", "ok", "ok"), recommended=SAKURA), confirm=answer)
    g.card("qwen3.5:9b").button.click()
    assert calls["confirm"] == ["顯示卡可能不夠"] and calls["use"] == used


def test_a_broken_catalog_leaves_the_window_usable(qapp, caplog):
    def broken(settings):
        raise ValueError("bad local_models.json")
    g = lmp.LocalModelsGroup(lambda: _settings(), advice=broken, installed=lambda s: None, confirm=lambda *a: True)
    assert "讀不到 NVIDIA" in g._summary.contentLabel.text() and g.card("qwen3.5:9b") is None
    assert "Cannot judge" in caplog.text


def test_control_panel_hosts_the_group_and_forwards_the_request(qapp, monkeypatch):
    import copy

    import yaml

    import control_panel as cp

    monkeypatch.setattr(cp, "_load_saved_settings", lambda: None)
    monkeypatch.setattr(cp, "_save_settings", lambda s: copy.deepcopy(s))
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    p = cp.ControlPanel(config, saved_settings=_settings())
    monkeypatch.setattr(p, "_auto_save", lambda *a: None)
    got = []
    p.local_model_requested.connect(got.append)
    p._local_models_group.use_requested.emit("qwen3.5:9b")
    assert got == ["qwen3.5:9b"]
    assert p._local_models_group.parent() is not None


def test_settings_button_downloads_without_asking_again(qapp):
    import local_model_prep as prep
    import local_model_ui

    asked, seen = [], {}

    def prepare_fn(settings, entry, log_dir, *, ask_download, on_progress, cancelled, free_gb_at_start, asr_loaded):
        seen["answer"] = ask_download("m", "M", 1.0)
        return prep.PrepResult(port=11434, model="m", installed={"m"})
    finished = []
    s = local_model_ui.LocalModelStarter(apply_settings=lambda x: None, warm=lambda e: None,
                                         notify=lambda *a: None, log_dir=Path("logs"), get_settings=dict,
                                         prepare=prepare_fn, ask=lambda label, size: asked.append(label) or False)
    s.finished.connect(finished.append)
    s.start({}, {"model": "m"}, user_request=True)
    _wait(qapp, lambda: finished)
    assert seen == {"answer": True} and asked == []


def test_main_switches_through_the_starter():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "panel.local_model_requested.connect(_use_local_model)" in src
    assert "user_request=True" in src and "panel.notify_local_model_busy()" in src
    assert "local_starter.finished.connect(" in src
