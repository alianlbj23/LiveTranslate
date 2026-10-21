"""Settings → Home → Maintenance → Check environment (env_check.py / env_check_ui.py), which
replaced the launcher's console check. Processes, Ollama and HTTP are faked."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import env_check as ec  # noqa: E402
import ollama_server  # noqa: E402

_REAL_CLIENT = httpx.Client
LOCAL = {"name": "Qwen3.5 9B", "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama", "model": "qwen3.5:9b"}


@pytest.fixture(autouse=True)
def _zh_tw():
    import i18n

    before = i18n.get_lang()
    i18n.set_lang("zh_TW")
    yield
    i18n.set_lang(before)


def _done(stdout="", stderr="", code=0):
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr=stderr)


def test_driver_line_is_readable():
    r = ec.check_driver(lambda args, timeout: _done("NVIDIA GeForce RTX 5070, 581.57, 2048, 12227\n"))
    assert r.status == ec.OK and r.detail == "NVIDIA GeForce RTX 5070｜驅動 581.57｜顯示記憶體 2.0 / 11.9 GB"


@pytest.mark.parametrize("outcome, text", [
    (FileNotFoundError(), "找不到 NVIDIA"),
    (subprocess.TimeoutExpired("nvidia-smi", 15), "15 秒"),
    (_done(stderr="NVIDIA-SMI has failed", code=9), "NVIDIA-SMI has failed"),
])
def test_driver_problems_fail(outcome, text):
    def run(args, timeout):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    r = ec.check_driver(run)
    assert r.status == ec.FAIL and text in r.detail


def test_cuda_checks_run_in_a_child_python_and_read_its_json():
    calls = []

    def run(args, timeout):
        calls.append(args)
        if "torch" in args[-1] and "ctranslate2" not in args[-1]:
            return _done(json.dumps({"torch": "2.9.0", "cuda": True, "device": "RTX 5070"}) + "\n")
        return _done("noise\n" + json.dumps({"ct2": "4.6", "fw": "1.2", "devices": 1, "types": ["float16", "int8"]}))
    assert ec.check_torch(run) == ec.Check("PyTorch（CUDA）", ec.OK, "torch 2.9.0｜RTX 5070")
    ct2 = ec.check_ctranslate2(run)
    assert ct2.status == ec.OK and "float16, int8" in ct2.detail
    assert all(a[1] == "-I" and a[2] == "-c" for a in calls)


def test_cuda_missing_warns_and_a_broken_child_fails():
    no_cuda = ec.check_torch(lambda a, t: _done(json.dumps({"torch": "2.9.0+cpu", "cuda": False, "device": ""})))
    assert no_cuda.status == ec.WARN and "CPU" in no_cuda.detail
    broken = ec.check_ctranslate2(lambda a, t: _done(stderr="ModuleNotFoundError: No module named 'ctranslate2'", code=1))
    assert broken.status == ec.FAIL and "ctranslate2" in broken.detail

    def slow(a, t):
        raise subprocess.TimeoutExpired("python", t)
    assert ec.check_torch(slow).status == ec.FAIL


def _fake_ollama(monkeypatch, alive_ports, models=("qwen3.5:9b",), exe=True, tmp=None):
    class Fake:
        def __init__(self, root, http=None):
            self.port = int(root.rsplit(":", 1)[1])

        def version(self):
            return "0.12.6" if self.port in alive_ports else None

        def installed(self):
            return set(models)

    monkeypatch.setattr(ollama_server, "Ollama", Fake)
    monkeypatch.setattr(ollama_server, "port_open", lambda p: p in alive_ports)
    monkeypatch.setattr(ollama_server, "port_owner", lambda p: ("ollama.exe", 1))
    path = tmp / "ollama.exe"
    if exe:
        path.write_bytes(b"")
    monkeypatch.setattr(ollama_server, "OLLAMA_EXE", path)


def test_ollama_ports_and_models(monkeypatch, tmp_path):
    _fake_ollama(monkeypatch, {11434}, tmp=tmp_path)
    r = ec.check_ollama()
    assert r.status == ec.OK
    assert "連接埠 11434：Ollama 0.12.6（程序 ollama.exe）｜模型：qwen3.5:9b" in r.detail
    assert "連接埠 11435：沒有 Ollama 回應" in r.detail


def test_ollama_not_running_is_fine_until_it_is_missing(monkeypatch, tmp_path):
    _fake_ollama(monkeypatch, set(), tmp=tmp_path)
    r = ec.check_ollama()
    assert r.status == ec.WARN and "自動啟動" in r.detail
    missing = tmp_path / "missing"
    missing.mkdir()
    _fake_ollama(monkeypatch, set(), exe=False, tmp=missing)
    r = ec.check_ollama()
    assert r.status == ec.FAIL and "找不到 Windows 版 Ollama" in r.detail


def _mock_httpx(monkeypatch, handler):
    monkeypatch.setattr(httpx, "Client", lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler)))


def test_translation_test_uses_the_local_model_without_thinking(monkeypatch):
    monkeypatch.setattr(ollama_server, "port_open", lambda p: True)
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "今天天氣真好呢"}})
    _mock_httpx(monkeypatch, handler)
    r = ec.check_translation(LOCAL)
    assert r.status == ec.OK and r.detail.startswith("qwen3.5:9b：今天天氣真好呢（")
    assert seen["think"] is False and seen["model"] == "qwen3.5:9b" and ec.TEST_SENTENCE in seen["messages"][0]["content"]


def test_translation_test_problems(monkeypatch):
    assert ec.check_translation({"api_base": "https://api.openai.com/v1", "model": "gpt"}).status == ec.WARN
    monkeypatch.setattr(ollama_server, "port_open", lambda p: False)
    skipped = ec.check_translation(LOCAL)
    assert skipped.status == ec.WARN and "沒在執行" in skipped.detail
    monkeypatch.setattr(ollama_server, "port_open", lambda p: True)
    _mock_httpx(monkeypatch, lambda r: httpx.Response(200, json={"message": {"content": ""}}))
    assert ec.check_translation(LOCAL).status == ec.WARN
    _mock_httpx(monkeypatch, lambda r: httpx.Response(500, text="boom"))
    failed = ec.check_translation(LOCAL)
    assert failed.status == ec.FAIL and "失敗" in failed.detail


def test_a_crashing_check_does_not_stop_the_others(monkeypatch, caplog):
    monkeypatch.setattr(ec, "check_ollama", lambda: (_ for _ in ()).throw(KeyError("boom")))
    monkeypatch.setattr(ec, "check_translation", lambda entry: ec.Check("翻譯測試", ec.WARN, "skip"))
    got = []
    ec.run_all(None, got.append, run=lambda a, t: _done("x, 1, 1, 1"))
    assert [c.title for c in got][-2:] == ["檢查時發生錯誤", "翻譯測試"] and len(got) == 5
    assert got[-2].status == ec.FAIL and "KeyError" in got[-2].detail
    text = ec.as_text(got)
    assert "[✗] 檢查時發生錯誤" in text and "[!] 翻譯測試\nskip" in text


def test_dialog_lists_results_and_copies_them():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    import env_check_ui

    def fake_run(entry, on_result):
        on_result(ec.Check("A", ec.OK, "fine"))
        on_result(ec.Check("B", ec.FAIL, "broken"))
    d = env_check_ui.EnvCheckDialog(LOCAL, run_all=fake_run)
    end = time.monotonic() + 5
    while not d.copy_btn.isEnabled() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    assert [r.title for r in d.results] == ["A", "B"] and d.status.text() == "檢查完成"
    d.copy_btn.click()
    assert QApplication.clipboard().text() == "[✓] A\nfine\n\n[✗] B\nbroken"


def test_home_page_offers_the_check():
    src = (ROOT / "control_panel.py").read_text(encoding="utf-8")
    assert 't("label_env_check")' in src and "EnvCheckDialog(self.get_active_model()" in src
