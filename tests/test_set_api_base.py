"""launcher/set_api_base.py runs at every launch: it points the local Ollama model
entries at the launcher's port, keeps one entry per installed catalog model
(local_models.json) so the overlay's model combo can switch between them, and makes the
selected model the active one. Auto-generated names follow the catalog label."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "launcher" / "set_api_base.py"
API = "http://127.0.0.1:11434/v1"

import local_models  # noqa: E402

LABEL = {c["model"]: c["label"] for c in local_models.load_catalog()["models"]}


def _settings(tmp_path, models):
    p = tmp_path / "user_settings.json"
    p.write_text(json.dumps({"models": models, "active_model": 0}, ensure_ascii=False), encoding="utf-8")
    return p


def _run(*args):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _models(p):
    return _load(p)["models"]


def _active(p):
    d = _load(p)
    return d["models"][d["active_model"]]


def _ollama(name, model):
    return {"name": name, "api_base": API, "api_key": "ollama", "model": model, "proxy": "none"}


def test_switch_adds_an_entry_per_installed_model_and_activates_the_selected_one(tmp_path):
    p = _settings(tmp_path, [_ollama("Ollama qwen3.5:9b", "qwen3.5:9b")])
    r = _run(p, API, "qwen3.5:4b", "qwen3.5:9b,qwen3.5:4b")
    assert r.returncode == 0, r.stderr
    assert {m["model"] for m in _models(p)} == {"qwen3.5:9b", "qwen3.5:4b"}
    assert _active(p)["model"] == "qwen3.5:4b"
    assert _active(p)["name"] == LABEL["qwen3.5:4b"]


def test_stale_auto_name_is_repaired_even_if_model_already_matches(tmp_path):
    """The bug seen on 2026-10-19: model already 4b, name still says 9b."""
    p = _settings(tmp_path, [_ollama("Ollama qwen3.5:9b", "qwen3.5:4b")])
    r = _run(p, API, "qwen3.5:4b", "qwen3.5:4b")
    assert r.returncode == 0, r.stderr
    assert _active(p)["name"] == LABEL["qwen3.5:4b"]


def test_custom_name_is_kept(tmp_path):
    p = _settings(tmp_path, [_ollama("我的本機翻譯", "qwen3.5:9b")])
    assert _run(p, API, "qwen3.5:4b", "qwen3.5:9b,qwen3.5:4b").returncode == 0
    by = {m["model"]: m for m in _models(p)}
    assert by["qwen3.5:9b"]["name"] == "我的本機翻譯"
    assert _active(p)["model"] == "qwen3.5:4b"


def test_unreachable_ollama_keeps_entries_and_ensures_the_selected_model(tmp_path):
    """Error path: without an installed list and with no Ollama answering, nothing is
    removed, the selected model is added and active; the run must not hang or fail."""
    p = _settings(tmp_path, [_ollama("Ollama qwen3.5:9b", "qwen3.5:9b")])
    r = _run(p, "http://127.0.0.1:9/v1", "qwen3.5:4b")
    assert r.returncode == 0, r.stderr
    assert {m["model"] for m in _models(p)} == {"qwen3.5:9b", "qwen3.5:4b"}
    assert _active(p)["model"] == "qwen3.5:4b"


def test_non_ollama_entries_untouched(tmp_path):
    cloud = {"name": "Cloud", "api_base": "https://api.example.com/v1", "api_key": "sk-x", "model": "gpt-x"}
    p = _settings(tmp_path, [cloud, _ollama("Ollama qwen3.5:9b", "qwen3.5:9b")])
    assert _run(p, API, "qwen3.5:4b", "qwen3.5:9b,qwen3.5:4b").returncode == 0
    assert _models(p)[0] == cloud


def test_invalid_model_name_is_rejected_and_file_unchanged(tmp_path):
    p = _settings(tmp_path, [_ollama("Ollama qwen3.5:9b", "qwen3.5:9b")])
    before = p.read_bytes()
    r = _run(p, API, 'x"; calc')
    assert r.returncode != 0
    assert p.read_bytes() == before


def test_broken_settings_file_reports_error_without_traceback(tmp_path):
    p = tmp_path / "user_settings.json"
    p.write_text("{not json", encoding="utf-8")
    r = _run(p, API, "qwen3.5:4b")
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "user_settings.json" in r.stderr
    assert p.read_text(encoding="utf-8") == "{not json"


def test_output_is_utf8_for_the_launcher(tmp_path):
    """The launcher reads the helper's output as UTF-8; Chinese model names and Windows
    error messages must not come out in the system code page (cp932 here)."""
    p = _settings(tmp_path, [_ollama("我的本機翻譯", "qwen3.5:9b")])
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), str(p), "http://127.0.0.1:9/v1", "qwen3.5:4b"],
                       capture_output=True, timeout=60)
    assert r.returncode == 0
    r.stdout.decode("utf-8")
    r.stderr.decode("utf-8")
    assert "我的本機翻譯".encode("utf-8") in r.stdout
