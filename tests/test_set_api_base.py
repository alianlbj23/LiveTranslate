"""launcher/set_api_base.py points the Ollama model entry at the launcher's port and
model. The auto-generated display name ("Ollama <model>") must follow the model,
otherwise the overlay's model combo keeps showing the old one."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "launcher" / "set_api_base.py"
API = "http://127.0.0.1:11434/v1"


def _settings(tmp_path, models):
    p = tmp_path / "user_settings.json"
    p.write_text(json.dumps({"models": models, "active_model": 0}, ensure_ascii=False), encoding="utf-8")
    return p


def _run(*args):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


def _models(p):
    return json.loads(p.read_text(encoding="utf-8"))["models"]


def _ollama(name, model):
    return {"name": name, "api_base": API, "api_key": "ollama", "model": model, "proxy": "none"}


def test_auto_generated_name_follows_model_switch(tmp_path):
    p = _settings(tmp_path, [_ollama("Ollama qwen3.5:9b", "qwen3.5:9b")])
    r = _run(p, API, "qwen3.5:4b")
    assert r.returncode == 0, r.stderr
    m = _models(p)[0]
    assert m["model"] == "qwen3.5:4b"
    assert m["name"] == "Ollama qwen3.5:4b"


def test_stale_auto_name_is_repaired_even_if_model_already_matches(tmp_path):
    """The bug seen on 2026-10-19: model already 4b, name still says 9b."""
    p = _settings(tmp_path, [_ollama("Ollama qwen3.5:9b", "qwen3.5:4b")])
    r = _run(p, API, "qwen3.5:4b")
    assert r.returncode == 0, r.stderr
    assert _models(p)[0]["name"] == "Ollama qwen3.5:4b"


def test_custom_name_is_kept(tmp_path):
    p = _settings(tmp_path, [_ollama("我的本機翻譯", "qwen3.5:9b")])
    assert _run(p, API, "qwen3.5:4b").returncode == 0
    m = _models(p)[0]
    assert m["model"] == "qwen3.5:4b"
    assert m["name"] == "我的本機翻譯"


def test_non_ollama_entries_untouched(tmp_path):
    cloud = {"name": "Cloud", "api_base": "https://api.example.com/v1", "api_key": "sk-x", "model": "gpt-x"}
    p = _settings(tmp_path, [cloud, _ollama("Ollama qwen3.5:9b", "qwen3.5:9b")])
    assert _run(p, API, "qwen3.5:4b").returncode == 0
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
