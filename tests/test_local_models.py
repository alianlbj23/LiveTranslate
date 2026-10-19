"""Local (Ollama) translation models offered in the model combo: catalog in
local_models.json, kept in sync with what is installed, and the choice made in the app
remembered for the next launch (launcher/selected_model.txt)."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import local_models as lm  # noqa: E402

API = "http://127.0.0.1:11434/v1"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")  # launcher/model.ps1 rule
SAKURA = "hf.co/SakuraLLM/Sakura-GalTransl-7B-v3:IQ4_XS"


def _legacy(model="qwen3.5:9b", **extra):
    entry = {"name": f"Ollama {model}", "api_base": "http://127.0.0.1:11435/v1", "api_key": "ollama",
             "model": model, "proxy": "none", "thinking_style": "openai", "streaming": True, "context_turns": 3}
    entry.update(extra)
    return {"models": [entry], "active_model": 0}


def _by_model(settings):
    return {m["model"]: m for m in settings["models"]}


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def test_catalog_lists_qwen_and_sakura_with_valid_names():
    cat = lm.load_catalog()
    models = {c["model"]: c for c in cat["models"]}
    assert cat["default"] == "qwen3.5:9b"
    assert {"qwen3.5:9b", "qwen3.5:4b", SAKURA} <= set(models)
    for c in cat["models"]:
        assert NAME_RE.match(c["model"]), c["model"]
        assert c["label"] and c["size_gb"] > 0
    assert models[SAKURA]["settings"]["prompt_style"] == "sakura"
    assert models[SAKURA]["settings"]["thinking_style"] == "off"


# ---------------------------------------------------------------------------
# sync_local_models
# ---------------------------------------------------------------------------

def test_sync_adds_every_installed_model_and_activates_the_selected_one():
    s = lm.sync_local_models(_legacy(context_turns=5), API, "qwen3.5:9b", ["qwen3.5:9b", "qwen3.5:4b"])
    by = _by_model(s)
    assert set(by) == {"qwen3.5:9b", "qwen3.5:4b"}
    assert s["models"][s["active_model"]]["model"] == "qwen3.5:9b"
    assert by["qwen3.5:9b"]["context_turns"] == 5, "existing per-model tuning is kept"
    assert all(m["api_base"] == API for m in s["models"])
    labels = {c["model"]: c["label"] for c in lm.load_catalog()["models"]}
    assert by["qwen3.5:4b"]["name"] == labels["qwen3.5:4b"]
    assert by["qwen3.5:9b"]["name"] == labels["qwen3.5:9b"], "auto name 'Ollama qwen3.5:9b' follows the catalog label"


def test_sync_adds_selected_model_even_before_it_is_installed():
    """The launcher pulls the selected model; its entry must exist (e.g. Sakura)."""
    s = lm.sync_local_models(_legacy(), API, SAKURA, ["qwen3.5:9b"])
    by = _by_model(s)
    assert s["models"][s["active_model"]]["model"] == SAKURA
    sak = by[SAKURA]
    assert sak["prompt_style"] == "sakura" and sak["thinking_style"] == "off"
    assert sak["overrides"] == {"temperature": 0.3, "top_p": 0.8}
    assert sak["api_key"] == "ollama" and sak["api_base"] == API


def test_sync_drops_uninstalled_catalog_entries_but_keeps_user_entries():
    cloud = {"name": "Cloud", "api_base": "https://api.example.com/v1", "api_key": "sk-x", "model": "gpt-x"}
    custom_local = {"name": "My llama", "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama", "model": "llama3:8b"}
    settings = _legacy()
    settings["models"] += [dict(cloud), dict(custom_local),
                           {"name": "Qwen3.5 4B", "api_base": API, "api_key": "ollama", "model": "qwen3.5:4b"}]
    s = lm.sync_local_models(settings, API, "qwen3.5:9b", ["qwen3.5:9b", "llama3:8b"])
    by = _by_model(s)
    assert "qwen3.5:4b" not in by, "not installed any more"
    assert by["gpt-x"] == cloud
    assert by["llama3:8b"]["name"] == "My llama"
    assert s["models"][s["active_model"]]["model"] == "qwen3.5:9b"


def test_sync_keeps_custom_names():
    s = lm.sync_local_models(_legacy(name="我的翻譯"), API, "qwen3.5:9b", ["qwen3.5:9b"])
    assert s["models"][0]["name"] == "我的翻譯"


def test_sync_tolerates_damaged_settings():
    for bad in (None, {"models": "x"}, {"models": [1, "a", None]}):
        s = lm.sync_local_models(bad, API, "qwen3.5:9b", [])
        assert s["models"][s["active_model"]]["model"] == "qwen3.5:9b"


def test_sync_rejects_invalid_selected_name():
    with pytest.raises(ValueError):
        lm.sync_local_models(_legacy(), API, 'x"; calc', [])


# ---------------------------------------------------------------------------
# remember_selected (choice made in the app -> next launch)
# ---------------------------------------------------------------------------

def test_tests_never_touch_the_real_selected_model_file():
    assert lm.SELECTED_MODEL_FILE.resolve() != (ROOT / "launcher" / "selected_model.txt").resolve()


def test_remember_selected_writes_local_models_only(tmp_path, monkeypatch):
    f = tmp_path / "selected_model.txt"
    monkeypatch.setattr(lm, "SELECTED_MODEL_FILE", f)
    assert lm.remember_selected({"api_key": "ollama", "api_base": API, "model": SAKURA}) is True
    assert f.read_bytes() == SAKURA.encode()
    assert lm.remember_selected({"api_key": "sk-x", "api_base": "https://api.example.com/v1", "model": "gpt-x"}) is False
    assert f.read_bytes() == SAKURA.encode()


def test_remember_selected_rejects_bad_names(tmp_path, monkeypatch, caplog):
    f = tmp_path / "selected_model.txt"
    f.write_bytes(b"qwen3.5:9b")
    monkeypatch.setattr(lm, "SELECTED_MODEL_FILE", f)
    with caplog.at_level("WARNING"):
        assert lm.remember_selected({"api_key": "ollama", "api_base": API, "model": "bad name!"}) is False
    assert f.read_bytes() == b"qwen3.5:9b"
    assert "bad name!" in caplog.text


def test_catalog_file_is_valid_json():
    json.loads((ROOT / "local_models.json").read_text(encoding="utf-8"))


def test_unknown_installed_list_removes_nothing():
    """Error path: when Ollama could not be asked (installed=None), a transient failure
    must not delete other model entries or their tweaks."""
    settings = _legacy(context_turns=5)
    settings["models"].append({"name": "Qwen3.5 4B", "api_base": API, "api_key": "ollama", "model": "qwen3.5:4b",
                               "context_turns": 1})
    s = lm.sync_local_models(settings, API, "qwen3.5:9b", None)
    by = _by_model(s)
    assert set(by) == {"qwen3.5:9b", "qwen3.5:4b"}
    assert by["qwen3.5:4b"]["context_turns"] == 1
