"""Switching models in the running app: the new translator gets the model's prompt style,
the choice is remembered for the next launch, and a local model is warmed up in the
background. Editing a model in the dialog must not drop fields it does not show."""
import os
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAKURA = {"name": "Sakura GalTransl 7B", "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama",
          "model": "hf.co/SakuraLLM/Sakura-GalTransl-7B-v3:IQ4_XS", "proxy": "none", "thinking_style": "off",
          "streaming": True, "context_turns": 3, "prompt_style": "sakura",
          "overrides": {"temperature": 0.3, "top_p": 0.8}}


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_model_edit_dialog_keeps_fields_it_does_not_show(qapp):
    from dialogs import ModelEditDialog

    d = ModelEditDialog(model_data=dict(SAKURA))
    data = d.get_data()
    assert data["prompt_style"] == "sakura"
    assert data["model"] == SAKURA["model"]
    assert data["overrides"] == SAKURA["overrides"]


def _app():
    import main as main_module

    app = main_module.LiveTranslateApp.__new__(main_module.LiveTranslateApp)
    app._panel = None
    app._target_language = "zh"
    app._config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return app, main_module


def test_switching_to_sakura_builds_a_sakura_translator_and_remembers_it(monkeypatch):
    app, main_module = _app()
    remembered, warmed = [], []
    monkeypatch.setattr(main_module.local_models, "remember_selected", lambda m: remembered.append(m["model"]) or True)
    monkeypatch.setattr(main_module, "_warm_local_model", lambda m: warmed.append(m["model"]))
    app._on_model_changed(dict(SAKURA))
    assert app._translator._prompt_style == "sakura"
    assert remembered == [SAKURA["model"]]
    assert warmed == [SAKURA["model"]]


def test_cloud_model_is_not_warmed(monkeypatch):
    app, main_module = _app()
    warmed = []
    monkeypatch.setattr(main_module.local_models, "remember_selected", lambda m: False)
    monkeypatch.setattr(main_module, "_warm_local_model", lambda m: warmed.append(m["model"]))
    app._on_model_changed({"name": "Cloud", "api_base": "https://api.example.com/v1", "api_key": "sk-x", "model": "gpt-x"})
    assert warmed == []


def test_warm_failure_is_logged_not_raised(monkeypatch, caplog):
    """Error path: Ollama not answering while warming must only log."""
    import main as main_module

    with caplog.at_level("WARNING"):
        main_module._warm_local_model_now({"api_base": "http://127.0.0.1:9/v1", "model": "qwen3.5:9b"})
    assert "warm" in caplog.text.lower()
