"""Fun-ASR-Nano / Fun-ASR-MLT-Nano language control.

Regression 2026-10-20: with Fun-ASR-MLT-Nano and the ASR language set to Japanese, some
Japanese lines came out in Korean (「위여.」「빨리 써.」). The engine passed the setting
code straight to the model, so the prompt read "语音转写成ja：". The model card only
documents Chinese language names in the prompt ("日文", "韩文", "英文"), so "ja" most
likely gave no hint and the 31-language MLT model guessed.

The engine now sends the documented name (asr_language.prompt_language_name) and labels
results with the forced language; main.py judges each sentence by its script
(tests/test_language_filter.py)."""
import logging

import numpy as np
import pytest

from asr_funasr_nano import FunASRNanoEngine


class _FakeModel:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [{"text": self.text}]


def _engine(engine_type="funasr-mlt-nano", text="こんにちは"):
    eng = FunASRNanoEngine.__new__(FunASRNanoEngine)  # skip loading the real model
    eng._engine_type = engine_type
    eng._model = _FakeModel(text)
    eng.language = None
    eng._prompt_language = None
    return eng


def _run(eng):
    result = eng.transcribe(np.zeros(1600, dtype=np.float32))
    return result, eng._model.calls[-1]


@pytest.mark.parametrize("code, name", [("ja", "日文"), ("ko", "韩文"), ("en", "英文"), ("zh", "中文")])
def test_mlt_prompt_gets_the_chinese_language_name(code, name):
    eng = _engine()
    eng.set_language(code)
    _, kwargs = _run(eng)
    assert kwargs["language"] == name


@pytest.mark.parametrize("code, name", [("ja", "日文"), ("en", "英文"), ("zh", "中文")])
def test_nano_prompt_gets_the_chinese_language_name(code, name):
    eng = _engine("funasr-nano")
    eng.set_language(code)
    _, kwargs = _run(eng)
    assert kwargs["language"] == name


def test_auto_sends_no_language_hint():
    eng = _engine()
    eng.set_language("auto")
    _, kwargs = _run(eng)
    assert "language" not in kwargs


@pytest.mark.parametrize("engine_type, code", [("funasr-nano", "ko"), ("funasr-mlt-nano", "fr")])
def test_language_the_model_does_not_know_sends_no_hint_and_warns(engine_type, code, caplog):
    eng = _engine(engine_type)
    with caplog.at_level(logging.WARNING, logger="LiveTranslate.FunASR-Nano"):
        eng.set_language(code)
    _, kwargs = _run(eng)
    assert "language" not in kwargs
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "LiveTranslate.FunASR-Nano"]
    assert len(warnings) == 1 and code in warnings[0].getMessage()


def test_supported_language_does_not_warn(caplog):
    eng = _engine()
    with caplog.at_level(logging.WARNING, logger="LiveTranslate.FunASR-Nano"):
        eng.set_language("ja")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_switching_to_auto_clears_the_previous_hint():
    eng = _engine(text="위여.")
    eng.set_language("ja")
    eng.set_language("auto")
    result, kwargs = _run(eng)
    assert "language" not in kwargs
    assert result["language"] == "ko"  # auto: guessed from the text


def test_switching_to_an_unsupported_language_clears_the_previous_hint():
    eng = _engine("funasr-nano")
    eng.set_language("ja")
    eng.set_language("ko")
    _, kwargs = _run(eng)
    assert "language" not in kwargs


@pytest.mark.parametrize("bad", [["ja"], 1, {"ja": 1}])
def test_invalid_language_value_is_rejected_and_keeps_the_previous_state(bad):
    eng = _engine()
    eng.set_language("ja")
    with pytest.raises(ValueError):
        eng.set_language(bad)
    assert eng.language == "ja"
    _, kwargs = _run(eng)
    assert kwargs["language"] == "日文"


@pytest.mark.parametrize("value", [None, ""])
def test_empty_language_means_auto(value):
    eng = _engine()
    eng.set_language("ja")
    eng.set_language(value)
    assert eng.language is None
    _, kwargs = _run(eng)
    assert "language" not in kwargs


def test_forced_language_labels_the_result_and_main_judges_the_text():
    """The engine does not relabel: one label for a whole interim buffer would drop the
    Japanese sentences of a mostly-Korean buffer (and pass a Korean sentence in a mostly-
    Japanese one). main.py checks every sentence instead."""
    eng = _engine(text="위여.")
    eng.set_language("ja")
    result, _ = _run(eng)
    assert result == {"text": "위여.", "language": "ja", "language_name": "ja"}
