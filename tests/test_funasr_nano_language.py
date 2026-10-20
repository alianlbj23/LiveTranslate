"""Fun-ASR-Nano / Fun-ASR-MLT-Nano language control.

Regression 2026-10-20: with Fun-ASR-MLT-Nano and the ASR language set to Japanese, some
Japanese lines came out in Korean (「위여.」「빨리 써.」). The engine passed the setting
code straight to the model, so the prompt read "语音转写成ja：". The models take the
language as a Chinese name in the prompt ("日文", "韩文", "英文"; see the model card),
so "ja" was no hint at all and the 31-language MLT model guessed. The result was also
labelled with the setting ("ja"), so main.py's language filter never caught it."""
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
    assert any(code in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("forced, text, expected", [
    # Japanese forced: Korean output is reported as Korean so main.py drops it
    ("ja", "위여.", "ko"),
    ("ja", "여보, 루미나.", "ko"),
    ("ja", "本当に", "ja"),            # kanji only is still Japanese
    ("ja", "本当", "ja"),
    ("ja", "OK、行こう", "ja"),         # romaji / English words are fine
    ("ja", "ソウルで 먹었다", "ja"),     # mostly Japanese, a Korean word quoted
    # Korean forced
    ("ko", "빨리 써.", "ko"),
    ("ko", "早く書いて", "ja"),
    ("ko", "OK", "ko"),
    # English forced
    ("en", "Hello world", "en"),
    ("en", "Hello 世界", "en"),
    ("en", "こんにちは", "ja"),
    ("en", "안녕하세요", "ko"),
])
def test_result_in_another_writing_system_is_labelled_with_that_language(forced, text, expected):
    eng = _engine(text=text)
    eng.set_language(forced)
    result, _ = _run(eng)
    assert result["language"] == expected
    assert result["text"] == text
