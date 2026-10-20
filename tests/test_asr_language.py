"""asr_language: prompt language names for Fun-ASR-Nano / MLT-Nano and the script check
main.py uses to drop text written in another language than the forced one."""
import pytest

from asr_language import FUNASR_LANGUAGE_NAMES, prompt_language_name, script_language


@pytest.mark.parametrize("forced, text, expected", [
    # Japanese forced: Hangul text is Korean
    ("ja", "위여.", "ko"),
    ("ja", "여보, 루미나.", "ko"),
    ("ja", "ㅋㅋㅋ", "ko"),             # compatibility jamo
    ("ja", "本当に", "ja"),            # kanji + kana
    ("ja", "本当", "ja"),              # kanji only is still Japanese
    ("ja", "OK、行こう", "ja"),         # romaji / English words are neutral
    ("ja", "Thank you.", "ja"),        # pure Latin stays (design choice, documented)
    ("ja", "ソウルで 먹었다", "ja"),     # kana outnumber Hangul (4 vs 3)
    ("ja", "本当 먹어", "ja"),          # tie stays with the forced language
    ("ja", "", "ja"),
    ("ja", "。", "ja"),
    # Korean forced
    ("ko", "빨리 써.", "ko"),
    ("ko", "早く書いて", "ja"),
    ("ko", "ｱﾘｶﾞﾄｳ", "ja"),            # half-width katakana
    ("ko", "你好世界", "zh"),
    ("ko", "먹어 本当", "ko"),          # tie stays with the forced language
    ("ko", "OK", "ko"),
    # English forced
    ("en", "Hello world", "en"),
    ("en", "Hello 世界", "en"),
    ("en", "こんにちは", "ja"),
    ("en", "안녕하세요", "ko"),
    ("en", "你好世界", "zh"),
    ("en", "안녕 日本", "ko"),          # Hangul ties CJK: Korean
    # other forced languages are not judged by script
    ("zh", "위여", "zh"),
    ("fr", "こんにちは", "fr"),
])
def test_script_language(forced, text, expected):
    assert script_language(text, forced) == expected


@pytest.mark.parametrize("engine_type, code, name", [
    ("funasr-mlt-nano", "ja", "日文"),
    ("funasr-mlt-nano", "ko", "韩文"),
    ("funasr-mlt-nano", "en", "英文"),
    ("funasr-mlt-nano", "zh", "中文"),
    ("funasr-mlt-nano", "fr", None),   # not one of MLT's 31 languages
    ("funasr-nano", "ja", "日文"),
    ("funasr-nano", "ko", None),       # Nano: zh / en / ja only
    ("funasr-nano", None, None),
    ("funasr-mlt-nano", "", None),
])
def test_prompt_language_name(engine_type, code, name):
    assert prompt_language_name(engine_type, code) == name


def test_every_mapped_code_is_an_app_language():
    from i18n import LANGUAGES

    codes = {code for code, _ in LANGUAGES}
    assert set(FUNASR_LANGUAGE_NAMES) <= codes


def test_engine_type_from_the_model_profile_matches_the_mapping():
    """prompt_language_name keys on the legacy engine name that asr_funasr.py passes to
    FunASRNanoEngine (profile["legacy_engine"]), not on the model key."""
    from model_manager import funasr_profile

    assert prompt_language_name(funasr_profile("funasr-nano-2512")["legacy_engine"], "ko") is None
    assert prompt_language_name(funasr_profile("funasr-mlt-nano-2512")["legacy_engine"], "ko") == "韩文"
