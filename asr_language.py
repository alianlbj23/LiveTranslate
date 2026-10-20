"""ASR language helpers shared by the Fun-ASR-Nano worker and main.py (standard library
only, so the GUI process can use them without importing an ASR backend)."""

# Fun-ASR-Nano / MLT-Nano take the language as a name inside the prompt
# ("语音转写成日文：", funasr_nano/model.py get_prompt). The model card ("Language control")
# documents only these Simplified-Chinese names, copied verbatim: do not convert them to
# Traditional characters. A code such as "ja" is undocumented and did not stop MLT from
# answering Japanese audio in Korean. Keys are the app's ASR codes (i18n.LANGUAGES) that
# MLT-Nano supports; add one when i18n.LANGUAGES gains another of the card's 31 languages.
FUNASR_LANGUAGE_NAMES = {
    "zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文",
    "vi": "越南语", "id": "印尼语", "th": "泰语", "ms": "马来语", "ar": "阿拉伯语", "hi": "印地语",
    "cs": "捷克语", "da": "丹麦语", "nl": "荷兰语", "fi": "芬兰语", "el": "希腊语", "hu": "匈牙利语",
    "pl": "波兰语", "pt": "葡萄牙语", "ro": "罗马尼亚语", "sv": "瑞典语",
}
# Fun-ASR-Nano-2512: zh / en / ja only (model table and inference example), although
# the card's prose also mentions 31 languages (that is MLT)
_NANO_LANGUAGES = {"zh", "en", "ja"}

# Languages whose text can be told apart by script; others are never judged
SCRIPT_CHECKED_LANGUAGES = {"ja", "ko", "en"}


def prompt_language_name(engine_type: str, code: str | None) -> str | None:
    """Prompt language name for a Fun-ASR engine ("funasr-nano" / "funasr-mlt-nano", the
    profile's legacy_engine), or None (no hint) when the model does not support it."""
    if engine_type == "funasr-nano" and code not in _NANO_LANGUAGES:
        return None
    return FUNASR_LANGUAGE_NAMES.get(code)


def _script_counts(text: str):
    hangul = kana = han = latin = 0
    for c in text:
        o = ord(c)
        if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            hangul += 1
        elif 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
            kana += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            han += 1
        elif c.isascii() and c.isalpha():
            latin += 1
    return hangul, kana, han, latin


def script_language(text: str, forced: str) -> str:
    """The language `text` is written in, judged against a forced ASR language.

    Returns `forced` unless another language's script dominates. Only ja / ko / en are
    judged (SCRIPT_CHECKED_LANGUAGES): Hangul and kana are unambiguous, but Han is shared
    by zh and ja and Latin by en and other languages, so other forced languages are
    returned unchanged. Han counts as Japanese under ja (kanji-only lines) and against
    Korean under ko. Latin is neutral under ja / ko (romaji, English words), so an English
    hallucination under ja is not caught. A tie between the forced language's script and
    the others keeps the forced language; under en a Hangul / CJK tie counts as Korean."""
    if forced not in SCRIPT_CHECKED_LANGUAGES:
        return forced
    hangul, kana, han, latin = _script_counts(text)
    cjk = kana + han
    cjk_lang = "ja" if kana else "zh"
    if forced == "ja":
        return "ko" if hangul > cjk else "ja"
    if forced == "ko":
        return cjk_lang if cjk > hangul else "ko"
    # en
    if hangul + cjk <= latin:
        return "en"
    return "ko" if hangul >= cjk else cjk_lang
