"""ASR language helpers shared by the Fun-ASR-Nano worker, main.py and the benchmark
(standard library only, so the GUI process can use them without importing an ASR
backend): the prompt names of the forced language, and the filter that drops sentences
heard in another language."""
from types import MappingProxyType

# Fun-ASR-Nano / MLT-Nano take the language as a name inside the prompt
# ("语音转写成日文：", funasr_nano/model.py get_prompt). The model card ("Language control")
# documents only these Simplified-Chinese names, copied verbatim: do not convert them to
# Traditional characters. A code such as "ja" is undocumented and did not stop MLT from
# answering Japanese audio in Korean. Keys are the app's ASR codes (i18n.LANGUAGES) that
# MLT-Nano supports; add one when i18n.LANGUAGES gains another of the card's 31 languages.
FUNASR_LANGUAGE_NAMES = MappingProxyType({
    "zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文",
    "vi": "越南语", "id": "印尼语", "th": "泰语", "ms": "马来语", "ar": "阿拉伯语", "hi": "印地语",
    "cs": "捷克语", "da": "丹麦语", "nl": "荷兰语", "fi": "芬兰语", "el": "希腊语", "hu": "匈牙利语",
    "pl": "波兰语", "pt": "葡萄牙语", "ro": "罗马尼亚语", "sv": "瑞典语",
})
# Fun-ASR-Nano-2512: zh / en / ja only (model table and inference example), although
# the card's prose also mentions 31 languages (that is MLT)
_NANO_LANGUAGES = frozenset({"zh", "en", "ja"})
FUNASR_PROMPT_ENGINES = frozenset({"funasr-nano", "funasr-mlt-nano"})

# Languages whose text can be told apart by script; others are never judged by script
# (a foreign engine label still drops them)
SCRIPT_CHECKED_LANGUAGES = frozenset({"ja", "ko", "en"})


def prompt_language_name(engine_type: str, code: str | None) -> str | None:
    """Prompt language name for a Fun-ASR engine ("funasr-nano" / "funasr-mlt-nano", the
    profile's legacy_engine), or None (no hint) when the model does not support it."""
    if engine_type not in FUNASR_PROMPT_ENGINES:
        raise ValueError(f"{engine_type!r} takes no language prompt")
    if engine_type == "funasr-nano" and code not in _NANO_LANGUAGES:
        return None
    return FUNASR_LANGUAGE_NAMES.get(code)


def hint_missing(profile: dict, forced: str) -> bool:
    """True when a FunASR model (model_manager.funasr_profile) cannot be told the forced
    language: a Nano-family model without a prompt name for it (e.g. Nano + ko). Its
    output is then not judged by script."""
    if forced == "auto" or profile.get("family") != "funasr-nano":
        return False
    return prompt_language_name(profile["legacy_engine"], forced) is None


def engine_label(label):
    """The engine's language label, or None when it gave none (SenseVoice answers "auto"
    without a language tag, the remote engine "unknown")."""
    return label if label and label not in ("auto", "unknown") else None


def heard_other_language(text: str, label: str, forced: str, hint_missing: bool = False):
    """The language `text` was heard in when it is not the forced one, else None. A label
    naming another language wins; otherwise ja / ko / en are judged by the text's own
    script, except when the model got no language hint (its output is not judged)."""
    if forced == "auto":
        return None
    named = engine_label(label)
    if named and named != forced:
        return named
    if hint_missing:
        return None
    heard = script_language(text, forced)
    return heard if heard != forced else None


def without_other_language(text: str, label: str, forced: str, split, hint_missing: bool = False):
    """(kept text, [(dropped sentence, heard language)]). One label covers a whole interim
    buffer, so every sentence (split(text, forced)) is judged by itself; the whole text
    is judged at once when the label names another language, the model got no hint, or
    the forced language is not script-checked. Kept sentences are joined without spaces
    for ja, with one space otherwise; an untouched text comes back as it was."""
    if forced == "auto":
        return text, []
    named = engine_label(label)
    if (named and named != forced) or hint_missing or forced not in SCRIPT_CHECKED_LANGUAGES:
        heard = heard_other_language(text, label, forced, hint_missing)
        return ("", [(text, heard)]) if heard else (text, [])
    parts = [p.strip() for p in split(text, forced) if p.strip()]
    if len(parts) <= 1:
        parts = [text]
    kept, dropped = [], []
    for part in parts:
        heard = heard_other_language(part, label, forced, hint_missing)
        if heard:
            dropped.append((part, heard))
        else:
            kept.append(part)
    if not dropped:
        return text, []
    return ("" if forced == "ja" else " ").join(kept), dropped


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
