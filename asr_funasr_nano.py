import logging
import os
import sys
import tempfile
import wave
import re
import numpy as np
import torch

log = logging.getLogger("LiveTranslate.FunASR-Nano")

# Add bundled code to path so model.py can resolve its imports (ctc, tools.utils)
_NANO_DIR = os.path.join(os.path.dirname(__file__), "funasr_nano")

# The models take the language as a Chinese name inside the prompt ("语音转写成日文："),
# not as a code: "ja" means nothing to them (model card, "Language control"). Keys are the
# app's ASR language codes (i18n.LANGUAGES) that Fun-ASR-MLT-Nano-2512 supports.
FUNASR_LANGUAGE_NAMES = {
    "zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文",
    "vi": "越南语", "id": "印尼语", "th": "泰语", "ms": "马来语", "ar": "阿拉伯语", "hi": "印地语",
    "cs": "捷克语", "da": "丹麦语", "nl": "荷兰语", "fi": "芬兰语", "el": "希腊语", "hu": "匈牙利语",
    "pl": "波兰语", "pt": "葡萄牙语", "ro": "罗马尼亚语", "sv": "瑞典语",
}
# Fun-ASR-Nano-2512 (non-MLT) only knows Chinese, English and Japanese
_NANO_LANGUAGES = {"zh", "en", "ja"}


def prompt_language_name(engine_type: str, code):
    """Chinese language name for the prompt, or None (no hint) when the model lacks it."""
    if not code:
        return None
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
    """The forced language, unless the text is mostly written in another language's
    script (the prompt is only a hint: MLT sometimes answers Japanese audio in Korean).
    Latin letters are neutral for ja / ko (romaji, English words)."""
    hangul, kana, han, latin = _script_counts(text)
    if forced == "ja":
        return "ko" if hangul > kana + han else "ja"
    if forced == "ko":
        if kana + han > hangul:
            return "ja" if kana else "zh"
        return "ko"
    if forced == "en":
        if hangul + kana + han > latin:
            if hangul >= kana + han:
                return "ko"
            return "ja" if kana else "zh"
        return "en"
    return forced


class FunASRNanoEngine:
    """Speech-to-text using Fun-ASR-Nano-2512 or Fun-ASR-MLT-Nano-2512."""

    def __init__(self, device="cuda", hub="ms", engine_type="funasr-nano"):
        if _NANO_DIR not in sys.path:
            sys.path.insert(0, _NANO_DIR)

        # Pre-register FunASRNano model class before AutoModel looks it up
        import model as _nano_model  # noqa: F401

        from funasr import AutoModel
        from model_manager import (
            ASR_MODEL_IDS,
            ensure_qwen_weights,
            get_local_model_path,
            neutralize_funasr_requirements,
        )

        model_name = ASR_MODEL_IDS[engine_type]
        local = get_local_model_path(engine_type, hub=hub)
        model = local or model_name

        if local:
            # Safety net; the download flow normally fetches these up-front.
            ensure_qwen_weights(local, hub=hub)
            neutralize_funasr_requirements(local)

        self._engine_type = engine_type
        self._prompt_language = None
        prev_cwd = os.getcwd()
        if local:
            os.chdir(local)
        try:
            self._model = AutoModel(
                model=model,
                trust_remote_code=True,
                device=device,
                hub=hub,
                disable_update=True,
            )
        finally:
            os.chdir(prev_cwd)
        self.language = None
        log.info(f"{engine_type} loaded: {model_name} on {device} (hub={hub})")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language and language != "auto" else None
        self._prompt_language = prompt_language_name(self._engine_type, self.language)
        if self.language and not self._prompt_language:
            log.warning(
                f"{self._engine_type} does not support language '{self.language}'; "
                "recognizing without a language hint"
            )
        log.info(f"Fun-ASR-Nano language: {old} -> {self.language} (prompt: {self._prompt_language or 'none'})")

    def to_device(self, device: str):
        self._model.model.to(device)
        log.info(f"Fun-ASR-Nano moved to {device}")

    def unload(self):
        if hasattr(self, "_model") and self._model is not None:
            try:
                self._model.model.to("cpu")
            except Exception:
                pass
            self._model = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        """Transcribe audio segment (float32, 16kHz mono)."""
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            audio_16bit = (audio * 32767).astype(np.int16)
            with wave.open(tmp, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_16bit.tobytes())

            kwargs = {"input": [tmp], "batch_size": 1, "disable_pbar": True}
            if self._prompt_language:
                kwargs["language"] = self._prompt_language

            with torch.inference_mode():
                result = self._model.generate(**kwargs)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        if not result or not result[0].get("text"):
            return None

        # "text" keeps punctuation; "text_tn" strips it all via regex
        text = result[0]["text"]

        # Clean special tags
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        if not text or text == "sil":
            return None

        if self.language:
            # main.py's language filter drops results whose language differs from the setting
            detected_lang = script_language(text, self.language)
            if detected_lang != self.language:
                log.info(f"Expected '{self.language}' but the text is '{detected_lang}': {text[:60]}")
        else:
            detected_lang = self._guess_language(text)

        log.debug(f"ASR: {text}")
        return {
            "text": text,
            "language": detected_lang,
            "language_name": detected_lang,
        }

    def _guess_language(self, text: str) -> str:
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        jp = sum(
            1 for c in text if "\u3040" <= c <= "\u30ff" or "\u31f0" <= c <= "\u31ff"
        )
        ko = sum(1 for c in text if "\uac00" <= c <= "\ud7af")
        total = len(text)
        if total == 0:
            return "auto"
        if jp > 0:
            return "ja"
        if ko > total * 0.3:
            return "ko"
        if cjk > total * 0.3:
            return "zh"
        return "en"
