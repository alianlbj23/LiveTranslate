import logging
import os
import sys
import tempfile
import wave
import re
import numpy as np
import torch

from asr_language import prompt_language_name

log = logging.getLogger("LiveTranslate.FunASR-Nano")

# Add bundled code to path so model.py can resolve its imports (ctc, tools.utils)
_NANO_DIR = os.path.join(os.path.dirname(__file__), "funasr_nano")

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
        self._prompt_language = None
        log.info(f"{engine_type} loaded: {model_name} on {device} (hub={hub})")

    def set_language(self, language: str | None):
        if language is not None and not isinstance(language, str):
            raise ValueError(f"ASR language must be a string, got {type(language).__name__}")
        new = language if language and language != "auto" else None
        prompt = prompt_language_name(self._engine_type, new)
        if new and not prompt:
            # main.py reports this in its own log and notifies (_note_asr_language_support)
            log.warning(
                f"{self._engine_type} does not support language '{new}'; "
                "recognizing without a language hint"
            )
        old = self.language
        self.language, self._prompt_language = new, prompt
        log.info(f"Fun-ASR-Nano language: {old} -> {new} (prompt: {prompt or 'none'})")

    def to_device(self, device: str):
        self._model.model.to(device)
        log.info(f"Fun-ASR-Nano moved to {device}")

    def unload(self):
        if hasattr(self, "_model") and self._model is not None:
            try:
                self._model.model.to("cpu")
            except Exception as e:
                log.warning(f"Could not move Fun-ASR-Nano to CPU before unloading: {e}")
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
            except OSError as e:
                log.warning(f"Could not delete temp audio {tmp}: {e}")

        if not result or not result[0].get("text"):
            return None

        # "text" keeps punctuation; "text_tn" strips it all via regex
        text = result[0]["text"]

        # Clean special tags
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        if not text or text == "sil":
            return None

        # A forced language labels the result; main.py checks each sentence's script
        # (one label for a whole interim buffer cannot be right for every sentence)
        language = self.language or self._guess_language(text)

        log.debug(f"ASR: {text}")
        return {
            "text": text,
            "language": language,
            "language_name": language,
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
