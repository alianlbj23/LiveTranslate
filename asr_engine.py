import logging
import gc
import os

import numpy as np
from faster_whisper import WhisperModel

from translator import LANGUAGE_DISPLAY

log = logging.getLogger("LiveTranslate.ASR")

SAMPLE_RATE = 16000
DEFAULT_PAD_SECONDS = 0.5
PAD_SECONDS_ENV = "LIVETRANS_WHISPER_PAD_SECONDS"

LANGUAGE_NAMES = {**LANGUAGE_DISPLAY, "auto": "auto"}


class ASREngine:
    """Speech-to-text using faster-whisper."""

    def __init__(
        self,
        model_size="medium",
        device="cuda",
        device_index=0,
        compute_type="float16",
        language="auto",
        download_root=None,
        pad_seconds=None,
        beam_size=5,
        whisper_options=None,
    ):
        self.language = language if language != "auto" else None
        # beam_size=1 (greedy) measured ~20% faster than the upstream default of 5
        # (RTX 5070, large-v3-turbo float16, 11s clip: 273ms -> 213ms)
        try:
            self._beam_size = max(1, int(beam_size))
        except (TypeError, ValueError):
            self._beam_size = 5
        self._model = WhisperModel(
            model_size,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            download_root=download_root,
        )
        opts = dict(whisper_options or {})
        self._condition_on_previous_text = bool(opts.get("condition_on_previous_text", True))
        self._log_prob_threshold = opts.get("log_prob_threshold", -1.0)
        self._no_speech_threshold = opts.get("no_speech_threshold", 0.6)
        self._initial_prompt = (opts.get("initial_prompt") or "").strip() or None
        self._hotwords = (opts.get("hotwords") or "").strip() or None
        self._hallucination_phrases = {
            self._normalize_text(p) for p in (opts.get("hallucination_phrases") or []) if p
        }
        self._set_input_padding(pad_seconds, log_change=False)
        log.info(
            f"Model loaded: {model_size} on {device} ({compute_type}, beam_size={self._beam_size}, "
            f"condition_on_previous_text={self._condition_on_previous_text}, "
            f"hallucination_filter={len(self._hallucination_phrases)} phrases)"
        )
        self._log_input_padding()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(ch for ch in text if ch.isalnum()).lower()

    def _is_hallucination(self, text: str) -> bool:
        if not self._hallucination_phrases:
            return False
        norm = self._normalize_text(text)
        if not norm or norm in self._hallucination_phrases:
            return bool(norm)
        # The segment is nothing but one known phrase repeated
        return any(p and norm.replace(p, "") == "" for p in self._hallucination_phrases)

    @staticmethod
    def _read_pad_seconds(value=None) -> float:
        if value is None:
            value = os.environ.get(PAD_SECONDS_ENV)
        if value is None:
            return DEFAULT_PAD_SECONDS
        try:
            return float(value)
        except (TypeError, ValueError):
            log.warning(
                f"Invalid Whisper pad seconds={value!r}; "
                f"using default {DEFAULT_PAD_SECONDS:g}s"
            )
            return DEFAULT_PAD_SECONDS

    def _set_input_padding(self, pad_seconds=None, log_change=True):
        self._pad_seconds = self._read_pad_seconds(pad_seconds)
        self._pad_quantum = int(round(SAMPLE_RATE * self._pad_seconds))
        if log_change:
            self._log_input_padding()

    def _log_input_padding(self):
        if self._pad_quantum > 0:
            log.info(
                "Whisper input padding enabled: "
                f"bucket={self._pad_seconds:g}s, quantum={self._pad_quantum} samples"
            )
        else:
            log.info("Whisper input padding disabled")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"ASR language: {old} -> {self.language}")

    def set_input_padding(self, pad_seconds):
        old_quantum = self._pad_quantum
        self._set_input_padding(pad_seconds, log_change=False)
        if self._pad_quantum != old_quantum:
            self._log_input_padding()

    def to_device(self, device: str):
        # ctranslate2 doesn't support device migration; must reload
        return False

    def unload(self):
        model = self._model
        self._model = None
        if model is None:
            return

        for attr in ("model", "feature_extractor", "hf_tokenizer"):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except Exception as e:
                    log.debug(f"Failed to delete WhisperModel.{attr}: {e}")
        del model
        gc.collect()

    def _prepare_audio_input(self, audio: np.ndarray) -> np.ndarray:
        if self._pad_quantum <= 0 or audio.size == 0:
            return audio

        original_samples = audio.shape[0]
        remainder = original_samples % self._pad_quantum
        if remainder == 0:
            return audio

        padded_samples = original_samples + self._pad_quantum - remainder
        padded = np.pad(audio, (0, padded_samples - original_samples), mode="constant")
        log.debug(f"Whisper input padded: {original_samples} -> {padded_samples} samples")
        return padded

    def transcribe(self, audio: np.ndarray, word_timestamps: bool = False) -> dict | None:
        """Transcribe audio segment.

        Args:
            audio: float32 numpy array, 16kHz mono
            word_timestamps: if True, include per-word timestamps in result

        Returns:
            dict with 'text', 'language', 'language_name' (and 'words' if word_timestamps) or None.
        """
        audio_input = self._prepare_audio_input(audio)
        segments, info = self._model.transcribe(
            audio_input,
            language=self.language,
            beam_size=self._beam_size,
            vad_filter=False,
            word_timestamps=word_timestamps,
            condition_on_previous_text=self._condition_on_previous_text,
            log_prob_threshold=self._log_prob_threshold,
            no_speech_threshold=self._no_speech_threshold,
            initial_prompt=self._initial_prompt,
            hotwords=self._hotwords,
        )

        text_parts = []
        words = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            if word_timestamps and seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})

        full_text = " ".join(text_parts).strip()
        if not full_text:
            return None
        if self._is_hallucination(full_text):
            log.info(f"Dropped known hallucination: {full_text!r}")
            return None

        detected_lang = info.language
        result = {
            "text": full_text,
            "language": detected_lang,
            "language_name": LANGUAGE_NAMES.get(detected_lang, detected_lang),
        }
        if word_timestamps and words:
            result["words"] = words
        return result
