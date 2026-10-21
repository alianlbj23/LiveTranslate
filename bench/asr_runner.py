"""Stage 1 ASR candidates: worker configs (built like main.py builds them), model
downloads, timed transcription through ASRClient, and the same subtitle filters the app
applies, so CER is measured on what would actually be shown.

Copied from main.py (tests/test_bench_asr.py compares them against main.py so they cannot
drift silently): the worker_config keys (LiveTranslateApp._switch_asr_engine), the
sentence splitter (_split_sentences) and the punctuation / noise filters
(_process_segment). The language filter itself is shared (asr_language)."""
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bench import paths

log = logging.getLogger("LiveTranslate.Bench.ASR")

TURBO_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
ASR_EMPTY = "asr_empty"  # error of a translation record whose ASR showed nothing (not a model failure)


ENGINE_TYPES = ("funasr", "anime-whisper", "whisper")  # also the app's asr_engine values


@dataclass(frozen=True)
class AsrCandidate:
    id: str
    label: str
    engine_type: str
    funasr_model: str | None = None
    download_bytes: int = 0         # rough size, for the "download first?" question
    default_item_s: float = 0.5     # seconds per sentence before anything was measured

    def __post_init__(self):
        if self.engine_type not in ENGINE_TYPES:
            raise ValueError(f"{self.id}: unknown engine type {self.engine_type!r}")
        if (self.engine_type == "funasr") != bool(self.funasr_model):
            raise ValueError(f"{self.id}: funasr_model is required for FunASR models and only for them")

    def app_settings(self) -> dict:
        """The user_settings.json keys that select this ASR in LiveTranslate (Whisper
        turbo also needs whisper_model_size, the local folder: settings_io adds it)."""
        out = {"asr_engine": self.engine_type}
        if self.funasr_model:
            out["funasr_model"] = self.funasr_model
        return out


ASR_CANDIDATES = [
    AsrCandidate("sensevoice", "SenseVoice Small", "funasr", funasr_model="sensevoice-small",
                 download_bytes=900_000_000, default_item_s=0.15),
    AsrCandidate("mlt-nano", "Fun-ASR-MLT-Nano", "funasr", funasr_model="funasr-mlt-nano-2512",
                 download_bytes=3_500_000_000, default_item_s=0.6),
    AsrCandidate("anime-whisper", "Anime-Whisper", "anime-whisper", download_bytes=1_600_000_000, default_item_s=0.5),
    AsrCandidate("whisper-turbo", "Whisper large-v3-turbo", "whisper", download_bytes=1_600_000_000,
                 default_item_s=0.35),
]
CANDIDATES_BY_ID = {c.id: c for c in ASR_CANDIDATES}


def find_turbo_path() -> str | None:
    from model_manager import list_local_faster_whisper_models

    for item in list_local_faster_whisper_models():
        if TURBO_REPO.split("/")[1] in item["name"]:
            return item["path"]
    return None


def is_cached(cand: AsrCandidate, hub: str) -> bool:
    from model_manager import is_asr_cached

    if cand.engine_type == "whisper":
        return find_turbo_path() is not None
    return is_asr_cached(cand.engine_type, cand.funasr_model or "medium", hub)


def download(cand: AsrCandidate, hub: str, proxy: str):
    """Download a missing model; raises on failure (the caller logs and skips it)."""
    if cand.engine_type == "whisper":
        script = paths.ROOT / "launcher" / "download_whisper_turbo.py"
        result = subprocess.run([sys.executable, "-I", str(script), str(paths.ROOT)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        if result.returncode != 0 or not find_turbo_path():
            raise RuntimeError(f"Whisper turbo download failed: {result.stderr.strip()[-300:]}")
        return
    from model_manager import download_asr

    download_asr(cand.engine_type, model_size=cand.funasr_model or "medium", hub=hub, proxy=proxy)


def build_worker_config(cand: AsrCandidate, settings: dict, config: dict) -> dict:
    """The worker config main.py's _switch_asr_engine would build for this engine."""
    from model_manager import ASR_DISPLAY_NAMES, MODELS_DIR, funasr_display_name, normalize_funasr_model_key

    asr_cfg = config.get("asr", {})
    funasr_model = normalize_funasr_model_key(cand.funasr_model or settings.get("funasr_model"))
    if cand.engine_type == "whisper":
        model_size = find_turbo_path()
        if not model_size:  # never measure another Whisper model under the turbo name
            raise RuntimeError("找不到 Whisper large-v3-turbo 模型資料夾")
        display_name = "Whisper faster-whisper-large-v3-turbo"
        pad = settings.get("whisper_pad_seconds", asr_cfg.get("whisper_pad_seconds", 0.5))
    elif cand.engine_type == "funasr":
        model_size = funasr_model
        display_name = funasr_display_name(funasr_model)
        pad = settings.get("sensevoice_pad_seconds", asr_cfg.get("sensevoice_pad_seconds", 0.5))
    else:
        model_size = settings.get("whisper_model_size", asr_cfg.get("model_size"))
        display_name = ASR_DISPLAY_NAMES.get(cand.engine_type, cand.engine_type)
        pad = None
    return {
        "engine_type": cand.engine_type,
        "funasr_model": funasr_model,
        "model_size": model_size,
        "device": settings.get("asr_device") or "cuda:0",
        "compute_type": asr_cfg.get("compute_type", "float16"),
        "beam_size": asr_cfg.get("beam_size", 5),
        "whisper_options": asr_cfg.get("whisper") or {},
        "hub": settings.get("hub", "ms"),
        "language": settings.get("asr_language", asr_cfg.get("language", "auto")),
        "pad_seconds": pad,
        "download_root": str((MODELS_DIR / "huggingface" / "hub").resolve()),
        "display_name": display_name,
        "remote_asr_url": settings.get("remote_asr_url", asr_cfg.get("remote_asr_url", "http://127.0.0.1:8765")),
    }


# ---------------------------------------------------------------- subtitle filters (copied from main.py)

_segmenters = {}


def split_sentences(text: str, lang: str = "en") -> list:
    """Copy of LiveTranslateApp._split_sentences (yasbd + comma fallback)."""
    from yasbd import get_supported_langs, pysbd_adapter

    if lang not in _segmenters:
        seg_lang = lang if lang in get_supported_langs() else "en"
        _segmenters[lang] = pysbd_adapter.Segmenter(language=seg_lang, clean=False)
    parts = [p for p in _segmenters[lang].segment(text) if p.strip()]
    if len(parts) > 1:
        return parts
    min_len = 25 if any(c == "、" for c in text) else 60
    if len(text) > min_len:
        for i in range(len(text) - 8, 5, -1):
            if text[i] in ",，;；、":
                before = text[:i + 1].strip()
                after = text[i + 1:].strip()
                if before and after and len(before) > 15 and len(after) > 3:
                    return [before, after]
    return parts


def displayed_text(text: str, label: str, seg_seconds: float, forced: str, hint_missing: bool = False) -> tuple[str, str | None]:
    """What the app would show for an ASR result: (text, reason it was cut or None)."""
    import asr_language

    text = (text or "").strip()
    if not text or not any(c.isalnum() for c in text):
        return "", "empty"
    if seg_seconds >= 2.0 and sum(1 for c in text if c.isalnum()) <= 3:
        return "", "noise"
    kept, dropped = asr_language.without_other_language(text, label, forced, split_sentences, hint_missing)
    if not kept:
        return "", f"language:{dropped[0][1]}"
    return kept, (f"language-partial:{dropped[0][1]}" if dropped else None)


def result_text(result) -> tuple[str, str]:
    """(text, language label) of an ASR result; ("", "") for None."""
    if not result:
        return "", ""
    return result.get("text", ""), result.get("language", "")


def warm_up_filters(forced: str):
    """Build the sentence splitter once (yasbd takes ~0.5 s the first time) so that cost
    never lands on a timed sentence; the app also builds it once and keeps it."""
    displayed_text("今日はいい天気ですね。明日も晴れるといいな", forced, 1.0, forced)


def hint_missing_for(cand: AsrCandidate, forced: str) -> bool:
    """Whether the model cannot be told the forced language (asr_language.hint_missing,
    as main.py's _note_asr_language_support decides it)."""
    if cand.engine_type != "funasr":
        return False
    import asr_language
    from model_manager import funasr_profile

    return asr_language.hint_missing(funasr_profile(cand.funasr_model), forced)


# ---------------------------------------------------------------- session


@dataclass(eq=False)
class AsrSession:
    """One ASR worker: load() starts it, close() stops it (and may be called again).
    `client_factory(config)` builds the client (ASRClient in real runs, a fake in tests)."""
    cand: AsrCandidate
    config: dict
    client_factory: Callable[[dict], Any] | None = None
    client: Any = None
    load_s: float = 0.0
    restarts: int = 0

    def _new_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.config)
        from asr_client import ASRClient

        return ASRClient(self.config, ready_timeout=300.0, request_timeout=60.0)

    def load(self):
        self.close()  # never leave a second worker running
        t0 = time.perf_counter()
        self.client = self._new_client()
        try:
            self.client.start()
            self.client.wait_ready()
        except Exception:
            self.close()
            raise
        self.load_s = time.perf_counter() - t0
        return self.load_s

    def transcribe(self, audio):
        """(result dict or None, ms). A dead or timed-out worker is restarted once per call."""
        from asr_client import ASRWorkerExited, ASRWorkerTimeout

        if self.client is None:
            raise ASRWorkerExited(f"{self.cand.label} worker is not running")
        for attempt in (1, 2):
            t0 = time.perf_counter()
            try:
                result = self.client.transcribe(audio)
                return result, (time.perf_counter() - t0) * 1000
            except (ASRWorkerExited, ASRWorkerTimeout) as e:
                log.warning(f"{self.cand.label} worker failed ({e}); restarting")
                self.close()
                if attempt == 2:
                    raise
                self.restarts += 1
                self.load()

    def close(self):
        if self.client is not None:
            try:
                self.client.shutdown()
            except Exception as e:  # shutdown of a broken worker must not mask the real error
                log.warning(f"{self.cand.label} worker shutdown failed: {e}")
            self.client = None
