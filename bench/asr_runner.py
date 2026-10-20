"""Stage 1 ASR candidates: worker configs (built like main.py builds them), model
downloads, timed transcription through ASRClient, and the same subtitle filters the app
applies, so CER is measured on what would actually be shown.

Copied from main.py (tests/test_bench_asr.py compares them against main.py so they cannot
drift silently): the worker_config keys (LiveTranslateApp._switch_asr_engine), the
sentence splitter (_split_sentences) and the punctuation / noise / language filters
(_process_segment, _without_other_language)."""
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field

from bench import paths

log = logging.getLogger("LiveTranslate.Bench.ASR")

TURBO_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"


@dataclass(frozen=True)
class AsrCandidate:
    id: str
    label: str
    engine_type: str
    funasr_model: str | None = None
    whisper: str | None = None  # "turbo"


ASR_CANDIDATES = [
    AsrCandidate("sensevoice", "SenseVoice Small", "funasr", funasr_model="sensevoice-small"),
    AsrCandidate("mlt-nano", "Fun-ASR-MLT-Nano", "funasr", funasr_model="funasr-mlt-nano-2512"),
    AsrCandidate("anime-whisper", "Anime-Whisper", "anime-whisper"),
    AsrCandidate("whisper-turbo", "Whisper large-v3-turbo", "whisper", whisper="turbo"),
]
CANDIDATES_BY_ID = {c.id: c for c in ASR_CANDIDATES}

# Rough download sizes for the "download first?" question (bytes)
DOWNLOAD_BYTES = {"sensevoice": 900_000_000, "mlt-nano": 3_500_000_000,
                  "anime-whisper": 1_600_000_000, "whisper-turbo": 1_600_000_000}


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


def _engine_label(label):
    return label if label and label not in ("auto", "unknown") else None


def _heard_other(text, label, forced, hint_missing=False):
    import asr_language

    if forced == "auto":
        return None
    engine_label = _engine_label(label)
    if engine_label and engine_label != forced:
        return engine_label
    if hint_missing:
        return None
    heard = asr_language.script_language(text, forced)
    return heard if heard != forced else None


def without_other_language(text: str, label: str, forced: str, hint_missing: bool = False) -> tuple[str, list]:
    """main.py _without_other_language: (kept text, [(dropped sentence, heard language)])."""
    import asr_language

    if forced == "auto":
        return text, []
    engine_label = _engine_label(label)
    if (engine_label and engine_label != forced) or hint_missing or forced not in asr_language.SCRIPT_CHECKED_LANGUAGES:
        heard = _heard_other(text, label, forced, hint_missing)
        return ("", [(text, heard)]) if heard else (text, [])
    parts = [p.strip() for p in split_sentences(text, forced) if p.strip()]
    if len(parts) <= 1:
        parts = [text]
    kept, dropped = [], []
    for part in parts:
        heard = _heard_other(part, label, forced, hint_missing)
        (dropped.append((part, heard)) if heard else kept.append(part))
    if not dropped:
        return text, []
    return ("" if forced == "ja" else " ").join(kept), dropped


def displayed_text(text: str, label: str, seg_seconds: float, forced: str, hint_missing: bool = False) -> tuple[str, str | None]:
    """What the app would show for an ASR result: (text, reason it was cut or None)."""
    text = (text or "").strip()
    if not text or not any(c.isalnum() for c in text):
        return "", "empty"
    if seg_seconds >= 2.0 and sum(1 for c in text if c.isalnum()) <= 3:
        return "", "noise"
    kept, dropped = without_other_language(text, label, forced, hint_missing)
    if not kept:
        return "", f"language:{dropped[0][1]}"
    return kept, (f"language-partial:{dropped[0][1]}" if dropped else None)


def warm_up_filters(forced: str):
    """Build the sentence splitter once (yasbd takes ~0.5 s the first time) so that cost
    never lands on a timed sentence; the app also builds it once and keeps it."""
    displayed_text("今日はいい天気ですね。明日も晴れるといいな", forced, 1.0, forced)


def hint_missing_for(cand: AsrCandidate, forced: str) -> bool:
    """main.py _note_asr_language_support: Fun-ASR-Nano family without a prompt name."""
    if cand.engine_type != "funasr" or forced == "auto":
        return False
    import asr_language
    from model_manager import funasr_profile

    profile = funasr_profile(cand.funasr_model)
    return profile.get("family") == "funasr-nano" and \
        asr_language.prompt_language_name(profile["legacy_engine"], forced) is None


# ---------------------------------------------------------------- session


@dataclass
class AsrSession:
    """One loaded ASR worker. `client_factory(config)` builds the client (ASRClient in
    real runs, a fake in tests)."""
    cand: AsrCandidate
    config: dict
    client_factory: object = None
    client: object = None
    load_s: float = 0.0
    restarts: int = 0
    errors: list = field(default_factory=list)

    def _new_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.config)
        from asr_client import ASRClient

        return ASRClient(self.config, ready_timeout=300.0, request_timeout=60.0)

    def load(self):
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
        """(result dict or None, ms). A dead worker is restarted once per call."""
        from asr_client import ASRWorkerExited, ASRWorkerTimeout

        for attempt in (1, 2):
            t0 = time.perf_counter()
            try:
                result = self.client.transcribe(audio)
                return result, (time.perf_counter() - t0) * 1000
            except (ASRWorkerExited, ASRWorkerTimeout) as e:
                self.errors.append(f"{type(e).__name__}: {e}")
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
