"""Which speech + translation model pair this computer can run, and which one to recommend.

The speech model (ASR worker) and the translation model (Ollama) sit on the graphics card
together. When they do not fit, Ollama silently falls back to the CPU and every subtitle
takes 10 s or more, so the launcher marks pairs that will not fit and the installer picks
defaults the card can hold.

GPU memory per model comes from the best-combination benchmark (2026-10-20, RTX 5070):
peak while running minus the idle card. Translation models carry theirs in
local_models.json (vram_gb, measured with the launcher's Ollama settings); speech models
are in ASR_VRAM_GB below (measured where marked, the rest estimated from model size).
Judgments use the card's total memory, which does not change with other programs; the
free memory at start is checked separately (startup_check).

Standard library + system_stats / model_manager / local_models only: the launcher calls
this through launcher/hardware_advice.py before anything heavy is installed."""
import logging
import re
from pathlib import Path

log = logging.getLogger("LiveTranslate.Hardware")

RESERVE_GB = 1.5        # Windows desktop, browser and other programs (benchmark idle card: 1.7–2.1 GB)
TIGHT_GB = 1.0          # less spare memory than this: works, but other programs can push it over
FREE_MARGIN_GB = 0.5    # at start, free memory must cover the pair plus this

# Speech models: (label, GPU GB). Keys: "whisper:<size>", "funasr:<model key>", engine name
ASR_VRAM_GB = {
    "whisper:turbo": 2.5,                 # measured
    "whisper:large-v3": 3.6,
    "whisper:medium": 2.0,
    "whisper:small": 1.0,
    "whisper:base": 0.6,
    "whisper:tiny": 0.5,
    "funasr:sensevoice-small": 0.7,       # measured
    "funasr:funasr-nano-2512": 3.7,       # same size as MLT-Nano
    "funasr:funasr-mlt-nano-2512": 3.7,   # measured
    "anime-whisper": 1.8,                 # measured
    "remote-whisper": 0.0,                # runs on another computer
}
ASR_LABELS = {
    "whisper:turbo": "Whisper large-v3-turbo", "whisper:large-v3": "Whisper large-v3",
    "whisper:medium": "Whisper medium", "whisper:small": "Whisper small", "whisper:base": "Whisper base",
    "whisper:tiny": "Whisper tiny", "funasr:sensevoice-small": "SenseVoice Small",
    "funasr:funasr-nano-2512": "Fun-ASR-Nano", "funasr:funasr-mlt-nano-2512": "Fun-ASR-MLT-Nano",
    "anime-whisper": "Anime-Whisper", "remote-whisper": "Remote-Whisper",  # model_manager.ASR_DISPLAY_NAMES
}
# The speech models the installer can set up, best first: (choice, ASR_VRAM_GB key)
INSTALL_ASR = (("whisper-turbo", "whisper:turbo"), ("sensevoice", "funasr:sensevoice-small"))


# ---------------------------------------------------------------- inputs

def translation_models(root=None) -> list:
    """Translation models of local_models.json (in `root`, default: next to this file)."""
    return _catalog(root)["models"]


def default_model(root=None) -> str | None:
    """The catalog's default translation model."""
    return _catalog(root)["default"]


def _catalog(root):
    import local_models

    return local_models.load_catalog(Path(root) / "local_models.json" if root else None)


def _whisper_key(size) -> str:
    from model_manager import _WHISPER_SIZES

    text = str(size or "medium")
    if "turbo" in text.lower():
        return "whisper:turbo"
    if text in _WHISPER_SIZES:
        return f"whisper:{text}"
    # another local folder: assume the largest Whisper so the advice errs on the safe side
    return "whisper:large-v3"


def current_asr(settings: dict) -> dict:
    """{"key", "label", "vram_gb"} of the speech model `settings` (user_settings.json)
    select; vram_gb is 0 when it runs on the CPU or on another computer."""
    from model_manager import normalize_asr_engine_selection

    engine, funasr_model = normalize_asr_engine_selection(settings.get("asr_engine") or "funasr",
                                                          settings.get("funasr_model"))
    if engine == "funasr":
        key = f"funasr:{funasr_model}"
    elif engine == "whisper":
        key = _whisper_key(settings.get("whisper_model_size"))
    elif engine in ("anime-whisper", "remote-whisper"):
        key = engine
    else:
        log.warning(f"Unknown ASR engine {engine!r}; judging it like Whisper large-v3")
        key = "whisper:large-v3"
    label = ASR_LABELS[key]
    vram = ASR_VRAM_GB[key]
    if key != "remote-whisper" and str(settings.get("asr_device") or "cuda").startswith("cpu"):
        label, vram = f"{label}（CPU）", 0.0
    return {"key": key, "label": label, "vram_gb": vram}


def gpu_index(settings: dict) -> int:
    """GPU number of the ASR device ("cuda:1 (RTX ...)" -> 1); 0 when none is named."""
    device = str(settings.get("asr_device") or "cuda")
    m = re.match(r"cuda:(\d+)", device)
    if m:
        return int(m.group(1))
    if device.startswith("cuda:"):
        log.warning(f"Unreadable ASR device {device!r}; using GPU 0")
    return 0


def detect_gpu(index: int = 0):
    """{"name", "total_gb", "free_gb"} of NVIDIA card `index`, or None (no driver / card)."""
    import system_stats

    gpu = system_stats.NvmlGpu(index)
    stats = gpu.read()
    if stats is None:
        return None
    return {"name": gpu.name() or "", "total_gb": stats.total_mb / 1024,
            "free_gb": (stats.total_mb - stats.used_mb) / 1024}


# ---------------------------------------------------------------- judging

def level_for(total_gb, models_gb) -> str:
    """"ok", "tight" (fits with less than TIGHT_GB to spare) or "too_big" for models
    needing `models_gb` on a card of `total_gb`; "unknown" when either is not known."""
    if total_gb is None or models_gb is None:
        return "unknown"
    spare = total_gb - (models_gb + RESERVE_GB)
    if spare < 0:
        return "too_big"
    return "tight" if spare < TIGHT_GB else "ok"


def _total_need(asr_gb, model) -> float | None:
    return None if model["vram_gb"] is None else asr_gb + model["vram_gb"] + RESERVE_GB


def _by_rank(models):
    return sorted(models, key=lambda m: (m["rank"] is None, m["rank"] or 0))


def launcher_advice(settings: dict, catalog: list, gpu) -> dict:
    """What the launcher window shows: each catalog model's level next to the speech
    model in use, and the best-ranked model that fits ("ok", else "tight"; None when
    nothing fits or the card is unknown)."""
    asr = current_asr(settings)
    total = gpu["total_gb"] if gpu else None
    models = []
    for m in catalog:
        level = level_for(total, None if m["vram_gb"] is None else asr["vram_gb"] + m["vram_gb"])
        models.append({"model": m["model"], "label": m["label"], "vram_gb": m["vram_gb"], "rank": m["rank"],
                       "total_need_gb": _total_need(asr["vram_gb"], m), "level": level})
    recommended = None
    for wanted in ("ok", "tight"):
        fits = [m for m in _by_rank(models) if m["level"] == wanted]
        if fits:
            recommended = fits[0]["model"]
            break
    return {"gpu": gpu, "asr": asr, "models": models, "recommended": recommended}


def install_choice(total_gb, catalog: list, default: str | None = None) -> dict:
    """Speech + translation defaults for a fresh install on a card of `total_gb`: the
    first pair that fits comfortably (speech quality first: Whisper turbo is far more
    accurate than SenseVoice, the translation models differ less), else the first that
    fits at all, else the lightest pair with a message to use cloud services instead.
    Without a known card: Whisper turbo + `default` (the catalog default)."""
    ranked = [m for m in _by_rank(catalog) if m["vram_gb"] is not None]
    if not ranked:
        raise ValueError("local_models.json has no model with vram_gb")

    def pick(choice, model, level, need, message):
        return {"asr_choice": choice, "model": model["model"], "label": model["label"], "size_gb": model["size_gb"],
                "level": level, "total_need_gb": need, "message": message}

    if total_gb is None:
        fallback = next((m for m in catalog if m["model"] == default), catalog[0])
        return pick(INSTALL_ASR[0][0], fallback, "unknown", None,
                    f"讀不到顯示卡的記憶體大小，使用預設的 {_asr_name(INSTALL_ASR[0][0])}＋{fallback['label']}。")
    pairs = [(choice, ASR_VRAM_GB[key], m) for choice, key in INSTALL_ASR for m in ranked]
    for wanted in ("ok", "tight"):
        for choice, asr_gb, m in pairs:
            if level_for(total_gb, asr_gb + m["vram_gb"]) == wanted:
                note = "" if wanted == "ok" else "（接近顯示卡上限，執行時請少開其他吃顯示卡的程式）"
                return pick(choice, m, wanted, _total_need(asr_gb, m),
                            f"依顯示卡 {total_gb:.1f} GB 選用 {_asr_name(choice)}＋{m['label']}{note}。")
    choice, asr_gb = INSTALL_ASR[-1][0], ASR_VRAM_GB[INSTALL_ASR[-1][1]]
    lightest = min(ranked, key=lambda m: m["vram_gb"])
    return pick(choice, lightest, "too_big", _total_need(asr_gb, lightest),
                f"顯示卡只有 {total_gb:.1f} GB，放不下任何一組本機的辨識＋翻譯模型。先設成最輕的 "
                f"{_asr_name(choice)}＋{lightest['label']}，建議改用雲端翻譯 API 或遠端辨識"
                "（見 INSTALL_GUIDE.md、REMOTE_ASR.md）。")


def _asr_name(choice: str) -> str:
    return ASR_LABELS[dict(INSTALL_ASR)[choice]]


def startup_check(settings: dict, model: str, catalog: list) -> dict:
    """Free GPU memory the pair in use needs at start (free_need_gb), on GPU gpu_index.
    A model outside the catalog is counted like the largest catalog model."""
    asr = current_asr(settings)
    entry = next((m for m in catalog if m["model"] == model), None)
    sizes = [m["vram_gb"] for m in catalog if m["vram_gb"] is not None]
    tl_gb = entry["vram_gb"] if entry and entry["vram_gb"] is not None else max(sizes)
    return {"free_need_gb": asr["vram_gb"] + tl_gb + FREE_MARGIN_GB, "asr_label": asr["label"],
            "model_label": entry["label"] if entry else model, "gpu_index": gpu_index(settings)}
