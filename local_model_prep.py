"""Getting the local translation model ready when LiveTranslate starts. This used to be the
launcher's console steps and now runs inside the app: Ollama with the fast settings, the
model downloaded (after asking), the local model entries pointed at Ollama's port, and
the free graphics memory compared with what the speech + translation pair needs.

No Qt: local_model_ui runs prepare() on a thread; questions and progress go through
callbacks."""
import logging
from dataclasses import dataclass

import httpx

import hardware_profile
import local_models
import ollama_server

log = logging.getLogger("LiveTranslate.LocalModel")


@dataclass
class PrepResult:
    port: int | None = None
    settings: dict | None = None   # new user settings when the local model entries changed
    low_vram: str | None = None    # warning: free graphics memory is short for the pair
    error: str | None = None       # the model cannot be used yet (shown to the user)


def measure_free_gb(settings: dict) -> float | None:
    """Free memory of the speech model's card plus what Ollama's loaded models hold (they
    are replaced by this session's model). Measure before the speech model loads: the
    pair's need includes it. None when there is no NVIDIA card / driver."""
    gpu = hardware_profile.detect_gpu(hardware_profile.gpu_index(settings))
    if not gpu or gpu.get("free_gb") is None:
        return None
    loaded = 0
    for port in (ollama_server.DEFAULT_PORT, ollama_server.ALT_PORT):
        if not ollama_server.port_open(port):
            continue
        try:
            loaded += ollama_server.Ollama(ollama_server.root_url(port)).loaded_vram_bytes()
        except (httpx.HTTPError, ValueError) as e:  # no usable model list: count nothing
            log.warning(f"Cannot read the models Ollama holds on port {port}: {e}")
    return gpu["free_gb"] + loaded / 2**30


def low_vram_message(free_gb, need: dict) -> str | None:
    if free_gb is None or free_gb >= need["free_need_gb"]:
        return None
    return (f"可用顯示記憶體只有 {free_gb:.1f} GB，目前的組合（{need['asr_label']}＋{need['model_label']}）"
            f"大約需要 {need['free_need_gb']:.1f} GB；不夠時翻譯會變得極慢甚至像卡住。"
            "建議關閉遊戲、Wallpaper Engine 或多餘的瀏覽器分頁，或在設定的「本機翻譯模型」改用標示【推薦】的模型。")


def prepare(settings: dict, model_entry: dict, log_dir, *, ask_download, on_progress=None,
            cancelled=None, say=None, free_gb_at_start=None) -> PrepResult:
    """ask_download(model, label, size_gb) -> bool is asked before a download;
    on_progress(done_bytes, total_bytes, status) / cancelled() as in Ollama.pull."""
    say = say or log.info
    model = str(model_entry.get("model", ""))
    try:
        port = ollama_server.ensure_server(model, log_dir, say=say)
    except RuntimeError as e:
        log.error(f"Ollama could not be started: {e}")
        return PrepResult(error=f"無法啟動 Ollama：{e}")
    server = ollama_server.Ollama(ollama_server.root_url(port))
    try:
        installed = server.installed()
    except (httpx.HTTPError, ValueError) as e:  # unknown list: nothing is removed from the settings
        log.warning(f"Cannot list the models of Ollama on port {port}: {e}")
        installed = None

    catalog = hardware_profile.translation_models()
    entry = next((m for m in catalog if m["model"] == model), None)
    label = entry["label"] if entry else model
    if installed is not None and model not in installed:
        if not ask_download(model, label, entry["size_gb"] if entry else None):
            return PrepResult(port=port, error=f"翻譯模型 {label} 還沒下載，翻譯暫時無法使用；"
                                                "可以在設定的「本機翻譯模型」下載。")
        say(f"下載 {label}")
        try:
            server.pull(model, on_progress=on_progress, cancelled=cancelled)
        except ollama_server.PullCancelled:
            log.info(f"Download of {model} cancelled")
            return PrepResult(port=port, error=f"已取消下載 {label}，翻譯暫時無法使用。")
        except ollama_server.PullError as e:
            log.error(f"Download of {model} failed: {e}")
            return PrepResult(port=port, error=f"{label} 下載失敗：{e}")
        installed = set(installed) | {model}

    try:
        new = local_models.sync_local_models(settings, f"{ollama_server.root_url(port)}/v1", model, installed)
    except ValueError as e:  # a model name Ollama cannot have
        log.error(f"Cannot point the local model entries at port {port}: {e}")
        return PrepResult(port=port, error=f"翻譯模型名稱無法使用：{model}")
    result = PrepResult(port=port, settings=new if new != settings else None)
    if free_gb_at_start is not None and catalog:
        need = hardware_profile.startup_check(new, model, catalog)
        result.low_vram = low_vram_message(free_gb_at_start, need)
    return result
