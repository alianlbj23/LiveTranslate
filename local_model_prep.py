"""Getting a local translation model ready (at start, or when the settings' "use" button is
pressed). This used to be the launcher's console steps and now runs inside the app: Ollama
with the fast settings, the model downloaded (after asking), and the free graphics memory
compared with what the model needs. settings_after() then points the local model entries at
Ollama's port, applied on the GUI thread to the settings as they are by then.

No Qt: local_model_ui runs prepare() on a thread; questions and progress go through
callbacks."""
import logging
import time
from dataclasses import dataclass

import httpx

import hardware_profile
import local_models
import ollama_server
from i18n import t

log = logging.getLogger("LiveTranslate.LocalModel")


@dataclass
class PrepResult:
    port: int | None = None
    model: str = ""
    installed: set | None = None   # Ollama's models afterwards (None: Ollama could not tell)
    low_vram: str | None = None    # warning: free graphics memory is short for the model
    error: str | None = None       # the model cannot be used yet (shown to the user)

    @property
    def api_base(self) -> str | None:
        return f"{ollama_server.root_url(self.port)}/v1" if self.port else None


def measure_free_gb(settings: dict) -> float | None:
    """Free memory of the speech model's card plus what Ollama's loaded models hold (they
    are replaced by the model being prepared). None when there is no NVIDIA card / driver."""
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
    return t("lm_low_vram").format(free=free_gb, pair=f"{need['asr_label']}＋{need['model_label']}",
                                   need=need["free_need_gb"])


def _installed(server, port: int):
    """Ollama's model list; asked twice (a server that just started can be slow), None when
    it still cannot tell (then nothing is downloaded or removed)."""
    for attempt in (1, 2):
        try:
            return server.installed()
        except (httpx.HTTPError, ValueError) as e:
            log.warning(f"Cannot list the models of Ollama on port {port} (attempt {attempt}): {e}")
            if attempt == 1:
                time.sleep(1.0)
    return None


def prepare(settings: dict, model_entry: dict, log_dir, *, ask_download, on_progress=None,
            cancelled=None, say=None, free_gb_at_start=None, asr_loaded=False) -> PrepResult:
    """ask_download(model, label, size_gb) -> bool is asked before a download;
    on_progress(done_bytes, total_bytes, status) / cancelled() as in Ollama.pull.
    free_gb_at_start: free graphics memory (None skips the check); asr_loaded: the speech
    model already holds its share (a switch while running), so it is not counted again."""
    say = say or log.info
    model = str(model_entry.get("model", ""))
    if not local_models.MODEL_NAME_RE.match(model):
        log.error(f"Invalid local model name: {model!r}")
        return PrepResult(error=t("lm_err_bad_name").format(model=model))
    try:
        port = ollama_server.ensure_server(model, log_dir, say=say)
    except RuntimeError as e:
        log.error(f"Ollama could not be started: {e}")
        return PrepResult(error=t("lm_err_ollama_start").format(e=e))
    server = ollama_server.Ollama(ollama_server.root_url(port))
    installed = _installed(server, port)

    catalog = hardware_profile.translation_models()
    entry = next((m for m in catalog if m["model"] == model), None)
    label = entry["label"] if entry else model
    if installed is not None and model not in installed:
        if not ask_download(model, label, entry["size_gb"] if entry else None):
            return PrepResult(port=port, model=model, error=t("lm_err_not_downloaded").format(label=label))
        say(f"Downloading {model}")
        try:
            server.pull(model, on_progress=on_progress, cancelled=cancelled)
        except ollama_server.PullCancelled:
            log.info(f"Download of {model} cancelled")
            return PrepResult(port=port, model=model, error=t("lm_err_cancelled").format(label=label))
        except ollama_server.PullError as e:
            log.error(f"Download of {model} failed: {e}")
            return PrepResult(port=port, model=model, error=t("lm_err_download_failed").format(label=label, e=e))
        installed = set(installed) | {model}

    result = PrepResult(port=port, model=model, installed=installed)
    if free_gb_at_start is not None and catalog:
        need = hardware_profile.startup_check(settings, model, catalog)
        if asr_loaded:
            asr_gb = hardware_profile.current_asr(settings)["vram_gb"]
            need = dict(need, free_need_gb=need["free_need_gb"] - asr_gb)
        result.low_vram = low_vram_message(free_gb_at_start, need)
    return result


def _same_entry(a: dict, b: dict) -> bool:
    """The same model entry, ignoring the api_base the sync rewrites for local entries."""
    keys = ("name", "model", "api_key")
    return all(a.get(k) == b.get(k) for k in keys)


def settings_after(current: dict, result: PrepResult, activate: bool) -> dict | None:
    """`current` with the local entries pointed at result's Ollama (one per installed catalog
    model). activate: the prepared model becomes the active one (the user asked for it);
    otherwise the entry the user has active now stays active (a download can take minutes).
    None when nothing changes."""
    try:
        new = local_models.sync_local_models(current, result.api_base, result.model, result.installed)
    except ValueError as e:  # a model name Ollama cannot have (checked in prepare already)
        log.error(f"Cannot point the local model entries at {result.api_base}: {e}")
        return None
    models = current.get("models", [])
    idx = current.get("active_model", 0)
    if not activate and isinstance(idx, int) and 0 <= idx < len(models):
        was = models[idx]
        if not (local_models.is_local_ollama(was) and was.get("model") == result.model):
            keep = next((i for i, m in enumerate(new["models"]) if _same_entry(m, was)), None)
            if keep is not None:
                new["active_model"] = keep
    return None if new == current else new
