"""「本機翻譯模型」 group on the control panel's translation page: the local_models.json
catalog with its download state and graphics card advice (recommended / near the limit /
may not fit) and a button to use a model, downloading it first when needed. It replaces
the launcher's model list; the work itself is done by local_model_ui.LocalModelStarter
(the panel emits use_requested, main.py starts the preparation)."""
import logging
import threading

import httpx
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import MessageBox, PushButton, SettingCard, SettingCardGroup

import hardware_profile
import local_models
import ollama_server
from i18n import t

log = logging.getLogger("LiveTranslate.LocalModels")


def advice_for(settings: dict) -> dict:
    """hardware_profile advice for the speech model in `settings`, with each model's
    download size and note from the catalog."""
    catalog = hardware_profile.translation_models()
    gpu = hardware_profile.detect_gpu(hardware_profile.gpu_index(settings))
    advice = hardware_profile.launcher_advice(settings, catalog, gpu)
    by_model = {c["model"]: c for c in catalog}
    for m in advice["models"]:
        m["size_gb"] = by_model[m["model"]].get("size_gb")
        m["note"] = by_model[m["model"]].get("note") or ""
    return advice


def ollama_port(settings: dict) -> int:
    """Port of the local Ollama the model entries point at (the default one when none does)."""
    for m in settings.get("models", []):
        if local_models.is_local_ollama(m):
            base = local_models.ollama_root(str(m.get("api_base", "")))
            try:
                return int(base.rsplit(":", 1)[1])
            except (IndexError, ValueError):
                log.warning(f"Local model entry without a port: {m.get('api_base')!r}")
    return ollama_server.DEFAULT_PORT


def installed_models(settings: dict):
    """Model names Ollama has, or None when it is not running / does not answer."""
    port = ollama_port(settings)
    if not ollama_server.port_open(port):
        return None
    try:
        return ollama_server.Ollama(ollama_server.root_url(port)).installed()
    except (httpx.HTTPError, ValueError) as e:  # only means "unknown" here
        log.warning(f"Cannot list the models of Ollama on port {port}: {e}")
        return None


def short_gpu_name(gpu: dict) -> str:
    name = str(gpu.get("name") or "")
    for prefix in ("NVIDIA ", "GeForce "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    total = f"{gpu['total_gb']:.1f} GB"
    return f"{name}（{total}）" if name else total


def summary_text(advice: dict) -> str:
    gpu = advice.get("gpu")
    if not gpu:
        return t("lm_hw_unknown")
    rec = next((m for m in advice["models"] if m["model"] == advice.get("recommended")), None)
    card, asr = short_gpu_name(gpu), advice["asr"]["label"]
    if rec:
        return t("lm_hw_summary").format(gpu=card, asr=asr, rec=rec["label"])
    return t("lm_hw_none_fits").format(gpu=card, asr=asr)


def tag_for(model: dict, recommended) -> str:
    if model["model"] == recommended:
        return t("lm_tag_recommended")
    return {"tight": t("lm_tag_tight"), "too_big": t("lm_tag_too_big")}.get(model["level"], "")


def too_big_text(model: dict, advice: dict) -> str | None:
    """The confirmation for a model the card may not fit (None when it fits or is unknown)."""
    if model["level"] != "too_big" or not advice.get("gpu"):
        return None
    rec = next((m for m in advice["models"] if m["model"] == advice.get("recommended")), None)
    tip = t("lm_tip_recommended").format(label=rec["label"]) if rec else t("lm_tip_cloud")
    return t("lm_confirm_too_big_text").format(total=advice["gpu"]["total_gb"], label=model["label"],
                                               asr=advice["asr"]["label"], need=model["total_need_gb"] or 0, tip=tip)


class _ModelCard(SettingCard):
    def __init__(self, model: str, parent=None):
        super().__init__(FIF.ROBOT, model, "", parent)
        self.model = model
        self.button = PushButton(t("lm_btn_use"), self)
        self.hBoxLayout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)


class LocalModelsGroup(SettingCardGroup):
    """`get_settings()` returns the current user settings. `confirm(title, text)` asks
    before using a model the card may not fit (a Fluent MessageBox by default)."""

    use_requested = pyqtSignal(str)          # catalog model tag
    _installed_ready = pyqtSignal(object)    # set of names, or None (from a thread)

    def __init__(self, get_settings, parent=None, advice=None, installed=None, confirm=None):
        super().__init__(t("group_local_models"), parent)
        self._get_settings = get_settings
        self._advice_fn = advice or advice_for
        self._installed_fn = installed or installed_models
        self._confirm = confirm or self._confirm_box
        self._installed = None
        self._advice = {"gpu": None, "asr": {"label": ""}, "models": [], "recommended": None}
        self._summary = SettingCard(FIF.SPEED_HIGH, t("lm_hw_title"), "", self)
        self._refresh_btn = PushButton(t("lm_btn_refresh"), self._summary)
        self._refresh_btn.clicked.connect(self.refresh)
        self._summary.hBoxLayout.addWidget(self._refresh_btn, 0, Qt.AlignmentFlag.AlignRight)
        self._summary.hBoxLayout.addSpacing(16)
        self.addSettingCard(self._summary)
        self._cards = {}
        self._asked_ollama = False
        self._installed_ready.connect(self._on_installed)
        self._update_advice()
        self._render()

    # -- data --

    def showEvent(self, event):
        super().showEvent(event)
        if not self._asked_ollama:  # not while building the window: Ollama is asked when shown
            self.refresh()

    def _update_advice(self):
        try:
            self._advice = self._advice_fn(self._get_settings())
        except Exception as e:  # a broken catalog or driver must not break the settings window
            log.warning(f"Cannot judge the local translation models: {e}")

    def refresh(self):
        """Graphics card advice now (NVML is fast), Ollama's model list on a thread."""
        self._asked_ollama = True
        self._update_advice()
        self._render()
        settings = self._get_settings()
        threading.Thread(target=lambda: self._installed_ready.emit(self._installed_fn(settings)),
                         name="LocalModelsList", daemon=True).start()

    def _on_installed(self, installed):
        self._installed = installed
        self._render()

    def _active_model(self):
        s = self._get_settings()
        models = s.get("models", [])
        idx = s.get("active_model", 0)
        if 0 <= idx < len(models) and local_models.is_local_ollama(models[idx]):
            return models[idx].get("model")
        return None

    # -- view --

    def _render(self):
        self._summary.setContent(summary_text(self._advice))
        active = self._active_model()
        for m in self._advice["models"]:
            card = self._cards.get(m["model"])
            if card is None:
                card = _ModelCard(m["model"], self)
                card.button.clicked.connect(lambda _c=False, model=m["model"]: self._on_use(model))
                self._cards[m["model"]] = card
                self.addSettingCard(card)
            card.setTitle(f"{m['label']} {tag_for(m, self._advice.get('recommended'))}".strip())
            if self._installed is None:
                state, have = t("lm_state_unknown"), True
            else:
                have = m["model"] in self._installed
                state = t("lm_state_installed") if have else t("lm_state_missing")
            detail = t("lm_row_detail").format(size=m.get("size_gb") or 0, vram=m.get("vram_gb") or 0, state=state)
            card.setContent(f"{m['note']}｜{detail}" if m.get("note") else detail)
            if m["model"] == active:
                card.button.setText(t("lm_btn_active"))
                card.button.setEnabled(False)
            else:
                card.button.setText(t("lm_btn_use") if have else t("lm_btn_download_use"))
                card.button.setEnabled(True)

    def _on_use(self, model: str):
        m = next((x for x in self._advice["models"] if x["model"] == model), None)
        text = too_big_text(m, self._advice) if m else None
        if text and not self._confirm(t("lm_confirm_too_big_title"), text):
            return
        self.use_requested.emit(model)

    def _confirm_box(self, title: str, text: str) -> bool:
        host = self.window() if isinstance(self.window(), QWidget) else None
        return bool(MessageBox(title, text, host).exec())

    def card(self, model: str) -> _ModelCard | None:
        return self._cards.get(model)
