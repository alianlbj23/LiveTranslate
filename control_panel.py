import json
import logging
import os
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFontComboBox,
    QHBoxLayout,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    InfoBar,
    MessageBox,
    BodyLabel,
    CaptionLabel,
    CardWidget,
    ColorPickerButton,
    ComboBox,
    DoubleSpinBox,
    FluentIcon as FIF,
    FluentWindow,
    LineEdit,
    ListWidget,
    NavigationItemPosition,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    SettingCard,
    SettingCardGroup,
    Slider,
    SpinBox,
    StrongBodyLabel,
    SwitchButton,
    TextBrowser,
    TextEdit,
    Theme,
    TitleLabel,
    qconfig,
    setTheme,
)

from benchmark import run_benchmark
from dialogs import (
    ModelEditDialog,
    available_screen_height,
)
from model_manager import (
    DEFAULT_FUNASR_MODEL,
    MODELS_DIR,
    _WHISPER_SIZES,
    dir_size,
    funasr_model_options,
    funasr_supports_padding,
    format_size,
    get_cache_entries,
    list_local_faster_whisper_models,
    migrate_funasr_settings,
    normalize_funasr_model_key,
    resolve_custom_whisper_model,
)
from i18n import t, LANGUAGES
from local_model_panel import LocalModelsGroup
from subtitle_settings import SubtitleSettingsWidget

log = logging.getLogger("LiveTranslate.Panel")

SETTINGS_FILE = Path(__file__).parent / "user_settings.json"

UI_FONT_FAMILY = "Microsoft JhengHei UI"
MONO_FONT_FAMILY = "Consolas"

# Semantic text colours as (light theme, dark theme) pairs so both themes stay readable.
# Expressed as QColor channels on purpose: no hand-written stylesheet hex literals.
_TEXT_OK = (QColor(16, 124, 16), QColor(108, 203, 95))
_TEXT_WARN = (QColor(197, 15, 31), QColor(255, 153, 164))
_TEXT_MUTED = (QColor(96, 96, 96), QColor(160, 160, 160))


def _load_saved_settings() -> dict | None:
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            migrate_funasr_settings(data)
            log.info(f"Loaded saved settings from {SETTINGS_FILE}")
            return data
    except Exception as e:
        log.warning(f"Failed to load settings: {e}")
    return None


def _save_settings(settings: dict):
    try:
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(SETTINGS_FILE)
        log.info(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        log.warning(f"Failed to save settings: {e}")


# ── Fluent building blocks ──


class _Page(ScrollArea):
    """One navigation page: a transparent scroll area holding a vertical stack of cards."""

    def __init__(self, object_name: str, parent=None):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.view = QWidget(self)
        self.view.setObjectName(f"{object_name}_view")
        self.vbox = QVBoxLayout(self.view)
        self.vbox.setContentsMargins(24, 16, 24, 24)
        self.vbox.setSpacing(16)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.enableTransparentBackground()

    def add_group(self, group: SettingCardGroup):
        self.vbox.addWidget(group)

    def finish(self):
        self.vbox.addStretch(1)


class _WidgetCard(SettingCard):
    """SettingCard whose right side hosts an arbitrary control (replaces QLabel + control rows)."""

    def __init__(self, icon, title, widget: QWidget, content: str = None, parent=None, stretch: bool = False):
        super().__init__(icon, title, content, parent)
        self.widget = widget
        if stretch:
            widget.setMinimumWidth(220)
        self.hBoxLayout.addWidget(widget, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)


class _BlockCard(CardWidget):
    """Card for tall content (lists, editors) with a bold title on top."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.vbox = QVBoxLayout(self)
        self.vbox.setContentsMargins(16, 12, 16, 12)
        self.vbox.setSpacing(8)
        self.title_label = StrongBodyLabel(title, self)
        self.vbox.addWidget(self.title_label)


DEFAULT_MIN_SPEECH_DENSITY = 0.25


def _prompt_preset_index(prompt) -> int:
    """Index in the prompt preset combo (daily, esports, anime, webid, custom)."""
    from translator import DEFAULT_PROMPT, PROMPT_PRESETS

    text = (prompt or "").strip()
    if text == DEFAULT_PROMPT.strip():
        return 0
    for i, key in enumerate(["daily", "esports", "anime", "webid"]):
        if text == PROMPT_PRESETS[key].strip():
            return i
    return 4  # custom


def _density_percent(value) -> int:
    """vad_min_speech_density from user_settings.json (0-0.9) as a whole percent;
    anything else falls back to the 25 % default."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 0.9:
        return int(round(value * 100))
    return int(round(DEFAULT_MIN_SPEECH_DENSITY * 100))


def _set_text_color(label, pair):
    label.setTextColor(pair[0], pair[1])


class ControlPanel(FluentWindow):
    """Settings and monitoring window (Win11 Fluent design, left navigation)."""

    settings_changed = pyqtSignal(dict)
    model_changed = pyqtSignal(dict)
    local_model_requested = pyqtSignal(str)  # catalog model to use (downloaded first when needed)
    update_check_requested = pyqtSignal()
    update_apply_requested = pyqtSignal()
    models_list_changed = pyqtSignal(list, int)
    subtitle_settings_changed = pyqtSignal(dict)
    _bench_result = pyqtSignal(str)
    _cache_result = pyqtSignal(list)
    reset_positions = pyqtSignal()
    # Home page: big start / stop button (main.py wires these to the pipeline)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    # update_home_status() may be called from the ASR / translation threads; the
    # widget updates must run on the GUI thread, so they travel through this signal
    _home_status_signal = pyqtSignal(object)

    def __init__(self, config, saved_settings=None):
        # Dark theme + UI font for every Fluent widget created from here on.
        # save=False: never write qfluentwidgets' own config.json into the project folder.
        setTheme(Theme.DARK, save=False)
        qconfig.set(qconfig.fontFamilies, [UI_FONT_FAMILY, "Microsoft JhengHei", "Segoe UI"], save=False)
        super().__init__()
        self._config = config
        self.setWindowTitle(t("window_control_panel"))
        self.setMinimumSize(760, 520)
        self.resize(940, min(720, available_screen_height(self)))
        self.setMicaEffectEnabled(True)
        self.navigationInterface.setExpandWidth(200)

        saved = migrate_funasr_settings(saved_settings) or _load_saved_settings()
        if saved:
            self._current_settings = saved
        else:
            tc = config["translation"]
            self._current_settings = {
                "vad_mode": "silero",
                "vad_threshold": config["asr"]["vad_threshold"],
                "energy_threshold": 0.02,
                "min_speech_duration": config["asr"]["min_speech_duration"],
                "max_speech_duration": config["asr"]["max_speech_duration"],
                "silence_mode": "auto",
                "silence_duration": 0.8,
                "asr_language": config["asr"].get("language", "auto"),
                "asr_engine": "funasr",
                "funasr_model": config["asr"].get(
                    "funasr_model", DEFAULT_FUNASR_MODEL
                ),
                "asr_device": "cuda",
                "sensevoice_pad_seconds": config["asr"].get(
                    "sensevoice_pad_seconds", 0.5
                ),
                "whisper_pad_seconds": config["asr"].get(
                    "whisper_pad_seconds", 0.5
                ),
                "models": [
                    {
                        "name": f"{tc['model']}",
                        "api_base": tc["api_base"],
                        "api_key": tc["api_key"],
                        "model": tc["model"],
                    }
                ],
                "active_model": 0,
                "hub": "ms",
            }

        if "models" not in self._current_settings:
            tc = config["translation"]
            self._current_settings["models"] = [
                {
                    "name": f"{tc['model']}",
                    "api_base": tc["api_base"],
                    "api_key": tc["api_key"],
                    "model": tc["model"],
                }
            ]
            self._current_settings["active_model"] = 0

        self._current_settings.setdefault(
            "funasr_model",
            config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL),
        )
        self._current_settings["funasr_model"] = normalize_funasr_model_key(
            self._current_settings.get("funasr_model")
        )
        self._current_settings.setdefault(
            "sensevoice_pad_seconds",
            config["asr"].get("sensevoice_pad_seconds", 0.5),
        )
        self._current_settings.setdefault(
            "whisper_pad_seconds",
            config["asr"].get("whisper_pad_seconds", 0.5),
        )

        self._home_running = False

        # Pages (order = navigation order). The changelog sits at the bottom.
        self.home_interface = self._create_home_page()
        self.asr_interface = self._create_vad_tab()
        self.translation_interface = self._create_translation_tab()
        self.appearance_interface = self._create_appearance_page()
        self.benchmark_interface = self._create_benchmark_tab()
        self.cache_interface = self._create_cache_tab()
        self.changelog_interface = self._create_changelog_tab()

        self.addSubInterface(self.home_interface, FIF.HOME, t("nav_home"))
        self.addSubInterface(self.asr_interface, FIF.MICROPHONE, t("tab_vad_asr"))
        self.addSubInterface(self.translation_interface, FIF.LANGUAGE, t("tab_translation"))
        self.addSubInterface(self.appearance_interface, FIF.PALETTE, t("nav_appearance"))
        self.addSubInterface(self.benchmark_interface, FIF.SPEED_HIGH, t("tab_benchmark"))
        self.addSubInterface(self.cache_interface, FIF.CLOUD, t("nav_models_cache"))
        self.addSubInterface(
            self.changelog_interface, FIF.HISTORY, t("tab_changelog"), NavigationItemPosition.BOTTOM
        )
        self._pages = {
            "home": self.home_interface,
            "asr": self.asr_interface,
            "translation": self.translation_interface,
            "appearance": self.appearance_interface,
            "subtitle": self.appearance_interface,
            "style": self.appearance_interface,
            "benchmark": self.benchmark_interface,
            "cache": self.cache_interface,
            "changelog": self.changelog_interface,
        }
        self.stackedWidget.currentChanged.connect(self._on_page_changed)
        self.switchTo(self.home_interface)

        self._bench_result.connect(self._on_bench_result)
        self._cache_result.connect(self._on_cache_result)
        self._home_status_signal.connect(self._apply_home_status)

        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._do_auto_save)

        self._apply_ui_font()
        self._refresh_home_from_settings()

    # ── window helpers ──

    def _apply_ui_font(self):
        """Fluent widgets already use qconfig.fontFamilies; this covers plain Qt widgets
        (QFontComboBox, list items) so the whole window uses Microsoft JhengHei UI."""
        for w in [self] + self.findChildren(QWidget):
            f = w.font()
            if f.family() == MONO_FONT_FAMILY:
                continue
            f.setFamilies([UI_FONT_FAMILY])
            f.setFamily(UI_FONT_FAMILY)
            w.setFont(f)

    def _fit_height(self):
        """Kept for callers; the Fluent window has fixed navigation so no auto-fit is needed."""
        return None

    def show_page(self, name: str):
        page = self._pages.get(name)
        if page is None:
            log.warning(f"show_page: unknown page '{name}'")
            return
        self.switchTo(page)

    def _on_page_changed(self, _index):
        if self.stackedWidget.currentWidget() is self.cache_interface:
            self._refresh_cache()

    # ── Home page ──

    def _create_home_page(self):
        page = _Page("home_interface")

        title = TitleLabel("LiveTranslate", page.view)
        subtitle = CaptionLabel(t("home_hint"), page.view)
        _set_text_color(subtitle, _TEXT_MUTED)
        page.vbox.addWidget(title)
        page.vbox.addWidget(subtitle)

        control = _BlockCard(t("nav_home"), page.view)
        row = QHBoxLayout()
        self._home_toggle_btn = PrimaryPushButton(t("home_start"), control)
        self._home_toggle_btn.setIcon(FIF.PLAY)
        self._home_toggle_btn.setFixedSize(220, 52)
        self._home_toggle_btn.clicked.connect(self._on_home_toggle)
        row.addWidget(self._home_toggle_btn)
        self._home_status_label = StrongBodyLabel(t("home_status_paused"), control)
        _set_text_color(self._home_status_label, _TEXT_MUTED)
        row.addSpacing(16)
        row.addWidget(self._home_status_label, 1)
        control.vbox.addLayout(row)
        page.vbox.addWidget(control)

        info = SettingCardGroup(t("tab_vad_asr") + " / " + t("tab_translation"), page.view)
        self._home_engine_card = SettingCard(FIF.MICROPHONE, t("home_asr_engine"), t("home_na"), info)
        self._home_model_card = SettingCard(FIF.ROBOT, t("home_translation_model"), t("home_na"), info)
        self._home_asr_card = SettingCard(FIF.SPEED_HIGH, t("home_asr_latency"), t("home_na"), info)
        self._home_tl_card = SettingCard(FIF.SPEED_MEDIUM, t("home_tl_latency"), t("home_na"), info)
        for card in (self._home_engine_card, self._home_model_card, self._home_asr_card, self._home_tl_card):
            info.addSettingCard(card)
        page.add_group(info)

        maint = SettingCardGroup(t("group_maintenance"), page.view)
        self._env_check_btn = PushButton(t("btn_env_check"))
        self._env_check_btn.clicked.connect(self._on_env_check_clicked)
        maint.addSettingCard(_WidgetCard(
            FIF.DEVELOPER_TOOLS, t("label_env_check"), self._env_check_btn, content=t("desc_env_check"),
        ))
        update_box = QWidget()
        update_row = QHBoxLayout(update_box)
        update_row.setContentsMargins(0, 0, 0, 0)
        update_row.setSpacing(8)
        self._update_check_btn = PushButton(t("btn_update_check"))
        self._update_check_btn.clicked.connect(self.update_check_requested.emit)
        update_row.addWidget(self._update_check_btn)
        self._update_apply_btn = PrimaryPushButton(t("btn_update_apply"))
        self._update_apply_btn.clicked.connect(self.update_apply_requested.emit)
        self._update_apply_btn.setVisible(False)
        update_row.addWidget(self._update_apply_btn)
        self._update_card = _WidgetCard(FIF.SYNC, t("label_update"), update_box, content=t("update_not_checked"))
        maint.addSettingCard(self._update_card)
        self._factory_reset_btn = PushButton(t("btn_factory_reset"))
        self._factory_reset_btn.clicked.connect(self._on_factory_reset_clicked)
        maint.addSettingCard(_WidgetCard(
            FIF.HISTORY, t("label_factory_reset"), self._factory_reset_btn, content=t("desc_factory_reset"),
        ))
        page.add_group(maint)
        page.finish()
        return page

    def set_update_status(self, text: str, can_update: bool, busy: bool):
        self._update_card.setContent(text)
        self._update_apply_btn.setVisible(can_update)
        self._update_apply_btn.setEnabled(not busy)
        self._update_check_btn.setEnabled(not busy)

    def confirm(self, title: str, text: str) -> bool:
        return bool(MessageBox(title, text, self).exec())

    def _on_env_check_clicked(self):
        from env_check_ui import EnvCheckDialog

        EnvCheckDialog(self.get_active_model(), parent=self).exec()

    def _on_home_toggle(self):
        if self._home_running:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    @staticmethod
    def _fmt_ms(value) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return t("home_na")
        if v != v or v < 0:  # NaN or negative
            return t("home_na")
        return f"{v:.0f} ms"

    def update_home_status(self, running=None, asr_engine=None, model_name=None, asr_ms=None, tl_ms=None):
        """Feed live state into the home page (any argument may be omitted).
        Thread-safe: callers on worker threads are marshalled to the GUI thread."""
        self._home_status_signal.emit(
            {"running": running, "asr_engine": asr_engine, "model_name": model_name, "asr_ms": asr_ms, "tl_ms": tl_ms}
        )

    def _apply_home_status(self, state: dict):
        running = state.get("running")
        asr_engine = state.get("asr_engine")
        model_name = state.get("model_name")
        asr_ms = state.get("asr_ms")
        tl_ms = state.get("tl_ms")
        if running is not None:
            self._home_running = bool(running)
            self._home_toggle_btn.setText(t("home_stop") if self._home_running else t("home_start"))
            self._home_toggle_btn.setIcon(FIF.PAUSE if self._home_running else FIF.PLAY)
            self._home_status_label.setText(
                t("home_status_running") if self._home_running else t("home_status_paused")
            )
            _set_text_color(self._home_status_label, _TEXT_OK if self._home_running else _TEXT_MUTED)
        if asr_engine is not None:
            self._home_engine_card.setContent(str(asr_engine) or t("home_na"))
        if model_name is not None:
            self._home_model_card.setContent(str(model_name) or t("home_na"))
        if asr_ms is not None:
            self._home_asr_card.setContent(self._fmt_ms(asr_ms))
        if tl_ms is not None:
            self._home_tl_card.setContent(self._fmt_ms(tl_ms))

    def _refresh_home_from_settings(self):
        engine = self._asr_engine.currentText() if hasattr(self, "_asr_engine") else None
        active = self.get_active_model()
        model_name = f"{active.get('name', '')} ({active.get('model', '')})" if active else None
        self.update_home_status(asr_engine=engine, model_name=model_name)

    # ── VAD / ASR page ──

    def _create_vad_tab(self):
        page = _Page("asr_interface")
        s = self._current_settings

        asr_group = SettingCardGroup(t("group_asr_engine"), page.view)

        self._asr_engine = ComboBox()
        self._asr_engine.addItems(
            [
                f"[{t('asr_accurate')}] Whisper (faster-whisper)",
                f"[{t('asr_fast')}] FunASR",
                "Anime-Whisper (ja, anime/galgame)",
                "Remote Whisper (remote GPU server)",
            ]
        )
        engine_map_idx = {
            "whisper": 0,
            "funasr": 1,
            "anime-whisper": 2,
            "remote-whisper": 3,
        }
        engine_idx = engine_map_idx.get(s.get("asr_engine"), 0)
        self._asr_engine.setCurrentIndex(engine_idx)
        self._asr_engine.currentIndexChanged.connect(self._auto_save)
        asr_group.addSettingCard(_WidgetCard(FIF.MICROPHONE, t("label_engine"), self._asr_engine, stretch=True))

        self._asr_lang = ComboBox()
        for code, native in LANGUAGES:
            label = t("asr_lang_auto") if code == "auto" else native
            self._asr_lang.addItem(f"{code} - {label}", userData=code)
        lang = s.get("asr_language", self._config["asr"].get("language", "auto"))
        idx = self._asr_lang.findData(lang)
        if idx >= 0:
            self._asr_lang.setCurrentIndex(idx)
        self._asr_lang.currentIndexChanged.connect(self._auto_save)
        asr_group.addSettingCard(_WidgetCard(FIF.LANGUAGE, t("label_language_hint"), self._asr_lang, stretch=True))

        self._asr_device = ComboBox()
        devices = ["cuda", "cpu"]
        try:
            import torch

            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.insert(i, f"cuda:{i} ({name})")
            if torch.cuda.device_count() > 0:
                devices = [d for d in devices if d != "cuda"]
        except Exception as e:
            log.debug(f"torch device enumeration unavailable: {e}")
        self._asr_device.addItems(devices)
        saved_dev = s.get("asr_device", self._config["asr"].get("device", "cuda"))
        for i in range(self._asr_device.count()):
            if self._asr_device.itemText(i).startswith(saved_dev):
                self._asr_device.setCurrentIndex(i)
                break
        self._asr_device.currentIndexChanged.connect(self._auto_save)
        asr_group.addSettingCard(_WidgetCard(FIF.DEVELOPER_TOOLS, t("label_device"), self._asr_device, stretch=True))

        self._funasr_model_combo = ComboBox()
        for key, display_name in funasr_model_options():
            self._funasr_model_combo.addItem(display_name, userData=key)
        saved_funasr_model = normalize_funasr_model_key(
            s.get("funasr_model", DEFAULT_FUNASR_MODEL)
        )
        funasr_idx = self._funasr_model_combo.findData(saved_funasr_model)
        if funasr_idx >= 0:
            self._funasr_model_combo.setCurrentIndex(funasr_idx)
        self._funasr_model_combo.currentIndexChanged.connect(
            self._on_funasr_model_changed
        )
        self._funasr_model_card = _WidgetCard(FIF.ROBOT, t("label_funasr_model"), self._funasr_model_combo, stretch=True)
        asr_group.addSettingCard(self._funasr_model_card)

        self._whisper_pad_seconds = DoubleSpinBox()
        self._whisper_pad_seconds.setRange(0.0, 5.0)
        self._whisper_pad_seconds.setDecimals(2)
        self._whisper_pad_seconds.setSingleStep(0.1)
        try:
            whisper_pad_seconds = float(s.get("whisper_pad_seconds", 0.5))
        except (TypeError, ValueError):
            whisper_pad_seconds = 0.5
        self._whisper_pad_seconds.setValue(whisper_pad_seconds)
        self._whisper_pad_seconds.setSuffix(" s")
        self._whisper_pad_seconds.setSpecialValueText(t("whisper_padding_off"))
        self._whisper_pad_seconds.valueChanged.connect(self._auto_save)
        self._whisper_pad_card = _WidgetCard(
            FIF.ALIGNMENT, t("label_whisper_padding"), self._whisper_pad_seconds, t("whisper_padding_tooltip")
        )
        asr_group.addSettingCard(self._whisper_pad_card)

        self._sensevoice_pad_seconds = DoubleSpinBox()
        self._sensevoice_pad_seconds.setRange(0.0, 5.0)
        self._sensevoice_pad_seconds.setDecimals(2)
        self._sensevoice_pad_seconds.setSingleStep(0.1)
        try:
            sensevoice_pad_seconds = float(s.get("sensevoice_pad_seconds", 0.5))
        except (TypeError, ValueError):
            sensevoice_pad_seconds = 0.5
        self._sensevoice_pad_seconds.setValue(sensevoice_pad_seconds)
        self._sensevoice_pad_seconds.setSuffix(" s")
        self._sensevoice_pad_seconds.setSpecialValueText(t("sensevoice_padding_off"))
        self._sensevoice_pad_seconds.valueChanged.connect(self._auto_save)
        self._sensevoice_pad_card = _WidgetCard(
            FIF.ALIGNMENT, t("label_sensevoice_padding"), self._sensevoice_pad_seconds, t("sensevoice_padding_tooltip")
        )
        asr_group.addSettingCard(self._sensevoice_pad_card)

        self._audio_device = ComboBox()
        self._audio_device.addItem(t("audio_disabled"))
        self._audio_device.addItem(t("system_default"))
        try:
            from audio_capture import list_output_devices

            for name in list_output_devices():
                self._audio_device.addItem(name)
        except Exception as e:
            log.warning(f"Could not list output devices: {e}")
        saved_audio = s.get("audio_device")
        if saved_audio == "__disabled__":
            self._audio_device.setCurrentIndex(0)
        elif saved_audio:
            idx = self._audio_device.findText(saved_audio)
            if idx >= 0:
                self._audio_device.setCurrentIndex(idx)
        else:
            self._audio_device.setCurrentIndex(1)  # system default
        self._audio_device.currentIndexChanged.connect(self._auto_save)
        asr_group.addSettingCard(_WidgetCard(FIF.VOLUME, t("label_audio"), self._audio_device, stretch=True))

        self._mic_device = ComboBox()
        self._mic_device.addItem(t("mic_disabled"))
        self._mic_device.addItem(t("system_default"))
        try:
            from audio_capture import list_input_devices

            for name in list_input_devices():
                self._mic_device.addItem(name)
        except Exception as e:
            log.warning(f"Could not list input devices: {e}")
        saved_mic = s.get("mic_device")
        if saved_mic:
            if saved_mic in ("__default__", "default"):
                self._mic_device.setCurrentIndex(1)
            else:
                idx = self._mic_device.findText(saved_mic)
                if idx >= 0:
                    self._mic_device.setCurrentIndex(idx)
        self._mic_device.currentIndexChanged.connect(self._auto_save)
        asr_group.addSettingCard(_WidgetCard(FIF.HEADPHONE, t("label_mic"), self._mic_device, stretch=True))

        self._hub_combo = ComboBox()
        self._hub_combo.addItems([t("hub_modelscope"), t("hub_huggingface")])
        saved_hub = s.get("hub", "ms")
        self._hub_combo.setCurrentIndex(0 if saved_hub == "ms" else 1)
        self._hub_combo.currentIndexChanged.connect(self._auto_save)
        asr_group.addSettingCard(_WidgetCard(FIF.CLOUD, t("label_hub"), self._hub_combo, stretch=True))

        self._ui_lang_combo = ComboBox()
        self._ui_lang_combo.addItems(["English", "简体中文", "繁體中文"])
        from i18n import get_lang

        saved_lang = s.get("ui_lang", get_lang())
        self._ui_lang_combo.setCurrentIndex({"en": 0, "zh": 1, "zh_TW": 2}.get(saved_lang, 1))
        self._ui_lang_combo.currentIndexChanged.connect(self._on_ui_lang_changed)
        asr_group.addSettingCard(_WidgetCard(FIF.GLOBE, t("label_ui_lang"), self._ui_lang_combo, stretch=True))

        page.add_group(asr_group)

        # Whisper model download — only visible when engine is Whisper
        self._whisper_group = SettingCardGroup(t("group_download_whisper"), page.view)
        self._whisper_size_combo = ComboBox()
        saved_size = s.get(
            "whisper_model_size", self._config["asr"].get("model_size", "medium")
        )
        self._populate_whisper_models(saved_size)
        self._whisper_size_combo.currentIndexChanged.connect(
            self._on_whisper_size_changed
        )
        whisper_row = QWidget()
        whisper_layout = QHBoxLayout(whisper_row)
        whisper_layout.setContentsMargins(0, 0, 0, 0)
        whisper_layout.addWidget(self._whisper_size_combo)
        self._whisper_status = CaptionLabel("", whisper_row)
        _set_text_color(self._whisper_status, _TEXT_MUTED)
        whisper_layout.addWidget(self._whisper_status)
        self._whisper_dl_btn = PushButton(t("btn_download_whisper"), whisper_row)
        self._whisper_dl_btn.clicked.connect(self._download_whisper)
        whisper_layout.addWidget(self._whisper_dl_btn)
        self._whisper_group.addSettingCard(_WidgetCard(FIF.DOWNLOAD, t("group_download_whisper"), whisper_row))
        page.add_group(self._whisper_group)
        self._whisper_group.setVisible(engine_idx == 0)
        self._asr_engine.currentIndexChanged.connect(
            self._on_engine_changed_whisper_vis
        )
        self._update_whisper_size_label()

        # Remote ASR server URL — only visible when engine is Remote Whisper
        self._remote_group = SettingCardGroup("Remote ASR Server", page.view)
        self._remote_url_edit = LineEdit()
        self._remote_url_edit.setText(s.get("remote_asr_url", "http://127.0.0.1:8765"))
        self._remote_url_edit.setPlaceholderText("http://127.0.0.1:8765")
        self._remote_url_edit.setMinimumWidth(260)
        self._remote_url_edit.editingFinished.connect(self._auto_save)
        self._remote_group.addSettingCard(_WidgetCard(FIF.CONNECT, "URL", self._remote_url_edit))
        page.add_group(self._remote_group)
        self._remote_group.setVisible(engine_idx == 3)

        vad_group = SettingCardGroup(t("group_vad_mode"), page.view)
        self._vad_mode = ComboBox()
        self._vad_mode.addItems([t("vad_silero"), t("vad_energy"), t("vad_disabled")])
        mode_map = {"silero": 0, "energy": 1, "disabled": 2}
        self._vad_mode.setCurrentIndex(mode_map.get(s.get("vad_mode", "energy"), 1))
        self._vad_mode.currentIndexChanged.connect(self._on_vad_mode_changed)
        self._vad_mode.currentIndexChanged.connect(self._auto_save)
        vad_group.addSettingCard(_WidgetCard(FIF.SETTING, t("group_vad_mode"), self._vad_mode, stretch=True))

        self._vad_threshold_slider = Slider(Qt.Orientation.Horizontal)
        self._vad_threshold_slider.setRange(0, 100)
        vad_pct = int(s.get("vad_threshold", 0.5) * 100)
        self._vad_threshold_slider.setValue(vad_pct)
        self._vad_threshold_slider.setMinimumWidth(220)
        self._vad_threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._vad_threshold_slider.sliderReleased.connect(self._auto_save)
        self._vad_threshold_label = StrongBodyLabel(f"{vad_pct}%")
        self._vad_threshold_label.setFont(QFont(MONO_FONT_FAMILY, 11, QFont.Weight.Bold))
        vad_group.addSettingCard(
            self._slider_card(FIF.BRIGHTNESS, t("group_silero_threshold"), self._vad_threshold_slider, self._vad_threshold_label)
        )

        self._energy_slider = Slider(Qt.Orientation.Horizontal)
        self._energy_slider.setRange(1, 100)
        energy_pm = int(s.get("energy_threshold", 0.03) * 1000)
        self._energy_slider.setValue(energy_pm)
        self._energy_slider.setMinimumWidth(220)
        self._energy_slider.valueChanged.connect(self._on_energy_changed)
        self._energy_slider.sliderReleased.connect(self._auto_save)
        self._energy_label = StrongBodyLabel(f"{energy_pm}‰")
        self._energy_label.setFont(QFont(MONO_FONT_FAMILY, 11, QFont.Weight.Bold))
        vad_group.addSettingCard(
            self._slider_card(FIF.VOLUME, t("group_energy_threshold"), self._energy_slider, self._energy_label)
        )
        page.add_group(vad_group)

        timing_group = SettingCardGroup(t("group_timing"), page.view)
        self._min_speech = DoubleSpinBox()
        self._min_speech.setRange(0.1, 5.0)
        self._min_speech.setSingleStep(0.1)
        self._min_speech.setValue(s.get("min_speech_duration", 2.0))
        self._min_speech.setSuffix(" s")
        self._min_speech.valueChanged.connect(self._on_timing_changed)
        self._min_speech.valueChanged.connect(self._auto_save)
        timing_group.addSettingCard(_WidgetCard(FIF.SPEED_MEDIUM, t("label_min_speech"), self._min_speech))

        self._max_speech = DoubleSpinBox()
        self._max_speech.setRange(2.0, 30.0)
        self._max_speech.setSingleStep(1.0)
        self._max_speech.setValue(s.get("max_speech_duration", 6.0))
        self._max_speech.setSuffix(" s")
        self._max_speech.valueChanged.connect(self._on_timing_changed)
        self._max_speech.valueChanged.connect(self._auto_save)
        timing_group.addSettingCard(_WidgetCard(FIF.SPEED_HIGH, t("label_max_speech"), self._max_speech))

        self._silence_mode = ComboBox()
        self._silence_mode.addItems([t("silence_auto"), t("silence_fixed")])
        saved_smode = s.get("silence_mode", "auto")
        self._silence_mode.setCurrentIndex(0 if saved_smode == "auto" else 1)
        self._silence_mode.currentIndexChanged.connect(self._on_silence_mode_changed)
        self._silence_mode.currentIndexChanged.connect(self._on_timing_changed)
        self._silence_mode.currentIndexChanged.connect(self._auto_save)
        timing_group.addSettingCard(_WidgetCard(FIF.PAUSE, t("label_silence"), self._silence_mode, stretch=True))

        self._silence_duration = DoubleSpinBox()
        self._silence_duration.setRange(0.1, 3.0)
        self._silence_duration.setSingleStep(0.1)
        self._silence_duration.setValue(s.get("silence_duration", 0.8))
        self._silence_duration.setSuffix(" s")
        self._silence_duration.setEnabled(saved_smode != "auto")
        self._silence_duration.valueChanged.connect(self._on_timing_changed)
        self._silence_duration.valueChanged.connect(self._auto_save)
        timing_group.addSettingCard(_WidgetCard(FIF.PAUSE, t("label_silence_dur"), self._silence_duration))

        # Share of a segment that must be speech; lower keeps short replies (「え？」)
        self._min_density_spin = SpinBox()
        self._min_density_spin.setRange(0, 90)
        self._min_density_spin.setSingleStep(5)
        self._min_density_spin.setSuffix("%")
        self._min_density_spin.setValue(_density_percent(s.get("vad_min_speech_density")))
        self._min_density_spin.valueChanged.connect(self._on_timing_changed)
        self._min_density_spin.valueChanged.connect(self._auto_save)
        timing_group.addSettingCard(_WidgetCard(
            FIF.FILTER, t("label_min_speech_density"), self._min_density_spin,
            content=t("desc_min_speech_density"),
        ))

        self._incremental_asr_cb = SwitchButton()
        self._incremental_asr_cb.setOnText(t("label_incremental_asr"))
        self._incremental_asr_cb.setOffText(t("label_incremental_asr"))
        self._incremental_asr_cb.setChecked(s.get("incremental_asr", False))
        self._incremental_asr_cb.checkedChanged.connect(self._on_timing_changed)
        self._incremental_asr_cb.checkedChanged.connect(self._auto_save)
        timing_group.addSettingCard(
            _WidgetCard(FIF.PLAY, t("label_incremental_asr"), self._incremental_asr_cb, t("incremental_asr_tooltip"))
        )

        self._interim_interval_spin = DoubleSpinBox()
        self._interim_interval_spin.setRange(0.3, 10.0)
        self._interim_interval_spin.setSingleStep(0.1)
        self._interim_interval_spin.setValue(s.get("interim_interval", 2.0))
        self._interim_interval_spin.setSuffix(" s")
        self._interim_interval_spin.setEnabled(s.get("incremental_asr", False))
        self._interim_interval_spin.valueChanged.connect(self._on_timing_changed)
        self._interim_interval_spin.valueChanged.connect(self._auto_save)
        self._incremental_asr_cb.checkedChanged.connect(self._interim_interval_spin.setEnabled)
        timing_group.addSettingCard(_WidgetCard(FIF.HISTORY, t("label_interim_interval"), self._interim_interval_spin))

        self._live_preview_cb = SwitchButton()
        self._live_preview_cb.setOnText(t("label_live_preview"))
        self._live_preview_cb.setOffText(t("label_live_preview"))
        self._live_preview_cb.setChecked(s.get("live_preview", True))
        self._live_preview_cb.setEnabled(s.get("incremental_asr", False))
        self._incremental_asr_cb.checkedChanged.connect(self._live_preview_cb.setEnabled)
        self._live_preview_cb.checkedChanged.connect(self._on_timing_changed)
        self._live_preview_cb.checkedChanged.connect(self._auto_save)
        timing_group.addSettingCard(
            _WidgetCard(FIF.CHAT, t("label_live_preview"), self._live_preview_cb, t("live_preview_tooltip"))
        )
        page.add_group(timing_group)

        self._on_engine_changed_whisper_vis(engine_idx)
        page.finish()
        return page

    @staticmethod
    def _slider_card(icon, title, slider, value_label):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addSpacing(8)
        layout.addWidget(value_label)
        return _WidgetCard(icon, title, row)

    # ── Translation page ──

    def _create_translation_tab(self):
        page = _Page("translation_interface")
        s = self._current_settings

        # Local Ollama models with graphics card advice (was the launcher's model list)
        self._local_models_group = LocalModelsGroup(self.get_settings, page.view)
        self._local_models_group.use_requested.connect(self.local_model_requested.emit)
        page.add_group(self._local_models_group)

        models_card = _BlockCard(t("group_model_configs"), page.view)
        self._model_list = ListWidget()
        self._model_list.setFont(QFont(MONO_FONT_FAMILY, 9))
        self._model_list.setMinimumHeight(160)
        self._model_list.itemDoubleClicked.connect(self._on_model_double_clicked)
        self._refresh_model_list()
        models_card.vbox.addWidget(self._model_list)

        btn_row = QHBoxLayout()
        add_btn = PushButton(t("btn_add"))
        add_btn.setIcon(FIF.ADD)
        add_btn.clicked.connect(self._add_model)
        btn_row.addWidget(add_btn)
        edit_btn = PushButton(t("btn_edit"))
        edit_btn.setIcon(FIF.EDIT)
        edit_btn.clicked.connect(self._edit_model)
        btn_row.addWidget(edit_btn)
        dup_btn = PushButton(t("btn_duplicate"))
        dup_btn.setIcon(FIF.COPY)
        dup_btn.clicked.connect(self._dup_model)
        btn_row.addWidget(dup_btn)
        del_btn = PushButton(t("btn_remove"))
        del_btn.setIcon(FIF.DELETE)
        del_btn.clicked.connect(self._remove_model)
        btn_row.addWidget(del_btn)
        btn_row.addStretch(1)
        models_card.vbox.addLayout(btn_row)
        page.vbox.addWidget(models_card)

        from translator import DEFAULT_PROMPT

        prompt_group = SettingCardGroup(t("group_system_prompt"), page.view)
        self._prompt_preset = ComboBox()
        self._prompt_preset.addItem(t("prompt_daily"), userData="daily")
        self._prompt_preset.addItem(t("prompt_esports"), userData="esports")
        self._prompt_preset.addItem(t("prompt_anime"), userData="anime")
        self._prompt_preset.addItem(t("prompt_webid"), userData="webid")
        self._prompt_preset.addItem(t("prompt_custom"), userData="custom")

        current_prompt = s.get("system_prompt", DEFAULT_PROMPT)
        self._prompt_preset.setCurrentIndex(_prompt_preset_index(current_prompt))
        self._prompt_preset.currentIndexChanged.connect(self._on_prompt_preset_changed)
        prompt_group.addSettingCard(_WidgetCard(FIF.DICTIONARY, t("label_prompt_preset"), self._prompt_preset, stretch=True))
        page.add_group(prompt_group)

        prompt_card = _BlockCard(t("group_system_prompt"), page.view)
        self._prompt_edit = TextEdit()
        self._prompt_edit.setFont(QFont(MONO_FONT_FAMILY, 9))
        self._prompt_edit.setMinimumHeight(140)
        self._prompt_edit.setPlainText(current_prompt)
        self._prompt_debounce = QTimer()
        self._prompt_debounce.setSingleShot(True)
        self._prompt_debounce.setInterval(600)
        self._prompt_debounce.timeout.connect(self._apply_prompt)
        self._prompt_edit.textChanged.connect(self._prompt_debounce.start)
        prompt_card.vbox.addWidget(self._prompt_edit)
        page.vbox.addWidget(prompt_card)

        net_group = SettingCardGroup(t("group_network"), page.view)
        self._timeout_spin = SpinBox()
        self._timeout_spin.setRange(1, 60)
        self._timeout_spin.setValue(s.get("timeout", 5))
        self._timeout_spin.setSuffix(" s")
        self._timeout_spin.valueChanged.connect(
            lambda v: self._current_settings.update({"timeout": v})
        )
        self._timeout_spin.valueChanged.connect(self._auto_save)
        net_group.addSettingCard(_WidgetCard(FIF.WIFI, t("label_timeout"), self._timeout_spin))
        page.add_group(net_group)

        page.finish()
        return page

    # ── Appearance page (overlay style + subtitle window) ──

    def _create_appearance_page(self):
        page = _Page("appearance_interface")
        self._build_style_groups(page)
        subtitle_card = _BlockCard(t("tab_subtitle"), page.view)
        subtitle_card.vbox.addWidget(self._create_subtitle_tab())
        page.vbox.addWidget(subtitle_card)
        page.finish()
        return page

    def _build_style_groups(self, page: _Page):
        from subtitle_overlay import DEFAULT_STYLE

        s = self._current_settings.get("style", dict(DEFAULT_STYLE))

        preset_group = SettingCardGroup(t("tab_style"), page.view)
        self._style_preset = ComboBox()
        preset_names = [
            ("default", t("preset_default")),
            ("transparent", t("preset_transparent")),
            ("compact", t("preset_compact")),
            ("light", t("preset_light")),
            ("dracula", t("preset_dracula")),
            ("nord", t("preset_nord")),
            ("monokai", t("preset_monokai")),
            ("solarized", t("preset_solarized")),
            ("gruvbox", t("preset_gruvbox")),
            ("tokyo_night", t("preset_tokyo_night")),
            ("catppuccin", t("preset_catppuccin")),
            ("one_dark", t("preset_one_dark")),
            ("everforest", t("preset_everforest")),
            ("kanagawa", t("preset_kanagawa")),
            ("custom", t("preset_custom")),
        ]
        self._preset_keys = [k for k, _ in preset_names]
        for _, label in preset_names:
            self._style_preset.addItem(label)
        current_preset = s.get("preset", "default")
        if current_preset in self._preset_keys:
            self._style_preset.setCurrentIndex(self._preset_keys.index(current_preset))
        else:
            self._style_preset.setCurrentIndex(self._preset_keys.index("custom"))
        self._style_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_group.addSettingCard(_WidgetCard(FIF.PALETTE, t("group_preset"), self._style_preset, stretch=True))

        reset_row = QWidget()
        reset_layout = QHBoxLayout(reset_row)
        reset_layout.setContentsMargins(0, 0, 0, 0)
        reset_btn = PushButton(t("btn_reset_style"))
        reset_btn.clicked.connect(self._reset_style)
        reset_layout.addWidget(reset_btn)
        reset_pos_btn = PushButton(t("btn_reset_positions"))
        reset_pos_btn.clicked.connect(self.reset_positions.emit)
        reset_layout.addWidget(reset_pos_btn)
        preset_group.addSettingCard(_WidgetCard(FIF.SETTING, t("btn_reset_style"), reset_row))
        page.add_group(preset_group)

        bg_group = SettingCardGroup(t("group_background"), page.view)
        self._bg_color_btn = self._make_color_btn(s.get("bg_color", DEFAULT_STYLE["bg_color"]), t("label_bg_color"))
        bg_group.addSettingCard(_WidgetCard(FIF.BRUSH, t("label_bg_color"), self._bg_color_btn))

        self._bg_opacity = SpinBox()
        self._bg_opacity.setRange(0, 100)
        self._bg_opacity.setSuffix("%")
        self._bg_opacity.setValue(round(s.get("bg_opacity", DEFAULT_STYLE["bg_opacity"]) / 255 * 100))
        self._bg_opacity.valueChanged.connect(self._on_style_value_changed)
        self._bg_opacity.valueChanged.connect(self._auto_save)
        bg_group.addSettingCard(_WidgetCard(FIF.TRANSPARENT, t("label_bg_opacity"), self._bg_opacity))

        self._header_color_btn = self._make_color_btn(
            s.get("header_color", DEFAULT_STYLE["header_color"]), t("label_header_color")
        )
        bg_group.addSettingCard(_WidgetCard(FIF.BRUSH, t("label_header_color"), self._header_color_btn))

        self._header_opacity = SpinBox()
        self._header_opacity.setRange(0, 100)
        self._header_opacity.setSuffix("%")
        self._header_opacity.setValue(round(s.get("header_opacity", DEFAULT_STYLE["header_opacity"]) / 255 * 100))
        self._header_opacity.valueChanged.connect(self._on_style_value_changed)
        self._header_opacity.valueChanged.connect(self._auto_save)
        bg_group.addSettingCard(_WidgetCard(FIF.TRANSPARENT, t("label_header_opacity"), self._header_opacity))

        self._border_radius = SpinBox()
        self._border_radius.setRange(0, 30)
        self._border_radius.setValue(s.get("border_radius", DEFAULT_STYLE["border_radius"]))
        self._border_radius.setSuffix(" px")
        self._border_radius.valueChanged.connect(self._on_style_value_changed)
        self._border_radius.valueChanged.connect(self._auto_save)
        bg_group.addSettingCard(_WidgetCard(FIF.TILES, t("label_border_radius"), self._border_radius))
        page.add_group(bg_group)

        text_group = SettingCardGroup(t("group_text"), page.view)
        self._orig_font_combo = QFontComboBox()
        self._orig_font_combo.setCurrentFont(
            QFont(s.get("original_font_family", DEFAULT_STYLE["original_font_family"]))
        )
        self._orig_font_combo.setMinimumWidth(220)
        self._orig_font_combo.currentFontChanged.connect(self._on_style_value_changed)
        self._orig_font_combo.currentFontChanged.connect(self._auto_save)
        text_group.addSettingCard(_WidgetCard(FIF.FONT, t("label_original_font"), self._orig_font_combo))

        self._orig_font_size = SpinBox()
        self._orig_font_size.setRange(6, 24)
        self._orig_font_size.setValue(s.get("original_font_size", DEFAULT_STYLE["original_font_size"]))
        self._orig_font_size.setSuffix(" pt")
        self._orig_font_size.valueChanged.connect(self._on_style_value_changed)
        self._orig_font_size.valueChanged.connect(self._auto_save)
        text_group.addSettingCard(_WidgetCard(FIF.ZOOM_IN, t("label_original_font_size"), self._orig_font_size))

        self._orig_color_btn = self._make_color_btn(
            s.get("original_color", DEFAULT_STYLE["original_color"]), t("label_original_color")
        )
        text_group.addSettingCard(_WidgetCard(FIF.BRUSH, t("label_original_color"), self._orig_color_btn))

        self._trans_font_combo = QFontComboBox()
        self._trans_font_combo.setCurrentFont(
            QFont(s.get("translation_font_family", DEFAULT_STYLE["translation_font_family"]))
        )
        self._trans_font_combo.setMinimumWidth(220)
        self._trans_font_combo.currentFontChanged.connect(self._on_style_value_changed)
        self._trans_font_combo.currentFontChanged.connect(self._auto_save)
        text_group.addSettingCard(_WidgetCard(FIF.FONT, t("label_translation_font"), self._trans_font_combo))

        self._trans_font_size = SpinBox()
        self._trans_font_size.setRange(6, 24)
        self._trans_font_size.setValue(s.get("translation_font_size", DEFAULT_STYLE["translation_font_size"]))
        self._trans_font_size.setSuffix(" pt")
        self._trans_font_size.valueChanged.connect(self._on_style_value_changed)
        self._trans_font_size.valueChanged.connect(self._auto_save)
        text_group.addSettingCard(_WidgetCard(FIF.ZOOM_IN, t("label_translation_font_size"), self._trans_font_size))

        self._trans_color_btn = self._make_color_btn(
            s.get("translation_color", DEFAULT_STYLE["translation_color"]), t("label_translation_color")
        )
        text_group.addSettingCard(_WidgetCard(FIF.BRUSH, t("label_translation_color"), self._trans_color_btn))

        self._ts_color_btn = self._make_color_btn(
            s.get("timestamp_color", DEFAULT_STYLE["timestamp_color"]), t("label_timestamp_color")
        )
        text_group.addSettingCard(_WidgetCard(FIF.BRUSH, t("label_timestamp_color"), self._ts_color_btn))
        page.add_group(text_group)

        win_group = SettingCardGroup(t("group_window"), page.view)
        self._window_opacity = SpinBox()
        self._window_opacity.setRange(30, 100)
        self._window_opacity.setSuffix("%")
        self._window_opacity.setValue(s.get("window_opacity", DEFAULT_STYLE["window_opacity"]))
        self._window_opacity.valueChanged.connect(self._on_style_value_changed)
        self._window_opacity.valueChanged.connect(self._auto_save)
        win_group.addSettingCard(_WidgetCard(FIF.TRANSPARENT, t("label_window_opacity"), self._window_opacity))
        page.add_group(win_group)

    def _make_color_btn(self, color: str, title: str) -> ColorPickerButton:
        btn = ColorPickerButton(QColor(color), title)
        btn.setProperty("hex_color", color)
        btn.colorChanged.connect(lambda c, b=btn: self._on_color_picked(b, c))
        return btn

    def _on_color_picked(self, btn: ColorPickerButton, color: QColor):
        if not color.isValid():
            return
        btn.setProperty("hex_color", color.name())
        self._on_style_value_changed()
        self._auto_save()

    def _set_color_btn(self, btn: ColorPickerButton, color: str):
        btn.setProperty("hex_color", color)
        btn.setColor(QColor(color))

    def _collect_style(self) -> dict:
        return {
            "preset": self._preset_keys[self._style_preset.currentIndex()],
            "bg_color": self._bg_color_btn.property("hex_color"),
            "bg_opacity": round(self._bg_opacity.value() / 100 * 255),
            "header_color": self._header_color_btn.property("hex_color"),
            "header_opacity": round(self._header_opacity.value() / 100 * 255),
            "border_radius": self._border_radius.value(),
            "original_font_family": self._orig_font_combo.currentFont().family(),
            "translation_font_family": self._trans_font_combo.currentFont().family(),
            "original_font_size": self._orig_font_size.value(),
            "translation_font_size": self._trans_font_size.value(),
            "original_color": self._orig_color_btn.property("hex_color"),
            "translation_color": self._trans_color_btn.property("hex_color"),
            "timestamp_color": self._ts_color_btn.property("hex_color"),
            "window_opacity": self._window_opacity.value(),
        }

    def _apply_style_to_controls(self, s: dict):
        """Update all style controls to match a style dict, without triggering auto-save."""
        self._set_color_btn(self._bg_color_btn, s["bg_color"])
        self._bg_opacity.setValue(round(s["bg_opacity"] / 255 * 100))
        self._set_color_btn(self._header_color_btn, s["header_color"])
        self._header_opacity.setValue(round(s["header_opacity"] / 255 * 100))
        self._border_radius.setValue(s["border_radius"])
        self._orig_font_combo.setCurrentFont(QFont(s["original_font_family"]))
        self._trans_font_combo.setCurrentFont(QFont(s["translation_font_family"]))
        self._orig_font_size.setValue(s["original_font_size"])
        self._trans_font_size.setValue(s["translation_font_size"])
        self._set_color_btn(self._orig_color_btn, s["original_color"])
        self._set_color_btn(self._trans_color_btn, s["translation_color"])
        self._set_color_btn(self._ts_color_btn, s["timestamp_color"])
        self._window_opacity.setValue(s["window_opacity"])

    def _on_preset_changed(self, index):
        from subtitle_overlay import STYLE_PRESETS

        key = self._preset_keys[index]
        if key == "custom":
            return
        preset = STYLE_PRESETS.get(key)
        if not preset:
            return
        self._block_style_signals(True)
        self._apply_style_to_controls(preset)
        self._block_style_signals(False)
        self._auto_save()

    def _on_style_value_changed(self, *_args):
        """When any style control changes manually, switch preset to Custom."""
        custom_idx = len(self._preset_keys) - 1
        if self._style_preset.currentIndex() != custom_idx:
            self._style_preset.blockSignals(True)
            self._style_preset.setCurrentIndex(custom_idx)
            self._style_preset.blockSignals(False)
        self._auto_save()

    def _reset_style(self):
        from subtitle_overlay import DEFAULT_STYLE

        self._style_preset.blockSignals(True)
        self._style_preset.setCurrentIndex(0)  # default
        self._style_preset.blockSignals(False)
        self._block_style_signals(True)
        self._apply_style_to_controls(DEFAULT_STYLE)
        self._block_style_signals(False)
        self._auto_save()

    def _block_style_signals(self, block: bool):
        for w in (
            self._bg_opacity,
            self._header_opacity,
            self._border_radius,
            self._orig_font_combo,
            self._trans_font_combo,
            self._orig_font_size,
            self._trans_font_size,
            self._window_opacity,
            self._bg_color_btn,
            self._header_color_btn,
            self._orig_color_btn,
            self._trans_color_btn,
            self._ts_color_btn,
        ):
            w.blockSignals(block)

    # ── Subtitle window settings (embedded in the appearance page) ──

    def _create_subtitle_tab(self):
        subtitle_settings = self._current_settings.get("subtitle_mode") or {}
        self._subtitle_widget = SubtitleSettingsWidget(subtitle_settings)
        self._subtitle_widget.settings_changed.connect(self._on_subtitle_settings_changed)
        return self._subtitle_widget

    # Changed at runtime by menus / dragging, never by the subtitle settings page. The
    # page keeps an older copy of them, so its edits must not send those values back
    # (e.g. "固定位置" from the right-click menu was undone by any edit on the page).
    _SUBWIN_RUNTIME_KEYS = ("enabled", "locked", "window_x", "window_y", "window_height",
                            "show_original", "show_translation", "show_previous")

    def _on_subtitle_settings_changed(self, s):
        current = self._current_settings.get("subtitle_mode") or {}
        if current is not s:
            for key in self._SUBWIN_RUNTIME_KEYS:
                if key in current:
                    s[key] = current[key]  # s is the page's own dict, so its copy is synced too
        self._current_settings["subtitle_mode"] = s
        self._auto_save()
        self.subtitle_settings_changed.emit(s)

    def update_subtitle_settings(self, s):
        self._current_settings["subtitle_mode"] = s
        self._subtitle_widget.update_settings(s)

    # ── Benchmark page ──

    def _create_benchmark_tab(self):
        page = _Page("benchmark_interface")

        ctrl_group = SettingCardGroup(t("tab_benchmark"), page.view)
        self._bench_lang = ComboBox()
        self._bench_lang.addItems(["ja", "en", "zh", "ko", "fr", "de"])
        self._bench_lang.setCurrentIndex(0)
        ctrl_group.addSettingCard(_WidgetCard(FIF.LANGUAGE, t("label_source"), self._bench_lang))
        self._bench_target = ComboBox()
        self._bench_target.addItems(["zh", "en", "ja", "ko", "fr", "de", "es", "ru"])
        ctrl_group.addSettingCard(_WidgetCard(FIF.LANGUAGE, t("target_label"), self._bench_target))
        self._bench_btn = PrimaryPushButton(t("btn_test_all"))
        self._bench_btn.clicked.connect(self._run_benchmark)
        ctrl_group.addSettingCard(_WidgetCard(FIF.SPEED_HIGH, t("btn_test_all"), self._bench_btn))
        page.add_group(ctrl_group)

        out_card = _BlockCard(t("tab_benchmark"), page.view)
        self._bench_output = TextEdit()
        self._bench_output.setReadOnly(True)
        self._bench_output.setFont(QFont(MONO_FONT_FAMILY, 9))
        self._bench_output.setMinimumHeight(260)
        out_card.vbox.addWidget(self._bench_output)
        page.vbox.addWidget(out_card)
        page.finish()
        return page

    # ── Changelog page (navigation bottom) ──

    def _create_changelog_tab(self):
        from dialogs import _load_latest_changelog

        page = _Page("changelog_interface")
        _, html = _load_latest_changelog()
        browser = TextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        browser.setFont(QFont(UI_FONT_FAMILY, 10))
        browser.setMinimumHeight(400)
        page.vbox.addWidget(browser, 1)
        return page

    # ── Models & cache page ──

    def _create_cache_tab(self):
        page = _Page("cache_interface")
        s = self._current_settings

        ts_group = SettingCardGroup(t("group_transcript"), page.view)
        self._auto_save_transcript_cb = SwitchButton()
        self._auto_save_transcript_cb.setOnText(t("label_auto_save_transcript"))
        self._auto_save_transcript_cb.setOffText(t("label_auto_save_transcript"))
        self._auto_save_transcript_cb.setChecked(s.get("auto_save_transcript", True))
        self._auto_save_transcript_cb.checkedChanged.connect(self._auto_save)
        ts_group.addSettingCard(
            _WidgetCard(FIF.SAVE, t("label_auto_save_transcript"), self._auto_save_transcript_cb, t("auto_save_transcript_tooltip"))
        )
        ts_open_btn = PushButton(t("btn_open_transcripts"))
        ts_open_btn.setIcon(FIF.FOLDER)
        ts_open_btn.clicked.connect(self._open_transcripts_folder)
        ts_group.addSettingCard(_WidgetCard(FIF.FOLDER, t("btn_open_transcripts"), ts_open_btn))
        page.add_group(ts_group)

        cache_card = _BlockCard(t("tab_cache"), page.view)
        top_row = QHBoxLayout()
        self._cache_total = BodyLabel("")
        self._cache_total.setFont(QFont(MONO_FONT_FAMILY, 9, QFont.Weight.Bold))
        top_row.addWidget(self._cache_total, 1)
        open_btn = PushButton(t("btn_open_folder"))
        open_btn.setIcon(FIF.FOLDER)
        open_btn.clicked.connect(
            lambda: (
                MODELS_DIR.mkdir(parents=True, exist_ok=True),
                os.startfile(str(MODELS_DIR)),
            )
        )
        top_row.addWidget(open_btn)
        delete_all_btn = PushButton(t("btn_delete_all_exit"))
        delete_all_btn.setIcon(FIF.DELETE)
        delete_all_btn.clicked.connect(self._delete_all_and_exit)
        top_row.addWidget(delete_all_btn)
        cache_card.vbox.addLayout(top_row)

        self._cache_list = ListWidget()
        self._cache_list.setFont(QFont(MONO_FONT_FAMILY, 9))
        self._cache_list.setMinimumHeight(220)
        cache_card.vbox.addWidget(self._cache_list, 1)
        page.vbox.addWidget(cache_card)

        self._cache_entries = []
        self._refresh_cache()
        page.finish()
        return page

    def _open_transcripts_folder(self):
        ts_dir = Path(__file__).parent / "transcripts"
        ts_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(ts_dir))

    def _refresh_cache(self):
        self._cache_list.clear()
        self._cache_total.setText(t("scanning"))

        def _scan():
            try:
                entries = get_cache_entries()
                results = []
                for name, path in entries:
                    size = dir_size(path)
                    results.append((name, str(path), size))
            except Exception as e:
                log.warning(f"cache scan failed: {e}")
                results = []
            self._cache_result.emit(results)

        threading.Thread(target=_scan, daemon=True).start()

    def _on_cache_result(self, results):
        self._cache_list.clear()
        self._cache_entries = results
        total = 0
        for name, path, size in results:
            total += size
            self._cache_list.addItem(f"{name}  —  {format_size(size)}")
        if not results:
            self._cache_list.addItem(t("no_cached_models"))
        self._cache_total.setText(
            t("cache_total").format(size=format_size(total), count=len(results))
        )

    def _delete_all_and_exit(self):
        if not self._cache_entries:
            return
        import shutil

        total_size = sum(s for _, _, s in self._cache_entries)
        ret = QMessageBox.warning(
            self,
            t("dialog_delete_title"),
            t("dialog_delete_msg").format(
                count=len(self._cache_entries), size=format_size(total_size)
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        for name, path, _ in self._cache_entries:
            try:
                shutil.rmtree(path)
                log.info(f"Deleted: {path}")
            except Exception as e:
                log.error(f"Failed to delete {path}: {e}")
        QApplication.instance().quit()

    # ── ASR helpers ──

    def _get_asr_lang_code(self) -> str:
        """Get the language code from the ASR language combo (stored as userData)."""
        return self._asr_lang.currentData() or "auto"

    def _on_engine_changed_whisper_vis(self, index):
        self._whisper_group.setVisible(index == 0)
        is_funasr = index == 1
        if hasattr(self, "_funasr_model_card"):
            self._funasr_model_card.setVisible(is_funasr)
        if hasattr(self, "_whisper_pad_card"):
            self._whisper_pad_card.setVisible(index == 0)
        if hasattr(self, "_sensevoice_pad_card"):
            show_funasr_pad = is_funasr and funasr_supports_padding(
                self._selected_funasr_model()
            )
            self._sensevoice_pad_card.setVisible(show_funasr_pad)
        if hasattr(self, "_remote_group"):
            self._remote_group.setVisible(index == 3)
        if hasattr(self, "_home_engine_card"):
            self.update_home_status(asr_engine=self._asr_engine.currentText())
        QTimer.singleShot(0, self._fit_height)

    def _selected_funasr_model(self) -> str:
        value = self._funasr_model_combo.currentData()
        return normalize_funasr_model_key(str(value) if value else None)

    def _on_funasr_model_changed(self):
        self._current_settings["funasr_model"] = self._selected_funasr_model()
        self._on_engine_changed_whisper_vis(self._asr_engine.currentIndex())
        self._auto_save()

    def _selected_whisper_model(self) -> str:
        value = self._whisper_size_combo.currentData()
        return str(value) if value else self._whisper_size_combo.currentText()

    def _populate_whisper_models(self, saved_value: str):
        self._whisper_size_combo.clear()
        for size in _WHISPER_SIZES:
            self._whisper_size_combo.addItem(size, userData=size)

        local_prefix = t("whisper_local_prefix")
        for item in list_local_faster_whisper_models():
            self._whisper_size_combo.addItem(
                f"{local_prefix}: {item['name']}", userData=item["path"]
            )

        selected = resolve_custom_whisper_model(saved_value) or saved_value
        idx = self._whisper_size_combo.findData(selected)
        if idx < 0:
            idx = self._whisper_size_combo.findText(saved_value)
        if idx < 0 and selected:
            label = f"{t('whisper_missing_local')}: {Path(str(selected)).name}"
            idx = self._whisper_size_combo.count()
            self._whisper_size_combo.addItem(label, userData=selected)
        if idx >= 0:
            self._whisper_size_combo.setCurrentIndex(idx)

    def _update_whisper_size_label(self):
        from model_manager import is_asr_cached, _MODEL_SIZE_BYTES

        size = self._selected_whisper_model()
        cached = is_asr_cached("whisper", size, self._current_settings.get("hub", "ms"))
        if size not in _WHISPER_SIZES:
            if cached:
                self._whisper_status.setText(t("whisper_local_ready"))
                _set_text_color(self._whisper_status, _TEXT_OK)
            else:
                self._whisper_status.setText(t("whisper_invalid_local"))
                _set_text_color(self._whisper_status, _TEXT_WARN)
            self._whisper_dl_btn.setEnabled(False)
            return
        if cached:
            self._whisper_status.setText(t("whisper_already_cached"))
            _set_text_color(self._whisper_status, _TEXT_OK)
            self._whisper_dl_btn.setEnabled(False)
        else:
            est = _MODEL_SIZE_BYTES.get(f"whisper-{size}", 0)
            self._whisper_status.setText(f"~{format_size(est)}")
            _set_text_color(self._whisper_status, _TEXT_MUTED)
            self._whisper_dl_btn.setEnabled(True)

    def _on_whisper_size_changed(self):
        self._current_settings["whisper_model_size"] = (
            self._selected_whisper_model()
        )
        self._update_whisper_size_label()
        # If already cached, switch engine immediately
        from model_manager import is_asr_cached

        size = self._selected_whisper_model()
        if is_asr_cached("whisper", size, self._current_settings.get("hub", "ms")):
            self._auto_save()

    def _download_whisper(self):
        from model_manager import is_asr_cached, get_missing_models

        size = self._selected_whisper_model()
        if size not in _WHISPER_SIZES:
            return
        hub = self._current_settings.get("hub", "ms")
        if is_asr_cached("whisper", size, hub):
            return
        missing = get_missing_models("whisper", size, hub)
        missing = [m for m in missing if m["type"] != "silero-vad"]
        if not missing:
            return
        from dialogs import ModelDownloadDialog

        dlg = ModelDownloadDialog(missing, hub=hub, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self._update_whisper_size_label()
            # Switch to Whisper engine with the downloaded size
            self._auto_save()

    # ── Model Management ──

    def _refresh_model_list(self):
        self._model_list.clear()
        active = self._current_settings.get("active_model", 0)
        for i, m in enumerate(self._current_settings.get("models", [])):
            prefix = ">>> " if i == active else "    "
            proxy = m.get("proxy", "none")
            proxy_tag = f"  [proxy: {proxy}]" if proxy != "none" else ""
            text = (
                f"{prefix}{m['name']}{proxy_tag}\n     {m['api_base']}  |  {m['model']}"
            )
            item = QListWidgetItem(text)
            if i == active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._model_list.addItem(item)
        if hasattr(self, "_home_model_card"):
            self._refresh_home_from_settings()
        if hasattr(self, "_local_models_group"):
            self._local_models_group._render()  # the model in use changed

    def refresh_local_models(self):
        """Ask Ollama again (after a download or a switch)."""
        self._local_models_group.refresh()

    def notify_local_model_busy(self):
        InfoBar.warning(t("lm_title_model"), t("lm_busy"), parent=self, duration=4000)

    def replace_local_models(self, settings: dict):
        """Take the model list and active model from `settings` (the local Ollama entries
        pointed at the port the app started Ollama on) and apply them like an edit in the
        list: saved, overlay combo updated, translator re-created."""
        self._current_settings["models"] = list(settings.get("models", []))
        self._current_settings["active_model"] = settings.get("active_model", 0)
        self._refresh_model_list()
        _save_settings(self._current_settings)
        self._emit_models_list_changed()
        active = self.get_active_model()
        if active:
            self.model_changed.emit(active)

    def _emit_models_list_changed(self):
        models = self._current_settings.get("models", [])
        active_idx = self._current_settings.get("active_model", 0)
        self.models_list_changed.emit(models, active_idx)

    def _add_model(self):
        dlg = ModelEditDialog(self)
        if dlg.exec():
            data = dlg.get_data()
            if data["name"] and data["model"]:
                self._current_settings.setdefault("models", []).append(data)
                self._refresh_model_list()
                _save_settings(self._current_settings)
                self._emit_models_list_changed()

    def _edit_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models):
            return
        dlg = ModelEditDialog(self, models[row])
        if dlg.exec():
            data = dlg.get_data()
            if data["name"] and data["model"]:
                models[row] = data
                self._refresh_model_list()
                _save_settings(self._current_settings)
                self._emit_models_list_changed()
                # Re-apply if editing the active model
                active = self._current_settings.get("active_model", 0)
                if row == active:
                    self.model_changed.emit(data)

    def _dup_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models):
            return
        dup = dict(models[row])
        dup["name"] = dup["name"] + " (copy)"
        models.append(dup)
        self._refresh_model_list()
        _save_settings(self._current_settings)
        self._emit_models_list_changed()

    def _remove_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models) or len(models) <= 1:
            return
        models.pop(row)
        active = self._current_settings.get("active_model", 0)
        if active >= len(models):
            self._current_settings["active_model"] = len(models) - 1
        self._refresh_model_list()
        self._model_list.setCurrentRow(min(row, len(models) - 1))
        _save_settings(self._current_settings)
        self._emit_models_list_changed()

    def _on_model_double_clicked(self, item):
        row = self._model_list.row(item)
        models = self._current_settings.get("models", [])
        if 0 <= row < len(models):
            self._model_list.setCurrentRow(row)
            self._edit_model()

    def _run_benchmark(self):
        models = self._current_settings.get("models", [])
        if not models:
            return

        source_lang = self._bench_lang.currentText()
        target_lang = self._bench_target.currentText()
        timeout_s = self._current_settings.get("timeout", 5)

        self._bench_btn.setEnabled(False)
        self._bench_btn.setText(t("testing"))
        self._bench_output.clear()

        from translator import DEFAULT_PROMPT, LANGUAGE_DISPLAY

        src = LANGUAGE_DISPLAY.get(source_lang, source_lang)
        tgt = LANGUAGE_DISPLAY.get(target_lang, target_lang)
        prompt = self._current_settings.get("system_prompt", DEFAULT_PROMPT)
        try:
            prompt = prompt.format(source_lang=src, target_lang=tgt)
        except (KeyError, IndexError):
            pass

        run_benchmark(
            models, source_lang, target_lang, timeout_s, prompt, self._bench_result.emit
        )

    def _on_bench_result(self, text: str):
        if text == "__DONE__":
            self._bench_btn.setEnabled(True)
            self._bench_btn.setText(t("btn_test_all"))
        else:
            self._bench_output.append(text)

    # ── Shared logic ──

    def _on_silence_mode_changed(self, index):
        self._silence_duration.setEnabled(index == 1)

    def _on_vad_mode_changed(self, index):
        modes = ["silero", "energy", "disabled"]
        self._current_settings["vad_mode"] = modes[index]

    def _on_threshold_changed(self, value):
        val = value / 100.0
        self._current_settings["vad_threshold"] = val
        self._vad_threshold_label.setText(f"{value}%")
        if not self._vad_threshold_slider.isSliderDown():
            self._auto_save()

    def _on_energy_changed(self, value):
        val = value / 1000.0
        self._current_settings["energy_threshold"] = val
        self._energy_label.setText(f"{value}‰")
        if not self._energy_slider.isSliderDown():
            self._auto_save()

    def _on_timing_changed(self, *_args):
        self._current_settings["min_speech_duration"] = round(self._min_speech.value(), 2)
        self._current_settings["max_speech_duration"] = round(self._max_speech.value(), 2)
        self._current_settings["silence_mode"] = (
            "auto" if self._silence_mode.currentIndex() == 0 else "fixed"
        )
        self._current_settings["silence_duration"] = round(self._silence_duration.value(), 2)
        if hasattr(self, "_min_density_spin"):
            self._current_settings["vad_min_speech_density"] = round(self._min_density_spin.value() / 100, 2)
        self._current_settings["incremental_asr"] = self._incremental_asr_cb.isChecked()
        self._current_settings["interim_interval"] = round(self._interim_interval_spin.value(), 2)
        self._current_settings["live_preview"] = self._live_preview_cb.isChecked()

    def _on_ui_lang_changed(self, index):
        lang = ("en", "zh", "zh_TW")[index] if 0 <= index < 3 else "zh"
        self._current_settings["ui_lang"] = lang
        _save_settings(self._current_settings)
        from i18n import set_lang

        set_lang(lang)
        QMessageBox.information(
            self,
            "LiveTranslate",
            "Language changed. Please restart the application.\n"
            "语言已更改，请重启应用程序。",
        )

    def _auto_save(self, *_args):
        self._save_timer.start()

    def _do_auto_save(self):
        self._apply_settings()
        _save_settings(self._current_settings)

    def _on_prompt_preset_changed(self, index):
        from translator import DEFAULT_PROMPT, PROMPT_PRESETS
        key = self._prompt_preset.itemData(index)
        if key == "custom":
            return
        prompt = PROMPT_PRESETS.get(key, DEFAULT_PROMPT)
        self._prompt_edit.setPlainText(prompt)
        self._apply_prompt()

    def _apply_prompt(self):
        text = self._prompt_edit.toPlainText().strip()
        if text:
            self._current_settings["system_prompt"] = text
            active = self.get_active_model()
            if active:
                self.model_changed.emit(active)
            _save_settings(self._current_settings)
            log.info("System prompt updated")
            # Update preset combo to reflect current state
            from translator import PROMPT_PRESETS
            self._prompt_preset.blockSignals(True)
            matched = 4  # custom
            for i, key in enumerate(["daily", "esports", "anime", "webid"]):
                if text.strip() == PROMPT_PRESETS[key].strip():
                    matched = i
                    break
            self._prompt_preset.setCurrentIndex(matched)
            self._prompt_preset.blockSignals(False)

    def _apply_settings(self):
        self._current_settings["asr_language"] = self._get_asr_lang_code()
        engine_map = {
            0: "whisper",
            1: "funasr",
            2: "anime-whisper",
            3: "remote-whisper",
        }
        self._current_settings["asr_engine"] = engine_map.get(
            self._asr_engine.currentIndex(), "whisper"
        )
        self._current_settings["funasr_model"] = self._selected_funasr_model()
        if hasattr(self, "_remote_url_edit"):
            url = self._remote_url_edit.text().strip()
            if url:
                self._current_settings["remote_asr_url"] = url
        self._current_settings["whisper_model_size"] = (
            self._selected_whisper_model()
        )
        dev_text = self._asr_device.currentText()
        self._current_settings["asr_device"] = dev_text.split(" (")[0]
        audio_idx = self._audio_device.currentIndex()
        if audio_idx == 0:
            self._current_settings["audio_device"] = "__disabled__"
        elif audio_idx == 1:
            self._current_settings["audio_device"] = None
        else:
            self._current_settings["audio_device"] = self._audio_device.currentText()
        mic_idx = self._mic_device.currentIndex()
        if mic_idx == 0:
            self._current_settings["mic_device"] = None
        elif mic_idx == 1:
            self._current_settings["mic_device"] = "__default__"
        else:
            self._current_settings["mic_device"] = self._mic_device.currentText()
        self._current_settings["hub"] = (
            "ms" if self._hub_combo.currentIndex() == 0 else "hf"
        )
        self._current_settings["sensevoice_pad_seconds"] = round(
            self._sensevoice_pad_seconds.value(), 2
        )
        self._current_settings["whisper_pad_seconds"] = round(
            self._whisper_pad_seconds.value(), 2
        )
        prompt_text = self._prompt_edit.toPlainText().strip()
        if prompt_text:
            self._current_settings["system_prompt"] = prompt_text
        self._current_settings["timeout"] = self._timeout_spin.value()
        if hasattr(self, "_incremental_asr_cb"):
            self._on_timing_changed()
        if hasattr(self, "_auto_save_transcript_cb"):
            self._current_settings["auto_save_transcript"] = (
                self._auto_save_transcript_cb.isChecked()
            )
        if hasattr(self, "_style_preset"):
            self._current_settings["style"] = self._collect_style()
        safe = {
            k: v
            for k, v in self._current_settings.items()
            if k not in ("models", "system_prompt")
        }
        log.info(f"Settings applied: {safe}")
        self.settings_changed.emit(dict(self._current_settings))

    def get_settings(self):
        return dict(self._current_settings)

    # --- 回復原廠設定 ---
    def _on_factory_reset_clicked(self):
        box = MessageBox(t("factory_reset_title"), t("factory_reset_confirm"), self)
        if not box.exec():
            return
        lang_changed = self.restore_factory_settings()
        InfoBar.success(
            t("factory_reset_title"),
            t("factory_reset_done") + (t("factory_reset_restart_lang") if lang_changed else ""),
            parent=self, duration=4000,
        )

    def restore_factory_settings(self) -> bool:
        """Put every tunable value back to the install defaults (factory_defaults.py),
        keep the model list / connection, ASR model and audio devices, apply it at once
        and reset both window positions. Returns True when the UI language changed
        (that one needs a restart)."""
        from factory_defaults import factory_reset

        old_lang = self._current_settings.get("ui_lang")
        new = factory_reset(self._current_settings)
        self._current_settings = new
        self._load_tunable_into_widgets(new)
        self._apply_settings()  # widgets -> settings, emits settings_changed (VAD, translator, overlay style)
        _save_settings(self._current_settings)
        # Subtitle window: push the factory subtitle settings through the normal page path
        self.update_subtitle_settings(dict(new.get("subtitle_mode") or {}))
        self._subtitle_widget.get_settings()  # emits -> _on_subtitle_settings_changed -> main
        self.reset_positions.emit()
        log.info("Factory settings restored (models, ASR model and audio devices kept)")
        return new.get("ui_lang") != old_lang

    def _load_tunable_into_widgets(self, s: dict):
        """Show the values of `s` in every tunable control without triggering auto-save."""
        widgets = [
            self._asr_lang, self._whisper_pad_seconds, self._sensevoice_pad_seconds, self._ui_lang_combo,
            self._vad_mode, self._vad_threshold_slider, self._energy_slider, self._min_speech, self._max_speech,
            self._silence_mode, self._silence_duration, self._min_density_spin, self._incremental_asr_cb,
            self._interim_interval_spin, self._live_preview_cb, self._prompt_preset, self._prompt_edit,
            self._timeout_spin, self._auto_save_transcript_cb, self._style_preset,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            idx = self._asr_lang.findData(s["asr_language"])
            if idx >= 0:
                self._asr_lang.setCurrentIndex(idx)
            self._whisper_pad_seconds.setValue(s["whisper_pad_seconds"])
            self._sensevoice_pad_seconds.setValue(s["sensevoice_pad_seconds"])
            self._ui_lang_combo.setCurrentIndex({"en": 0, "zh": 1, "zh_TW": 2}.get(s["ui_lang"], 2))
            self._vad_mode.setCurrentIndex({"silero": 0, "energy": 1, "disabled": 2}.get(s["vad_mode"], 0))
            vad_pct = round(s["vad_threshold"] * 100)
            self._vad_threshold_slider.setValue(vad_pct)
            self._vad_threshold_label.setText(f"{vad_pct}%")
            energy_pm = round(s["energy_threshold"] * 1000)
            self._energy_slider.setValue(energy_pm)
            self._energy_label.setText(f"{energy_pm}\u2030")
            self._min_speech.setValue(s["min_speech_duration"])
            self._max_speech.setValue(s["max_speech_duration"])
            fixed = s["silence_mode"] == "fixed"
            self._silence_mode.setCurrentIndex(1 if fixed else 0)
            self._silence_duration.setValue(s["silence_duration"])
            self._silence_duration.setEnabled(fixed)
            self._min_density_spin.setValue(_density_percent(s["vad_min_speech_density"]))
            self._incremental_asr_cb.setChecked(s["incremental_asr"])
            self._interim_interval_spin.setValue(s["interim_interval"])
            self._interim_interval_spin.setEnabled(s["incremental_asr"])
            self._live_preview_cb.setChecked(s["live_preview"])
            self._live_preview_cb.setEnabled(s["incremental_asr"])
            self._prompt_edit.setPlainText(s["system_prompt"])
            self._prompt_preset.setCurrentIndex(_prompt_preset_index(s["system_prompt"]))
            self._timeout_spin.setValue(s["timeout"])
            self._auto_save_transcript_cb.setChecked(s["auto_save_transcript"])
            preset = s["style"].get("preset", "custom")
            if preset in self._preset_keys:
                self._style_preset.setCurrentIndex(self._preset_keys.index(preset))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._block_style_signals(True)
        try:
            self._apply_style_to_controls(s["style"])
        finally:
            self._block_style_signals(False)

    def get_active_model(self) -> dict | None:
        models = self._current_settings.get("models", [])
        idx = self._current_settings.get("active_model", 0)
        if 0 <= idx < len(models):
            return models[idx]
        return None

    def has_saved_settings(self) -> bool:
        return SETTINGS_FILE.exists()
