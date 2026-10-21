import ctypes
import logging
import os
import re
import sys
import time
from ctypes import wintypes

import psutil

import system_stats
from i18n import t, LANGUAGES
from PyQt6.QtCore import QEvent, QPoint, QPropertyAnimation, QEasingCurve, QRect, QRectF, QSize, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QCursor, QFont, QGuiApplication, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    ComboBox,
    FluentIcon as FIF,
    IconWidget,
    ToolTipFilter,
    ToolTipPosition,
    TransparentPushButton,
    TransparentToggleToolButton,
    TransparentToolButton,
    VerticalSeparator,
)

log = logging.getLogger("LiveTranslate.Overlay")

_GWL_EXSTYLE = -20
_GWL_STYLE = -16
_WS_CAPTION = 0x00C00000
_WS_SYSMENU = 0x00080000
_WS_MINIMIZEBOX = 0x00020000
_WM_NCCALCSIZE = 0x0083
_SWP_FRAMECHANGED_ONLY = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020  # no size/move/z-order/activate
_user32 = ctypes.windll.user32 if sys.platform == "win32" else None


class _MSG(ctypes.Structure):
    _fields_ = [("hWnd", wintypes.HWND), ("message", wintypes.UINT), ("wParam", wintypes.WPARAM),
                ("lParam", wintypes.LPARAM), ("time", wintypes.DWORD), ("pt", wintypes.POINT)]
_WS_EX_TRANSPARENT = 0x20

# Translucent dark card: about 85 % opaque, 12 px corners; the header only tints the card
DEFAULT_STYLE = {
    "preset": "default",
    "bg_color": "#16161d",
    "bg_opacity": 217,
    "header_color": "#ffffff",
    "header_opacity": 8,
    "border_radius": 12,
    "original_font_family": "Microsoft YaHei",
    "translation_font_family": "Microsoft YaHei",
    "original_font_size": 10,
    "translation_font_size": 15,
    "original_color": "#9a9aa8",
    "translation_color": "#ffffff",
    "timestamp_color": "#6c6c7c",
    "window_opacity": 100,
}

_BASE = DEFAULT_STYLE

STYLE_PRESETS = {
    "default": dict(_BASE),
    "transparent": {
        **_BASE,
        "preset": "transparent",
        "bg_opacity": 120,
        "header_opacity": 120,
        "window_opacity": 70,
    },
    "compact": {
        **_BASE,
        "preset": "compact",
        "original_font_size": 9,
        "translation_font_size": 11,
    },
    "light": {
        **_BASE,
        "preset": "light",
        "bg_color": "#e8e8f0",
        "bg_opacity": 230,
        "header_color": "#c8c8d8",
        "header_opacity": 220,
        "original_color": "#333333",
        "translation_color": "#111111",
        "timestamp_color": "#666688",
    },
    "dracula": {
        **_BASE,
        "preset": "dracula",
        "bg_color": "#282a36",
        "bg_opacity": 235,
        "header_color": "#44475a",
        "header_opacity": 230,
        "original_color": "#f8f8f2",
        "translation_color": "#f8f8f2",
        "timestamp_color": "#6272a4",
    },
    "nord": {
        **_BASE,
        "preset": "nord",
        "bg_color": "#2e3440",
        "bg_opacity": 235,
        "header_color": "#3b4252",
        "header_opacity": 230,
        "original_color": "#d8dee9",
        "translation_color": "#eceff4",
        "timestamp_color": "#4c566a",
    },
    "monokai": {
        **_BASE,
        "preset": "monokai",
        "bg_color": "#272822",
        "bg_opacity": 235,
        "header_color": "#3e3d32",
        "header_opacity": 230,
        "original_color": "#f8f8f2",
        "translation_color": "#f8f8f2",
        "timestamp_color": "#75715e",
    },
    "solarized": {
        **_BASE,
        "preset": "solarized",
        "bg_color": "#002b36",
        "bg_opacity": 235,
        "header_color": "#073642",
        "header_opacity": 230,
        "original_color": "#839496",
        "translation_color": "#eee8d5",
        "timestamp_color": "#586e75",
    },
    "gruvbox": {
        **_BASE,
        "preset": "gruvbox",
        "bg_color": "#282828",
        "bg_opacity": 235,
        "header_color": "#3c3836",
        "header_opacity": 230,
        "original_color": "#ebdbb2",
        "translation_color": "#fbf1c7",
        "timestamp_color": "#928374",
    },
    "tokyo_night": {
        **_BASE,
        "preset": "tokyo_night",
        "bg_color": "#1a1b26",
        "bg_opacity": 235,
        "header_color": "#24283b",
        "header_opacity": 230,
        "original_color": "#a9b1d6",
        "translation_color": "#c0caf5",
        "timestamp_color": "#565f89",
    },
    "catppuccin": {
        **_BASE,
        "preset": "catppuccin",
        "bg_color": "#1e1e2e",
        "bg_opacity": 235,
        "header_color": "#313244",
        "header_opacity": 230,
        "original_color": "#cdd6f4",
        "translation_color": "#cdd6f4",
        "timestamp_color": "#6c7086",
    },
    "one_dark": {
        **_BASE,
        "preset": "one_dark",
        "bg_color": "#282c34",
        "bg_opacity": 235,
        "header_color": "#3e4452",
        "header_opacity": 230,
        "original_color": "#abb2bf",
        "translation_color": "#e5c07b",
        "timestamp_color": "#636d83",
    },
    "everforest": {
        **_BASE,
        "preset": "everforest",
        "bg_color": "#2d353b",
        "bg_opacity": 235,
        "header_color": "#343f44",
        "header_opacity": 230,
        "original_color": "#d3c6aa",
        "translation_color": "#d3c6aa",
        "timestamp_color": "#859289",
    },
    "kanagawa": {
        **_BASE,
        "preset": "kanagawa",
        "bg_color": "#1f1f28",
        "bg_opacity": 235,
        "header_color": "#2a2a37",
        "header_opacity": 230,
        "original_color": "#dcd7ba",
        "translation_color": "#dcd7ba",
        "timestamp_color": "#54546d",
    },
}


def _hex_to_rgba(hex_color: str, opacity: int) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_STYLE_COLOR_KEYS = ("bg_color", "header_color", "original_color", "translation_color", "timestamp_color")
_STYLE_INT_RANGES = {
    "bg_opacity": (0, 255),
    "header_opacity": (0, 255),
    "border_radius": (0, 30),
    "original_font_size": (6, 48),
    "translation_font_size": (6, 48),
    "window_opacity": (20, 100),
}


def _sanitize_style(style) -> dict:
    """The style comes from user_settings.json: replace values of the wrong type or
    format with the defaults and clamp numbers, so a bad entry cannot crash the UI."""
    s = {**DEFAULT_STYLE, **(style if isinstance(style, dict) else {})}
    for key in _STYLE_COLOR_KEYS:
        if not (isinstance(s.get(key), str) and _HEX_COLOR_RE.match(s[key])):
            s[key] = DEFAULT_STYLE[key]
    for key, (lo, hi) in _STYLE_INT_RANGES.items():
        v = s.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            s[key] = DEFAULT_STYLE[key]
        else:
            s[key] = int(min(hi, max(lo, v)))
    for key in ("original_font_family", "translation_font_family"):
        if not (isinstance(s.get(key), str) and s[key].strip()):
            s[key] = DEFAULT_STYLE[key]
    return s


class ChatMessage(QWidget):
    """Single chat message widget with original + async translation."""

    _current_style = dict(DEFAULT_STYLE)
    _compact_mode = False

    def __init__(
        self,
        msg_id: int,
        timestamp: str,
        original: str,
        source_lang: str,
        asr_ms: float,
        parent=None,
    ):
        super().__init__(parent)
        self.msg_id = msg_id
        self._original = original
        self._translated = ""
        self._timestamp = timestamp
        self._source_lang = source_lang
        self._asr_ms = asr_ms
        self._translate_ms = 0.0
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 5, 10, 5)
        self._layout.setSpacing(1)

        s = self._current_style
        self._header_label = QLabel(self._build_header_html(s))
        self._header_label.setFont(
            QFont(s["original_font_family"], s["original_font_size"])
        )
        self._header_label.setTextFormat(Qt.TextFormat.RichText)
        self._header_label.setWordWrap(True)
        self._header_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._header_label)

        self._trans_label = QLabel(self._note_html(s, t("translating")))
        self._trans_label.setFont(
            QFont(s["translation_font_family"], s["translation_font_size"])
        )
        self._trans_label.setTextFormat(Qt.TextFormat.RichText)
        self._trans_label.setWordWrap(True)
        self._trans_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._trans_label)

    def _build_header_html(self, s):
        """Original line: small grey text; time, language and ASR latency as faint tags."""
        tag = f'font-size:8pt; color:{s["timestamp_color"]};'
        lang = f'<span style="{tag}">{_escape(str(self._source_lang))}</span>'
        orig = f'<span style="color:{s["original_color"]};">{_escape(self._original)}</span>'
        if self._compact_mode:
            return f"{lang}&nbsp;&nbsp;{orig}"
        return (
            f'<span style="{tag}">{_escape(str(self._timestamp))}</span>&nbsp;&nbsp;{lang}&nbsp;&nbsp;{orig}'
            f'&nbsp;&nbsp;<span style="{tag}">ASR {self._asr_ms:.0f}ms</span>'
        )

    def _translation_html(self, s, text: str, translate_ms=None):
        """Translation line: larger, white, semi-bold; TL latency as a faint tag."""
        html = f'<span style="color:{s["translation_color"]}; font-weight:600;">{_escape(text)}</span>'
        if translate_ms is not None and not self._compact_mode:
            html += (
                f'&nbsp;&nbsp;<span style="font-size:8pt; font-weight:400; color:{s["timestamp_color"]};">'
                f"TL {translate_ms:.0f}ms</span>"
            )
        return html

    @staticmethod
    def _note_html(s, text: str):
        return f'<span style="color:{s["timestamp_color"]}; font-style:italic;">{_escape(text)}</span>'

    def update_streaming(self, partial_text: str):
        """Update translation label with partial streaming text (throttled)."""
        self._pending_streaming = partial_text
        if not hasattr(self, "_streaming_timer"):
            self._streaming_timer = QTimer()
            self._streaming_timer.setSingleShot(True)
            self._streaming_timer.setInterval(50)
            self._streaming_timer.timeout.connect(self._flush_streaming)
        if not self._streaming_timer.isActive():
            self._flush_streaming()
            self._streaming_timer.start()

    def _flush_streaming(self):
        text = getattr(self, "_pending_streaming", None)
        if text is None:
            return
        self._pending_streaming = None
        self._trans_label.setText(self._translation_html(self._current_style, text))

    def set_translation(self, translated: str, translate_ms: float):
        self._translated = translated or ""
        self._translate_ms = translate_ms
        # Stop streaming throttle if active
        if hasattr(self, "_streaming_timer"):
            self._streaming_timer.stop()
            self._pending_streaming = None
        s = self._current_style
        if translated:
            self._trans_label.setText(self._translation_html(s, translated, translate_ms))
        else:
            self._trans_label.setText(self._note_html(s, t("same_language")))

    def apply_style(self, s: dict):
        self._header_label.setText(self._build_header_html(s))
        self._header_label.setFont(
            QFont(s["original_font_family"], s["original_font_size"])
        )
        self._trans_label.setFont(
            QFont(s["translation_font_family"], s["translation_font_size"])
        )
        if self._translated:
            self._trans_label.setText(self._translation_html(s, self._translated, self._translate_ms))

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #2a2a3a; color: #ddd; border: 1px solid #555; }
            QMenu::item:selected { background: #444; }
            QMenu::separator { height: 1px; background: #555; margin: 4px 0; }
        """)
        copy_orig = menu.addAction(t("copy_original"))
        copy_trans = menu.addAction(t("copy_translation"))
        copy_all = menu.addAction(t("copy_all"))
        menu.addSeparator()
        export_menu = menu.addMenu(t("export_menu"))
        export_orig = export_menu.addAction(t("export_original"))
        export_trans = export_menu.addAction(t("export_translation"))
        export_both = export_menu.addAction(t("export_all"))
        menu.addSeparator()
        clear_list = menu.addAction(t("clear_list"))
        action = menu.exec(event.globalPos())
        if action == copy_orig:
            QApplication.clipboard().setText(self._original)
        elif action == copy_trans:
            QApplication.clipboard().setText(self._translated)
        elif action == copy_all:
            QApplication.clipboard().setText(f"{self._original}\n{self._translated}")
        elif action == clear_list:
            overlay = self.window()
            if hasattr(overlay, '_on_clear'):
                overlay._on_clear()
        elif action in (export_orig, export_trans, export_both):
            mode = {export_orig: "original", export_trans: "translation", export_both: "both"}[action]
            overlay = self.window()
            if hasattr(overlay, "export_messages"):
                overlay.export_messages(mode, parent=self)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_BTN_CSS = """
    QPushButton {
        background: rgba(255,255,255,20);
        border: 1px solid rgba(255,255,255,40);
        border-radius: 3px;
        color: #aaa;
        font-size: 11px;
        padding: 0 6px;
    }
    QPushButton:hover {
        background: rgba(255,255,255,40);
        color: #ddd;
    }
"""

_BAR_CSS_TPL = """
    QProgressBar {{
        background: rgba(255,255,255,15);
        border: 1px solid rgba(255,255,255,30);
        border-radius: 3px;
        text-align: center;
        font-size: 8pt;
        color: #aaa;
    }}
    QProgressBar::chunk {{
        background: {color};
        border-radius: 2px;
    }}
"""


class MonitorBar(QWidget):
    """Compact system monitor displayed in the overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(6)

        # MIC bar (hidden when mic is disabled)
        self._mic_lbl = QLabel("MIC")
        self._mic_lbl.setFixedWidth(26)
        self._mic_lbl.setFont(QFont("Consolas", 8))
        self._mic_lbl.setStyleSheet("color: #888; background: transparent;")
        self._mic_lbl.setVisible(False)
        row1.addWidget(self._mic_lbl)

        self._mic_bar = QProgressBar()
        self._mic_bar.setRange(0, 100)
        self._mic_bar.setFixedHeight(14)
        self._mic_bar.setTextVisible(True)
        self._mic_bar.setFormat("%v%")
        self._mic_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#c586c0"))
        self._mic_bar.setVisible(False)
        row1.addWidget(self._mic_bar)

        rms_lbl = QLabel("RMS:")
        rms_lbl.setFixedWidth(26)
        rms_lbl.setFont(QFont("Consolas", 8))
        rms_lbl.setStyleSheet("color: #888; background: transparent;")
        row1.addWidget(rms_lbl)

        self._rms_bar = QProgressBar()
        self._rms_bar.setRange(0, 100)
        self._rms_bar.setFixedHeight(14)
        self._rms_bar.setTextVisible(True)
        self._rms_bar.setFormat("%v%")
        self._rms_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#4ec9b0"))
        row1.addWidget(self._rms_bar)

        vad_lbl = QLabel("VAD:")
        vad_lbl.setFixedWidth(26)
        vad_lbl.setFont(QFont("Consolas", 8))
        vad_lbl.setStyleSheet("color: #888; background: transparent;")
        row1.addWidget(vad_lbl)

        self._vad_bar = QProgressBar()
        self._vad_bar.setRange(0, 100)
        self._vad_bar.setFixedHeight(14)
        self._vad_bar.setTextVisible(True)
        self._vad_bar.setFormat("%v%")
        self._vad_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#dcdcaa"))
        row1.addWidget(self._vad_bar)

        layout.addLayout(row1)

        self._stats_label = QLabel()
        self._stats_label.setFont(QFont("Consolas", 8))
        self._stats_label.setStyleSheet("color: #888; background: transparent;")
        self._stats_label.setTextFormat(Qt.TextFormat.RichText)
        self._stats_label.setWordWrap(True)
        self._stats_label.setToolTip(t("ovl_stats_tip"))
        layout.addWidget(self._stats_label)

        # The work happens outside this process (ASR worker subprocess, Ollama), so:
        # CPU = whole machine, RAM = this process tree, GPU = whole card (system_stats)
        self._proc = psutil.Process(os.getpid())
        system_stats.system_cpu_percent()  # prime the counter
        self._gpu = system_stats.NvmlGpu(0)
        self._cpu_ram_failed = False
        self._cpu = 0
        self._ram_mb = 0.0
        self._gpu_text = "N/A"
        self._asr_device = ""
        self._asr_count = 0
        self._tl_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = 0.0

        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._update_system)
        self._sys_timer.start(1000)
        self._update_system()
        self._refresh_stats()

    def update_audio(self, rms: float, vad: float, mic_rms=None):
        self._rms_bar.setValue(min(100, int(rms * 500)))
        self._vad_bar.setValue(min(100, int(vad * 100)))
        mic_active = mic_rms is not None
        if self._mic_lbl.isVisible() != mic_active:
            self._mic_lbl.setVisible(mic_active)
            self._mic_bar.setVisible(mic_active)
        if mic_active:
            self._mic_bar.setValue(min(100, int(mic_rms * 500)))

    def update_asr_device(self, device: str):
        self._asr_device = device
        m = re.search(r"cuda:(\d+)", device or "")
        if m and int(m.group(1)) != self._gpu.index:
            self._gpu = system_stats.NvmlGpu(int(m.group(1)))
        self._refresh_stats()

    def update_pipeline_stats(
        self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0
    ):
        self._asr_count = asr_count
        self._tl_count = tl_count
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cost = cost
        self._refresh_stats()

    def _update_system(self):
        try:
            self._cpu = int(round(system_stats.system_cpu_percent()))
            self._ram_mb = system_stats.process_tree_rss_mb(self._proc)
            self._cpu_ram_failed = False
        except psutil.Error as e:
            if not self._cpu_ram_failed:  # once per failure streak, not every second
                log.warning(f"CPU/RAM stats unavailable: {e}")
            self._cpu_ram_failed = True
        self._gpu_text = system_stats.format_gpu(self._gpu.read())
        self._refresh_stats()

    def _refresh_stats(self):
        total = self._prompt_tokens + self._completion_tokens
        tokens_str = f"{total / 1000:.1f}k" if total >= 1000 else str(total)
        dev_str = ""
        if self._asr_device:
            dev_color = "#4ec9b0" if "cuda" in self._asr_device.lower() else "#dcdcaa"
            dev_str = (
                f'<span style="color:{dev_color};">{self._asr_device}</span> '
                f'<span style="color:#555;">|</span> '
            )
        cost_str = ""
        if self._cost > 0:
            from i18n import get_lang
            symbol = "¥" if get_lang() == "zh" else "$"
            cost_str = f' <span style="color:#fa5;">{symbol}{self._cost:.4f}</span>'
        self._stats_label.setText(
            f"{dev_str}"
            f'<span style="color:#6cf;">CPU</span> {self._cpu}% '
            f'<span style="color:#6cf;">RAM</span> {self._ram_mb:.0f}MB '
            f'<span style="color:#6cf;">GPU</span> {self._gpu_text} '
            f'<span style="color:#555;">|</span> '
            f'<span style="color:#8b8;">ASR</span> {self._asr_count} '
            f'<span style="color:#db8;">TL</span> {self._tl_count} '
            f'<span style="color:#c9c;">Tok</span> {tokens_str} '
            f'<span style="color:#666;">({self._prompt_tokens}\u2191{self._completion_tokens}\u2193)</span>'
            f'{cost_str}'
        )


class _DragArea(QWidget):
    """Small draggable area (title + grip)."""

    drag_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self._drag_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        if self._drag_pos:
            self._drag_pos = None
            self.drag_finished.emit()


def _set_tooltip(widget, text: str):
    """Fluent tooltip, shown only while hovering (its window stays above the topmost overlay)."""
    widget.setToolTip(text)
    widget.installEventFilter(ToolTipFilter(widget, showDelay=300, position=ToolTipPosition.BOTTOM))


def _icon_button(icon, tip: str, toggle: bool = False, checked: bool = False):
    b = TransparentToggleToolButton(icon) if toggle else TransparentToolButton(icon)
    b.setFixedSize(28, 28)
    b.setIconSize(QSize(14, 14))
    b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    if toggle:
        b.setChecked(checked)
    _set_tooltip(b, tip)
    return b


def _label_text(key: str) -> str:
    return t(key).rstrip(":：").strip()


_PAUSED_ICON_COLOR = QColor("#e5b454")  # amber play icon: "paused" must stand out
_QUIT_ICON_COLOR = QColor("#e06c6c")


class DragHandle(QWidget):
    """Top bar: row 1 = title + icon buttons, row 2 = toggle icons + model / language combos."""

    _FULL_H = 72
    _COMPACT_H = 40

    settings_clicked = pyqtSignal()
    subtitle_clicked = pyqtSignal()
    subtitle_lock_toggled = pyqtSignal(bool)
    subtitle_click_through_toggled = pyqtSignal(bool)
    subtitle_line_visibility_toggled = pyqtSignal(bool, bool)  # (show original, show translation)
    subtitle_show_previous_toggled = pyqtSignal(bool)
    subtitle_settings_requested = pyqtSignal()
    click_through_toggled = pyqtSignal(bool)
    topmost_toggled = pyqtSignal(bool)
    auto_scroll_toggled = pyqtSignal(bool)
    target_language_changed = pyqtSignal(str)
    source_language_changed = pyqtSignal(str)
    model_changed = pyqtSignal(int)
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    minimize_clicked = pyqtSignal()
    quit_clicked = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "full" or "compact"
    position_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "full"
        self._running = False
        self.setObjectName("overlayHeader")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(self._FULL_H)
        self.setStyleSheet("#overlayHeader { background: rgba(255,255,255,8); border-radius: 8px; }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # Row 1: drag title + action buttons
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(2)

        drag = _DragArea()
        drag.drag_finished.connect(self.position_changed)
        drag_layout = QHBoxLayout(drag)
        drag_layout.setContentsMargins(4, 0, 4, 0)
        drag_layout.setSpacing(6)
        title = QLabel("LiveTranslate")
        title.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        title.setStyleSheet("color: rgba(255,255,255,150); background: transparent;")
        drag_layout.addWidget(title)
        drag_layout.addStretch()
        row1.addWidget(drag, 1)

        # Checked while the subtitle window is shown (state is set by main via set_subtitle_checked)
        self._subtitle_btn = _icon_button(FIF.FONT, t("ovl_tip_subtitle"), toggle=True)
        self._subtitle_btn.clicked.connect(self.subtitle_clicked.emit)
        row1.addWidget(self._subtitle_btn)

        # Subtitle window options: lock position / click-through / shown lines / settings
        self._subtitle_menu_btn = TransparentToolButton(FIF.CHEVRON_DOWN_MED)
        self._subtitle_menu_btn.setFixedSize(18, 28)
        self._subtitle_menu_btn.setIconSize(QSize(10, 10))
        self._subtitle_menu_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        _set_tooltip(self._subtitle_menu_btn, t("subwin_menu_tip"))
        self._subtitle_menu = QMenu(self._subtitle_menu_btn)
        self._subtitle_lock_action = self._subtitle_menu.addAction(t("subwin_menu_lock"))
        self._subtitle_lock_action.setCheckable(True)
        self._subtitle_lock_action.triggered.connect(
            lambda checked: self.subtitle_lock_toggled.emit(bool(checked))
        )
        self._subtitle_ct_action = self._subtitle_menu.addAction(t("subwin_menu_click_through"))
        self._subtitle_ct_action.setCheckable(True)
        self._subtitle_ct_action.triggered.connect(
            lambda checked: self.subtitle_click_through_toggled.emit(bool(checked))
        )
        self._subtitle_menu.addSeparator()
        self._subtitle_show_orig_action = self._subtitle_menu.addAction(t("subwin_menu_show_original"))
        self._subtitle_show_orig_action.setCheckable(True)
        self._subtitle_show_orig_action.setChecked(True)
        self._subtitle_show_orig_action.triggered.connect(lambda _c: self._emit_line_visibility())
        self._subtitle_show_trans_action = self._subtitle_menu.addAction(t("subwin_menu_show_translation"))
        self._subtitle_show_trans_action.setCheckable(True)
        self._subtitle_show_trans_action.setChecked(True)
        self._subtitle_show_trans_action.triggered.connect(lambda _c: self._emit_line_visibility())
        self._subtitle_show_prev_action = self._subtitle_menu.addAction(t("subwin_menu_show_previous"))
        self._subtitle_show_prev_action.setCheckable(True)
        self._subtitle_show_prev_action.setChecked(True)
        self._subtitle_show_prev_action.triggered.connect(
            lambda checked: self.subtitle_show_previous_toggled.emit(bool(checked))
        )
        self._subtitle_menu.addSeparator()
        self._subtitle_settings_action = self._subtitle_menu.addAction(t("subwin_menu_settings"))
        self._subtitle_settings_action.triggered.connect(self.subtitle_settings_requested.emit)
        self._subtitle_menu_btn.clicked.connect(
            lambda: self._subtitle_menu.exec(
                self._subtitle_menu_btn.mapToGlobal(QPoint(0, self._subtitle_menu_btn.height()))
            )
        )
        row1.addWidget(self._subtitle_menu_btn)

        self._start_stop_btn = _icon_button(FIF.PLAY, t("ovl_tip_start"))
        self._start_stop_btn.clicked.connect(self._on_start_stop)
        row1.addWidget(self._start_stop_btn)

        self._clear_btn = _icon_button(FIF.BROOM, t("ovl_tip_clear"))
        self._clear_btn.clicked.connect(self.clear_clicked.emit)
        row1.addWidget(self._clear_btn)

        self._mode_btn = _icon_button(FIF.BACK_TO_WINDOW, t("ovl_tip_compact"))
        self._mode_btn.clicked.connect(self._toggle_mode)
        row1.addWidget(self._mode_btn)

        self._settings_btn = _icon_button(FIF.SETTING, t("ovl_tip_settings"))
        self._settings_btn.clicked.connect(self.settings_clicked.emit)
        row1.addWidget(self._settings_btn)

        self._minimize_btn = _icon_button(FIF.REMOVE, t("ovl_tip_minimize"))  # the usual "—" minimize glyph
        self._minimize_btn.clicked.connect(self.minimize_clicked.emit)
        row1.addWidget(self._minimize_btn)

        self._quit_btn = _icon_button(FIF.POWER_BUTTON, t("ovl_tip_quit"))
        self._quit_btn.setIcon(FIF.POWER_BUTTON.icon(color=_QUIT_ICON_COLOR))
        self._quit_btn.clicked.connect(self.quit_clicked.emit)
        row1.addWidget(self._quit_btn)

        outer.addLayout(row1)

        # Row 2: window toggles + model / source / target combos
        self._row2_widget = QWidget()
        row2 = QHBoxLayout(self._row2_widget)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(2)

        # Attribute names are used by main.py to sync the tray menu (setChecked / toggled)
        self._ct_check = _icon_button(FIF.TRANSPARENT, t("click_through"), toggle=True)
        self._ct_check.toggled.connect(self.click_through_toggled.emit)
        row2.addWidget(self._ct_check)

        self._topmost_check = _icon_button(FIF.PIN, t("top_most"), toggle=True, checked=True)
        self._topmost_check.toggled.connect(self.topmost_toggled.emit)
        row2.addWidget(self._topmost_check)

        self._auto_scroll = _icon_button(FIF.DOWN, t("auto_scroll"), toggle=True, checked=True)
        self._auto_scroll.toggled.connect(self.auto_scroll_toggled.emit)
        row2.addWidget(self._auto_scroll)

        sep = VerticalSeparator()
        sep.setFixedHeight(18)
        row2.addSpacing(4)
        row2.addWidget(sep)
        row2.addSpacing(4)

        def _combo(tip: str):
            c = ComboBox()
            c.setFixedHeight(28)
            c.setMinimumWidth(70)
            c.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            _set_tooltip(c, tip)
            return c

        self._model_combo = _combo(_label_text("model_label"))
        self._model_combo.currentIndexChanged.connect(self.model_changed.emit)
        row2.addWidget(self._model_combo, 3)

        self._source_lang = _combo(_label_text("source_label"))
        for code, native in LANGUAGES:
            label = t("asr_lang_auto") if code == "auto" else native
            self._source_lang.addItem(f"{code} - {label}", userData=code)
        self._source_lang.currentIndexChanged.connect(
            lambda idx: self.source_language_changed.emit(
                self._source_lang.currentData() or "auto"
            )
        )
        row2.addWidget(self._source_lang, 2)

        arrow = IconWidget(FIF.RIGHT_ARROW)
        arrow.setFixedSize(12, 12)
        row2.addWidget(arrow)

        self._target_lang = _combo(_label_text("target_label"))
        for code, native in LANGUAGES:
            if code == "auto":
                continue
            self._target_lang.addItem(f"{code} - {native}", userData=code)
        self._target_lang.currentIndexChanged.connect(
            lambda idx: self.target_language_changed.emit(
                self._target_lang.currentData() or "zh"
            )
        )
        row2.addWidget(self._target_lang, 2)

        outer.addWidget(self._row2_widget)
        self.set_running(False)

    def _on_start_stop(self):
        if self._running:
            self.stop_clicked.emit()
        else:
            self.start_clicked.emit()

    def set_target_language(self, lang: str):
        idx = self._target_lang.findData(lang)
        if idx >= 0:
            self._target_lang.blockSignals(True)
            self._target_lang.setCurrentIndex(idx)
            self._target_lang.blockSignals(False)

    def set_source_language(self, lang: str):
        idx = self._source_lang.findData(lang)
        if idx >= 0:
            self._source_lang.blockSignals(True)
            self._source_lang.setCurrentIndex(idx)
            self._source_lang.blockSignals(False)

    def set_models(self, models: list, active_index: int = 0):
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in models or []:
            if not isinstance(m, dict):
                continue
            self._model_combo.addItem(str(m.get("name", m.get("model", "?"))))
        if 0 <= active_index < self._model_combo.count():
            self._model_combo.setCurrentIndex(active_index)
        self._model_combo.blockSignals(False)

    @property
    def auto_scroll(self) -> bool:
        return self._auto_scroll.isChecked()

    def set_running(self, running: bool):
        self._running = running
        if running:
            self._start_stop_btn.setIcon(FIF.PAUSE)
            self._start_stop_btn.setToolTip(t("ovl_tip_pause"))
        else:
            self._start_stop_btn.setIcon(FIF.PLAY.icon(color=_PAUSED_ICON_COLOR))
            self._start_stop_btn.setToolTip(t("ovl_tip_start"))

    def _toggle_mode(self):
        new_mode = "compact" if self._mode == "full" else "full"
        self._apply_mode(new_mode)
        self.mode_changed.emit(new_mode)

    def _apply_mode(self, mode: str):
        self._mode = mode
        compact = mode == "compact"
        self._row2_widget.setVisible(not compact)
        self._clear_btn.setVisible(not compact)
        self._subtitle_btn.setVisible(not compact)
        self._subtitle_menu_btn.setVisible(not compact)
        self._mode_btn.setIcon(FIF.FULL_SCREEN if compact else FIF.BACK_TO_WINDOW)
        self._mode_btn.setToolTip(t("ovl_tip_full") if compact else t("ovl_tip_compact"))
        self.setFixedHeight(self._COMPACT_H if compact else self._FULL_H)

    def set_mode(self, mode: str):
        if mode != self._mode:
            self._apply_mode(mode)

    def set_subtitle_lock_state(self, locked: bool, click_through: bool):
        """Sync the dropdown's check marks from the real window state (no signals)."""
        self._subtitle_lock_action.setChecked(bool(locked))
        self._subtitle_ct_action.setChecked(bool(click_through))

    def _emit_line_visibility(self):
        self.subtitle_line_visibility_toggled.emit(
            self._subtitle_show_orig_action.isChecked(), self._subtitle_show_trans_action.isChecked()
        )

    def set_subtitle_line_visibility(self, show_original: bool, show_translation: bool):
        """Sync the show-original / show-translation check marks from the window state (no signals)."""
        self._subtitle_show_orig_action.setChecked(bool(show_original))
        self._subtitle_show_trans_action.setChecked(bool(show_translation))

    def set_subtitle_show_previous(self, show: bool):
        """Sync the show-previous check mark from the window state (no signals)."""
        self._subtitle_show_prev_action.setChecked(bool(show))

    def set_subtitle_checked(self, checked: bool):
        self._subtitle_btn.setChecked(bool(checked))


class _EdgeResizeHandle(QWidget):
    """Invisible strip along the top or bottom edge of the card (inside the shadow
    margin) that stretches the frameless window up or down. Uses the native system
    resize where the platform supports it, otherwise resizes the window itself."""

    def __init__(self, edge: Qt.Edge, parent=None):
        super().__init__(parent)
        self._edge = edge
        self._start = None  # (mouse global y, window geometry) during a manual resize
        self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))

    def edge(self) -> Qt.Edge:
        return self._edge

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        win = self.window()
        handle = win.windowHandle()
        if handle is not None and handle.startSystemResize(self._edge):
            self._start = None
        else:
            self._start = (event.globalPosition().toPoint().y(), QRect(win.geometry()))
        event.accept()

    def mouseMoveEvent(self, event):
        if self._start is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return
        y0, geo = self._start
        dy = event.globalPosition().toPoint().y() - y0
        win = self.window()
        lo, hi = win.minimumHeight(), win.maximumHeight()
        if self._edge == Qt.Edge.BottomEdge:
            h = max(lo, min(hi, geo.height() + dy))
            win.setGeometry(geo.x(), geo.y(), geo.width(), h)
        else:
            h = max(lo, min(hi, geo.height() - dy))
            win.setGeometry(geo.x(), geo.y() + geo.height() - h, geo.width(), h)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._start is not None:
            self._start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _StatusSection(QWidget):
    """Collapsible "狀態" block holding the MonitorBar (MIC/RMS/VAD bars, CPU/GPU, counters).
    Collapsed by default so the card shows only the conversation."""

    def __init__(self, monitor: "MonitorBar", parent=None):
        super().__init__(parent)
        self._monitor = monitor
        self._expanded = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._toggle_btn = TransparentPushButton(FIF.CHEVRON_RIGHT_MED, t("status_section"))
        self._toggle_btn.setFixedHeight(24)
        self._toggle_btn.setIconSize(QSize(10, 10))
        self._toggle_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._toggle_btn.clicked.connect(lambda: self.set_expanded(not self._expanded))
        row.addWidget(self._toggle_btn)
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(monitor)
        monitor.setVisible(False)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool):
        self._expanded = bool(expanded)
        self._monitor.setVisible(self._expanded)
        self._toggle_btn.setIcon(FIF.CHEVRON_DOWN_MED if self._expanded else FIF.CHEVRON_RIGHT_MED)


class SubtitleOverlay(QWidget):
    """Chat-style overlay window for displaying live transcription."""

    add_message_signal = pyqtSignal(int, str, str, str, float)
    update_translation_signal = pyqtSignal(int, str, float)
    update_streaming_signal = pyqtSignal(int, str)
    clear_signal = pyqtSignal()
    # Monitor signals (thread-safe)
    update_monitor_signal = pyqtSignal(float, float, object)
    update_stats_signal = pyqtSignal(int, int, int, int, float)
    update_asr_device_signal = pyqtSignal(str)

    settings_requested = pyqtSignal()
    target_language_changed = pyqtSignal(str)
    source_language_changed = pyqtSignal(str)
    model_switch_requested = pyqtSignal(int)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    subtitle_toggled = pyqtSignal()
    subtitle_lock_toggled = pyqtSignal(bool)
    subtitle_click_through_toggled = pyqtSignal(bool)
    subtitle_line_visibility_toggled = pyqtSignal(bool, bool)  # (show original, show translation)
    subtitle_show_previous_toggled = pyqtSignal(bool)
    subtitle_settings_requested = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "full" or "compact"
    position_changed = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._messages = {}
        self._max_messages = 50
        self._click_through = False
        self._last_tray_toggle = None
        self._height_before_compact = None
        self._mode_anim = None
        self._pos_save_timer = QTimer(self)
        self._pos_save_timer.setSingleShot(True)
        self._pos_save_timer.setInterval(500)
        self._pos_save_timer.timeout.connect(lambda: self.position_changed.emit())
        self._last_saved_geo = None
        self._card_radius = DEFAULT_STYLE["border_radius"]
        self._setup_ui()

        self.add_message_signal.connect(self._on_add_message)
        self.update_translation_signal.connect(self._on_update_translation)
        self.update_streaming_signal.connect(self._on_update_streaming)
        self.clear_signal.connect(self._on_clear)
        self.update_monitor_signal.connect(self._on_update_monitor)
        self.update_stats_signal.connect(self._on_update_stats)
        self.update_asr_device_signal.connect(self._on_update_asr_device)

    def _setup_ui(self):
        # A normal window (not Tool): it has a taskbar button from the start. Windows only
        # iconifies this frameless window with the minimize-button hint.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.setWindowTitle("LiveTranslate")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        width = 620
        height = 500
        x = geo.right() - width - 20
        y = geo.bottom() - height - 60
        self.setGeometry(x, y, width, height)
        self.setMinimumSize(480, 200)

        main_layout = QVBoxLayout(self)
        # Room around the card for the soft shadow painted in paintEvent
        main_layout.setContentsMargins(self._SHADOW, self._SHADOW, self._SHADOW, self._SHADOW)
        main_layout.setSpacing(0)

        # The card. Its style sheet is scoped to #overlayCard so it does not cascade into
        # the Fluent widgets inside (an unscoped "background" rule would restyle all of them).
        self._container = QWidget()
        self._container.setObjectName("overlayCard")
        self._container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(6, 6, 6, 4)
        container_layout.setSpacing(2)

        # Drag handle
        self._handle = DragHandle()
        self._handle.settings_clicked.connect(self.settings_requested.emit)
        self._handle.subtitle_clicked.connect(self.subtitle_toggled.emit)
        self._handle.subtitle_lock_toggled.connect(self.subtitle_lock_toggled.emit)
        self._handle.subtitle_click_through_toggled.connect(self.subtitle_click_through_toggled.emit)
        self._handle.subtitle_line_visibility_toggled.connect(self.subtitle_line_visibility_toggled.emit)
        self._handle.subtitle_show_previous_toggled.connect(self.subtitle_show_previous_toggled.emit)
        self._handle.subtitle_settings_requested.connect(self.subtitle_settings_requested.emit)
        self._handle.click_through_toggled.connect(self._set_click_through)
        self._handle.topmost_toggled.connect(self._set_topmost)
        self._handle.target_language_changed.connect(self.target_language_changed.emit)
        self._handle.source_language_changed.connect(self.source_language_changed.emit)
        self._handle.model_changed.connect(self.model_switch_requested.emit)
        self._handle.start_clicked.connect(self.start_requested.emit)
        self._handle.stop_clicked.connect(self.stop_requested.emit)
        self._handle.clear_clicked.connect(self._on_clear)
        self._handle.quit_clicked.connect(self.quit_requested.emit)
        self._handle.minimize_clicked.connect(self.minimize_to_taskbar)
        self._handle.mode_changed.connect(self._on_mode_changed)
        self._handle.position_changed.connect(self.position_changed)
        container_layout.addWidget(self._handle)

        # Monitor bar inside the collapsible status section (collapsed by default)
        self._monitor = MonitorBar()
        self._status = _StatusSection(self._monitor)
        container_layout.addWidget(self._status)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                width: 6px; background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,60); border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self._msg_container = QWidget()
        self._msg_container.setStyleSheet("background: transparent;")
        self._msg_layout = QVBoxLayout(self._msg_container)
        self._msg_layout.setContentsMargins(0, 0, 0, 0)
        self._msg_layout.setSpacing(2)
        self._msg_layout.addStretch()

        self._scroll.setWidget(self._msg_container)
        self._scroll.viewport().setStyleSheet("background: transparent;")
        container_layout.addWidget(self._scroll)

        grip_row = QHBoxLayout()
        grip_row.addStretch()
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(16, 16)
        self._grip.setStyleSheet("background: transparent;")
        grip_row.addWidget(self._grip)
        container_layout.addLayout(grip_row)

        main_layout.addWidget(self._container)
        self._apply_card_style(_sanitize_style(DEFAULT_STYLE))

        # Stretch the window up / down by its top or bottom edge
        self._top_resize = _EdgeResizeHandle(Qt.Edge.TopEdge, self)
        self._bottom_resize = _EdgeResizeHandle(Qt.Edge.BottomEdge, self)
        self._place_resize_handles()

        self._ct_timer = QTimer(self)
        self._ct_timer.timeout.connect(self._check_click_through)
        self._ct_timer.start(50)

    _SHADOW = 10  # px of transparent margin around the card for the shadow

    def paintEvent(self, event):
        """Soft shadow around the card, drawn as thin rings outside it. Cheaper than a
        QGraphicsDropShadowEffect, which would re-render the whole message list through
        an offscreen pixmap on every streaming update."""
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Near-invisible fill: fully transparent pixels of a translucent window do not
        # receive the mouse on Windows, and the margin hosts the resize handles
        p.fillRect(self.rect(), QColor(0, 0, 0, 1))
        card = QRectF(self._container.geometry()).translated(0, 2)
        r = float(self._card_radius)
        n = self._SHADOW
        for i in range(1, n + 1):
            outer = QPainterPath()
            outer.addRoundedRect(card.adjusted(-i, -i, i, i), r + i, r + i)
            inner = QPainterPath()
            inner.addRoundedRect(card.adjusted(-(i - 1), -(i - 1), i - 1, i - 1), r + i - 1, r + i - 1)
            alpha = int(46 * (1 - i / (n + 1)) ** 2)
            p.fillPath(outer.subtracted(inner), QColor(0, 0, 0, alpha))
        p.end()

    def _schedule_pos_save(self):
        if not self.isVisible():
            return
        geo = (self.x(), self.y(), self.width(), self.height())
        if geo != self._last_saved_geo:
            self._last_saved_geo = geo
            self._pos_save_timer.start()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._schedule_pos_save()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_resize_handles()
        self._schedule_pos_save()

    def _place_resize_handles(self):
        if not hasattr(self, "_top_resize"):
            return
        m = self._SHADOW
        band = m + 4  # the shadow margin plus a few px of the card's own padding
        card_w = max(0, self.width() - 2 * m)
        self._top_resize.setGeometry(m, 0, card_w, band)
        self._bottom_resize.setGeometry(m, self.height() - band, card_w, band)
        self._top_resize.raise_()
        self._bottom_resize.raise_()

    def set_running(self, running: bool):
        self._handle.set_running(running)

    def _set_topmost(self, enabled: bool):
        flags = self.windowFlags()
        if enabled:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def minimize_to_taskbar(self):
        """Minimize to the Windows taskbar (animated, see _add_window_animation)."""
        if self.isMinimized():
            return
        # This translucent (layered) window is drawn back at its old place by any repaint
        # while minimized, so painting stops until it is restored (see changeEvent)
        self.setUpdatesEnabled(False)
        self.showMinimized()

    def restore_window(self):
        """Bring the window back from the taskbar or from being hidden (tray menu)."""
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()

    def toggle_from_tray(self):
        """Tray icon left click: minimize a shown window, bring back a minimized or hidden one.
        A second click within the double-click interval is ignored, so a double click
        does not minimize and restore at once."""
        now = time.monotonic()
        if (self._last_tray_toggle is not None
                and now - self._last_tray_toggle < QApplication.doubleClickInterval() / 1000):
            return
        self._last_tray_toggle = now
        if self.isVisible() and not self.isMinimized():
            self.minimize_to_taskbar()
        else:
            self.restore_window()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() != QEvent.Type.WindowStateChange:
            return
        if self.isMinimized():
            # also minimized from its taskbar button: no repaints, see minimize_to_taskbar
            self.setUpdatesEnabled(False)
            return
        if not self.updatesEnabled():
            self.setUpdatesEnabled(True)

    def showEvent(self, event):
        super().showEvent(event)
        self._add_window_animation()

    def _add_window_animation(self):
        """Windows plays the minimize / restore animation only for windows with a caption and
        a minimize box. Qt gives this frameless window neither and resets the style whenever
        the flags change (top-most toggle), so they are added on every show; nativeEvent keeps
        the caption from taking any room."""
        if _user32 is None or QGuiApplication.platformName() != "windows":
            return  # offscreen tests: winId() is not an HWND
        hwnd = int(self.winId())
        style = _user32.GetWindowLongW(hwnd, _GWL_STYLE)
        wanted = style | _WS_CAPTION | _WS_SYSMENU | _WS_MINIMIZEBOX
        if wanted != style:
            _user32.SetWindowLongW(hwnd, _GWL_STYLE, wanted)
            _user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, _SWP_FRAMECHANGED_ONLY)

    def nativeEvent(self, eventType, message):
        # The whole window is client area: no system title bar from the caption style above.
        # Do not call super() here: PyQt6 crashes on it (access violation).
        msg = _MSG.from_address(int(message))
        if msg.message == _WM_NCCALCSIZE and msg.wParam:
            return True, 0
        return False, 0

    def _set_click_through(self, enabled: bool):
        self._click_through = enabled
        if not enabled:
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            if style & _WS_EX_TRANSPARENT:
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style & ~_WS_EX_TRANSPARENT
                )

    def _check_click_through(self):
        if not self._click_through:
            return
        cursor = QCursor.pos()
        local = self.mapFromGlobal(cursor)
        hwnd = int(self.winId())
        style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)

        scroll_top = self._scroll.mapTo(self, QPoint(0, 0)).y()
        in_header = 0 <= local.x() <= self.width() and 0 <= local.y() < scroll_top

        if in_header:
            if style & _WS_EX_TRANSPARENT:
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style & ~_WS_EX_TRANSPARENT
                )
        else:
            if not (style & _WS_EX_TRANSPARENT):
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style | _WS_EX_TRANSPARENT
                )

    def _on_mode_changed(self, mode: str):
        compact = mode == "compact"
        self._status.setVisible(not compact)
        ChatMessage._compact_mode = compact
        s = ChatMessage._current_style
        for msg in self._messages.values():
            msg.apply_style(s)
        self.mode_changed.emit(mode)

        # Animate window height
        if self._mode_anim and self._mode_anim.state() != QPropertyAnimation.State.Stopped:
            self._mode_anim.stop()

        if compact:
            self._height_before_compact = self.height()
            target_h = self.minimumHeight()
        else:
            target_h = self._height_before_compact or 500

        actual_h = self.frameGeometry().height()
        if abs(actual_h - target_h) < 10:
            self.resize(self.width(), target_h)
        else:
            anim = QPropertyAnimation(self, b"size")
            anim.setDuration(200)
            anim.setStartValue(QSize(self.width(), actual_h))
            anim.setEndValue(QSize(self.width(), target_h))
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._mode_anim = anim
            anim.start()

    def set_mode(self, mode: str):
        self._handle.set_mode(mode)

    def set_subtitle_checked(self, checked: bool):
        self._handle.set_subtitle_checked(checked)

    def set_subtitle_lock_state(self, locked: bool, click_through: bool):
        self._handle.set_subtitle_lock_state(locked, click_through)

    def set_subtitle_line_visibility(self, show_original: bool, show_translation: bool):
        self._handle.set_subtitle_line_visibility(show_original, show_translation)

    def set_subtitle_show_previous(self, show: bool):
        self._handle.set_subtitle_show_previous(show)

    @pyqtSlot(float, float, object)
    def _on_update_monitor(self, rms: float, vad_conf: float, mic_rms):
        self._monitor.update_audio(rms, vad_conf, mic_rms)

    @pyqtSlot(int, int, int, int, float)
    def _on_update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost):
        self._monitor.update_pipeline_stats(
            asr_count, tl_count, prompt_tokens, completion_tokens, cost
        )

    @pyqtSlot(str)
    def _on_update_asr_device(self, device: str):
        self._monitor.update_asr_device(device)

    @pyqtSlot(int, str, str, str, float)
    def _on_add_message(self, msg_id, timestamp, original, source_lang, asr_ms):
        msg = ChatMessage(msg_id, timestamp, original, source_lang, asr_ms)
        self._messages[msg_id] = msg
        self._msg_layout.addWidget(msg)

        if len(self._messages) > self._max_messages:
            oldest_id = min(self._messages.keys())
            old_msg = self._messages.pop(oldest_id)
            self._msg_layout.removeWidget(old_msg)
            old_msg.deleteLater()

        QTimer.singleShot(50, self._scroll_to_bottom)

    @pyqtSlot(int, str, float)
    def _on_update_translation(self, msg_id, translated, translate_ms):
        msg = self._messages.get(msg_id)
        if msg:
            msg.set_translation(translated, translate_ms)
            QTimer.singleShot(50, self._scroll_to_bottom)

    def _on_update_streaming(self, msg_id, partial_text):
        msg = self._messages.get(msg_id)
        if msg:
            msg.update_streaming(partial_text)

    @pyqtSlot()
    def _on_clear(self):
        for msg in self._messages.values():
            self._msg_layout.removeWidget(msg)
            msg.deleteLater()
        self._messages.clear()

    def _scroll_to_bottom(self):
        if not self._handle.auto_scroll:
            return
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def apply_style(self, style: dict):
        style = style if isinstance(style, dict) else {}
        s = {**DEFAULT_STYLE, **style}
        # Migrate old single font_family to split fields
        if "font_family" in s and "original_font_family" not in style:
            s["original_font_family"] = s["font_family"]
            s["translation_font_family"] = s["font_family"]
        s = _sanitize_style(s)
        self._apply_card_style(s)
        # Window opacity
        self.setWindowOpacity(s["window_opacity"] / 100.0)
        # Update all existing messages
        ChatMessage._current_style = s
        for msg in self._messages.values():
            msg.apply_style(s)

    def _apply_card_style(self, s: dict):
        """Card + header backgrounds, scoped by object name (see _setup_ui)."""
        self._card_radius = s["border_radius"]
        bg_rgba = _hex_to_rgba(s["bg_color"], s["bg_opacity"])
        self._container.setStyleSheet(
            f"#overlayCard {{ background-color: {bg_rgba}; border-radius: {s['border_radius']}px;"
            f" border: 1px solid rgba(255,255,255,18); }}"
        )
        hdr_rgba = _hex_to_rgba(s["header_color"], s["header_opacity"])
        self._handle.setStyleSheet(
            f"#overlayHeader {{ background: {hdr_rgba}; border-radius: {max(4, s['border_radius'] - 4)}px; }}"
        )
        self.update()

    def export_messages(self, mode: str, parent=None):
        """Export captured messages to a .txt file.

        mode: "original" | "translation" | "both"
        """
        if not self._messages:
            QMessageBox.information(parent or self, "LiveTranslate", t("export_empty"))
            return

        from datetime import datetime
        suffix = {"original": "original", "translation": "translation", "both": "all"}.get(mode, "all")
        default_name = f"livetrans_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}.txt"
        path, _ = QFileDialog.getSaveFileName(
            parent or self,
            t("export_dialog_title"),
            default_name,
            t("export_filter"),
        )
        if not path:
            return

        lines = []
        for msg_id in sorted(self._messages.keys()):
            msg = self._messages[msg_id]
            ts = msg._timestamp
            orig = msg._original or ""
            trans = msg._translated or ""
            if mode == "original":
                lines.append(f"[{ts}] {orig}")
            elif mode == "translation":
                if trans:
                    lines.append(f"[{ts}] {trans}")
            else:
                lines.append(f"[{ts}] {orig}")
                if trans:
                    lines.append(f"  -> {trans}")
                lines.append("")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines).rstrip() + "\n")
        except OSError as e:
            QMessageBox.critical(
                parent or self,
                "LiveTranslate",
                t("export_failed").format(error=str(e)),
            )

    # Thread-safe public API
    def add_message(self, msg_id, timestamp, original, source_lang, asr_ms):
        self.add_message_signal.emit(msg_id, timestamp, original, source_lang, asr_ms)

    def update_translation(self, msg_id, translated, translate_ms):
        self.update_translation_signal.emit(msg_id, translated, translate_ms)

    def update_streaming(self, msg_id, partial_text):
        self.update_streaming_signal.emit(msg_id, partial_text)

    def update_monitor(self, rms, vad_conf, mic_rms=None):
        self.update_monitor_signal.emit(rms, vad_conf, mic_rms)

    def update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0):
        self.update_stats_signal.emit(
            asr_count, tl_count, prompt_tokens, completion_tokens, cost
        )

    def update_asr_device(self, device: str):
        self.update_asr_device_signal.emit(device)

    def set_target_language(self, lang: str):
        self._handle.set_target_language(lang)

    def set_source_language(self, lang: str):
        self._handle.set_source_language(lang)

    def set_models(self, models: list, active_index: int = 0):
        self._handle.set_models(models, active_index)

    def clear(self):
        self.clear_signal.emit()
