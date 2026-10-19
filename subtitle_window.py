"""
Subtitle window - clean text-only window for OBS capture.
Uses QPainterPath for outlined text rendering.

Usage:
  - Left-drag (or middle-drag) to move; drag the right edge / bottom-right corner to change width
  - Double-click to lock (enables click-through); unlock via tray menu "Subtitle Click-through"
  - Configure via tray menu → Subtitle Mode → Settings
  - OBS: Window Capture → select "LiveTranslate Subtitle" → check "Allow Transparency"
"""

import ctypes
import time
from pathlib import Path

import json

from PyQt6.QtCore import (
    Qt, QPoint, QRect, pyqtSignal, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QParallelAnimationGroup, QEasingCurve, QTimer,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QPixmap,
)
from PyQt6.QtWidgets import QApplication, QMenu, QWidget, QVBoxLayout

from i18n import t

# Win32 extended window style bits for OS-level mouse click-through (same
# mechanism as the main overlay): a window with WS_EX_TRANSPARENT passes clicks
# to whatever is behind it, so it never blocks the video player / browser.
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x20


def _resolve_image_path(path: str) -> str:
    """Resolve image path (relative to project dir or absolute)."""
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        return str(p) if p.exists() else ""
    resolved = Path(__file__).parent / p
    return str(resolved) if resolved.exists() else ""

# Default subtitle window settings
DEFAULT_SUBTITLE_WIN_SETTINGS = {
    "enabled": False,
    "sentences": 1,
    "window_width": 1000,
    "line_spacing": 8,
    "bg_color": "#000000",
    "bg_opacity": 76,
    "bg_image": "",
    "border_radius": 8,
    "auto_hide_timeout": 5,
    "auto_hide_animation": "fade",
    "auto_hide_duration": 300,
    # Minimum time a sentence stays before the next one may replace it (ms).
    # Lower = subtitles keep up with fast talkers; pair with "sentences": 2.
    "min_display_ms": 1500,
    "click_through": False,
    # Lock position/size without click-through: clicks still hit the window, but it cannot be dragged or resized
    "locked": False,
    "lines": [
        {
            "type": "original",
            "enabled": True,
            "font_family": "Microsoft YaHei",
            "font_size": 24,
            "color": "#FFFFFF",
            "opacity": 255,
            "outline_enabled": True,
            "outline_color": "#000000",
            "outline_width": 2,
            "align": "center",
            "bg_image": "",
            "entry_animation": "none",
            "exit_animation": "none",
            "animation_duration": 300,
        },
        {
            "type": "translation",
            "lang": "zh",
            "enabled": True,
            "font_family": "Microsoft YaHei",
            "font_size": 28,
            "color": "#FFD700",
            "opacity": 255,
            "outline_enabled": True,
            "outline_color": "#000000",
            "outline_width": 2,
            "align": "center",
            "bg_image": "",
            "entry_animation": "none",
            "exit_animation": "none",
            "animation_duration": 300,
        },
    ],
}


def _merge_settings(base, override):
    result = {**base}
    for k, v in (override or {}).items():
        if k == "lines" and isinstance(v, list):
            result["lines"] = v
        else:
            result[k] = v
    return result


def _hex_to_rgba(hex_color: str, opacity: int) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


class _SubtitleTextWidget(QWidget):
    """Renders outlined text using QPainterPath, with automatic word-wrap.
    Supports entry/exit animations via custom properties.
    """

    height_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._wrapped_lines = []
        self._font = QFont("Microsoft YaHei", 24)
        self._color = QColor(255, 255, 255)
        self._outline_enabled = True
        self._outline_color = QColor(0, 0, 0)
        self._outline_width = 2
        self._align = "center"
        self._bg_pixmap = None
        self._text_cache = None
        self._last_width = 0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Animation state
        self._content_opacity_val = 1.0
        self._slide_offset_x_val = 0.0
        self._slide_offset_y_val = 0.0
        self._entry_animation = "none"
        self._exit_animation = "none"
        self._animation_duration = 300
        self._anim_group = None
        self._pending_text = None

    # --- pyqtProperty for content opacity ---
    def _get_content_opacity(self):
        return self._content_opacity_val

    def _set_content_opacity(self, val):
        self._content_opacity_val = val
        self.update()

    content_opacity = pyqtProperty(float, _get_content_opacity, _set_content_opacity)

    # --- pyqtProperty for slide offsets ---
    def _get_slide_offset_x(self):
        return self._slide_offset_x_val

    def _set_slide_offset_x(self, val):
        self._slide_offset_x_val = val
        self.update()

    slide_offset_x = pyqtProperty(float, _get_slide_offset_x, _set_slide_offset_x)

    def _get_slide_offset_y(self):
        return self._slide_offset_y_val

    def _set_slide_offset_y(self, val):
        self._slide_offset_y_val = val
        self.update()

    slide_offset_y = pyqtProperty(float, _get_slide_offset_y, _set_slide_offset_y)

    def set_config(self, cfg: dict):
        self._font = QFont(cfg.get("font_family", "Microsoft YaHei"), cfg.get("font_size", 24))
        c = QColor(cfg.get("color", "#FFFFFF"))
        c.setAlpha(cfg.get("opacity", 255))
        self._color = c
        self._outline_enabled = cfg.get("outline_enabled", True)
        self._outline_color = QColor(cfg.get("outline_color", "#000000"))
        self._outline_width = cfg.get("outline_width", 2)
        self._align = cfg.get("align", "center")
        resolved = _resolve_image_path(cfg.get("bg_image", ""))
        self._bg_pixmap = QPixmap(resolved) if resolved else None
        self._entry_animation = cfg.get("entry_animation", "none")
        self._exit_animation = cfg.get("exit_animation", "none")
        self._animation_duration = cfg.get("animation_duration", 300)
        self._text_cache = None
        self._update_height()
        self.update()

    def set_text(self, text: str):
        if self._text and text != self._text and self._exit_animation != "none":
            self._pending_text = text
            self._stop_all_animations()
            self.animate_out(callback=self._apply_pending_text)
            return

        self._apply_text_immediate(text)

    def _apply_pending_text(self):
        text = getattr(self, "_pending_text", "")
        self._pending_text = None
        self._apply_text_immediate(text)

    def _apply_text_immediate(self, text: str):
        # Stop any running animations and reset to final state
        self._stop_all_animations()
        self._content_opacity_val = 1.0
        self._slide_offset_x_val = 0.0
        self._slide_offset_y_val = 0.0
        self._pending_text = None

        self._text = text
        self._text_cache = None
        self._update_height()
        self.update()
        self.height_changed.emit()

        if text:
            self.animate_in()

    def _stop_all_animations(self):
        if self._anim_group and self._anim_group.state() != self._anim_group.State.Stopped:
            self._anim_group.stop()
        self._anim_group = None

    def animate_in(self):
        anim_type = self._entry_animation
        if anim_type == "none":
            self._content_opacity_val = 1.0
            self._slide_offset_x_val = 0.0
            self._slide_offset_y_val = 0.0
            self.update()
            return

        dur = self._animation_duration
        group = QParallelAnimationGroup(self)

        # Opacity animation (all types fade in)
        opacity_anim = QPropertyAnimation(self, b"content_opacity", self)
        opacity_anim.setDuration(dur)
        opacity_anim.setStartValue(0.0)
        opacity_anim.setEndValue(1.0)
        opacity_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        group.addAnimation(opacity_anim)

        w = self.width() or 200
        h = self.height() or 40

        if anim_type == "slide_left":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(dur)
            slide.setStartValue(float(-w))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_right":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(dur)
            slide.setStartValue(float(w))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_up":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(dur)
            slide.setStartValue(float(h))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_down":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(dur)
            slide.setStartValue(float(-h))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)

        self._content_opacity_val = 0.0
        self.update()
        self._anim_group = group
        group.start()

    def animate_out(self, callback=None, anim_type=None, duration=None):
        if anim_type is None:
            anim_type = self._exit_animation
        if duration is None:
            duration = self._animation_duration
        if anim_type == "none":
            self._content_opacity_val = 0.0
            self.update()
            if callback:
                callback()
            return

        self._stop_all_animations()

        group = QParallelAnimationGroup(self)

        opacity_anim = QPropertyAnimation(self, b"content_opacity", self)
        opacity_anim.setDuration(duration)
        opacity_anim.setStartValue(self._content_opacity_val)
        opacity_anim.setEndValue(0.0)
        opacity_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        group.addAnimation(opacity_anim)

        w = self.width() or 200
        h = self.height() or 40

        if anim_type == "slide_left":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(-w))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_right":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(w))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_up":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(-h))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_down":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(h))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)

        if callback:
            group.finished.connect(callback)

        self._anim_group = group
        group.start()

    def split_text(self, text: str) -> list:
        """Split text into segments that fit within available width."""
        fm = QFontMetrics(self._font)
        ow = self._outline_width if self._outline_enabled else 0
        avail_w = self.width() - ow * 2
        if avail_w <= 0 or fm.horizontalAdvance(text) <= avail_w:
            return [text]

        segments = []
        while text:
            if fm.horizontalAdvance(text) <= avail_w:
                segments.append(text)
                break

            best = 0
            for i in range(1, len(text) + 1):
                if fm.horizontalAdvance(text[:i]) > avail_w:
                    break
                best = i
            if best == 0:
                best = 1

            # Prefer breaking at word/punctuation boundary
            break_at = best
            for j in range(best - 1, max(best // 2, 0), -1):
                if text[j] in ' ,，。、!！?？;；:：.':
                    break_at = j + 1
                    break

            segments.append(text[:break_at].rstrip())
            text = text[break_at:].lstrip()

        return segments or [text]

    def _rewrap(self):
        """Recalculate wrapped lines from current text."""
        if not self._text:
            self._wrapped_lines = []
        else:
            self._wrapped_lines = self.split_text(self._text)

    def desired_height(self) -> int:
        fm = QFontMetrics(self._font)
        ow = self._outline_width if self._outline_enabled else 0
        n = max(len(self._wrapped_lines), 1)
        return fm.lineSpacing() * n + ow * 2 + 4

    def _update_height(self):
        self._rewrap()
        self.setFixedHeight(self.desired_height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        if w != self._last_width:
            self._last_width = w
            self._rewrap()
            self._text_cache = None
            new_h = self.desired_height()
            if new_h != self.height():
                self.setFixedHeight(new_h)
                # Width changed the wrap count: let the window re-fit its height
                self.height_changed.emit()

    def _render_text_pixmap(self):
        lines = self._wrapped_lines or [self._text]
        w = self.width()
        h = self.desired_height()
        if w <= 0 or h <= 0:
            self._text_cache = None
            return

        dpr = self.devicePixelRatioF()
        pw, ph = int(w * dpr), int(h * dpr)
        if pw <= 0 or ph <= 0:
            self._text_cache = None
            return

        pix = QPixmap(pw, ph)
        pix.setDevicePixelRatio(dpr)
        pix.fill(QColor(0, 0, 0, 0))

        painter = QPainter(pix)
        if not painter.isActive():
            self._text_cache = None
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        fm = QFontMetrics(self._font)
        ow = self._outline_width if self._outline_enabled else 0
        y = ow + fm.ascent()

        path = QPainterPath()
        for line in lines:
            text_w = fm.horizontalAdvance(line)
            if self._align == "center":
                lx = (w - text_w) / 2
            elif self._align == "right":
                lx = w - text_w - ow
            else:
                lx = ow
            path.addText(lx, y, self._font, line)
            y += fm.lineSpacing()

        if self._outline_enabled and self._outline_width > 0:
            pen = QPen(self._outline_color, self._outline_width * 2,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._color))
        painter.drawPath(path)
        painter.end()

        self._text_cache = pix

    def paintEvent(self, event):
        if not self._text:
            return

        if self._text_cache is None:
            self._render_text_pixmap()
        if self._text_cache is None:
            return

        painter = QPainter(self)
        painter.setOpacity(self._content_opacity_val)

        if self._bg_pixmap and not self._bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), self._bg_pixmap)

        painter.drawPixmap(
            int(self._slide_offset_x_val),
            int(self._slide_offset_y_val),
            self._text_cache,
        )

        painter.end()


class _EditFrame(QWidget):
    """Dashed border + corner grip drawn over the subtitle window while it is
    unlocked and hovered, so the user can see it is draggable / resizable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 200), 2, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 6, 6)
        # Corner grip: three short diagonal lines
        p.setPen(QPen(QColor(255, 255, 255, 230), 2))
        w, h = self.width(), self.height()
        for i in (4, 9, 14):
            p.drawLine(w - 2 - i, h - 2, w - 2, h - 2 - i)
        p.end()


class SubtitleWindow(QWidget):
    """Clean text-only subtitle window for OBS capture.

    Middle-click anywhere to drag. Window width is fixed (set in settings),
    height auto-fits to text content.
    """

    update_text_signal = pyqtSignal(str, str)  # original, translations_json
    position_changed = pyqtSignal()
    window_closed = pyqtSignal()
    # Emitted when the user double-clicks to lock the window (main syncs tray/settings)
    click_through_requested = pyqtSignal(bool)
    # Right-click menu: lock position toggle and open settings (main syncs state)
    lock_requested = pyqtSignal(bool)
    settings_requested = pyqtSignal()
    # Live preview of the sentence being spoken: (original, provisional translation, flags)
    live_signal = pyqtSignal(str, str, int, int)

    _EDGE = 12       # px zone on the right edge used for width resizing
    _CORNER = 18     # px zone at the bottom-right corner
    _MIN_WIDTH = 200
    _MAX_WIDTH = 3840

    def __init__(self, settings=None):
        super().__init__()
        self._settings = _merge_settings(DEFAULT_SUBTITLE_WIN_SETTINGS, settings)
        self._text_widgets = []
        self._sentences = []  # [(original, {lang: text, ...}), ...]
        self._drag_pos = None
        self._resize_start = None  # (mouse x, width) while dragging the right edge
        self._hover = False
        self._bg_pixmap = None
        self._click_through = bool(self._settings.get("click_through", False))
        self._locked = bool(self._settings.get("locked", False))
        # Qt clears the extended style on show/raise/flag changes, so re-assert it
        # on a timer rather than only once.
        self._ct_timer = QTimer(self)
        self._ct_timer.setInterval(500)
        self._ct_timer.timeout.connect(self._apply_click_through)
        # Auto-hide state
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self._on_auto_hide_timeout)
        self._is_hidden_by_timeout = False
        # Pending overflow segments for delayed insertion
        self._pending_segment_timers = []
        # Minimum display time: queue rapid updates instead of replacing instantly
        self._last_insert_time = 0.0
        self._min_display_ms = int(self._settings.get("min_display_ms", 1500))  # minimum ms before a sentence can be replaced
        self._height_anim = None
        # Live (unfinished) sentence shown after the committed ones
        self._live = None
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(4000)  # stale preview safety clear
        self._live_seq_shown = 0
        self._live_timer.timeout.connect(self._on_live_timeout)

        self._setup_ui()
        self.update_text_signal.connect(self._on_update_text)
        self.live_signal.connect(self._on_live)

    @staticmethod
    def _is_pos_visible(x, y, margin=50):
        for screen in QApplication.screens():
            geo = screen.availableGeometry()
            if geo.left() <= x + margin and x < geo.right() and geo.top() <= y + margin and y < geo.bottom():
                return True
        return False

    def _clamp_to_screen(self):
        x, y = self.x(), self.y()
        if self._is_pos_visible(x, y):
            return
        screen = QApplication.screenAt(QPoint(x, y))
        if screen is None:
            screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        nx = max(geo.left(), min(x, geo.right() - self.width()))
        ny = max(geo.top(), min(y, geo.bottom() - self.height()))
        self.move(nx, ny)

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowTitle("LiveTranslate Subtitle")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        s = self._settings
        w = s.get("window_width", 1000)
        saved_x = s.get("window_x")
        saved_y = s.get("window_y")
        if saved_x is not None and saved_y is not None:
            if self._is_pos_visible(saved_x, saved_y):
                self.move(saved_x, saved_y)
            else:
                self.move(100, 100)
        else:
            self.move(100, 100)
        self.setFixedWidth(w)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # Content area
        self._content = QWidget()
        self._content.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(16, 8, 16, 8)
        self._content_layout.setSpacing(s.get("line_spacing", 8))

        self._rebuild_text_widgets()

        self._main_layout.addWidget(self._content)

        # Dashed frame + corner grip shown while hovering an unlocked window
        self._edit_frame = _EditFrame(self)
        self._edit_frame.hide()
        self.setMouseTracking(True)

        self._apply_background()
        self._fit_height_animated()
        self.set_click_through(self._click_through)

    def _rebuild_text_widgets(self):
        for w in self._text_widgets:
            self._content_layout.removeWidget(w)
            w.deleteLater()
        self._text_widgets = []

        for line_cfg in self._settings.get("lines", []):
            if not line_cfg.get("enabled", True):
                continue
            tw = _SubtitleTextWidget()
            tw.set_config(line_cfg)
            tw.height_changed.connect(self._fit_height_animated)
            self._text_widgets.append(tw)
            self._content_layout.addWidget(tw)

    def _apply_background(self):
        s = self._settings
        resolved = _resolve_image_path(s.get("bg_image", ""))
        if resolved:
            self._bg_pixmap = QPixmap(resolved)
            self._content.setStyleSheet("background: transparent;")
        else:
            self._bg_pixmap = None
            color = s.get("bg_color", "#000000")
            opacity = s.get("bg_opacity", 0)
            if opacity == 0:
                self._content.setStyleSheet("background: transparent;")
            else:
                rgba = _hex_to_rgba(color, opacity)
                radius = s.get("border_radius", 8)
                self._content.setStyleSheet(f"background: {rgba}; border-radius: {radius}px;")
        self.update()

    def _calc_target_height(self):
        margins = self._content_layout.contentsMargins()
        spacing = self._content_layout.spacing()
        total = margins.top() + margins.bottom()
        for i, tw in enumerate(self._text_widgets):
            total += tw.desired_height()
            if i > 0:
                total += spacing
        return max(total, 20)

    def _fit_height_snap(self):
        new_h = self._calc_target_height()
        old_h = self.height()
        if new_h == old_h:
            return
        if self._height_anim and self._height_anim.state() != QPropertyAnimation.State.Stopped:
            self._height_anim.stop()
        self.move(self.x(), self.y() - (new_h - old_h) // 2)
        self.setFixedHeight(new_h)
        self._clamp_to_screen()
        self.position_changed.emit()

    def _fit_height_animated(self):
        new_h = self._calc_target_height()
        old_h = self.height()
        if new_h == old_h:
            return
        if self._height_anim and self._height_anim.state() != QPropertyAnimation.State.Stopped:
            self._height_anim.stop()

        target_y = self.y() - (new_h - old_h) // 2
        self.setMinimumHeight(min(old_h, new_h))
        self.setMaximumHeight(max(old_h, new_h))

        anim = QPropertyAnimation(self, b"geometry")
        anim.setDuration(150)
        anim.setStartValue(QRect(self.x(), self.y(), self.width(), old_h))
        anim.setEndValue(QRect(self.x(), target_y, self.width(), new_h))
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def on_finished():
            self.setFixedHeight(new_h)
            self._clamp_to_screen()
            self.position_changed.emit()
        anim.finished.connect(on_finished)

        self._height_anim = anim
        anim.start()

    def apply_settings(self, settings: dict):
        self._settings = _merge_settings(DEFAULT_SUBTITLE_WIN_SETTINGS, settings)

        for w in self._text_widgets:
            self._content_layout.removeWidget(w)
            w.deleteLater()
        self._text_widgets = []

        for line_cfg in self._settings.get("lines", []):
            if not line_cfg.get("enabled", True):
                continue
            tw = _SubtitleTextWidget()
            tw.set_config(line_cfg)
            tw.height_changed.connect(self._fit_height_animated)
            self._text_widgets.append(tw)
            self._content_layout.addWidget(tw)

        self._content_layout.setSpacing(self._settings.get("line_spacing", 8))
        self._min_display_ms = int(self._settings.get("min_display_ms", 1500))

        w = self._settings.get("window_width", 1000)
        self.setFixedWidth(w)

        self._apply_background()
        self._refresh_display()
        self.set_click_through(self._settings.get("click_through", False))
        self.set_locked(self._settings.get("locked", False))

        # Reset auto-hide timer with new settings
        self._restart_auto_hide_timer()

    # --- Auto-hide ---
    def _restart_auto_hide_timer(self):
        timeout = self._settings.get("auto_hide_timeout", 0)
        self._auto_hide_timer.stop()
        if timeout > 0 and self._sentences:
            self._auto_hide_timer.setInterval(timeout * 1000)
            self._auto_hide_timer.start()

    def _on_auto_hide_timeout(self):
        if self._is_hidden_by_timeout:
            return
        self._is_hidden_by_timeout = True
        anim_type = self._settings.get("auto_hide_animation", "fade")
        duration = self._settings.get("auto_hide_duration", 300)
        for tw in self._text_widgets:
            tw.animate_out(anim_type=anim_type, duration=duration)

    def _restore_from_auto_hide(self):
        if not self._is_hidden_by_timeout:
            return
        self._is_hidden_by_timeout = False
        anim_type = self._settings.get("auto_hide_animation", "fade")
        duration = self._settings.get("auto_hide_duration", 300)
        # Reverse the hide animation type for restore
        restore_type = anim_type
        if anim_type == "slide_down":
            restore_type = "slide_up"
        elif anim_type == "slide_up":
            restore_type = "slide_down"
        elif anim_type == "slide_left":
            restore_type = "slide_right"
        elif anim_type == "slide_right":
            restore_type = "slide_left"

        for tw in self._text_widgets:
            tw._stop_all_animations()
            # Set hidden state
            tw._content_opacity_val = 0.0
            if restore_type == "slide_left":
                tw._slide_offset_x_val = float(-(tw.width() or 200))
            elif restore_type == "slide_right":
                tw._slide_offset_x_val = float(tw.width() or 200)
            elif restore_type == "slide_up":
                tw._slide_offset_y_val = float(tw.height() or 40)
            elif restore_type == "slide_down":
                tw._slide_offset_y_val = float(-(tw.height() or 40))
            tw.update()
            # Use the entry animation mechanism with overridden type
            old_entry = tw._entry_animation
            old_dur = tw._animation_duration
            tw._entry_animation = restore_type if restore_type != "none" else "fade"
            tw._animation_duration = duration
            tw.animate_in()
            tw._entry_animation = old_entry
            tw._animation_duration = old_dur

    # --- Click-through ---
    def set_click_through(self, enabled: bool):
        self._click_through = bool(enabled)
        if self._click_through:
            if not self._ct_timer.isActive():
                self._ct_timer.start()
        else:
            self._ct_timer.stop()
        self._apply_click_through()
        self._update_edit_frame()

    def is_click_through(self) -> bool:
        return self._click_through

    def set_locked(self, enabled: bool):
        """Lock position and width (independent of click-through)."""
        self._locked = bool(enabled)
        self._settings["locked"] = self._locked
        self._drag_pos = None
        self._resize_start = None
        if self._locked:
            self.unsetCursor()
        self._update_edit_frame()

    def is_locked(self) -> bool:
        return self._locked

    def _build_context_menu(self) -> QMenu:
        menu = QMenu(self)
        lock = menu.addAction(t("subwin_menu_lock"))
        lock.setCheckable(True)
        lock.setChecked(self._locked)
        lock.triggered.connect(lambda checked: self.lock_requested.emit(bool(checked)))
        ct = menu.addAction(t("subwin_menu_click_through"))
        ct.setCheckable(True)
        ct.setChecked(self._click_through)
        ct.triggered.connect(lambda checked: self.click_through_requested.emit(bool(checked)))
        menu.addSeparator()
        settings = menu.addAction(t("subwin_menu_settings"))
        settings.triggered.connect(self.settings_requested.emit)
        hide = menu.addAction(t("subwin_menu_hide"))
        hide.triggered.connect(self.close)
        return menu

    def contextMenuEvent(self, event):
        if self._click_through:
            return
        self._build_context_menu().exec(event.globalPos())

    def _apply_click_through(self):
        try:
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            if self._click_through:
                if not (style & _WS_EX_TRANSPARENT):
                    ctypes.windll.user32.SetWindowLongW(
                        hwnd, _GWL_EXSTYLE, style | _WS_EX_TRANSPARENT
                    )
            elif style & _WS_EX_TRANSPARENT:
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style & ~_WS_EX_TRANSPARENT
                )
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        # Qt resets the extended style across hide/show, so re-assert it here.
        self._apply_click_through()

    # --- Drag / resize / lock ---
    _DRAG_BUTTONS = Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton

    def _edge_zone(self, pos):
        """Return "corner", "right" or None for a local position."""
        x, y = pos.x(), pos.y()
        if x >= self.width() - self._CORNER and y >= self.height() - self._CORNER:
            return "corner"
        if x >= self.width() - self._EDGE:
            return "right"
        return None

    def _update_cursor(self, pos):
        if self._click_through or self._locked:
            self.unsetCursor()
            return
        zone = self._edge_zone(pos)
        if zone == "corner":
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif zone == "right":
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)

    def _update_edit_frame(self):
        if not hasattr(self, "_edit_frame"):
            return
        self._edit_frame.setGeometry(self.rect())
        show = self._hover and not self._click_through and not self._locked
        self._edit_frame.setVisible(show)
        if show:
            self._edit_frame.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_edit_frame()

    def enterEvent(self, event):
        self._hover = True
        self._update_edit_frame()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self._update_edit_frame()
        self.unsetCursor()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() & self._DRAG_BUTTONS and not self._click_through and not self._locked:
            local = event.position().toPoint()
            if event.button() == Qt.MouseButton.LeftButton and self._edge_zone(local):
                self._resize_start = (event.globalPosition().toPoint().x(), self.width())
                self._drag_pos = None
            else:
                self._resize_start = None
                self._drag_pos = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            start_x, start_w = self._resize_start
            new_w = start_w + (event.globalPosition().toPoint().x() - start_x)
            new_w = max(self._MIN_WIDTH, min(self._MAX_WIDTH, new_w))
            if new_w != self.width():
                self.setFixedWidth(new_w)   # text widgets re-wrap in their resizeEvent
                self._fit_height_snap()
            event.accept()
            return
        if self._drag_pos and event.buttons() & self._DRAG_BUTTONS:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        self._update_cursor(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() & self._DRAG_BUTTONS and (self._drag_pos or self._resize_start is not None):
            self._drag_pos = None
            self._resize_start = None
            self._settings["window_width"] = self.width()
            self.position_changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Double-click = lock in place: enable click-through so the mouse passes through
        if event.button() == Qt.MouseButton.LeftButton and not self._click_through and not self._locked:
            self._drag_pos = None
            self._resize_start = None
            self._settings["click_through"] = True
            self.set_click_through(True)
            self.click_through_requested.emit(True)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def closeEvent(self, event):
        self.window_closed.emit()
        super().closeEvent(event)

    def paintEvent(self, event):
        if self._bg_pixmap and not self._bg_pixmap.isNull():
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self._bg_pixmap)
            painter.end()
        super().paintEvent(event)

    # --- Text updates ---
    def update_text(self, original: str, translations: dict | str):
        """Thread-safe text update.

        translations: dict mapping lang code to translated text,
                      or a plain string (backward compat, treated as primary target).
        """
        if isinstance(translations, str):
            # Backward compat: wrap in dict with empty key
            translations = {"": translations}
        self.update_text_signal.emit(original, json.dumps(translations, ensure_ascii=False))

    # live_signal flags
    _LIVE_HAS_TL = 1     # translation argument carries a value
    _LIVE_COMPLETE = 2   # translation is the complete provisional result (end of stream)

    def update_live(self, original: str, translation=None, complete: bool = False, seq: int = 0):
        """Thread-safe: show/update the sentence still being spoken.

        translation=None keeps whatever translation is currently shown (no "..." blink);
        complete=False marks a streaming partial, which only replaces the shown text
        when it is at least as long (text grows like YouTube captions, never shrinks).
        seq identifies the preview generation; updates older than the shown one are dropped.
        """
        flags = 0
        if translation is not None:
            flags |= self._LIVE_HAS_TL
        if complete:
            flags |= self._LIVE_COMPLETE
        self.live_signal.emit(original or "", translation or "", flags, int(seq))

    def clear_live(self):
        self.live_signal.emit("", "", 0, 0)

    def _drop_live(self):
        """Remove the live line. _live_seq_shown stays monotonic on purpose so late
        partials of an already finalized generation are still recognised as stale."""
        self._live = None
        self._live_timer.stop()

    def _on_live_timeout(self):
        self._on_live("", "", 0, 0)

    @pyqtSlot(str, str, int, int)
    def _on_live(self, original: str, translation: str, flags: int, seq: int):
        if not original:
            if self._live is None:
                return
            self._drop_live()
            self._refresh_display()
            return

        if seq:
            if self._live is None and seq <= self._live_seq_shown:
                return  # late update of a generation that was already finalized/cleared
            if self._live is not None and seq < self._live_seq_shown:
                return  # stale update from an earlier preview generation
            self._live_seq_shown = max(self._live_seq_shown, seq)

        cur_o, cur_t = self._live if self._live is not None else ("", "")
        new_o, new_t = cur_o, cur_t
        ignore_translation = False

        if original != cur_o:
            if cur_o and cur_o.startswith(original):
                if flags & self._LIVE_HAS_TL and flags & self._LIVE_COMPLETE and seq >= self._live_seq_shown:
                    # ASR settled on the shorter text for the newest generation and its
                    # complete translation is here: take both
                    new_o, new_t = original, translation
                    ignore_translation = True
                else:
                    # ASR momentarily produced a shorter prefix: keep the longer text, and
                    # its translation would describe the shorter text, so ignore it
                    new_o = cur_o
                    ignore_translation = True
            elif cur_o and (original.startswith(cur_o) or original.startswith(cur_o[:max(4, len(cur_o) // 2)])):
                # Same sentence growing / being refined: keep the shown translation for now
                new_o = original
            else:
                # A different sentence: reset the translation
                new_o = original
                new_t = ""

        if flags & self._LIVE_HAS_TL and not ignore_translation:
            if flags & self._LIVE_COMPLETE or len(translation) >= len(new_t):
                new_t = translation

        if (new_o, new_t) == (cur_o, cur_t) and self._live is not None:
            self._live_timer.start()
            return
        self._live = (new_o, new_t)
        self._live_timer.start()
        if self._is_hidden_by_timeout:
            self._restore_from_auto_hide()
        self._restart_auto_hide_timer()
        self._refresh_display()

    @pyqtSlot(str, str)
    def _on_update_text(self, original: str, translations_json: str):
        translations = json.loads(translations_json)
        # A sentence still waiting for its minimum-display slot is shown right away
        # instead of being dropped, so fast speech never loses subtitles.
        self._flush_pending_segments()

        # Respect minimum display time: delay if previous sentence was inserted recently
        now_ms = time.monotonic() * 1000
        elapsed = now_ms - self._last_insert_time
        base_delay = max(0, int(self._min_display_ms - elapsed)) if self._last_insert_time > 0 else 0

        if base_delay == 0:
            self._insert_sentence(original, translations)
        else:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(base_delay)
            timer._payload = (original, translations)
            timer.timeout.connect(lambda tm=timer: self._fire_pending(tm))
            timer.start()
            self._pending_segment_timers.append(timer)

    def _fire_pending(self, timer):
        if timer in self._pending_segment_timers:
            self._pending_segment_timers.remove(timer)
        original, translations = timer._payload
        timer.deleteLater()
        self._insert_sentence(original, translations)

    def _flush_pending_segments(self):
        """Insert every queued sentence immediately (oldest first)."""
        pending = self._pending_segment_timers
        self._pending_segment_timers = []
        for timer in pending:
            timer.stop()
            payload = getattr(timer, "_payload", None)
            timer.deleteLater()
            if payload:
                self._insert_sentence(*payload)

    def _insert_sentence(self, original: str, translations: dict):
        """Insert a single sentence and refresh display."""

        max_sentences = self._settings.get("sentences", 1)
        self._sentences.append((original, translations))
        if len(self._sentences) > max_sentences:
            self._sentences = self._sentences[-max_sentences:]
        # The final version of the previewed sentence replaces the preview
        if self._live is not None:
            lo = self._live[0]
            same = bool(lo and original) and (
                lo == original
                or (len(lo) >= 4 and (original.startswith(lo) or lo.startswith(original)))
                # leading characters re-recognised differently but the tail matches
                or (len(lo) >= 6 and lo[-6:] in original)
            )
            if same:
                self._drop_live()

        if self._is_hidden_by_timeout:
            self._restore_from_auto_hide()

        self._refresh_display()
        self._restart_auto_hide_timer()
        self._last_insert_time = time.monotonic() * 1000

    def _cancel_pending_segments(self):
        """Cancel any pending delayed segment insertions."""
        for timer in self._pending_segment_timers:
            timer.stop()
            timer.deleteLater()
        self._pending_segment_timers.clear()


    def _refresh_display(self):
        items = list(self._sentences)
        if self._live is not None:
            live_o, live_t = self._live
            items.append((live_o, {"": live_t if live_t else "\u2026"}))
        if not items:
            for tw in self._text_widgets:
                tw.set_text("")
            self._fit_height_snap()
            return

        lines_cfg = [ln for ln in self._settings.get("lines", []) if ln.get("enabled", True)]
        wi = 0

        for cfg in lines_cfg:
            if wi >= len(self._text_widgets):
                break
            tw = self._text_widgets[wi]
            line_type = cfg.get("type", "original")

            if line_type == "original":
                texts = [s[0] for s in items if s[0]]
            else:
                lang = cfg.get("lang", "")
                texts = []
                for _, tl_dict in items:
                    if isinstance(tl_dict, str):
                        if tl_dict:
                            texts.append(tl_dict)
                    elif lang and lang in tl_dict:
                        texts.append(tl_dict[lang])
                    elif "" in tl_dict and tl_dict[""]:
                        texts.append(tl_dict[""])
                    else:
                        for v in tl_dict.values():
                            if v:
                                texts.append(v)
                                break
            tw.set_text(" | ".join(texts) if len(texts) > 1 else (texts[0] if texts else ""))
            wi += 1

    def get_target_languages(self) -> set:
        """Return set of unique target language codes from enabled translation lines."""
        langs = set()
        for cfg in self._settings.get("lines", []):
            if cfg.get("enabled", True) and cfg.get("type") == "translation":
                lang = cfg.get("lang", "")
                if lang:
                    langs.add(lang)
        return langs

    def clear(self):
        self._sentences.clear()
        self._drop_live()
        self._cancel_pending_segments()
        self._auto_hide_timer.stop()
        self._is_hidden_by_timeout = False
        for tw in self._text_widgets:
            tw._stop_all_animations()
            tw._content_opacity_val = 1.0
            tw._slide_offset_x_val = 0.0
            tw._slide_offset_y_val = 0.0
            tw.set_text("")
        self._fit_height_snap()
