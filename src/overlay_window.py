"""
The caption display window.

Each line is its own small rounded-rect box. Qt schedules the fade and height
animations directly, so the overlay does not need a polling timer.

No drag/close logic here anymore - that moved to drag_handle.py (see its
docstring for why). This window can optionally be click-through
(WA_TransparentForMouseEvents), controlled by main_app.py.

CLICK-THROUGH CAVEAT: `Qt.WindowType.WindowTransparentForInput` is the
flag Qt provides for OS-level input transparency; I have not been able to
verify it behaves correctly on Windows in this exact PyQt6 version. If
--click-through doesn't actually let clicks pass through to whatever's
behind the captions, tell me what you observe (nothing happens / errors /
window disappears / etc.) and we'll adjust.
"""

import logging
import time
from typing import List, Optional

from line_splitter import split_into_chunks
from PyQt6.QtCore import QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import (QGraphicsOpacityEffect, QLabel, QVBoxLayout,
                             QWidget)

MAX_LINES = 2
MAX_RENDERED_LINES = 2
BASE_DURATION_S = 2
SECONDS_PER_WORD = 0.2
MIN_DURATION_S = 2.0
MAX_DURATION_S = 6.0
FADE_WINDOW_S = 1.0

BOX_MARGIN = 12  # horizontal margin inside each line's box, each side
BOX_SPACING = 3  # vertical gap between boxes
PANEL_MARGIN = 4
SPLITTER_SAFETY_PX = 4
logger = logging.getLogger(__name__)


def _line_duration(text: str) -> float:
    word_count = max(1, len(text.split()))
    duration = BASE_DURATION_S + SECONDS_PER_WORD * word_count
    return max(MIN_DURATION_S, min(MAX_DURATION_S, duration))


class SubtitleLineWidget(QWidget):
    """One caption line, its own rounded-rect box, independently fadeable
    and shrinkable."""

    def __init__(self, text: str, text_color: QColor, box_width: int, font: QFont):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.label = QLabel(self)
        self.label.setWordWrap(True)
        self.label.setFont(font)
        self.label.setStyleSheet(f"color: {text_color.name()}; background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(BOX_MARGIN, 6, BOX_MARGIN, 6)
        layout.addWidget(self.label)

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)
        self._fade_animation: Optional[QPropertyAnimation] = None
        self._height_animation: Optional[QPropertyAnimation] = None

        self._box_width = box_width
        self.set_text(text)
        self._natural_height = self.sizeHint().height()
        self.setFixedWidth(box_width)
        self.setMinimumHeight(0)
        self.setMaximumHeight(self._natural_height)

    def set_text(self, text: str) -> None:
        self.label.setText(text)
        self.label.setFixedWidth(self._box_width - 2 * BOX_MARGIN)

    def recompute_natural_height(self) -> None:
        """Call after set_text() if the widget should resize to fit new
        (e.g. growing partial) text."""
        self._natural_height = self.sizeHint().height()
        self.setMaximumHeight(self._natural_height)
        self.setMinimumHeight(0)

    def start_fade(self, duration_ms: int) -> None:
        duration_ms = max(1, duration_ms)
        if self._fade_animation is not None:
            self._fade_animation.stop()
        if self._height_animation is not None:
            self._height_animation.stop()
        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_animation.setDuration(duration_ms)
        self._fade_animation.setStartValue(self._opacity_effect.opacity())
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.start()

        self._height_animation = QPropertyAnimation(self, b"maximumHeight", self)
        self._height_animation.setDuration(duration_ms)
        self._height_animation.setStartValue(self.maximumHeight())
        self._height_animation.setEndValue(0)
        self._height_animation.start()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 10, 10)
        super().paintEvent(event)


class _LineMeta:
    __slots__ = ("created_at", "duration", "fade_at", "end_at", "fade_timer", "end_timer")

    def __init__(self, text: str):
        self.created_at = time.monotonic()
        self.duration = _line_duration(text)
        self.end_at = self.created_at + self.duration
        self.fade_at = self.end_at - FADE_WINDOW_S
        self.fade_timer: Optional[QTimer] = None
        self.end_timer: Optional[QTimer] = None


class SubtitleOverlay(QWidget):
    def __init__(
        self,
        new_text_color: str = "#F6FF00",
        old_text_color: str = "#E5E5E5",
        width: int = 900,
        height: int = 160,
        click_through: bool = False,
    ):
        super().__init__()
        self._handle_ref: Optional[QWidget] = None

        self._new_text_color = QColor(new_text_color)
        self._old_text_color = QColor(old_text_color)
        self._box_width = width - 2 * PANEL_MARGIN
        # self._font = QFont("Segoe UI", 18)
        self._font = QFont("Inter", 18)
        self._split_width = max(1, self._box_width - 2 * BOX_MARGIN - SPLITTER_SAFETY_PX)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        line_height = QFontMetrics(self._font).lineSpacing()
        two_line_box_height = 2 * line_height + 12
        minimum_height = (
            2 * PANEL_MARGIN
            + (MAX_LINES + 1) * two_line_box_height
            + MAX_LINES * BOX_SPACING
        )
        self.setFixedSize(width, max(height, minimum_height))

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(BOX_SPACING)
        self._layout.addStretch()  # keeps boxes bottom-aligned within the reserved height

        self._committed: List[tuple[_LineMeta, SubtitleLineWidget]] = []
        self._partial_widget: Optional[SubtitleLineWidget] = None
        self._consumed_prefix = ""

        self.set_click_through(click_through)

    def set_drag_handle(self, handle: QWidget) -> None:
            """Keep a reference to the handle so we can keep it on top."""
            self._handle_ref = handle

    def mousePressEvent(self, event) -> None:
            """When the overlay is clicked, keep the handle on top."""
            super().mousePressEvent(event)
            if self._handle_ref:
                self._handle_ref.raise_()

    def set_click_through(self, enabled: bool) -> None:
        """Toggle OS-level input transparency. See module docstring for the
        verification caveat."""
        was_visible = self.isVisible()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enabled)
        if was_visible:
            # Re-applying window flags on a visible widget needs a re-show
            # to take effect on most platforms.
            self.show()

    def update_partial(self, text: str) -> None:
        if self._consumed_prefix and not text.startswith(self._consumed_prefix):
            logger.warning("ASR partial revised text before the consumed display prefix")
            self._consumed_prefix = ""

        remainder = text[len(self._consumed_prefix):].lstrip()
        if not text:
            self._remove_partial()
            return

        chunks = self._split(remainder)
        if len(chunks) > 1:
            self._remove_partial()
            for chunk in chunks[:-1]:
                self._add_committed_line(chunk.replace("\n", " "))
            promoted = " ".join(chunk.replace("\n", " ") for chunk in chunks[:-1])
            self._consumed_prefix = f"{self._consumed_prefix} {promoted}".strip()
            remainder = chunks[-1]

        if not remainder:
            self._remove_partial()
            return
        if self._partial_widget is None:
            self._partial_widget = SubtitleLineWidget(
                remainder, self._new_text_color, self._box_width, self._font
            )
            self._layout.addWidget(self._partial_widget)
        else:
            self._partial_widget.set_text(remainder)
            self._partial_widget.recompute_natural_height()
            self._partial_widget.setMaximumHeight(self._partial_widget._natural_height)

    def commit_line(self, text: str) -> None:
        self._remove_partial()
        remainder = text
        if self._consumed_prefix:
            if text.startswith(self._consumed_prefix):
                remainder = text[len(self._consumed_prefix):].lstrip()
            else:
                logger.warning("ASR final text revised the consumed display prefix")
            self._consumed_prefix = ""

        for chunk in self._split(remainder):
            self._add_committed_line(chunk.replace("\n", " "))

    def _split(self, text: str) -> list[str]:
        return split_into_chunks(
            text,
            QFontMetrics(self._font).horizontalAdvance,
            self._split_width,
            MAX_RENDERED_LINES,
        )

    def _add_committed_line(self, text: str) -> None:
        if text:
            widget = SubtitleLineWidget(text, self._old_text_color, self._box_width, self._font)
            insert_at = self._layout.count()  # partial was just removed, so append at end
            self._layout.insertWidget(insert_at, widget)
            meta = _LineMeta(text)
            self._committed.append((meta, widget))

            for older_meta, older_widget in self._committed[:-1]:
                if older_meta.end_at > meta.end_at:
                    older_meta.end_at = meta.end_at
                    older_meta.fade_at = meta.fade_at
                    self._schedule_line(older_meta, older_widget)
            self._schedule_line(meta, widget)

            while len(self._committed) > MAX_LINES:
                _, old_widget = self._committed.pop(0)
                self._layout.removeWidget(old_widget)
                old_widget.deleteLater()

    def _remove_partial(self) -> None:
        if self._partial_widget is not None:
            self._layout.removeWidget(self._partial_widget)
            self._partial_widget.deleteLater()
            self._partial_widget = None

    def _schedule_line(self, meta: _LineMeta, widget: SubtitleLineWidget) -> None:
        if meta.fade_timer is not None:
            meta.fade_timer.stop()
        if meta.end_timer is not None:
            meta.end_timer.stop()

        now = time.monotonic()
        fade_delay_ms = max(0, int((meta.fade_at - now) * 1000))
        end_delay_ms = max(0, int((meta.end_at - now) * 1000))

        meta.fade_timer = QTimer(widget)
        meta.fade_timer.setSingleShot(True)
        meta.fade_timer.timeout.connect(
            lambda: widget.start_fade(max(1, int((meta.end_at - time.monotonic()) * 1000)))
        )
        meta.fade_timer.start(fade_delay_ms)

        meta.end_timer = QTimer(widget)
        meta.end_timer.setSingleShot(True)
        meta.end_timer.timeout.connect(lambda: self._remove_line(meta, widget))
        meta.end_timer.start(end_delay_ms)

    def _remove_line(self, meta: _LineMeta, widget: SubtitleLineWidget) -> None:
        for index, (current_meta, current_widget) in enumerate(self._committed):
            if current_meta is meta and current_widget is widget:
                self._committed.pop(index)
                self._layout.removeWidget(widget)
                widget.deleteLater()
                return

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(10, 18, 24, 220))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 14, 14)
        super().paintEvent(event)
