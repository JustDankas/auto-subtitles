"""
Phase 7 — Natural Subtitle Rendering Engine.
Features:
- Sentence boundary detection via ASR punctuation (. ! ?) and capitalization.
- Configurable multi-line buffer with line-by-line time decay.
- Overflow wrapping based on configurable character thresholds.
- Isolated OverlayConfig dataclass for easy parameter tuning.
- Native Windows click-through toggling (WS_EX_TRANSPARENT).
"""

import ctypes
import queue
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QVBoxLayout, QWidget)

from audio_pipeline import AudioPipeline

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020


@dataclass
class OverlayConfig:
    """Central configuration panel for tuning subtitle layout and timing."""
    # --- Line & Word Limits ---
    # NOTE: as of the partial/final ASR pipeline, each utterance_id maps to
    # exactly one on-screen line and QLabel's word-wrap handles overflow, so
    # the fields below are no longer read anywhere. Left in place so existing
    # config calls don't break; safe to delete once you've confirmed you
    # don't want them back for some other purpose.
    max_chars_per_line: int = 50          # unused (deprecated)
    max_visible_lines: int = 3            # still used — caps stacked lines on screen
    min_chars_before_cap_split: int = 12  # unused (deprecated)

    # --- Timers & Decay ---
    line_expiry_seconds: float = 3.0      # How long a completed top line remains before disappearing
    silence_autohide_seconds: float = 5.0 # Complete silence duration before clearing all text

    # --- ASR Sentence Parsing Rules (unused/deprecated, see note above) ---
    split_on_punctuation: bool = True
    split_on_capitalization: bool = True
    punctuation_marks: tuple = ('.', '!', '?', '…')

    # --- Visual Style ---
    font_size: int = 20                   # Subtitle font size in pixels
    bg_opacity: int = 180                 # Background opacity (0 = transparent, 255 = solid)
    text_color: str = "#FFFFFF"           # Hex text color


def set_native_click_through(hwnd: int, enable: bool):
    if sys.platform != "win32":
        return
    try:
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if enable:
            style |= WS_EX_TRANSPARENT
        else:
            style &= ~WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    except Exception as e:
        print(f"Error updating native window styles: {e}")


class DragHandle(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 220);
                border-radius: 4px;
            }
            QFrame:hover {
                background-color: rgba(60, 60, 60, 240);
            }
        """)


class SubtitleOverlayApp(QMainWindow):
    def __init__(self, pipeline: AudioPipeline, config: OverlayConfig):
        super().__init__()
        self.pipeline = pipeline
        self.config = config

        self.is_dragging = False
        self.drag_start_position = QPoint()
        self.is_native_passthrough = False

        # Active lines tracking: List[Dict{"text": str, "completed_at": float|None, "utterance_id": int|None}]
        self.lines: List[Dict] = []
        self.current_utterance_id = None  # the utterance_id still being updated in place, if any
        self.last_audio_activity_time = time.time()

        self._init_window_flags()
        self._init_ui()
        self._init_timers()

    def _init_window_flags(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def _init_ui(self):
        self.resize(850, 180)
        self.setMinimumSize(400, 100)

        central_widget = QWidget(self)
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 5, 10, 10)

        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(0, 0, 0, 0)

        self.handle = DragHandle(self)
        handle_layout = QHBoxLayout(self.handle)
        handle_layout.setContentsMargins(8, 2, 8, 2)

        handle_label = QLabel("⋮⋮ Subtitles", self.handle)
        handle_label.setStyleSheet("color: #AAAAAA; font-size: 11px; font-weight: bold;")
        handle_layout.addWidget(handle_label)

        close_btn = QPushButton("✕", self.handle)
        close_btn.setFixedSize(18, 18)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #CCCCCC;
                border: none;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #FF5555;
            }
        """)
        close_btn.clicked.connect(self.close)
        handle_layout.addWidget(close_btn)

        top_bar_layout.addStretch()
        top_bar_layout.addWidget(self.handle)
        top_bar_layout.addStretch()

        self.subtitle_label = QLabel("Listening for audio...", self)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: {self.config.text_color};
                font-size: {self.config.font_size}px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, {self.config.bg_opacity});
                border-radius: 8px;
                padding: 10px 15px;
                line-height: 130%;
            }}
        """)

        layout.addLayout(top_bar_layout)
        layout.addWidget(self.subtitle_label, stretch=1)

    def _init_timers(self):
        # Poll hover state for OS click-through
        self.hover_timer = QTimer(self)
        self.hover_timer.setInterval(50)
        self.hover_timer.timeout.connect(self._check_hover_state)
        self.hover_timer.start()

        # Engine tick: polls queue & manages line lifecycle
        self.engine_timer = QTimer(self)
        self.engine_timer.setInterval(40)
        self.engine_timer.timeout.connect(self._engine_tick)
        self.engine_timer.start()

    def _check_hover_state(self):
        if self.is_dragging:
            return

        cursor_pos = QCursor.pos()
        handle_rect = QRect(
            self.handle.mapToGlobal(QPoint(0, 0)),
            self.handle.size()
        )

        is_over_handle = handle_rect.contains(cursor_pos)
        hwnd = int(self.winId())

        if is_over_handle and self.is_native_passthrough:
            set_native_click_through(hwnd, False)
            self.is_native_passthrough = False
        elif not is_over_handle and not self.is_native_passthrough:
            set_native_click_through(hwnd, True)
            self.is_native_passthrough = True

    def _process_incoming_update(self, text: str, is_final: bool, utterance_id: int):
        """
        Handles both interim (partial) and final ASR results for a given
        utterance_id. A partial replaces the in-place text of the line
        currently being built; a final locks that line's text and starts
        its decay timer. NOTE: with segment-level (not fragment-level)
        ASR output, the old punctuation/capitalization chunk-merging logic
        is no longer needed — QLabel's word wrap handles line breaking,
        and each utterance_id maps to exactly one on-screen line.
        """
        self.last_audio_activity_time = time.time()

        if utterance_id != self.current_utterance_id:
            # A new utterance has started — open a fresh, uncommitted line.
            self.lines.append({"text": text, "completed_at": None, "utterance_id": utterance_id})
            self.current_utterance_id = utterance_id
        else:
            # Same utterance as before — update its text in place.
            for line in reversed(self.lines):
                if line.get("utterance_id") == utterance_id:
                    line["text"] = text
                    break

        if is_final:
            for line in reversed(self.lines):
                if line.get("utterance_id") == utterance_id:
                    line["completed_at"] = time.time()
                    break
            # Any further messages (there shouldn't be any) start a new line
            # rather than silently overwriting this now-locked one.
            self.current_utterance_id = None

        while len(self.lines) > self.config.max_visible_lines:
            self.lines.pop(0)

    def _purge_expired_lines(self):
        now = time.time()
        
        # Check if entire display should auto-hide due to long silence
        if now - self.last_audio_activity_time >= self.config.silence_autohide_seconds:
            if self.lines:
                self.lines.clear()
            return

        # Decay non-active lines (top/completed lines) after line_expiry_seconds
        remaining_lines = []
        for line in self.lines:
            if line["completed_at"] is not None:
                if now - line["completed_at"] < self.config.line_expiry_seconds:
                    remaining_lines.append(line)
            else:
                remaining_lines.append(line)

        self.lines = remaining_lines

    def _engine_tick(self):
        has_updates = False

        # 1. Pull new ASR chunks from pipeline Queue B
        while not self.pipeline.text_queue.empty():
            try:
                text, _latency, is_final, utterance_id = self.pipeline.text_queue.get_nowait()
                if text:
                    self._process_incoming_update(text, is_final, utterance_id)
                    has_updates = True
            except queue.Empty:
                break

        # 2. Check line expirations / time decay
        prev_count = len(self.lines)
        self._purge_expired_lines()
        if len(self.lines) != prev_count:
            has_updates = True

        # 3. Update UI if state changed
        if has_updates or not self.lines:
            if self.lines:
                display_text = "\n".join(line["text"] for line in self.lines)
                self.subtitle_label.setText(display_text)
            else:
                self.subtitle_label.setText("")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.drag_start_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.is_dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_start_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            event.accept()

    def closeEvent(self, event):
        self.hover_timer.stop()
        self.engine_timer.stop()
        self.pipeline.stop()
        event.accept()
        QApplication.quit()


def main():
    app = QApplication(sys.argv)

    # --- Easily tune your settings here ---
    config = OverlayConfig(
        max_chars_per_line=48,
        max_visible_lines=3,
        line_expiry_seconds=3.2,
        silence_autohide_seconds=5.0,
        split_on_punctuation=True,
        split_on_capitalization=True,
        font_size=20
    )

    pipeline = AudioPipeline(model_size="base")
    pipeline.start()

    overlay = SubtitleOverlayApp(pipeline, config)
    overlay.show()

    app.aboutToQuit.connect(pipeline.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()