import ctypes
import queue
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt6.QtCore import (QEasingCurve, QPoint, QPropertyAnimation, QRect, Qt,
                          QTimer)
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (QApplication, QFrame, QGraphicsOpacityEffect,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from audio_pipeline import AudioPipeline

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020


@dataclass
class OverlayConfig:
    """Central configuration panel for tuning subtitle layout and animation behavior."""
    max_visible_lines: int = 3            # Caps active stacked lines on screen

    # --- Timers & Dynamic Decay ---
    base_line_expiry: float = 1.8         # Minimum duration a completed line remains (seconds)
    seconds_per_word: float = 0.22        # Additional display time granted per word
    silence_autohide_seconds: float = 5.0 # Complete silence duration before clearing all lines

    # --- Visual Style & Layout Bounds ---
    font_size: int = 20                   # Subtitle font size in pixels
    bg_opacity: int = 180                 # Background opacity (0 = transparent, 255 = solid)
    text_color: str = "#FFFFFF"           # Hex text color for stable text
    new_word_color: str = "#FFD700"       # Hex text color for newly added partial words (Gold)
    line_spacing: int = 6                 # Pixel gap between stacked line bubbles
    fade_duration_ms: int = 300           # Fade-out animation length in milliseconds
    min_line_height: int = 38             # Minimum height per line box to stop layout vertical jumps
    label_width: int = 800                # Fixed width for subtitle lines to prevent layout resize twitch


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

        # Active lines tracking list:
        # Each entry: {"utterance_id": int, "label": QLabel, "raw_text": str,
        #              "completed_at": Optional[float], "expiry_duration": float,
        #              "fading": bool, "anim": Optional[QPropertyAnimation]}
        self.lines: List[Dict] = []
        self.current_utterance_id: Optional[int] = None
        self.last_audio_activity_time = time.time()
        self.placeholder_label: Optional[QLabel] = None

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
        # Calculate maximum container height to reserve space and prevent vertical cropping
        max_container_height = (self.config.min_line_height + self.config.line_spacing) * self.config.max_visible_lines + 80
        window_width = self.config.label_width + 40

        self.resize(window_width, max_container_height)
        self.setMinimumSize(window_width, max_container_height)

        central_widget = QWidget(self)
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 5, 10, 10)

        # Top Bar (Drag handle and close button)
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
        main_layout.addLayout(top_bar_layout)

        # Subtitle Lines Container
        self.lines_container = QWidget(self)
        self.lines_container.setStyleSheet("background: transparent;")
        self.lines_layout = QVBoxLayout(self.lines_container)
        self.lines_layout.setContentsMargins(5, 5, 5, 5)
        self.lines_layout.setSpacing(self.config.line_spacing)
        self.lines_layout.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)

        main_layout.addWidget(self.lines_container, stretch=1)

        # Initial Placeholder Indicator
        self.placeholder_label = self._create_line_label("Listening for audio...")
        self.lines_layout.addWidget(self.placeholder_label)

    def _create_line_label(self, initial_text: str = "") -> QLabel:
        """Creates a standardized, fixed-width left-aligned subtitle label bubble."""
        label = QLabel(self.lines_container)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setFixedWidth(self.config.label_width)
        label.setMinimumHeight(self.config.min_line_height)
        label.setStyleSheet(f"""
            QLabel {{
                color: {self.config.text_color};
                font-size: {self.config.font_size}px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, {self.config.bg_opacity});
                border-radius: 6px;
                padding: 6px 12px;
            }}
        """)
        if initial_text:
            label.setText(f'<span style="color: {self.config.text_color};">{initial_text}</span>')
        return label

    def _format_diff_text(self, old_text: str, new_text: str) -> str:
        """Compares old and new text word-by-word to highlight newly appended partial words."""
        old_words = old_text.strip().split()
        new_words = new_text.strip().split()

        if not old_words:
            return " ".join(
                f'<span style="color: {self.config.new_word_color};">{w}</span>' for w in new_words
            )

        common_len = 0
        min_len = min(len(old_words), len(new_words))
        while common_len < min_len and old_words[common_len] == new_words[common_len]:
            common_len += 1

        stable_part = " ".join(new_words[:common_len])
        new_part = " ".join(new_words[common_len:])

        formatted_str = ""
        if stable_part:
            formatted_str += f'<span style="color: {self.config.text_color};">{stable_part}</span>'
        if new_part:
            if formatted_str:
                formatted_str += " "
            formatted_str += f'<span style="color: {self.config.new_word_color};">{new_part}</span>'

        return formatted_str

    def _init_timers(self):
        # Poll hover state for OS click-through
        self.hover_timer = QTimer(self)
        self.hover_timer.setInterval(50)
        self.hover_timer.timeout.connect(self._check_hover_state)
        self.hover_timer.start()

        # Engine tick: polls queue & manages line lifecycle
        self.engine_timer = QTimer(self)
        self.engine_timer.setInterval(30)
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

    def _remove_placeholder(self):
        if self.placeholder_label is not None:
            self.lines_layout.removeWidget(self.placeholder_label)
            self.placeholder_label.deleteLater()
            self.placeholder_label = None

    def _fade_out_line(self, line_entry: Dict):
        """Triggers a smooth 300ms fade-out animation before deleting the widget."""
        if line_entry["fading"]:
            return

        line_entry["fading"] = True
        label = line_entry["label"]

        effect = QGraphicsOpacityEffect(label)
        label.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(self.config.fade_duration_ms)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutQuad)

        def cleanup():
            if label is not None:
                self.lines_layout.removeWidget(label)
                label.deleteLater()
            if line_entry in self.lines:
                self.lines.remove(line_entry)

        anim.finished.connect(cleanup)
        line_entry["anim"] = anim  # Retain animation object reference
        anim.start()

    def _process_incoming_update(self, text: str, is_final: bool, utterance_id: int):
        self.last_audio_activity_time = time.time()
        self._remove_placeholder()

        # 1. Check if this is a brand new utterance
        if utterance_id != self.current_utterance_id:
            label = self._create_line_label()
            formatted_html = (
                f'<span style="color: {self.config.text_color};">{text}</span>'
                if is_final
                else self._format_diff_text("", text)
            )
            label.setText(formatted_html)
            self.lines_layout.addWidget(label)

            line_entry = {
                "utterance_id": utterance_id,
                "label": label,
                "raw_text": text,
                "completed_at": None,
                "expiry_duration": 0.0,
                "fading": False,
                "anim": None,
            }
            self.lines.append(line_entry)
            self.current_utterance_id = utterance_id
        else:
            # 2. Update existing active line in place with word diffing
            for entry in reversed(self.lines):
                if entry["utterance_id"] == utterance_id and not entry["fading"]:
                    if is_final:
                        entry["label"].setText(
                            f'<span style="color: {self.config.text_color};">{text}</span>'
                        )
                    else:
                        formatted_html = self._format_diff_text(entry["raw_text"], text)
                        entry["label"].setText(formatted_html)
                    entry["raw_text"] = text
                    break

        # 3. Finalize utterance lifecycle on segment end
        if is_final:
            for entry in reversed(self.lines):
                if entry["utterance_id"] == utterance_id and not entry["fading"]:
                    entry["completed_at"] = time.time()
                    word_count = len(text.split())
                    entry["expiry_duration"] = self.config.base_line_expiry + (
                        word_count * self.config.seconds_per_word
                    )
                    break
            self.current_utterance_id = None

        # 4. Enforce visible lines capacity with smooth fade out
        active_lines = [entry for entry in self.lines if not entry["fading"]]
        while len(active_lines) > self.config.max_visible_lines:
            oldest_entry = active_lines.pop(0)
            self._fade_out_line(oldest_entry)

    def _purge_expired_lines(self):
        now = time.time()

        # Silence autohide: fade out all lines after prolonged inactivity
        if now - self.last_audio_activity_time >= self.config.silence_autohide_seconds:
            for entry in list(self.lines):
                if not entry["fading"]:
                    self._fade_out_line(entry)
            return

        # Check line decay timers for completed segments
        for entry in list(self.lines):
            if entry["completed_at"] is not None and not entry["fading"]:
                elapsed = now - entry["completed_at"]
                if elapsed >= entry["expiry_duration"]:
                    self._fade_out_line(entry)

    def _engine_tick(self):
        # 1. Poll incoming ASR results
        while not self.pipeline.text_queue.empty():
            try:
                text, _latency, is_final, utterance_id = self.pipeline.text_queue.get_nowait()
                if text:
                    self._process_incoming_update(text, is_final, utterance_id)
            except queue.Empty:
                break

        # 2. Process lifecycle animations & line decay
        self._purge_expired_lines()

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

    config = OverlayConfig(
        max_visible_lines=3,
        base_line_expiry=1.8,
        seconds_per_word=0.22,
        silence_autohide_seconds=5.0,
        font_size=20,
        new_word_color="#FFD700",
        min_line_height=38,
        label_width=800,
        line_spacing=6,
        fade_duration_ms=300
    )

    pipeline = AudioPipeline(model_size="base")
    pipeline.start()

    overlay = SubtitleOverlayApp(pipeline, config)
    overlay.show()

    app.aboutToQuit.connect(pipeline.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()