"""
Phase 6 & 7 — Integrated Subtitle Overlay Application.
Features:
- Native Windows OS click-through toggling (WS_EX_TRANSPARENT).
- Handle drag/positioning with auto/manual click-through mode.
- Auto-hiding subtitles after 4 seconds of silence.
- Clean application lifecycle and thread termination.
"""

import ctypes
import queue
import sys

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QVBoxLayout, QWidget)

from audio_pipeline import AudioPipeline

# --- Windows Native API Constants ---
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000


def set_native_click_through(hwnd: int, enable: bool):
    """Sets or unsets Windows WS_EX_TRANSPARENT flag for OS-level click passthrough."""
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
    def __init__(self, pipeline: AudioPipeline):
        super().__init__()
        self.pipeline = pipeline
        self.is_dragging = False
        self.drag_start_position = QPoint()
        self.is_native_passthrough = False

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
        self.resize(800, 150)
        self.setMinimumSize(400, 100)

        central_widget = QWidget(self)
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 5, 10, 10)

        # --- Top Control Bar ---
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

        # --- Subtitle Label ---
        self.subtitle_label = QLabel("Listening for audio...", self)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 22px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 170);
                border-radius: 8px;
                padding: 10px 15px;
            }
        """)

        layout.addLayout(top_bar_layout)
        layout.addWidget(self.subtitle_label, stretch=1)

    def _init_timers(self):
        # Poll hover state to dynamically update native Windows click-through
        self.hover_timer = QTimer(self)
        self.hover_timer.setInterval(50)
        self.hover_timer.timeout.connect(self._check_hover_state)
        self.hover_timer.start()

        # Poll Queue B for new transcription text
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(50)
        self.poll_timer.timeout.connect(self._poll_subtitles)
        self.poll_timer.start()

        # Auto-hide timer: clears text after 4s of silence
        self.autohide_timer = QTimer(self)
        self.autohide_timer.setInterval(4000)
        self.autohide_timer.setSingleShot(True)
        self.autohide_timer.timeout.connect(self._clear_subtitles)

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

    def _poll_subtitles(self):
        while not self.pipeline.text_queue.empty():
            try:
                text, _ = self.pipeline.text_queue.get_nowait()
                if text:
                    self.subtitle_label.setText(text)
                    self.autohide_timer.start()  # Reset 4s auto-hide countdown
            except queue.Empty:
                break

    def _clear_subtitles(self):
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
        self.poll_timer.stop()
        self.autohide_timer.stop()
        self.pipeline.stop()
        event.accept()
        QApplication.quit()


def main():
    app = QApplication(sys.argv)

    pipeline = AudioPipeline(model_size="base")
    pipeline.start()

    overlay = SubtitleOverlayApp(pipeline)
    overlay.show()

    app.aboutToQuit.connect(pipeline.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()