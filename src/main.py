"""
Phase 6 — Full Pipeline & Overlay Integration.
Connects the AudioPipeline (WASAPI -> Silero VAD -> Faster-Whisper)
to the PyQt6 translucent overlay window using a main-thread QTimer.
"""

import queue
import sys

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QVBoxLayout, QWidget)

from audio_pipeline import AudioPipeline


class DragHandle(QFrame):
    """Small visible handle widget used to position and interact with the overlay."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(40, 40, 40, 200);
                border-radius: 4px;
            }
            QFrame:hover {
                background-color: rgba(70, 70, 70, 230);
            }
        """)


class SubtitleOverlayApp(QMainWindow):
    def __init__(self, pipeline: AudioPipeline):
        super().__init__()
        self.pipeline = pipeline
        self.is_dragging = False
        self.drag_start_position = QPoint()

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
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def _init_ui(self):
        self.resize(800, 150)
        self.setMinimumSize(400, 100)

        central_widget = QWidget(self)
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 5, 10, 10)

        # --- Top Bar Controls ---
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

        # --- Subtitle Text Display ---
        self.subtitle_label = QLabel("Listening for audio...", self)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 22px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 160);
                border-radius: 8px;
                padding: 10px 15px;
            }
        """)

        layout.addLayout(top_bar_layout)
        layout.addWidget(self.subtitle_label, stretch=1)

    def _init_timers(self):
        # Poll cursor to handle window drag vs click-through
        self.hover_timer = QTimer(self)
        self.hover_timer.setInterval(50)
        self.hover_timer.timeout.connect(self._check_hover_state)
        self.hover_timer.start()

        # Poll Queue B for transcription results (Qt main thread safe)
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(50)
        self.poll_timer.timeout.connect(self._poll_subtitles)
        self.poll_timer.start()

    def _check_hover_state(self):
        if self.is_dragging:
            return

        cursor_pos = QCursor.pos()
        handle_global_rect = QRect(
            self.handle.mapToGlobal(QPoint(0, 0)),
            self.handle.size()
        )

        is_over_handle = handle_global_rect.contains(cursor_pos)
        current_passthrough = self.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        if is_over_handle and current_passthrough:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        elif not is_over_handle and not current_passthrough:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def _poll_subtitles(self):
        """Pulls generated transcriptions from Queue B and updates the UI."""
        while not self.pipeline.text_queue.empty():
            try:
                text, _ = self.pipeline.text_queue.get_nowait()
                if text:
                    self.subtitle_label.setText(text)
            except queue.Empty:
                break

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
        """Safely shut down background audio and ASR threads."""
        self.pipeline.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)

    # Instantiate and start the audio pipeline
    pipeline = AudioPipeline(model_size="base")
    pipeline.start()

    overlay = SubtitleOverlayApp(pipeline)
    overlay.show()

    # Register application quit hook
    app.aboutToQuit.connect(pipeline.stop)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()