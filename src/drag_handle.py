"""
Small, always-on-top, always-interactive drag handle + close button.

Kept as a SEPARATE top-level window from the caption box, because true
OS-level click-through (mouse events passing through to whatever's behind
the captions - e.g. a video player) requires the WHOLE native window to be
click-through on Windows; there's no way to make only part of one window
click-through. So: two windows, moved together by main_app.py via the
`moved` signal. This handle is never click-through, regardless of the
caption box's setting - it's always how you drag and always how you close.
"""

from typing import Optional

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class DragHandle(QWidget):
    moved = pyqtSignal(int, int)  # dx, dy - so the caption box can follow along
    close_requested = pyqtSignal()

    def __init__(self, width: int = 180, height: int = 30):
        super().__init__()
        self._drag_offset: Optional[QPoint] = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(width, height)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 10, 3)

        grip = QLabel("\u2237\u2237 Drag to Move")  # simple grip glyph
        grip.setStyleSheet("color: white; font-size: 13px; font-weight: bold;")
        layout.addWidget(grip)
        layout.addStretch()

        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(18, 18)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            "QPushButton { color: white; background: rgba(255,255,255,70); "
            "border: none; border-radius: 9px; font-weight: bold; font-size: 10px; }"
            "QPushButton:hover { background: rgba(220,60,60,200); }"
        )
        close_btn.clicked.connect(self.close_requested.emit)
        layout.addWidget(close_btn)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(20, 20, 20, 210))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 8, 8)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            new_pos = event.globalPosition().toPoint() - self._drag_offset
            old_pos = self.pos()
            self.move(new_pos)
            self.moved.emit(new_pos.x() - old_pos.x(), new_pos.y() - old_pos.y())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
