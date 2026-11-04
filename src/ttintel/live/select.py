"""Temporary click-and-drag desktop selector with per-monitor pixel mapping."""
from __future__ import annotations

from .core import Region, region_from_drag


def monitor_regions() -> list[Region]:
    import mss
    with mss.MSS() as grabber:
        # MSS v10 dictionaries and MSS v11 Monitor objects both supported.
        def get(m: object, name: str) -> int:
            return int(getattr(m, name) if hasattr(m, name) else m[name])
        return [
            Region(get(m, "left"), get(m, "top"), get(m, "width"), get(m, "height"))
            for m in grabber.monitors[1:]
        ]


def select_screen_region(app: object, monitor: int = 1) -> Region | None:
    """Qt selection window closes before the first screenshot is taken."""
    from PySide6 import QtCore, QtGui, QtWidgets
    regions = monitor_regions()
    if not 1 <= monitor <= len(regions):
        raise ValueError(f"monitor must be between 1 and {len(regions)}")
    physical = regions[monitor - 1]
    screens = app.screens()
    if not screens:
        raise RuntimeError("no graphical screens are available")
    # Qt virtual geometry is expressed in device-independent pixels, whereas
    # MSS uses physical pixels. Match likely display by relative monitor order,
    # and transform the drag using measured window width/height, not guessed DPI.
    ranked = sorted(screens, key=lambda s: (s.geometry().x(), s.geometry().y()))
    sorted_regions = sorted(enumerate(regions, 1), key=lambda item:
                            (item[1].left, item[1].top))
    rank = next(i for i, (index, _) in enumerate(sorted_regions) if index == monitor)
    screen = ranked[min(rank, len(ranked) - 1)]

    class Selector(QtWidgets.QWidget):
        complete = QtCore.Signal()

        def __init__(self) -> None:
            super().__init__(None, QtCore.Qt.WindowType.Tool |
                             QtCore.Qt.WindowType.FramelessWindowHint |
                             QtCore.Qt.WindowType.WindowStaysOnTopHint)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setMouseTracking(True)
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            self.start: QtCore.QPointF | None = None
            self.end: QtCore.QPointF | None = None
            self.region: Region | None = None
            self.setGeometry(screen.geometry())

        def mousePressEvent(self, event: object) -> None:
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                self.complete.emit()
            elif event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.start = event.position()
                self.end = event.position()

        def mouseMoveEvent(self, event: object) -> None:
            if self.start is not None:
                self.end = event.position()
                self.update()

        def mouseReleaseEvent(self, event: object) -> None:
            if self.start is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
                return
            self.end = event.position()
            self.region = region_from_drag(
                physical, (self.start.x(), self.start.y()),
                (self.end.x(), self.end.y()), (self.width(), self.height()),
            )
            self.start = None
            if self.region is not None:
                self.complete.emit()
            else:
                self.update()

        def keyPressEvent(self, event: object) -> None:
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self.complete.emit()

        def paintEvent(self, event: object) -> None:
            p = QtGui.QPainter(self)
            p.fillRect(self.rect(), QtGui.QColor(4, 14, 27, 135))
            p.setPen(QtGui.QPen(QtGui.QColor("#62e9da"), 2))
            p.setFont(QtGui.QFont("Segoe UI", 18, QtGui.QFont.Weight.Bold))
            p.drawText(24, 48, "TABLE TENNIS  /  SELECT VIDEO")
            p.setFont(QtGui.QFont("Segoe UI", 11))
            p.setPen(QtGui.QColor("#d7e5ed"))
            p.drawText(24, 76, "Drag over browser playback  |  Esc / right click to cancel")
            if self.start is not None and self.end is not None:
                box = QtCore.QRectF(self.start, self.end).normalized()
                p.setPen(QtGui.QPen(QtGui.QColor("#62e9da"), 3))
                p.setBrush(QtGui.QColor(39, 232, 197, 35))
                p.drawRect(box)

    selector = Selector()
    loop = QtCore.QEventLoop()
    selector.complete.connect(loop.quit)
    selector.show()
    selector.raise_()
    selector.activateWindow()
    selector.setFocus()
    loop.exec()
    region = selector.region
    selector.close()
    return region
