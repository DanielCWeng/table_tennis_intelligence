"""Dark PySide6 + PyQtGraph diagnostic UI. Import only for the live command."""
from __future__ import annotations

from collections import deque
import csv
from pathlib import Path
import time

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np

from .core import ROI
from .sources import LiveEngine, Result


class EditableROI(QtWidgets.QGraphicsRectItem):
    """Movable scene-pixel region, with a visible bottom-right resize grip."""
    def __init__(self, roi: ROI, color: str, changed: object):
        super().__init__(0, 0, roi.width, roi.height)
        self.setPos(roi.x, roi.y)
        self.color = QtGui.QColor(color)
        self.changed = changed
        self._resizing = False
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(5)

    def paint(self, painter: object, option: object, widget: object = None) -> None:
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(self.color, 2))
        painter.drawRect(self.rect())
        rect = self.rect()
        painter.fillRect(QtCore.QRectF(rect.right() - 12, rect.bottom() - 12, 12, 12),
                         self.color)

    def itemChange(self, change: object, value: object) -> object:
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            bounds = self.scene().sceneRect()
            x = max(bounds.left(), min(float(value.x()), bounds.right() - self.rect().width()))
            y = max(bounds.top(), min(float(value.y()), bounds.bottom() - self.rect().height()))
            return QtCore.QPointF(x, y)
        return super().itemChange(change, value)

    def mousePressEvent(self, event: object) -> None:
        r = self.rect()
        self._resizing = (r.right() - event.pos().x() <= 18 and
                          r.bottom() - event.pos().y() <= 18)
        if self._resizing:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: object) -> None:
        if self._resizing:
            scene = self.scene().sceneRect()
            max_width = max(24., scene.right() - self.pos().x())
            max_height = max(24., scene.bottom() - self.pos().y())
            w = max(24., min(max_width, event.pos().x()))
            h = max(24., min(max_height, event.pos().y()))
            self.setRect(0, 0, w, h)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: object) -> None:
        if self._resizing:
            self._resizing = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        self.changed()

    def roi(self) -> ROI:
        p, r = self.pos(), self.rect()
        return ROI(round(p.x()), round(p.y()), round(r.width()), round(r.height()))


class VideoView(QtWidgets.QGraphicsView):
    regions_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        # Parent the scene to the view; otherwise the Python wrapper can be
        # garbage collected while the C++ view still references the scene.
        scene = QtWidgets.QGraphicsScene(self)
        self.setScene(scene)
        self._scene = scene
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QtGui.QColor("#0b1421"))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setMinimumSize(580, 340)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pix = scene.addPixmap(QtGui.QPixmap())
        self.pix.setZValue(-1)
        self.target: EditableROI | None = None
        self.reference: EditableROI | None = None
        self.mode: str | None = None
        self._origin: QtCore.QPointF | None = None
        self._rubber: QtWidgets.QGraphicsRectItem | None = None
        self.dots: list[QtWidgets.QGraphicsEllipseItem] = []
        self.shape: tuple[int, int] | None = None

    def set_frame(self, frame: np.ndarray, points: tuple, references: tuple) -> None:
        h, w = frame.shape[:2]
        if (w, h) != self.shape:
            self.shape = (w, h)
            self.scene().setSceneRect(0, 0, w, h)
            self.clear_rois()
        rgb = np.ascontiguousarray(frame)
        image = QtGui.QImage(rgb.data, w, h, rgb.strides[0],
                             QtGui.QImage.Format.Format_RGB888).copy()
        self.pix.setPixmap(QtGui.QPixmap.fromImage(image))
        for dot in self.dots:
            self.scene().removeItem(dot)
        self.dots.clear()
        for sequence, color in ((points[:60], "#64e6cf"), (references[:35], "#ebb878")):
            for x, y in sequence:
                dot = self.scene().addEllipse(x - 2, y - 2, 4, 4,
                                               QtGui.QPen(QtGui.QColor(color)),
                                               QtGui.QBrush(QtGui.QColor(color)))
                dot.setZValue(2)
                self.dots.append(dot)
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        if self.shape:
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def choose(self, name: str) -> None:
        self.mode = name
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def clear_rois(self) -> None:
        for item in (self.target, self.reference):
            if item is not None:
                self.scene().removeItem(item)
        self.target = self.reference = None
        self.regions_changed.emit()

    def _point(self, mouse: object) -> QtCore.QPointF:
        pt = self.mapToScene(mouse)
        bounds = self.sceneRect()
        return QtCore.QPointF(
            max(bounds.left(), min(bounds.right(), pt.x())),
            max(bounds.top(), min(bounds.bottom(), pt.y())),
        )

    def mousePressEvent(self, event: object) -> None:
        if self.mode and self.shape and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._origin = self._point(event.pos())
            self._rubber = self.scene().addRect(QtCore.QRectF(self._origin, self._origin),
                                                 QtGui.QPen(QtGui.QColor("#f3df8a"), 2))
            self._rubber.setZValue(8)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: object) -> None:
        if self._origin is not None and self._rubber is not None:
            self._rubber.setRect(QtCore.QRectF(self._origin, self._point(event.pos())).normalized())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: object) -> None:
        if self._origin is None or self._rubber is None:
            super().mouseReleaseEvent(event)
            return
        rect = QtCore.QRectF(self._origin, self._point(event.pos())).normalized()
        self.scene().removeItem(self._rubber)
        self._rubber = None
        self._origin = None
        if rect.width() >= 24 and rect.height() >= 24:
            roi = ROI(round(rect.x()), round(rect.y()),
                      round(rect.width()), round(rect.height()))
            key = self.mode
            old = self.target if key == "target" else self.reference
            if old is not None:
                self.scene().removeItem(old)
            item = EditableROI(roi, "#63ead3" if key == "target" else "#f6b76c",
                               self.regions_changed.emit)
            self.scene().addItem(item)
            if key == "target":
                self.target = item
            else:
                self.reference = item
            self.regions_changed.emit()
        self.mode = None
        self.unsetCursor()
        event.accept()

    def regions(self) -> tuple[ROI | None, ROI | None]:
        return (self.target.roi() if self.target else None,
                self.reference.roi() if self.reference else None)


class LiveWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: LiveEngine) -> None:
        super().__init__()
        self.engine = engine
        self.samples: deque[tuple] = deque(maxlen=1800)
        self.first_timestamp: float | None = None
        self.setWindowTitle("TABLE TENNIS INTELLIGENCE  /  LIVE LAB")
        self.resize(1260, 800)
        self.setStyleSheet("""
            QWidget {background:#0c1521;color:#deebf0;font-family:'Segoe UI';font-size:12px}
            QPushButton {background:#203248;border:1px solid #345064;border-radius:7px;
                        padding:9px 13px;font-weight:600}
            QPushButton:hover {background:#294c5d}
            QLabel#heading {font-size:19px;color:#64e6cf;font-weight:700}
            QLabel#value {font-size:16px;font-weight:600;color:#f0f6f6}
            QLabel#muted {color:#91a6b4}
        """)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("TTI  /  LIVE LAB")
        title.setObjectName("heading")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QtWidgets.QLabel(
            "IMAGE-SPACE MOTION  •  NOT VERIFIED PLAYER / BALL / SHOT DETECTION"))
        outer.addLayout(header)
        body = QtWidgets.QHBoxLayout()
        outer.addLayout(body, 1)
        left = QtWidgets.QVBoxLayout()
        body.addLayout(left, 3)
        self.video = VideoView()
        self.video.regions_changed.connect(self._regions_changed)
        left.addWidget(self.video, 1)
        row = QtWidgets.QHBoxLayout()
        left.addLayout(row)
        for label, callback in (
            ("Select player ROI", lambda: self.video.choose("target")),
            ("Background reference", lambda: self.video.choose("reference")),
            ("Clear ROIs", self.video.clear_rois),
        ):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        note = QtWidgets.QLabel(
            "Drag to draw. Move an ROI by its border; resize with bottom-right grip. "
            "Place this window away from the captured browser video.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        left.addWidget(note)
        controls = QtWidgets.QHBoxLayout()
        left.addLayout(controls)
        for label, callback in (
            ("START / RESUME", self._start),
            ("PAUSE", self._pause),
            ("RESET", self._reset),
            ("EXPORT CSV", self._export),
            ("EXIT", self.close),
        ):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(callback)
            controls.addWidget(b)
        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, 2)
        self.status = QtWidgets.QLabel("Ready  /  no capture active")
        self.status.setObjectName("value")
        right.addWidget(self.status)
        self.velocity = QtWidgets.QLabel("Lateral movement: —")
        self.velocity.setObjectName("value")
        right.addWidget(self.velocity)
        self.quality = QtWidgets.QLabel("Tracking confidence: —")
        right.addWidget(self.quality)
        self.quiet = QtWidgets.QLabel("Quiet interval: —")
        right.addWidget(self.quiet)
        self.telemetry = QtWidgets.QLabel("Capture —  •  Processing —  •  Display —")
        self.telemetry.setWordWrap(True)
        right.addWidget(self.telemetry)
        self.event = QtWidgets.QLabel("Latest candidate: none")
        self.event.setWordWrap(True)
        right.addWidget(self.event)
        self.models = QtWidgets.QLabel(
            "ACTIVE: sparse LK flow + reference compensation\n"
            "POSE: unavailable  •  BALL: unavailable\n"
            "TABLE CALIBRATION: unavailable  •  SHOTS: unverified")
        self.models.setObjectName("muted")
        right.addWidget(self.models)
        pg.setConfigOptions(antialias=True, background="#101e2c", foreground="#bbd0da")
        self.plot = pg.PlotWidget(title="Signed lateral displacement  /  processing pixels")
        self.plot.setLabel("left", "relative px")
        self.plot.showGrid(x=True, y=True, alpha=.20)
        self.curve = self.plot.plot(pen=pg.mkPen("#62ead2", width=2))
        right.addWidget(self.plot, 2)
        self.intensity_plot = pg.PlotWidget(title="Motion intensity  /  px per second")
        self.intensity_curve = self.intensity_plot.plot(pen=pg.mkPen("#eab875", width=2))
        self.intensity_plot.showGrid(x=True, y=True, alpha=.20)
        right.addWidget(self.intensity_plot, 2)
        self.footer = QtWidgets.QLabel(
            "Unknown means unknown. Optical-flow quality is heuristic, not calibrated. "
            "Nothing is recorded unless CSV export is requested.")
        self.footer.setObjectName("muted")
        self.footer.setWordWrap(True)
        outer.addWidget(self.footer)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

    def _start(self) -> None:
        self.engine.start()
        self.status.setText("LIVE  /  capture active")

    def _pause(self) -> None:
        self.engine.pause()
        self.status.setText("PAUSED  /  source held")

    def _reset(self) -> None:
        self.engine.reset()
        self.samples.clear()
        self.first_timestamp = None
        self.curve.setData([], [])
        self.intensity_curve.setData([], [])
        self.event.setText("Latest candidate: none")
        self.status.setText("RESET  /  tracking restarted")

    def _regions_changed(self) -> None:
        self.engine.set_regions(*self.video.regions())
        self.samples.clear()
        self.first_timestamp = None

    def _poll(self) -> None:
        result = self.engine.latest()
        if result is None:
            if self.engine.errors:
                self.status.setText("ERROR  /  " + self.engine.errors[-1])
            elif self.engine.finished and not self.engine.running:
                self.status.setText("SOURCE ENDED  /  exit or reset")
            return
        self._render_result(result)

    def _render_result(self, result: Result) -> None:
        sample = result.sample
        self.video.set_frame(result.packet.image, sample.points, sample.reference_points)
        if self.first_timestamp is None:
            self.first_timestamp = sample.timestamp
        t = sample.timestamp - self.first_timestamp
        pos = sample.position.value
        intensity = sample.intensity.value
        self.samples.append((t, pos.x if pos else None,
                             intensity, sample.displacement.confidence,
                             sample.reason, sample.candidate))
        if pos is not None:
            self.velocity.setText(
                f"Lateral: {pos.x:+.1f} relative px  |  Vertical: {pos.y:+.1f} px")
            self.quality.setText(
                f"Tracking quality: {sample.displacement.confidence:.0%}  "
                "(heuristic, not calibrated)")
        else:
            self.velocity.setText(f"Lateral movement: UNKNOWN  ({sample.reason})")
            self.quality.setText("Tracking confidence: UNKNOWN")
        self.quiet.setText(
            f"Low-motion interval: {sample.quiet_seconds:.2f}s (candidate)"
            if sample.quiet_seconds is not None else "Low-motion interval: —")
        if sample.candidate:
            self.event.setText(f"Latest candidate @ {t:.2f}s  /  {sample.candidate}")
        now = time.monotonic()
        self.telemetry.setText(
            f"Capture {self.engine.rate(self.engine.capture_times):.1f} FPS"
            f"  |  Analysis {self.engine.rate(self.engine.processing_times):.1f} FPS"
            f"\nAnalysis {result.analysis_ms:.1f} ms"
            f"  |  Display age {(now - result.grabbed_at) * 1000:.0f} ms"
            f"\nDropped capture {self.engine.dropped_capture}"
            f"  |  Dropped display {self.engine.dropped_display}"
        )
        self.status.setText(
            f"{'SYNTHETIC' if self.engine.source == 'demo' else 'LIVE'}"
            f"  /  {sample.reason.upper()}  /  {t:.1f}s")
        data = list(self.samples)[-300:]
        self.curve.setData(
            [row[0] for row in data if row[1] is not None],
            [row[1] for row in data if row[1] is not None])
        self.intensity_curve.setData(
            [row[0] for row in data if row[2] is not None],
            [row[2] for row in data if row[2] is not None])

    def _export(self) -> None:
        if not self.samples:
            self.status.setText("Nothing to export")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export diagnostics (no video)", "ttintel_live_diagnostics.csv",
            "CSV (*.csv)")
        if not filename:
            return
        with Path(filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("elapsed_s", "relative_x_px", "intensity_px_s",
                             "tracking_quality_heuristic", "state", "candidate"))
            writer.writerows(self.samples)
        self.status.setText(f"Exported {len(self.samples)} diagnostic samples")

    def closeEvent(self, event: object) -> None:
        self.timer.stop()
        self.engine.close()
        super().closeEvent(event)
