"""Dependency-light live image-space motion observations.

Movement is optical-flow evidence in pixels, not a detected athlete, a shot,
or physical distance. Invalid frames produce unknown Estimates, never zeros
masquerading as measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math

import numpy as np

from ttintel.media import FramePacket
from ttintel.schemas import Estimate, InferenceType, Point2D, Visibility


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 32 or self.height < 32:
            raise ValueError("capture region must be at least 32x32")

    @classmethod
    def parse(cls, value: str) -> "Region":
        try:
            bits = [int(s.strip()) for s in value.split(",")]
            if len(bits) != 4:
                raise ValueError()
            return cls(*bits)
        except ValueError as exc:
            raise ValueError("region must be left,top,width,height") from exc

    def as_mss(self) -> dict[str, int]:
        return dict(left=self.left, top=self.top, width=self.width, height=self.height)


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def clamp(self, width: int, height: int) -> "ROI | None":
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(width, self.x + self.width), min(height, self.y + self.height)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        return ROI(x0, y0, x1 - x0, y1 - y0)


def region_from_drag(monitor: Region, start: tuple[float, float],
                     end: tuple[float, float], view_size: tuple[int, int]) -> Region | None:
    """Transform reversed or clipped GUI drag coordinates to physical screen px."""
    vw, vh = view_size
    if vw <= 0 or vh <= 0:
        raise ValueError("view size must be positive")
    x0, x1 = sorted((max(0, min(vw, start[0])), max(0, min(vw, end[0]))))
    y0, y1 = sorted((max(0, min(vh, start[1])), max(0, min(vh, end[1]))))
    left = monitor.left + round(x0 * monitor.width / vw)
    top = monitor.top + round(y0 * monitor.height / vh)
    right = monitor.left + round(x1 * monitor.width / vw)
    bottom = monitor.top + round(y1 * monitor.height / vh)
    if right - left < 32 or bottom - top < 32:
        return None
    return Region(left, top, right - left, bottom - top)


@dataclass(frozen=True)
class MotionSample:
    frame_id: int
    timestamp: float
    displacement: Estimate[Point2D]
    intensity: Estimate[float]  # image-space motion px/s, processed image scale
    position: Estimate[Point2D]  # integrated relative displacement, not world coordinates
    reason: str
    points: tuple[tuple[float, float], ...] = ()
    reference_points: tuple[tuple[float, float], ...] = ()
    candidate: str | None = None
    quiet_seconds: float | None = None


class LiveMotionTracker:
    """Sparse forward/backward LK flow, background compensated when supported.

    Source frames are RGB FramePackets. Confidence describes tracking quality,
    not the likelihood that the ROI contains a table-tennis player.
    """

    def __init__(self, width: int = 384, max_gap: float = 0.40,
                 cut_threshold: float = 65.0) -> None:
        if width < 128 or max_gap <= 0:
            raise ValueError("invalid tracker configuration")
        self.width = width
        self.max_gap = max_gap
        self.cut_threshold = cut_threshold
        self.roi: ROI | None = None
        self.reference: ROI | None = None
        self.reset()

    def reset(self) -> None:
        self._previous: np.ndarray | None = None
        self._timestamp: float | None = None
        self._position = np.zeros(2, np.float64)
        self._recent: deque[float] = deque(maxlen=10)
        self._direction = 0
        self._leg = 0.0
        self._last_turn: float | None = None
        self._quiet_since: float | None = None

    def set_regions(self, roi: ROI | None, reference: ROI | None = None) -> None:
        if (roi, reference) != (self.roi, self.reference):
            self.roi, self.reference = roi, reference
            self.reset()

    def _unknown(self, packet: FramePacket, reason: str) -> MotionSample:
        flags = ["heuristic", reason]
        return MotionSample(
            packet.frame_id, packet.timestamp,
            Estimate.unknown("live.lk_flow", *flags),
            Estimate.unknown("live.lk_flow", *flags),
            Estimate.unknown("live.lk_flow", *flags),
            reason,
        )

    @staticmethod
    def _features(gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        import cv2
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=120, qualityLevel=0.01, minDistance=5,
            blockSize=5, mask=mask,
        )

    @staticmethod
    def _track(old: np.ndarray, new: np.ndarray, points: np.ndarray | None
               ) -> tuple[np.ndarray, np.ndarray, float]:
        import cv2
        empty = np.empty((0, 2), np.float32)
        if points is None or len(points) < 3:
            return empty, empty, 0.0
        fwd, ok, _ = cv2.calcOpticalFlowPyrLK(old, new, points, None)
        if fwd is None or ok is None:
            return empty, empty, 0.0
        back, rev, _ = cv2.calcOpticalFlowPyrLK(new, old, fwd, None)
        if back is None or rev is None:
            return empty, empty, 0.0
        a, b, c = points[:, 0], fwd[:, 0], back[:, 0]
        mask = ((ok[:, 0] == 1) & (rev[:, 0] == 1)
                & np.isfinite(b).all(axis=1) & np.isfinite(c).all(axis=1)
                & (np.linalg.norm(a - c, axis=1) <= 1.5)
                & (np.linalg.norm(b - a, axis=1) < 45))
        return a[mask], b[mask], float(np.mean(mask))

    @staticmethod
    def _mask(shape: tuple[int, int], roi: ROI) -> np.ndarray:
        mask = np.zeros(shape, np.uint8)
        mask[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width] = 255
        return mask

    def process(self, packet: FramePacket) -> MotionSample:
        import cv2
        image = np.asarray(packet.image)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
            raise ValueError("live expects an RGB image at least 32x32")
        if not math.isfinite(packet.timestamp):
            self.reset()
            return self._unknown(packet, "invalid_timestamp")
        h, w = image.shape[:2]
        scale = min(1.0, self.width / w)
        small = (cv2.resize(image, (round(w * scale), round(h * scale)),
                            interpolation=cv2.INTER_AREA) if scale < 1 else image)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        sh, sw = gray.shape
        raw = self.roi or ROI(int(w * .19), int(h * .17), int(w * .62), int(h * .66))
        target = ROI(round(raw.x * scale), round(raw.y * scale),
                     round(raw.width * scale), round(raw.height * scale)).clamp(sw, sh)
        if target is None:
            return self._unknown(packet, "invalid_roi")
        tmask = self._mask((sh, sw), target)
        if self.reference:
            r = self.reference
            ref = ROI(round(r.x * scale), round(r.y * scale),
                      round(r.width * scale), round(r.height * scale)).clamp(sw, sh)
            rmask = self._mask((sh, sw), ref) if ref else np.zeros_like(tmask)
        else:
            rmask = cv2.bitwise_not(tmask)
            rmask = cv2.erode(rmask, np.ones((5, 5), np.uint8))
        rmask[tmask > 0] = 0  # foreground may never contaminate camera evidence
        prev, prev_time = self._previous, self._timestamp
        self._previous, self._timestamp = gray, packet.timestamp
        if prev is None or prev.shape != gray.shape:
            self._position[:] = 0
            return self._unknown(packet, "warmup")
        dt = packet.timestamp - float(prev_time)
        if dt <= 0 or dt > self.max_gap:
            self.reset()
            self._previous, self._timestamp = gray, packet.timestamp
            return self._unknown(packet, "frame_gap")
        if float(np.mean(cv2.absdiff(prev, gray))) >= self.cut_threshold:
            self.reset()
            self._previous, self._timestamp = gray, packet.timestamp
            return self._unknown(packet, "camera_cut")
        a, b, ratio = self._track(prev, gray, self._features(prev, tmask))
        ca, cb, _ = self._track(prev, gray, self._features(prev, rmask))
        if len(b):
            x = np.clip(b[:, 0].astype(int), 0, sw - 1)
            y = np.clip(b[:, 1].astype(int), 0, sh - 1)
            keep = tmask[y, x] > 0
            a, b = a[keep], b[keep]
        if len(cb):
            x = np.clip(cb[:, 0].astype(int), 0, sw - 1)
            y = np.clip(cb[:, 1].astype(int), 0, sh - 1)
            keep = rmask[y, x] > 0
            ca, cb = ca[keep], cb[keep]
        if len(b) < 5:
            self._direction, self._leg = 0, 0.0
            return self._unknown(packet, "tracking_loss")
        if len(cb) < 5:
            return self._unknown(packet, "reference_loss")
        camera = np.median(cb - ca, axis=0)
        residual = (b - a) - camera
        # Reduce influence of inconsistent points without inventing motion.
        central = np.median(residual, axis=0)
        errors = np.linalg.norm(residual - central, axis=1)
        keep = errors <= max(2.5, float(np.median(errors)) * 3)
        if np.count_nonzero(keep) < 5:
            return self._unknown(packet, "incoherent_flow")
        corrected = np.median(residual[keep], axis=0)
        coherence = float(np.mean(keep))
        quality = max(0., min(1., ratio * coherence * min(1., len(b) / 24)))
        self._position += corrected
        intensity = float(np.linalg.norm(corrected) / dt)
        event = None
        direction = 1 if corrected[0] >= 0.65 else -1 if corrected[0] <= -0.65 else 0
        if direction:
            if self._direction and direction != self._direction and self._leg >= 2.0:
                if self._last_turn is None or packet.timestamp - self._last_turn > .18:
                    event = "lateral_direction_change_candidate"
                    self._last_turn = packet.timestamp
            self._leg = self._leg + abs(float(corrected[0])) if direction == self._direction else abs(float(corrected[0]))
            self._direction = direction
            self._quiet_since = None
        elif intensity < 5.0:
            if self._quiet_since is None:
                self._quiet_since = packet.timestamp
        flags = ["heuristic"]
        displacement = Estimate(
            Point2D(float(corrected[0]), float(corrected[1])), quality,
            "live.lk_flow", Visibility.VISIBLE, InferenceType.TEMPORALLY_TRACKED, flags,
        )
        speed = Estimate(intensity, quality, "live.lk_flow", Visibility.VISIBLE,
                         InferenceType.DERIVED, flags)
        pos = Estimate(Point2D(float(self._position[0]), float(self._position[1])),
                       quality, "live.lk_flow", Visibility.VISIBLE,
                       InferenceType.DERIVED, flags)
        return MotionSample(
            packet.frame_id, packet.timestamp, displacement, speed, pos, "tracking",
            tuple((float(p[0] / scale), float(p[1] / scale)) for p in b[keep]),
            tuple((float(p[0] / scale), float(p[1] / scale)) for p in cb),
            event, packet.timestamp - self._quiet_since if self._quiet_since is not None else None,
        )
