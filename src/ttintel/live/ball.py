"""Human-seeded ball localisation for Live Lab.

A click is a user observation. Subsequent matches are visual-template
*proposals*, not TOTNet detections or ball/shot ground truth. Once evidence is
insufficient the trail breaks and the user must explicitly reseed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ttintel.media import FramePacket
from ttintel.schemas import Estimate, InferenceType, Point2D, Visibility


@dataclass(frozen=True)
class BallObservation:
    frame_id: int
    timestamp: float
    position: Estimate[Point2D]
    state: str  # manual, assisted, not_seeded, or a specific loss reason

    @property
    def valid(self) -> bool:
        return self.position.value is not None


class BallAssistTracker:
    """Conservative local template matcher, opt-in via an explicit user click.

    Coordinates are ORIGINAL RGB image pixels, independent of the downsampled
    movement tracker. Similarity/confidence describes patch consistency only;
    it is not a calibrated ball-identification probability.
    """

    def __init__(self, patch_radius: int = 9, max_gap: float = .40,
                 min_similarity: float = .76, min_uniqueness: float = .035):
        if patch_radius < 3 or max_gap <= 0 or not (0 < min_similarity <= 1):
            raise ValueError("invalid ball-assist configuration")
        self.patch_radius = patch_radius
        self.max_gap = max_gap
        self.min_similarity = min_similarity
        self.min_uniqueness = min_uniqueness
        self.reset()

    def reset(self) -> None:
        self.template: np.ndarray | None = None
        self.point: tuple[float, float] | None = None
        self.frame_id: int | None = None
        self.timestamp: float | None = None
        self.shape: tuple[int, int] | None = None
        self.seed_observation: BallObservation | None = None
        self.lost_reason = "not_seeded"

    @staticmethod
    def _gray(packet: FramePacket) -> np.ndarray:
        import cv2
        frame = np.asarray(packet.image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("ball assist needs RGB video")
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _unknown(packet: FramePacket, state: str) -> BallObservation:
        return BallObservation(
            packet.frame_id, packet.timestamp,
            Estimate.unknown("live.ball_assist" if state != "not_seeded"
                             else "live.ball_assist.unavailable",
                             "heuristic", state),
            state,
        )

    def seed(self, packet: FramePacket, point: tuple[float, float]) -> BallObservation:
        gray = self._gray(packet)
        h, w = gray.shape
        x, y = map(float, point)
        if not (math.isfinite(x) and math.isfinite(y)
                and 0 <= x < w and 0 <= y < h):
            raise ValueError("ball click is outside the frame")
        ix, iy = round(x), round(y)
        r = self.patch_radius
        x0, y0 = max(0, ix - r), max(0, iy - r)
        x1, y1 = min(w, ix + r + 1), min(h, iy + r + 1)
        patch = gray[y0:y1, x0:x1].copy()
        if min(patch.shape) < 7 or float(patch.std()) < 4.0:
            raise ValueError("ball patch has insufficient texture; click with visible background")
        self.template = patch
        self.point = (float(ix), float(iy))
        self.offset = (ix - x0, iy - y0)
        self.frame_id = packet.frame_id
        self.timestamp = packet.timestamp
        self.shape = (w, h)
        self.lost_reason = ""
        self.seed_observation = BallObservation(
            packet.frame_id, packet.timestamp,
            Estimate.observed(Point2D(float(ix), float(iy)), 1.0,
                              "live.ball_user_click", Visibility.VISIBLE, "user_annotated"),
            "manual",
        )
        return self.seed_observation

    def _lose(self, packet: FramePacket, reason: str) -> BallObservation:
        self.reset()
        self.lost_reason = reason
        return self._unknown(packet, reason)

    def process(self, packet: FramePacket, scene_cut: bool = False) -> BallObservation:
        if self.template is None:
            state = "not_seeded" if self.lost_reason == "not_seeded" else "needs_reseed"
            return self._unknown(packet, state)
        if packet.frame_id == self.frame_id and self.seed_observation is not None:
            return self.seed_observation
        if packet.frame_id <= self.frame_id:
            # A queued frame predating the click cannot undo a recent manual seed.
            return self._unknown(packet, "stale_before_seed")
        dt = packet.timestamp - float(self.timestamp)
        if scene_cut:
            return self._lose(packet, "scene_cut")
        if not math.isfinite(dt) or not 0 < dt <= self.max_gap:
            return self._lose(packet, "frame_gap")
        gray = self._gray(packet)
        h, w = gray.shape
        if (w, h) != self.shape:
            return self._lose(packet, "dimensions_changed")
        import cv2
        px, py = self.point
        th, tw = self.template.shape
        # Wide enough for a fast ball, but bounded to avoid matching remote
        # distractors across an entire broadcast frame.
        radius = min(170, max(36, round(3000 * dt)))
        x0 = max(0, round(px) - radius - tw)
        y0 = max(0, round(py) - radius - th)
        x1 = min(w, round(px) + radius + tw)
        y1 = min(h, round(py) + radius + th)
        crop = gray[y0:y1, x0:x1]
        if crop.shape[0] < th or crop.shape[1] < tw:
            return self._lose(packet, "out_of_frame")
        scores = cv2.matchTemplate(crop, self.template, cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(scores).all():
            return self._lose(packet, "invalid_similarity")
        _, peak, _, loc = cv2.minMaxLoc(scores)
        cx, cy = x0 + loc[0] + self.offset[0], y0 + loc[1] + self.offset[1]
        if math.hypot(cx - px, cy - py) > radius:
            return self._lose(packet, "motion_outside_search")
        # Other high-scoring peaks are evidence of ambiguity (white table
        # markings, bright shirts, repeated background patterns, etc.).
        rivals = scores.copy()
        block = max(4, min(th, tw) // 2)
        rivals[max(0, loc[1] - block):loc[1] + block + 1,
               max(0, loc[0] - block):loc[0] + block + 1] = -1
        second = float(np.max(rivals))
        uniqueness = float(peak - second)
        if peak < self.min_similarity or uniqueness < self.min_uniqueness:
            return self._lose(packet, "ambiguous_or_weak_match")
        self.point = (float(cx), float(cy))
        self.frame_id = packet.frame_id
        self.timestamp = packet.timestamp
        # Confidence is a patch-quality heuristic, NEVER ball-classification
        # accuracy, and there is no trajectory interpolation across losses.
        confidence = float(np.clip(min(peak, uniqueness / .20), 0, 1))
        return BallObservation(
            packet.frame_id, packet.timestamp,
            Estimate(Point2D(float(cx), float(cy)), confidence,
                     "live.ball_template_match", Visibility.VISIBLE,
                     InferenceType.TEMPORALLY_TRACKED, ["heuristic", "user_seeded"]),
            "assisted",
        )
