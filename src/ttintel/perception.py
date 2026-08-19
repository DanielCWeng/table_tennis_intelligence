"""Baseline perception providers and optional-model boundaries.

The default provider never invents a pose.  If a research model is not
installed, its output is absent/unknown and is recorded as such.  A small
bright-blob ball baseline and JSON annotation provider make the complete
pipeline runnable for fixtures and local development without downloading
large checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters.base import AdapterInfo
from .media import FramePacket
from .schemas import (
    BallState,
    BoundingBox,
    Estimate,
    InferenceType,
    JointState,
    Point2D,
    PoseObservation,
    RacketState,
    Visibility,
)


@dataclass
class FrameDetections:
    poses: list[PoseObservation]
    ball: BallState | None
    rackets: dict[str, RacketState]
    diagnostics: list[str]


class UnavailablePoseEstimator:
    info = AdapterInfo(
        name="pose.unavailable",
        role="2-D human pose",
        environment="none",
        license_status="not_applicable",
    )

    def estimate(self, packet: FramePacket) -> Sequence[PoseObservation]:
        return []


class UnavailableRacketEstimator:
    info = AdapterInfo(
        name="racket.unavailable",
        role="racket keypoints",
        environment="none",
        license_status="not_applicable",
    )

    def estimate(self, packet: FramePacket) -> dict[str, RacketState]:
        return {}


def _bright_components(image: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Return connected bright components as x, y, width, height, area."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return []
    rgb = array[..., :3].astype(np.float32)
    spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    mask = (np.min(rgb, axis=2) >= 205.0) & (spread <= 32.0)
    if not np.any(mask):
        return []
    # A dependency-light flood fill is adequate for the sparse candidate mask
    # used by this diagnostic baseline.
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    for row, col in zip(*np.nonzero(mask)):
        if visited[row, col]:
            continue
        stack = [(int(row), int(col))]
        visited[row, col] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if 2 <= len(points) <= 600:
            ys = [point[0] for point in points]
            xs = [point[1] for point in points]
            components.append((min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, len(points)))
    return components


class BrightBlobBallTracker:
    """Conservative white-ball candidate detector for debugging only."""

    info = AdapterInfo(
        name="ball.bright_blob_baseline",
        role="diagnostic ball candidate",
        environment="numpy",
        license_status="not_applicable",
    )

    def __init__(self) -> None:
        self._previous: Point2D | None = None

    def estimate(self, packet: FramePacket) -> BallState | None:
        components = _bright_components(packet.image)
        if not components:
            self._previous = None
            return BallState(
                image=Estimate.unknown(self.info.name, "no_candidate", "heuristic")
            )
        chosen = self._choose(components)
        x, y, width, height, area = chosen
        point = Point2D(x + width / 2.0, y + height / 2.0)
        roundness = min(width, height) / max(width, height, 1)
        area_confidence = max(0.0, min(1.0, 1.0 - abs(area - 25.0) / 100.0))
        confidence = max(0.05, min(0.75, 0.35 * roundness + 0.65 * area_confidence))
        self._previous = point
        return BallState(
            image=Estimate(
                value=point,
                confidence=confidence,
                source=self.info.name,
                visibility=Visibility.VISIBLE,
                inference_type=InferenceType.MODEL_INFERRED,
                quality_flags=["heuristic"],
            )
        )

    def _choose(self, components: Sequence[tuple[int, int, int, int, int]]) -> tuple[int, int, int, int, int]:
        if self._previous is None:
            return min(components, key=lambda item: abs(item[4] - 25))
        return min(
            components,
            key=lambda item: (
                (item[0] + item[2] / 2.0 - self._previous.x) ** 2
                + (item[1] + item[3] / 2.0 - self._previous.y) ** 2
            ),
        )


def _point_from_json(value: Any) -> Point2D:
    if isinstance(value, Mapping):
        return Point2D(float(value["x"]), float(value["y"]))
    return Point2D(float(value[0]), float(value[1]))


def _estimate_point(value: Any, source: str, confidence: float = 1.0) -> Estimate[Point2D]:
    return Estimate(
        value=_point_from_json(value),
        confidence=float(confidence),
        source=source,
        visibility=Visibility.VISIBLE,
        inference_type=InferenceType.OBSERVED,
    )


class AnnotationProvider:
    """Read small local JSON annotations without bringing in model weights.

    Accepted shape::

        {"frames": [{"frame_id": 0, "poses": [...], "ball": {...}}]}

    Pose boxes are ``[x1, y1, x2, y2]`` and joints are ``{"pelvis": [x, y]}``.
    """

    info = AdapterInfo(
        name="fixture.annotations",
        role="local observed annotations",
        environment="stdlib/json",
        license_status="local_data",
    )

    def __init__(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = payload.get("frames", payload) if isinstance(payload, Mapping) else payload
        self._frames: dict[int, Mapping[str, Any]] = {
            int(item["frame_id"]): item for item in entries
        }

    def infer(self, packet: FramePacket) -> FrameDetections:
        item = self._frames.get(packet.frame_id, {})
        poses: list[PoseObservation] = []
        for index, raw_pose in enumerate(item.get("poses", [])):
            raw_box = raw_pose.get("bbox")
            if raw_box is None:
                continue
            box = BoundingBox(*(float(value) for value in raw_box))
            bbox = Estimate.observed(box, float(raw_pose.get("confidence", 1.0)), self.info.name)
            joints: dict[str, JointState] = {}
            for name, raw_joint in raw_pose.get("joints", {}).items():
                joint_value = raw_joint.get("point", raw_joint) if isinstance(raw_joint, Mapping) else raw_joint
                confidence = raw_joint.get("confidence", 1.0) if isinstance(raw_joint, Mapping) else 1.0
                joints[name] = JointState(name, _estimate_point(joint_value, self.info.name, confidence))
            poses.append(
                PoseObservation(
                    observation_id=str(raw_pose.get("id", f"annotation-{packet.frame_id}-{index}")),
                    bbox=bbox,
                    joints=joints,
                    detector_track_id=str(raw_pose["track_id"]) if raw_pose.get("track_id") is not None else None,
                    source_model=self.info.name,
                )
            )
        ball = None
        if item.get("ball") is not None:
            raw_ball = item["ball"]
            raw_point = raw_ball.get("point", raw_ball.get("image", raw_ball))
            ball = BallState(
                image=_estimate_point(raw_point, self.info.name, raw_ball.get("confidence", 1.0)),
            )
        return FrameDetections(poses=poses, ball=ball, rackets={}, diagnostics=[])


class DefaultPerceptionProvider:
    """Compose local annotations with the lightweight ball fallback."""

    def __init__(
        self,
        annotations: AnnotationProvider | None = None,
        *,
        ball_tracker: Any | None = None,
        pose_estimator: Any | None = None,
        racket_estimator: Any | None = None,
        use_bright_blob: bool = True,
    ) -> None:
        self.annotations = annotations
        self.ball_tracker = (
            ball_tracker
            if ball_tracker is not None
            else (BrightBlobBallTracker() if use_bright_blob else None)
        )
        self.pose_estimator = pose_estimator or UnavailablePoseEstimator()
        self.racket_estimator = racket_estimator or UnavailableRacketEstimator()

    def infer(self, packet: FramePacket) -> FrameDetections:
        if self.annotations is not None:
            detections = self.annotations.infer(packet)
        else:
            detections = FrameDetections([], None, {}, [])
        if not detections.poses and not isinstance(self.pose_estimator, UnavailablePoseEstimator):
            detections.poses = list(self.pose_estimator.estimate(packet))
            detections.diagnostics.append(self.pose_estimator.info.name)
        elif isinstance(self.pose_estimator, UnavailablePoseEstimator):
            detections.diagnostics.append("pose_model_unavailable")
        if not detections.rackets and not isinstance(self.racket_estimator, UnavailableRacketEstimator):
            detections.rackets = dict(self.racket_estimator.estimate(packet))
            detections.diagnostics.append(self.racket_estimator.info.name)
        if detections.ball is None and self.ball_tracker is not None:
            detections.ball = self.ball_tracker.estimate(packet)
            detections.diagnostics.append(self.ball_tracker.info.name)
        return detections
