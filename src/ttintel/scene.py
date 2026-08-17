"""Shot boundaries and deterministic gameplay-segment classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .media import FramePacket
from .schemas import CameraSegment, SegmentType, VideoMetadata


@dataclass(frozen=True)
class CutCandidate:
    frame_id: int
    timestamp: float
    difference_score: float


@dataclass(frozen=True)
class GameplayEvidence:
    table_detected: bool = False
    calibration_succeeded: bool = False
    sufficient_table_visible: bool = False
    plausible_player_tracks: int = 0
    camera_static: bool = False
    ball_evidence: bool = False
    moving_camera: bool = False
    replay_evidence: bool = False


def _thumbnail(image: np.ndarray, width: int = 64, height: int = 36) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)
    if array.ndim != 2:
        raise ValueError("frame image must be a 2-D or 3-D array")
    y_indices = np.linspace(0, array.shape[0] - 1, height).astype(int)
    x_indices = np.linspace(0, array.shape[1] - 1, width).astype(int)
    return array[np.ix_(y_indices, x_indices)].astype(np.float32)


def frame_difference(previous: np.ndarray, current: np.ndarray) -> float:
    """Return a normalised mean absolute frame difference in [0, 1]."""

    left = _thumbnail(previous)
    right = _thumbnail(current)
    difference = float(np.mean(np.abs(left - right)))
    scale = max(1.0, float(np.max(np.abs(left))) - float(np.min(np.abs(left))))
    return max(0.0, min(1.0, difference / scale / 255.0 * 4.0))


def detect_cuts(
    frames: Sequence[FramePacket],
    *,
    threshold: float = 0.45,
    min_interval_seconds: float = 0.25,
) -> list[CutCandidate]:
    """Detect hard cuts with a small, reproducible thumbnail baseline.

    This is intentionally a baseline rather than a replacement for TransNetV2.
    Every candidate remains inspectable through its difference score.
    """

    cuts: list[CutCandidate] = []
    previous: FramePacket | None = None
    last_cut_time = float("-inf")
    for packet in frames:
        if previous is not None:
            score = frame_difference(previous.image, packet.image)
            if score >= threshold and packet.timestamp - last_cut_time >= min_interval_seconds:
                cuts.append(CutCandidate(packet.frame_id, packet.timestamp, score))
                last_cut_time = packet.timestamp
        previous = packet
    return cuts


def build_segments(
    video: VideoMetadata,
    frames: Sequence[FramePacket],
    cuts: Sequence[CutCandidate] = (),
) -> list[CameraSegment]:
    if not frames:
        end = float(video.duration or 0.0)
        return [CameraSegment(f"{video.video_id}-seg-000", video.video_id, 0.0, end)]
    boundaries = [float(frames[0].timestamp)] + [float(cut.timestamp) for cut in cuts]
    end_time = float(video.duration if video.duration is not None else frames[-1].timestamp)
    if end_time < frames[-1].timestamp:
        end_time = frames[-1].timestamp
    segments: list[CameraSegment] = []
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else end_time
        if end <= start:
            continue
        segments.append(
            CameraSegment(
                segment_id=f"{video.video_id}-seg-{index:03d}",
                video_id=video.video_id,
                start_time=start,
                end_time=end,
            )
        )
    return segments


def gameplay_quality_score(evidence: GameplayEvidence) -> float:
    """Score only evidence actually supplied by perception components."""

    score = 0.0
    score += 0.22 if evidence.table_detected else 0.0
    score += 0.24 if evidence.calibration_succeeded else 0.0
    score += 0.12 if evidence.sufficient_table_visible else 0.0
    score += 0.18 if evidence.plausible_player_tracks >= 2 else 0.09 if evidence.plausible_player_tracks == 1 else 0.0
    score += 0.12 if evidence.camera_static else 0.0
    score += 0.12 if evidence.ball_evidence else 0.0
    score -= 0.35 if evidence.moving_camera else 0.0
    score -= 0.20 if evidence.replay_evidence else 0.0
    return max(0.0, min(1.0, score))


def classify_segment(segment: CameraSegment, evidence: GameplayEvidence) -> CameraSegment:
    score = gameplay_quality_score(evidence)
    segment.gameplay_quality_score = score
    if evidence.moving_camera:
        segment.segment_type = SegmentType.MOVING_CAMERA
    elif evidence.replay_evidence:
        segment.segment_type = SegmentType.REPLAY
    elif score >= 0.70 and evidence.plausible_player_tracks >= 2:
        segment.segment_type = SegmentType.VALID_GAMEPLAY
    elif evidence.table_detected and evidence.plausible_player_tracks == 1:
        segment.segment_type = SegmentType.PLAYER_CLOSEUP
    elif evidence.table_detected and not evidence.calibration_succeeded:
        segment.segment_type = SegmentType.INVALID_TABLE_VIEW
    else:
        segment.segment_type = SegmentType.UNKNOWN
    return segment


def segment_for_timestamp(segments: Iterable[CameraSegment], timestamp: float) -> CameraSegment | None:
    for segment in segments:
        if segment.start_time <= timestamp <= segment.end_time:
            return segment
    return None
