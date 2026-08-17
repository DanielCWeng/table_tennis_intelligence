"""Deterministic measurements kept separate from interpretation."""

from __future__ import annotations

from math import acos, degrees, hypot
from typing import Any, Sequence

from .schemas import Estimate, Event, EventType, FrameState, InferenceType, Point2D, Visibility


def _angle(a: Point2D, b: Point2D, c: Point2D) -> float:
    first = (a.x - b.x, a.y - b.y)
    second = (c.x - b.x, c.y - b.y)
    denominator = hypot(*first) * hypot(*second)
    if denominator == 0.0:
        raise ValueError("angle points must not be coincident")
    cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / denominator))
    return degrees(acos(cosine))


def joint_angle(
    frame: FrameState,
    player_id: str,
    first: str,
    vertex: str,
    last: str,
) -> Estimate[float]:
    player = frame.players.get(player_id)
    if player is None:
        return Estimate.unknown("joint_angle", "player_missing")
    joints = [player.joint(name) for name in (first, vertex, last)]
    if any(joint is None or joint.image.value is None for joint in joints):
        return Estimate.unknown("joint_angle", "joint_missing")
    assert all(joint is not None for joint in joints)
    confidence = min(joint.image.confidence for joint in joints if joint is not None)
    try:
        value = _angle(
            joints[0].image.value,  # type: ignore[union-attr]
            joints[1].image.value,  # type: ignore[union-attr]
            joints[2].image.value,  # type: ignore[union-attr]
        )
    except ValueError:
        return Estimate.unknown("joint_angle", "degenerate_geometry")
    return Estimate(
        value=value,
        confidence=confidence,
        source="image_joint_geometry",
        visibility=Visibility.VISIBLE,
        inference_type=InferenceType.DERIVED,
        quality_flags=["image_space_measurement"],
    )


def _player_centre(frame: FrameState, player_id: str) -> Point2D | None:
    player = frame.players.get(player_id)
    if player is None:
        return None
    return player.centre_image()


def movement_onset(
    frames: Sequence[FrameState],
    player_id: str,
    *,
    reference_timestamp: float | None = None,
    displacement_px: float = 8.0,
    window_frames: int = 3,
) -> Estimate[float]:
    """Return the first timestamp with sustained image-space displacement."""

    points = [(frame.timestamp, _player_centre(frame, player_id)) for frame in frames]
    points = [(timestamp, point) for timestamp, point in points if point is not None]
    if len(points) < window_frames + 1:
        return Estimate.unknown("movement_onset", "insufficient_track")
    baseline = points[0][1]
    for index in range(window_frames, len(points)):
        current = points[index][1]
        distance = hypot(current.x - baseline.x, current.y - baseline.y)
        if distance >= displacement_px:
            timestamp = points[index][0]
            if reference_timestamp is not None:
                timestamp = timestamp - reference_timestamp
            return Estimate(
                value=timestamp,
                confidence=min(1.0, len(points) / 20.0),
                source="player_track_displacement",
                visibility=Visibility.VISIBLE,
                inference_type=InferenceType.DERIVED,
                quality_flags=["image_space_proxy"],
            )
    return Estimate.unknown("movement_onset", "threshold_not_reached")


def frame_measurements(frame: FrameState) -> dict[str, Any]:
    measurements: dict[str, Any] = {
        "frame_id": frame.frame_id,
        "timestamp": frame.timestamp,
        "players": {},
    }
    for player_id in frame.players:
        measurements["players"][player_id] = {
            "elbow_angle_left": joint_angle(frame, player_id, "left_shoulder", "left_elbow", "left_wrist"),
            "elbow_angle_right": joint_angle(frame, player_id, "right_shoulder", "right_elbow", "right_wrist"),
            "knee_angle_left": joint_angle(frame, player_id, "left_hip", "left_knee", "left_ankle"),
            "knee_angle_right": joint_angle(frame, player_id, "right_hip", "right_knee", "right_ankle"),
            "position_image": player_centre_estimate(frame, player_id),
        }
    return measurements


def player_centre_estimate(frame: FrameState, player_id: str) -> Estimate[Point2D]:
    point = _player_centre(frame, player_id)
    player = frame.players.get(player_id)
    if point is None or player is None:
        return Estimate.unknown("player_centre", "player_missing")
    confidence = player.bbox.confidence if player.bbox else 0.0
    return Estimate(
        value=point,
        confidence=confidence,
        source=player.source_model,
        visibility=Visibility.VISIBLE,
        inference_type=InferenceType.DERIVED,
        quality_flags=["image_space_position"],
    )


def compute_analytics(frames: Sequence[FrameState], events: Sequence[Event]) -> dict[str, Any]:
    player_ids = sorted({player_id for frame in frames for player_id in frame.players})
    per_player: dict[str, Any] = {}
    for player_id in player_ids:
        available = sum(1 for frame in frames if player_id in frame.players)
        per_player[player_id] = {
            "pose_availability": available / len(frames) if frames else 0.0,
            "movement_onset": movement_onset(frames, player_id),
        }
    event_counts: dict[str, int] = {}
    for event in events:
        event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1
    return {
        "frame_measurements": [frame_measurements(frame) for frame in frames],
        "summary": {
            "frame_count": len(frames),
            "ball_availability": sum(1 for frame in frames if frame.ball and frame.ball.image.value is not None) / len(frames) if frames else 0.0,
            "players": per_player,
            "event_counts": event_counts,
            "contact_candidates": event_counts.get(EventType.PLAYER_CONTACT.value, 0),
            "bounce_candidates": event_counts.get(EventType.TABLE_BOUNCE.value, 0),
        },
    }
