"""Conservative bounce/contact candidates with explicit evidence."""

from __future__ import annotations

from math import hypot
from typing import Iterable, Sequence

import numpy as np

from .calibration import image_point_to_table
from .schemas import (
    BallState,
    Event,
    EventType,
    FrameState,
    InferenceType,
    PlayerState,
    Point2D,
    QualityFlag,
    TableCalibration,
)


def _ball_point(frame: FrameState) -> Point2D | None:
    if frame.ball is None or frame.ball.image.value is None:
        return None
    return frame.ball.image.value


def _ball_confidence(frame: FrameState) -> float:
    return frame.ball.image.confidence if frame.ball is not None else 0.0


def _inside_table(point: Point2D, calibration: TableCalibration | None) -> tuple[bool, Point2D | None]:
    if calibration is None:
        return True, None
    try:
        table_point = image_point_to_table(calibration, point)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return False, None
    from .geometry import TABLE_LENGTH_M, TABLE_WIDTH_M

    tolerance = 0.08
    return (
        -tolerance <= table_point.x <= TABLE_LENGTH_M + tolerance
        and -tolerance <= table_point.y <= TABLE_WIDTH_M + tolerance,
        table_point,
    )


def detect_bounce_candidates(
    frames: Sequence[FrameState],
    *,
    calibration: TableCalibration | None = None,
    min_vertical_motion_px: float = 1.5,
) -> list[Event]:
    """Find image-space downward-to-upward reversals near the table plane.

    In the absence of 3-D ball reconstruction this remains a candidate event,
    never a confirmed bounce.  A calibration, when supplied, adds a table
    surface check and a table-coordinate point to the evidence path.
    """

    events: list[Event] = []
    for index in range(1, len(frames) - 1):
        previous, current, following = frames[index - 1], frames[index], frames[index + 1]
        p0, p1, p2 = _ball_point(previous), _ball_point(current), _ball_point(following)
        if p0 is None or p1 is None or p2 is None:
            continue
        incoming = p1.y - p0.y
        outgoing = p2.y - p1.y
        if incoming < min_vertical_motion_px or outgoing > -min_vertical_motion_px:
            continue
        inside, table_point = _inside_table(p1, calibration)
        if not inside:
            continue
        confidence = min(1.0, 0.25 + 0.20 * min(incoming, 8.0) / 8.0 + 0.20 * min(-outgoing, 8.0) / 8.0 + 0.35 * _ball_confidence(current))
        if calibration is None:
            confidence *= 0.65
        ball_after = current.ball
        if table_point is not None and ball_after is not None:
            from .schemas import Estimate, Visibility

            ball_after = BallState(
                image=ball_after.image,
                world=ball_after.world,
                table_xy=Estimate(
                    value=table_point,
                    confidence=confidence,
                    source="table_homography",
                    visibility=Visibility.VISIBLE,
                    inference_type=InferenceType.PHYSICS_INFERRED,
                    quality_flags=[QualityFlag.HEURISTIC.value],
                ),
                velocity=ball_after.velocity,
                blur_length=ball_after.blur_length,
                blur_direction=ball_after.blur_direction,
            )
        events.append(
            Event(
                event_id=f"bounce-{current.frame_id}",
                timestamp=current.timestamp,
                frame_id=current.frame_id,
                event_type=EventType.TABLE_BOUNCE,
                ball_before=previous.ball,
                ball_after=ball_after,
                confidence=confidence,
                evidence=[
                    "image_vertical_direction_reversal",
                    "ball_present_in_three_consecutive_frames",
                    "table_surface_check" if calibration is not None else "no_3d_surface_check",
                ],
                quality_flags=[QualityFlag.HEURISTIC.value],
            )
        )
    return events


def _candidate_player_contact(frame: FrameState, *, radius_px: float) -> tuple[str, float, str] | None:
    point = _ball_point(frame)
    if point is None:
        return None
    best: tuple[str, float, str] | None = None
    for player_id, player in frame.players.items():
        candidate_points: list[tuple[str, Point2D]] = []
        for joint_name in ("wrist", "left_wrist", "right_wrist", "hand", "left_hand", "right_hand"):
            joint = player.joint(joint_name)
            if joint and joint.image.value is not None:
                candidate_points.append((joint_name, joint.image.value))
        racket = frame.rackets.get(player_id)
        if racket and racket.centre and racket.centre.value is not None:
            candidate_points.append(("racket", racket.centre.value))
        for label, candidate in candidate_points:
            distance = hypot(candidate.x - point.x, candidate.y - point.y)
            if distance <= radius_px and (best is None or distance < best[1]):
                best = (player_id, distance, label)
    return best


def detect_contact_candidates(
    frames: Sequence[FrameState],
    *,
    radius_px: float = 70.0,
    cooldown_seconds: float = 0.12,
) -> list[Event]:
    events: list[Event] = []
    last_timestamp = float("-inf")
    for frame in frames:
        candidate = _candidate_player_contact(frame, radius_px=radius_px)
        if candidate is None or frame.timestamp - last_timestamp < cooldown_seconds:
            continue
        player_id, distance, evidence_joint = candidate
        player = frame.players.get(player_id)
        confidence = min(1.0, max(0.05, 0.65 * (1.0 - distance / radius_px) + 0.35 * _ball_confidence(frame)))
        events.append(
            Event(
                event_id=f"contact-{frame.frame_id}",
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                event_type=EventType.PLAYER_CONTACT,
                actor_player_id=player_id,
                ball_before=frame.ball,
                player_state=player,
                confidence=confidence,
                evidence=[
                    f"ball_near_{evidence_joint}",
                    "temporal_sequence_not_yet_confirmed",
                ],
                quality_flags=[QualityFlag.HEURISTIC.value],
            )
        )
        last_timestamp = frame.timestamp
    return events


def infer_events(frames: Sequence[FrameState], *, calibration: TableCalibration | None = None) -> list[Event]:
    events = detect_bounce_candidates(frames, calibration=calibration)
    events.extend(detect_contact_candidates(frames))
    events.sort(key=lambda event: (event.timestamp, event.event_type.value))
    return events


def add_rally_boundaries(events: Sequence[Event], *, gap_seconds: float = 1.5) -> list[Event]:
    """Add conservative rally start/end markers around contact candidates."""

    contacts = [event for event in events if event.event_type == EventType.PLAYER_CONTACT]
    if not contacts:
        return list(events)
    result = list(events)
    start = contacts[0]
    result.append(
        Event(
            event_id=f"rally-start-{start.frame_id}",
            timestamp=start.timestamp,
            frame_id=start.frame_id,
            event_type=EventType.RALLY_START,
            confidence=start.confidence * 0.8,
            evidence=["first_contact_in_observed_sequence"],
        )
    )
    for previous, current in zip(contacts, contacts[1:]):
        if current.timestamp - previous.timestamp > gap_seconds:
            result.append(
                Event(
                    event_id=f"rally-end-{previous.frame_id}",
                    timestamp=previous.timestamp,
                    frame_id=previous.frame_id,
                    event_type=EventType.RALLY_END,
                    confidence=0.45,
                    evidence=["contact_gap_exceeded_threshold"],
                )
            )
            result.append(
                Event(
                    event_id=f"rally-start-{current.frame_id}",
                    timestamp=current.timestamp,
                    frame_id=current.frame_id,
                    event_type=EventType.RALLY_START,
                    confidence=0.45,
                    evidence=["contact_gap_exceeded_threshold"],
                )
            )
    result.append(
        Event(
            event_id=f"rally-end-{contacts[-1].frame_id}",
            timestamp=contacts[-1].timestamp,
            frame_id=contacts[-1].frame_id,
            event_type=EventType.RALLY_END,
            confidence=contacts[-1].confidence * 0.8,
            evidence=["last_contact_in_observed_sequence"],
        )
    )
    result.sort(key=lambda event: (event.timestamp, event.event_type.value, event.event_id))
    return result
