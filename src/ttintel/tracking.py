"""Offline multi-hypothesis ball tracking for model heatmap candidates.

The model is allowed to be wrong on an individual frame.  This module keeps
the alternatives, then chooses a path over the whole clip using a second-order
dynamic program.  The second order matters: a plausible ball path has a
coherent velocity, whereas a logo or sock usually has either near-zero motion
or an implausible jump when the detector reacquires the ball.

This is deliberately image-space tracking.  The table homography is used as a
soft spatial prior, not as a claim that an airborne ball lies on z=0.  Motion
scale is estimated from the candidate field for each clip; no fixed pixel
threshold is used as a detector gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, log, log1p
from typing import Any, Sequence

import numpy as np

from .calibration import image_point_to_table
from .geometry import TABLE_LENGTH_M, TABLE_WIDTH_M, corner_sensitivity_m_per_px
from .media import FramePacket
from .schemas import Estimate, InferenceType, Point2D, TableCalibration, Visibility


@dataclass(frozen=True)
class BallCandidate:
    """One detector alternative for one frame."""

    frame_id: int
    timestamp: float
    position: Point2D
    confidence: float
    rank: int = 0
    table_position: Point2D | None = None
    table_penalty: float = 0.0


@dataclass(frozen=True)
class CandidateFrame:
    """All retained detector alternatives for one decoded frame."""

    frame_id: int
    timestamp: float
    candidates: tuple[BallCandidate, ...]


@dataclass(frozen=True)
class TrajectoryPoint:
    """The selected or inferred ball position at one frame."""

    frame_id: int
    timestamp: float
    position: Point2D | None
    confidence: float
    inference_type: InferenceType
    visibility: Visibility
    candidate_rank: int | None = None
    segment_id: int | None = None


@dataclass
class BallTrajectory:
    """Result of offline Viterbi-style linking."""

    points: list[TrajectoryPoint]
    breakpoints: list[int]
    selected_candidates: list[BallCandidate | None]
    score: float
    motion_scale_px_per_second: float

    @property
    def positions(self) -> list[Point2D | None]:
        return [point.position for point in self.points]


@dataclass(frozen=True)
class TrackingConfig:
    """Dimensionless regularisers for the offline path objective.

    These are not confidence thresholds.  They control the relative cost of
    leaving the heatmap unsupported, violating motion smoothness, and using a
    table-inconsistent alternative.  The actual motion scale and table margin
    are measured per clip.
    """

    top_k: int = 8
    nms_radius: int = 7
    missing_frame_cost: float = 0.75
    reacquisition_cost: float = 0.25
    smoothness_weight: float = 0.75
    speed_weight: float = 0.35
    table_weight: float = 0.45
    # A segment change is cheaper than forcing a false quadratic through a
    # paddle contact.  The observation and table terms still have to support
    # the new candidate; this is not a free teleport.
    breakpoint_loss: float = 0.65
    max_inferred_gap: int = 8


def _table_context(
    point: Point2D,
    calibration: TableCalibration | None,
) -> tuple[Point2D | None, float]:
    """Return table coordinates and a soft distance penalty.

    A homography is only exact for the table plane, so airborne candidates are
    allowed a margin.  The margin grows with the measured corner conditioning;
    this makes calibration uncertainty visible in the tracker instead of
    pretending every pixel maps equally reliably.
    """

    if calibration is None:
        return None, 0.0
    try:
        table_point = image_point_to_table(calibration, point)
        sensitivity = corner_sensitivity_m_per_px(calibration.image_corners)
    except (ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
        return None, 1.0
    if not (isfinite(table_point.x) and isfinite(table_point.y)):
        return table_point, 1.0

    # A few centimetres is the useful table-edge precision.  Conditioning is
    # supplied by geometry.py; the fixed component is only the ball's allowed
    # height/projection slack, not a detector cutoff.
    # Airborne projection can move well outside the plane rectangle; use a
    # broad soft margin so calibration informs the path without rejecting an
    # actual ball over the table.  Floor-level distractors remain much farther
    # away and receive the saturated penalty.
    margin = max(0.25, min(0.65, 0.12 + 3.0 * float(sensitivity)))
    dx = max(0.0, -table_point.x, table_point.x - TABLE_LENGTH_M)
    dy = max(0.0, -table_point.y, table_point.y - TABLE_WIDTH_M)
    outside = hypot(dx, dy)
    penalty = min(3.0, outside / margin)
    return table_point, float(penalty)


def _candidate_from_adapter_value(
    value: Any,
    *,
    frame_id: int,
    timestamp: float,
    rank: int,
    calibration: TableCalibration | None,
) -> BallCandidate:
    """Accept the adapter's dataclass as well as small test doubles/tuples."""

    if hasattr(value, "position"):
        position = value.position
        confidence = float(value.confidence)
    elif hasattr(value, "x") and hasattr(value, "y"):
        position = Point2D(float(value.x), float(value.y))
        confidence = float(value.confidence)
    elif isinstance(value, dict):
        position = value.get("position", value)
        if isinstance(position, dict):
            position = Point2D(float(position["x"]), float(position["y"]))
        else:
            position = Point2D(float(position[0]), float(position[1]))
        confidence = float(value["confidence"])
    else:
        x, y, confidence = value
        position = Point2D(float(x), float(y))
        confidence = float(confidence)
    if not isinstance(position, Point2D):
        if hasattr(position, "x") and hasattr(position, "y"):
            position = Point2D(float(position.x), float(position.y))
        else:
            position = Point2D(float(position[0]), float(position[1]))
    confidence = max(0.0, min(1.0, confidence))
    table_position, table_penalty = _table_context(position, calibration)
    return BallCandidate(
        frame_id=frame_id,
        timestamp=timestamp,
        position=position,
        confidence=confidence,
        rank=rank,
        table_position=table_position,
        table_penalty=table_penalty,
    )


def collect_top_k_candidates(
    packets: Sequence[FramePacket],
    adapter: Any,
    *,
    top_k: int = 8,
    nms_radius: int = 7,
    calibration: TableCalibration | None = None,
) -> list[CandidateFrame]:
    """Run the adapter once per frame and retain its NMS-separated peaks."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if hasattr(adapter, "reset"):
        adapter.reset()
    method = getattr(adapter, "estimate_candidates", None) or getattr(adapter, "estimate_top_k", None)
    if method is None:
        raise TypeError("offline tracking requires an adapter.estimate_candidates method")

    result: list[CandidateFrame] = []
    for packet in packets:
        raw = method(packet, top_k=top_k, nms_radius=nms_radius)
        candidates = tuple(
            _candidate_from_adapter_value(
                value,
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                rank=rank,
                calibration=calibration,
            )
            for rank, value in enumerate(raw)
        )
        result.append(CandidateFrame(packet.frame_id, packet.timestamp, candidates))
    return result


def _frame_dt(frames: Sequence[CandidateFrame], index: int) -> float:
    if index <= 0:
        return 1.0
    delta = float(frames[index].timestamp - frames[index - 1].timestamp)
    return delta if delta > 1e-6 else 1.0


def _motion_prior(frames: Sequence[CandidateFrame]) -> tuple[float, float]:
    """Estimate typical speed and its robust spread from the candidate field."""

    speeds: list[float] = []
    nearest_alternative_speeds: list[float] = []
    for index in range(1, len(frames)):
        if not frames[index - 1].candidates or not frames[index].candidates:
            continue
        first, second = frames[index - 1].candidates[0], frames[index].candidates[0]
        speeds.append(hypot(second.position.x - first.position.x, second.position.y - first.position.y) / _frame_dt(frames, index))
        pair_speeds = [
            hypot(current.position.x - previous.position.x, current.position.y - previous.position.y)
            / _frame_dt(frames, index)
            for previous in frames[index - 1].candidates
            for current in frames[index].candidates
        ]
        positive = [value for value in pair_speeds if value > 1e-6]
        if positive:
            nearest_alternative_speeds.append(min(positive))
    observed = speeds + nearest_alternative_speeds
    if not observed:
        return 1.0, 1.0
    values = np.asarray(observed, dtype=float)
    # The upper half is more informative than the median when the argmax is
    # intermittently stationary on a distractor.  The nearest alternative is
    # included so a moving lower-ranked candidate can establish the clip's
    # motion scale even when the argmax is a static advert for every frame.
    typical = float(np.percentile(values, 65.0))
    spread = float(np.median(np.abs(values - typical)))
    # A stationary distractor plus a moving ball is intentionally bimodal;
    # using the full MAD there would erase the very slow/fast distinction we
    # are trying to model.  Cap the robust scale relative to the measured
    # motion instead of introducing a pixel-space constant.
    scale = max(1.0, min(spread, typical * 0.35), typical * 0.20)
    return max(1.0, typical), scale


def _evidence_floor(frames: Sequence[CandidateFrame]) -> float:
    """Estimate the clip's detector floor from its per-frame argmax field.

    TOTNet's heatmap is a softmax without a null class, so even a frame with
    no ball has a rank-0 candidate.  A fixed confidence cutoff would encode
    one venue's heatmap scale into the tracker; the lower tail of this clip's
    own argmax confidences is the measured reference instead.  The numerical
    lower bound only prevents ``log(0)`` for degenerate test doubles.
    """

    top_confidences = [frame.candidates[0].confidence for frame in frames if frame.candidates]
    if not top_confidences:
        return float(np.finfo(float).tiny)
    return max(
        float(np.percentile(np.asarray(top_confidences, dtype=float), 10.0)),
        float(np.finfo(float).tiny),
    )


def _emission(candidate: BallCandidate, frame: CandidateFrame, config: TrackingConfig) -> float:
    """Score one candidate's support within its own frame.

    This term is deliberately rank-relative.  An absolute-confidence floor was
    tried here and removed: TOTNet's softmax has no null class, so its scale
    means different things on different clips, and gating candidate *selection*
    on the clip's lower confidence tail discarded genuine motion-blurred
    detections on Frankfurt while buying two frames on London.  Deciding that a
    frame carries no ball is left to the interpolation gate, which acts only on
    bridge anchors and was measured not to cost anything on either clip.
    """

    if not frame.candidates:
        return -config.missing_frame_cost
    frame_max = max(item.confidence for item in frame.candidates)
    relative = (candidate.confidence + 1e-5) / (frame_max + 1e-5)
    # Rank-relative evidence keeps a low-confidence but locally strongest
    # candidate usable while still preferring a genuinely strong peak.
    return (
        0.60
        + 0.20 * float(np.log(relative))
        + 0.08 * log1p(50.0 * candidate.confidence)
        - config.table_weight * candidate.table_penalty
    )


def _transition_score(
    previous_previous: BallCandidate | None,
    previous: BallCandidate | None,
    current: BallCandidate | None,
    *,
    previous_dt: float,
    current_dt: float,
    typical_speed: float,
    speed_scale: float,
    config: TrackingConfig,
) -> float:
    if current is None:
        # The observation term charges for a missing frame.  Motion should not
        # charge it a second time, otherwise a one-frame occlusion becomes
        # disproportionately expensive.
        return 0.0
    if previous is None:
        return -config.reacquisition_cost

    dx = current.position.x - previous.position.x
    dy = current.position.y - previous.position.y
    speed = hypot(dx, dy) / max(current_dt, 1e-6)
    slow_residual = max(0.0, typical_speed - speed) / max(speed_scale, 1.0)
    speed_cost = -config.speed_weight * log1p(slow_residual * slow_residual)
    # Excessive speed is softly discouraged, but never hard rejected: a
    # contact or bounce can be a real discontinuity and the later frames can
    # decide whether this is a true re-acquisition.
    fast_residual = max(0.0, speed - typical_speed - 3.0 * speed_scale) / max(typical_speed, 1.0)
    speed_cost -= 0.08 * log1p(fast_residual * fast_residual)

    if previous_previous is None:
        return speed_cost
    old_dx = previous.position.x - previous_previous.position.x
    old_dy = previous.position.y - previous_previous.position.y
    old_speed = hypot(old_dx, old_dy) / max(previous_dt, 1e-6)
    acceleration = hypot(dx / max(current_dt, 1e-6) - old_dx / max(previous_dt, 1e-6), dy / max(current_dt, 1e-6) - old_dy / max(previous_dt, 1e-6))
    acceleration_residual = acceleration / max(speed_scale, 1.0)
    # log1p is a robust loss: large residuals become a piecewise break rather
    # than making the entire rally pay for one paddle contact.
    smooth_cost = -config.smoothness_weight * log1p(acceleration_residual * acceleration_residual)
    # Keep the variable live in the expression's explanation: old_speed is a
    # useful diagnostic while the acceleration itself is vector-valued.
    _ = old_speed
    return speed_cost + max(smooth_cost, -config.breakpoint_loss)


def _state_candidate(frame: CandidateFrame, state: int) -> BallCandidate | None:
    return frame.candidates[state] if state >= 0 else None


def _best_path(
    frames: Sequence[CandidateFrame],
    *,
    config: TrackingConfig,
) -> tuple[list[int], float, float, float, float]:
    if not frames:
        return [], 0.0, 1.0, 1.0, float(np.finfo(float).tiny)
    typical_speed, speed_scale = _motion_prior(frames)
    evidence_floor = _evidence_floor(frames)
    state_counts = [len(frame.candidates) + 1 for frame in frames]  # final state is missing
    missing = lambda frame: len(frame.candidates)

    # At frame zero, the second-order state is represented by (previous,
    # current) = (missing, current).  Keys use -1 for missing.
    scores: dict[tuple[int, int], float] = {}
    backpointers: list[dict[tuple[int, int], tuple[int, int]]] = []
    first = frames[0]
    for current_state in range(state_counts[0]):
        current = _state_candidate(first, current_state) if current_state != missing(first) else None
        scores[(-1, current_state if current is not None else -1)] = (
            _emission(current, first, config)
            if current is not None
            else -config.missing_frame_cost
        )
    if len(frames) == 1:
        best = max(scores.items(), key=lambda item: item[1])
        return [best[0][1]], best[1], typical_speed, speed_scale, evidence_floor

    # Process frame one separately because the first state has only one real
    # predecessor.  Thereafter every state stores the last two candidate ids.
    second = frames[1]
    next_scores: dict[tuple[int, int], float] = {}
    first_back: dict[tuple[int, int], tuple[int, int]] = {}
    for current_state in range(state_counts[1]):
        current = _state_candidate(second, current_state) if current_state != missing(second) else None
        current_id = current_state if current is not None else -1
        best_value = float("-inf")
        best_previous = (-1, -1)
        for (_, previous_state), previous_score in scores.items():
            previous = _state_candidate(first, previous_state) if previous_state >= 0 else None
            value = previous_score + (
                _emission(current, second, config)
                if current is not None
                else -config.missing_frame_cost
            )
            value += _transition_score(
                None,
                previous,
                current,
                previous_dt=_frame_dt(frames, 1),
                current_dt=_frame_dt(frames, 1),
                typical_speed=typical_speed,
                speed_scale=speed_scale,
                config=config,
            )
            if value > best_value:
                best_value, best_previous = value, (previous_state, current_id)
        next_scores[(best_previous[1], current_id)] = best_value
        first_back[(best_previous[1], current_id)] = best_previous
    scores = next_scores
    backpointers.append(first_back)

    for index in range(2, len(frames)):
        frame = frames[index]
        previous_frame = frames[index - 1]
        next_scores = {}
        back: dict[tuple[int, int], tuple[int, int]] = {}
        for current_state in range(state_counts[index]):
            current = _state_candidate(frame, current_state) if current_state != missing(frame) else None
            current_id = current_state if current is not None else -1
            best_value = float("-inf")
            best_previous = (-1, -1)
            for (previous_previous_id, previous_id), previous_score in scores.items():
                previous_previous = _state_candidate(frames[index - 2], previous_previous_id) if previous_previous_id >= 0 else None
                previous = _state_candidate(previous_frame, previous_id) if previous_id >= 0 else None
                value = previous_score + (
                    _emission(current, frame, config)
                    if current is not None
                    else -config.missing_frame_cost
                )
                value += _transition_score(
                    previous_previous,
                    previous,
                    current,
                    previous_dt=_frame_dt(frames, index - 1),
                    current_dt=_frame_dt(frames, index),
                    typical_speed=typical_speed,
                    speed_scale=speed_scale,
                    config=config,
                )
                if value > best_value:
                    best_value = value
                    best_previous = (previous_previous_id, previous_id)
            next_scores[(best_previous[1], current_id)] = best_value
            back[(best_previous[1], current_id)] = best_previous
        scores = next_scores
        backpointers.append(back)

    final_state, final_score = max(scores.items(), key=lambda item: item[1])
    path = [-1] * len(frames)
    path[-2], path[-1] = final_state
    # backpointers[0] reconstructs frame 0/1; each subsequent table maps the
    # ending pair at frame i to the ending pair at frame i-1.
    # The first table is the special frame-0/frame-1 initialisation table; the
    # final pair already contains those two states once the later tables have
    # been unwound.  Stop at one so the pseudo-state used during
    # initialisation cannot overwrite frame 0.
    for pointer_index in range(len(backpointers) - 1, 0, -1):
        previous_pair = backpointers[pointer_index][(path[pointer_index], path[pointer_index + 1])]
        path[pointer_index - 1] = previous_pair[0]
        path[pointer_index] = previous_pair[1]
    return path, final_score, typical_speed, speed_scale, evidence_floor


def _clears_evidence_floor(
    confidence: float,
    evidence_floor: float,
    *,
    frame_max: float | None = None,
) -> bool:
    """Return whether an anchor has evidence separated from the measured floor.

    The margin is deliberately much smaller than the log gap used by the
    emission.  It distinguishes the London anchors at the bare floor from a
    genuine short-occlusion bridge.  A lower-ranked anchor may be below the
    p10 confidence itself when its frame's top-1 is clearly away from the
    floor; the rank-relative and motion terms established that frame-level
    evidence before this gate is reached.
    """

    candidate_ratio = log(max(confidence, float(np.finfo(float).tiny)) / evidence_floor)
    if candidate_ratio >= 0.05:
        return True
    if frame_max is None:
        return False
    frame_ratio = log(max(frame_max, float(np.finfo(float).tiny)) / evidence_floor)
    return candidate_ratio >= log(0.05) and abs(frame_ratio) >= 0.05


def _fill_inferred_points(
    frames: Sequence[CandidateFrame],
    selected: Sequence[BallCandidate | None],
    *,
    max_gap: int,
    evidence_floor: float,
) -> list[TrajectoryPoint]:
    points: list[TrajectoryPoint] = []
    for index, (frame, candidate) in enumerate(zip(frames, selected)):
        if candidate is not None:
            points.append(
                TrajectoryPoint(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    position=candidate.position,
                    confidence=candidate.confidence,
                    inference_type=InferenceType.TEMPORALLY_TRACKED,
                    visibility=Visibility.VISIBLE,
                    candidate_rank=candidate.rank,
                )
            )
            continue
        previous = next((j for j in range(index - 1, -1, -1) if selected[j] is not None), None)
        following = next((j for j in range(index + 1, len(selected)) if selected[j] is not None), None)
        gap = (following - previous - 1) if previous is not None and following is not None else max_gap + 1
        if gap <= max_gap and previous is not None and following is not None:
            before, after = selected[previous], selected[following]
            assert before is not None and after is not None
            before_frame = frames[previous]
            after_frame = frames[following]
            before_max = max((item.confidence for item in before_frame.candidates), default=0.0)
            after_max = max((item.confidence for item in after_frame.candidates), default=0.0)
            if gap == 1 and (
                (before.confidence < evidence_floor) != (after.confidence < evidence_floor)
            ):
                points.append(
                    TrajectoryPoint(
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        position=None,
                        confidence=0.0,
                        inference_type=InferenceType.UNKNOWN,
                        visibility=Visibility.UNKNOWN,
                    )
                )
                continue
            if not (
                _clears_evidence_floor(
                    before.confidence, evidence_floor, frame_max=before_max
                )
                and _clears_evidence_floor(after.confidence, evidence_floor, frame_max=after_max)
            ):
                points.append(
                    TrajectoryPoint(
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        position=None,
                        confidence=0.0,
                        inference_type=InferenceType.UNKNOWN,
                        visibility=Visibility.UNKNOWN,
                    )
                )
                continue
            ratio = (index - previous) / float(following - previous)
            position = Point2D(
                before.position.x + ratio * (after.position.x - before.position.x),
                before.position.y + ratio * (after.position.y - before.position.y),
            )
            confidence = min(before.confidence, after.confidence) * 0.55
            points.append(
                TrajectoryPoint(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    position=position,
                    confidence=confidence,
                    inference_type=InferenceType.INTERPOLATED,
                    visibility=Visibility.OCCLUDED,
                    segment_id=None,
                )
            )
        else:
            points.append(
                TrajectoryPoint(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    position=None,
                    confidence=0.0,
                    inference_type=InferenceType.UNKNOWN,
                    visibility=Visibility.UNKNOWN,
                )
            )
    return points


def _breakpoints(
    points: Sequence[TrajectoryPoint],
    *,
    speed_scale: float,
) -> tuple[list[int], list[int | None]]:
    observed = [point.position for point in points]
    accelerations: list[tuple[int, float]] = []
    for index in range(1, len(points) - 1):
        if observed[index - 1] is None or observed[index] is None or observed[index + 1] is None:
            continue
        dt_before = max(points[index].timestamp - points[index - 1].timestamp, 1e-6)
        dt_after = max(points[index + 1].timestamp - points[index].timestamp, 1e-6)
        v_before = np.asarray(
            [(observed[index].x - observed[index - 1].x) / dt_before, (observed[index].y - observed[index - 1].y) / dt_before]
        )
        v_after = np.asarray(
            [(observed[index + 1].x - observed[index].x) / dt_after, (observed[index + 1].y - observed[index].y) / dt_after]
        )
        accelerations.append((index, float(np.linalg.norm(v_after - v_before))))
    if not accelerations:
        return [], [None] * len(points)
    values = np.asarray([value for _, value in accelerations], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(2.0 * speed_scale, median + 3.0 * max(mad, 0.15 * speed_scale))
    breaks = [points[index].frame_id for index, value in accelerations if value > threshold]
    segment_for_index: list[int | None] = []
    segment = 0
    break_set = set(breaks)
    for point in points:
        segment_for_index.append(segment)
        if point.frame_id in break_set:
            segment += 1
    return breaks, segment_for_index


def link_ball_trajectory(
    candidate_frames: Sequence[CandidateFrame],
    *,
    calibration: TableCalibration | None = None,
    config: TrackingConfig | None = None,
) -> BallTrajectory:
    """Choose the best whole-clip path through candidate alternatives."""

    settings = config or TrackingConfig()
    if not candidate_frames:
        return BallTrajectory([], [], [], 0.0, 1.0)
    if calibration is not None:
        # Direct callers may construct CandidateFrame objects themselves rather
        # than going through collect_top_k_candidates.  Apply the same trusted
        # geometry prior at this boundary too.
        def calibrated(candidate: BallCandidate) -> BallCandidate:
            table_position, table_penalty = _table_context(candidate.position, calibration)
            return BallCandidate(
                frame_id=candidate.frame_id,
                timestamp=candidate.timestamp,
                position=candidate.position,
                confidence=candidate.confidence,
                rank=candidate.rank,
                table_position=table_position,
                table_penalty=table_penalty,
            )

        candidate_frames = [
            CandidateFrame(
                frame.frame_id,
                frame.timestamp,
                tuple(calibrated(candidate) for candidate in frame.candidates),
            )
            for frame in candidate_frames
        ]
    path, score, typical_speed, speed_scale, evidence_floor = _best_path(
        candidate_frames, config=settings
    )
    selected = [
        frame.candidates[state] if state >= 0 and state < len(frame.candidates) else None
        for frame, state in zip(candidate_frames, path)
    ]
    points = _fill_inferred_points(
        candidate_frames,
        selected,
        max_gap=settings.max_inferred_gap,
        evidence_floor=evidence_floor,
    )
    breaks, segment_for_index = _breakpoints(points, speed_scale=speed_scale)
    points = [
        TrajectoryPoint(
            frame_id=point.frame_id,
            timestamp=point.timestamp,
            position=point.position,
            confidence=point.confidence,
            inference_type=point.inference_type,
            visibility=point.visibility,
            candidate_rank=point.candidate_rank,
            segment_id=segment_for_index[index],
        )
        for index, point in enumerate(points)
    ]
    return BallTrajectory(points, breaks, selected, score, typical_speed)


def track_offline(
    packets: Sequence[FramePacket],
    adapter: Any,
    *,
    calibration: TableCalibration | None = None,
    config: TrackingConfig | None = None,
) -> BallTrajectory:
    """Collect TOTNet alternatives and link them with future-frame access."""

    settings = config or TrackingConfig()
    candidates = collect_top_k_candidates(
        packets,
        adapter,
        top_k=settings.top_k,
        nms_radius=settings.nms_radius,
        calibration=calibration,
    )
    return link_ball_trajectory(candidates, calibration=calibration, config=settings)


def trajectory_ball_states(
    trajectory: BallTrajectory,
    *,
    source: str = "totnet.ball_tracker.viterbi",
) -> list[Any]:
    """Convert trajectory points to schema BallState values without changing schemas."""

    from .schemas import BallState

    states: list[BallState] = []
    for point in trajectory.points:
        if point.position is None:
            estimate = Estimate.unknown(source, "no_supported_candidate")
        elif point.inference_type == InferenceType.INTERPOLATED:
            estimate = Estimate(
                value=point.position,
                confidence=point.confidence,
                source=source,
                visibility=Visibility.OCCLUDED,
                inference_type=InferenceType.INTERPOLATED,
                quality_flags=["occluded", "trajectory_interpolated"],
            )
        elif point.inference_type == InferenceType.PHYSICS_INFERRED:
            estimate = Estimate(
                value=point.position,
                confidence=point.confidence,
                source=source,
                visibility=Visibility.OCCLUDED,
                inference_type=InferenceType.PHYSICS_INFERRED,
                quality_flags=["occluded", "trajectory_inferred"],
            )
        else:
            estimate = Estimate(
                value=point.position,
                confidence=point.confidence,
                source=source,
                visibility=Visibility.VISIBLE,
                inference_type=InferenceType.TEMPORALLY_TRACKED,
            )
        states.append(BallState(image=estimate))
    return states


# Short aliases make the API easy to discover without taking away the more
# explicit names used in reports and tests.
collect_candidates = collect_top_k_candidates
link_trajectory = link_ball_trajectory


__all__ = [
    "BallCandidate",
    "BallTrajectory",
    "CandidateFrame",
    "TrackingConfig",
    "TrajectoryPoint",
    "collect_candidates",
    "collect_top_k_candidates",
    "link_ball_trajectory",
    "link_trajectory",
    "track_offline",
    "trajectory_ball_states",
]
