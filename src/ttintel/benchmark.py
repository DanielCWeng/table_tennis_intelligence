"""Small gold-set evaluation helpers for the MVP benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Sequence

from .schemas import Event, EventType, FrameState, Point2D, Session


@dataclass(frozen=True)
class GoldBallPoint:
    frame_id: int
    point: Point2D


@dataclass(frozen=True)
class BallBenchmark:
    visible_ball_recall: float
    mean_pixel_error: float | None
    false_positive_count: int
    longest_missed_sequence: int


@dataclass(frozen=True)
class EventBenchmark:
    precision: float
    recall: float
    mean_timing_error_seconds: float | None


@dataclass(frozen=True)
class BenchmarkReport:
    session_id: str
    ball: BallBenchmark
    bounce: EventBenchmark
    contact: EventBenchmark
    notes: list[str]


def evaluate_ball(
    predicted: Sequence[FrameState],
    gold: Sequence[GoldBallPoint],
    *,
    tolerance_px: float = 12.0,
) -> BallBenchmark:
    truth = {item.frame_id: item.point for item in gold}
    matched = 0
    errors: list[float] = []
    false_positives = 0
    missed_run = 0
    longest_missed = 0
    for frame in predicted:
        prediction = frame.ball.image.value if frame.ball else None
        expected = truth.get(frame.frame_id)
        if expected is None:
            if prediction is not None:
                false_positives += 1
            continue
        if prediction is not None and hypot(prediction.x - expected.x, prediction.y - expected.y) <= tolerance_px:
            matched += 1
            errors.append(hypot(prediction.x - expected.x, prediction.y - expected.y))
            missed_run = 0
        else:
            missed_run += 1
            longest_missed = max(longest_missed, missed_run)
    return BallBenchmark(
        visible_ball_recall=matched / len(gold) if gold else 0.0,
        mean_pixel_error=sum(errors) / len(errors) if errors else None,
        false_positive_count=false_positives,
        longest_missed_sequence=longest_missed,
    )


def _event_times(events: Iterable[Event], event_type: EventType) -> list[float]:
    return sorted(event.timestamp for event in events if event.event_type == event_type)


def evaluate_events(
    predicted: Sequence[Event],
    gold: Sequence[Event],
    event_type: EventType,
    *,
    tolerance_seconds: float = 0.08,
) -> EventBenchmark:
    predicted_times = _event_times(predicted, event_type)
    gold_times = _event_times(gold, event_type)
    remaining = list(gold_times)
    matches: list[float] = []
    for timestamp in predicted_times:
        if not remaining:
            break
        index = min(range(len(remaining)), key=lambda item: abs(remaining[item] - timestamp))
        error = abs(remaining[index] - timestamp)
        if error <= tolerance_seconds:
            matches.append(error)
            remaining.pop(index)
    return EventBenchmark(
        precision=len(matches) / len(predicted_times) if predicted_times else 0.0,
        recall=len(matches) / len(gold_times) if gold_times else 0.0,
        mean_timing_error_seconds=sum(matches) / len(matches) if matches else None,
    )


def evaluate_session(session: Session, gold_ball: Sequence[GoldBallPoint], gold_events: Sequence[Event] = ()) -> BenchmarkReport:
    return BenchmarkReport(
        session_id=session.session_id,
        ball=evaluate_ball(session.frames, gold_ball),
        bounce=evaluate_events(session.events, gold_events, EventType.TABLE_BOUNCE),
        contact=evaluate_events(session.events, gold_events, EventType.PLAYER_CONTACT),
        notes=[
            "Metrics are only meaningful for a manually labelled gold set.",
            "Model disagreement and missing labels must remain explicit in the benchmark notes.",
        ],
    )
