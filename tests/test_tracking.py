import numpy as np

from ttintel.adapters.totnet import _decode_heatmap
from ttintel.schemas import InferenceType, Point2D
from ttintel.tracking import (
    BallCandidate,
    CandidateFrame,
    TrackingConfig,
    link_ball_trajectory,
    trajectory_ball_states,
)


def _frames(points: list[tuple[float, float]], *, distractor: bool = False) -> list[CandidateFrame]:
    result = []
    for index, (x, y) in enumerate(points):
        candidates = [BallCandidate(index, index / 25.0, Point2D(x, y), 0.55, rank=0)]
        if distractor:
            candidates.insert(
                0,
                BallCandidate(index, index / 25.0, Point2D(100.0, 100.0), 0.9, rank=0),
            )
            candidates[1] = BallCandidate(index, index / 25.0, Point2D(x, y), 0.35, rank=1)
        result.append(CandidateFrame(index, index / 25.0, tuple(candidates)))
    return result


def test_heatmap_top_k_uses_nms_not_adjacent_pixels() -> None:
    heatmap = np.zeros((288, 512), dtype=np.float32)
    heatmap[40, 80] = 0.8
    heatmap[41, 81] = 0.7  # same response, should be suppressed
    heatmap[150, 300] = 0.6
    heatmap[230, 450] = 0.5

    peaks = _decode_heatmap(heatmap, top_k=3, nms_radius=5)

    assert [(peak.x, peak.y) for peak in peaks] == [(80, 40), (300, 150), (450, 230)]


def test_linker_prefers_fast_alternative_over_static_high_confidence_distractor() -> None:
    frames = _frames([(20.0 + 15.0 * index, 140.0) for index in range(10)], distractor=True)

    trajectory = link_ball_trajectory(frames)

    assert all(point.candidate_rank == 1 for point in trajectory.points)
    assert all(point.position is not None for point in trajectory.points)


def test_linker_marks_an_unsupported_frame_as_physics_inferred() -> None:
    frames = _frames([(20.0 + 12.0 * index, 140.0) for index in range(7)])
    frames[3] = CandidateFrame(3, 3 / 25.0, ())

    trajectory = link_ball_trajectory(frames, config=TrackingConfig(max_inferred_gap=2))
    states = trajectory_ball_states(trajectory)

    assert trajectory.points[3].position is not None
    assert trajectory.points[3].inference_type == InferenceType.PHYSICS_INFERRED
    assert states[3].image.inference_type == InferenceType.PHYSICS_INFERRED
    assert states[3].image.visibility.value == "occluded"


def test_linker_reports_piecewise_breakpoint_without_fitting_one_curve() -> None:
    points = [(20.0 + 10.0 * index, 120.0) for index in range(5)]
    points.extend([(60.0 - 12.0 * index, 120.0) for index in range(1, 6)])

    trajectory = link_ball_trajectory(_frames(points))

    assert trajectory.breakpoints
    assert trajectory.breakpoints[0] == 4
