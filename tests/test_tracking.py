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
        # Include two measured-floor frames so these synthetic clips have the
        # same separation between detector noise and a real candidate as the
        # footage.  A constant-confidence fixture is intentionally treated as
        # unsupported by the offline linker.
        if index < 2:
            candidates = [
                BallCandidate(index, index / 25.0, Point2D(x, y), 0.001, rank=0),
                BallCandidate(index, index / 25.0, Point2D(x, y), 0.55, rank=1),
            ]
        else:
            candidates = [BallCandidate(index, index / 25.0, Point2D(x, y), 0.55, rank=0)]
        if distractor:
            if index >= 2:
                candidates.insert(
                    0,
                    BallCandidate(index, index / 25.0, Point2D(100.0, 100.0), 0.9, rank=0),
                )
                candidates[1] = BallCandidate(index, index / 25.0, Point2D(x, y), 0.35, rank=1)
            else:
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


def test_linker_marks_an_unsupported_frame_as_interpolated() -> None:
    frames = _frames([(20.0 + 12.0 * index, 140.0) for index in range(7)])
    frames[3] = CandidateFrame(3, 3 / 25.0, ())

    trajectory = link_ball_trajectory(frames, config=TrackingConfig(max_inferred_gap=2))
    states = trajectory_ball_states(trajectory)

    assert trajectory.points[3].position is not None
    assert trajectory.points[3].inference_type == InferenceType.INTERPOLATED
    assert states[3].image.inference_type == InferenceType.INTERPOLATED
    assert states[3].image.visibility.value == "occluded"


def test_linker_reports_piecewise_breakpoint_without_fitting_one_curve() -> None:
    points = [(20.0 + 10.0 * index, 120.0) for index in range(5)]
    points.extend([(60.0 - 12.0 * index, 120.0) for index in range(1, 6)])

    trajectory = link_ball_trajectory(_frames(points))

    assert trajectory.breakpoints
    assert trajectory.breakpoints[0] == 4


def test_floor_only_candidate_clip_produces_no_positions() -> None:
    frames = [
        CandidateFrame(
            index,
            index / 25.0,
            (BallCandidate(index, index / 25.0, Point2D(40.0 + index, 100.0), 0.001, rank=0),),
        )
        for index in range(8)
    ]

    trajectory = link_ball_trajectory(frames)

    assert all(point.position is None for point in trajectory.points)


def test_real_detection_bridges_a_short_occlusion_when_anchors_clear_floor() -> None:
    frames = [
        CandidateFrame(
            index,
            index / 25.0,
            (
                BallCandidate(
                    index,
                    index / 25.0,
                    Point2D(10.0 + index, 120.0),
                    0.001 if index in (0, 9) else 0.20,
                    rank=0,
                ),
            ),
        )
        for index in range(10)
    ]
    frames[4] = CandidateFrame(4, 4 / 25.0, ())

    trajectory = link_ball_trajectory(frames, config=TrackingConfig(max_inferred_gap=2))

    assert trajectory.points[4].position is not None
    assert trajectory.points[4].inference_type == InferenceType.INTERPOLATED
