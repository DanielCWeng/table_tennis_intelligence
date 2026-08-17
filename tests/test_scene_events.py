from ttintel.events import detect_bounce_candidates
from ttintel.media import FramePacket
from ttintel.scene import detect_cuts
from ttintel.schemas import BallState, Estimate, EventType, FrameState, Point2D


def test_thumbnail_cut_baseline_finds_hard_cut() -> None:
    import numpy as np

    frames = [
        FramePacket(0, 0.0, np.zeros((20, 30, 3), dtype=np.uint8)),
        FramePacket(1, 0.1, np.zeros((20, 30, 3), dtype=np.uint8)),
        FramePacket(2, 0.2, np.full((20, 30, 3), 255, dtype=np.uint8)),
    ]
    cuts = detect_cuts(frames, threshold=0.4)
    assert [cut.frame_id for cut in cuts] == [2]


def test_bounce_candidate_keeps_candidate_confidence() -> None:
    frames = []
    for frame_id, y in enumerate((80.0, 100.0, 125.0, 100.0, 80.0)):
        frames.append(
            FrameState(
                frame_id=frame_id,
                timestamp=frame_id / 25.0,
                segment_id="segment",
                ball=BallState(
                    image=Estimate.observed(Point2D(100.0, y), 0.9, "fixture")
                ),
            )
        )
    events = detect_bounce_candidates(frames)
    assert len(events) == 1
    assert events[0].event_type == EventType.TABLE_BOUNCE
    assert events[0].confidence < 1.0
    assert "no_3d_surface_check" in events[0].evidence
