from pathlib import Path
import os

import numpy as np
import pytest

from ttintel.adapters.totnet import (
    DEFAULT_CHECKPOINT,
    TOTNetBallTracker,
    _decode_heatmap,
    _pad_temporal_window,
)
from ttintel.media import FramePacket
from ttintel.perception import DefaultPerceptionProvider
from ttintel.schemas import BallState, Estimate, InferenceType, Point2D, Visibility


def test_temporal_window_left_pads_first_frames_and_keeps_last_five() -> None:
    frames = [object(), object(), object(), object(), object(), object()]
    assert _pad_temporal_window(frames[:1]) == [frames[0]] * 5
    assert _pad_temporal_window(frames[:3]) == [frames[0], frames[0], frames[0], frames[1], frames[2]]
    assert _pad_temporal_window(frames) == frames[1:]


def test_heatmap_decode_uses_argmax_probability() -> None:
    heatmap = np.zeros((288, 512), dtype=np.float32)
    heatmap[17, 61] = 0.42
    x, y, confidence = _decode_heatmap(heatmap)
    assert (x, y) == (61, 17)
    assert confidence == pytest.approx(0.42)


def test_model_prediction_preserves_absence_and_provenance_without_loading_model() -> None:
    tracker = TOTNetBallTracker.__new__(TOTNetBallTracker)
    tracker.confidence_threshold = 0.5
    absent = tracker._state_from_prediction(61, 17, 0.42, np.zeros((720, 1280, 3), dtype=np.uint8))
    assert isinstance(absent, BallState)
    assert absent.image.value is None
    assert absent.image.inference_type == InferenceType.MODEL_INFERRED
    assert absent.image.quality_flags == ["absent"]


def test_perception_accepts_an_injected_ball_tracker() -> None:
    class FakeTracker:
        info = type("Info", (), {"name": "test.ball_tracker"})()

        def estimate(self, packet: FramePacket) -> BallState:
            return BallState(
                image=Estimate(
                    value=Point2D(12.0, 13.0),
                    confidence=0.8,
                    source=self.info.name,
                    visibility=Visibility.VISIBLE,
                    inference_type=InferenceType.MODEL_INFERRED,
                )
            )

    provider = DefaultPerceptionProvider(ball_tracker=FakeTracker(), use_bright_blob=False)
    detection = provider.infer(FramePacket(0, 0.0, np.zeros((20, 30, 3), dtype=np.uint8)))
    assert detection.ball is not None
    assert detection.ball.image.value == Point2D(12.0, 13.0)
    assert "test.ball_tracker" in detection.diagnostics


def test_totnet_integration_skips_without_checkpoint_or_cuda() -> None:
    if os.environ.get("TTINTEL_RUN_TOTNET_INTEGRATION") != "1":
        pytest.skip("TOTNet integration is opt-in because it loads a 94 MB checkpoint")
    if not Path(DEFAULT_CHECKPOINT).is_file():
        pytest.skip("TOTNet checkpoint is absent")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tracker = TOTNetBallTracker(device="cuda")
    packet = FramePacket(0, 0.0, np.zeros((720, 1280, 3), dtype=np.uint8))
    result = tracker.estimate(packet)
    assert result.image.inference_type == InferenceType.MODEL_INFERRED
