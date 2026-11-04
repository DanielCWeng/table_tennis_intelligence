"""Regression coverage for model-free live capture, geometry, and tracking."""
from __future__ import annotations

from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest

from ttintel.media import FramePacket
from ttintel.live.core import LiveMotionTracker, ROI, Region, region_from_drag
from ttintel.live.sources import LiveEngine, demo_frames, offer_latest, synthetic_frame


def test_region_drag_reversed_negative_monitor_and_dpi() -> None:
    mon = Region(-1920, -120, 1920, 1080)
    selection = region_from_drag(mon, (450, 300), (50, 50), (960, 540))
    assert selection == Region(-1820, -20, 800, 500)
    assert region_from_drag(mon, (4, 4), (5, 5), (960, 540)) is None
    assert Region.parse("-100, 20, 400, 250") == Region(-100, 20, 400, 250)
    with pytest.raises(ValueError):
        Region.parse("1,2,3")


def test_roi_clamp_and_invalid() -> None:
    assert ROI(90, 60, 80, 80).clamp(150, 120) == ROI(90, 60, 60, 60)
    assert ROI(148, 80, 20, 20).clamp(150, 120) is None


def patterned_scene(shift: int = 0, globally_shifted: int = 0) -> np.ndarray:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(11)
    bg = rng.integers(45, 120, (200, 320, 3), dtype=np.uint8)
    patch = rng.integers(80, 255, (70, 70, 3), dtype=np.uint8)
    bg[68:138, 70 + shift:140 + shift] = patch
    if globally_shifted:
        bg = cv2.warpAffine(bg, np.float32(
            [[1, 0, globally_shifted], [0, 1, 0]]), (320, 200))
    return bg


def test_known_lateral_translation_and_static_recovery() -> None:
    pytest.importorskip("cv2")
    tracker = LiveMotionTracker()
    tracker.set_regions(ROI(72, 68, 65, 70), ROI(190, 10, 100, 170))
    first = tracker.process(FramePacket(0, 0., patterned_scene()))
    assert first.position.value is None and first.reason == "warmup"
    sample = tracker.process(FramePacket(1, 1 / 30, patterned_scene(4)))
    assert sample.position.value is not None, sample.reason
    assert sample.displacement.value.x > 1., sample.displacement
    assert sample.displacement.inference_type.value == "temporally_tracked"
    static = tracker.process(FramePacket(2, 2 / 30, patterned_scene(4)))
    assert static.candidate is None
    assert static.intensity.value is not None


def test_cut_gap_and_tracking_loss_are_unknown_not_fabricated() -> None:
    pytest.importorskip("cv2")
    tracker = LiveMotionTracker()
    tracker.process(FramePacket(0, .01, patterned_scene()))
    cut = tracker.process(FramePacket(1, .04, np.full((200, 320, 3), 255, np.uint8)))
    assert cut.position.value is None and cut.reason == "camera_cut"
    gap = tracker.process(FramePacket(2, 2.0, patterned_scene()))
    assert gap.reason == "frame_gap"
    assert not gap.displacement.attempted is False  # run but invalid
    blank = np.zeros((200, 320, 3), np.uint8)
    tracker.process(FramePacket(3, 2.04, blank))
    loss = tracker.process(FramePacket(4, 2.08, blank))
    assert loss.position.value is None and loss.reason in ("tracking_loss", "camera_cut")


def test_no_shot_claim_from_motion_or_panning() -> None:
    pytest.importorskip("cv2")
    tracker = LiveMotionTracker()
    tracker.set_regions(ROI(90, 55, 80, 85), ROI(200, 40, 100, 120))
    tracker.process(FramePacket(0, .00, patterned_scene()))
    shifted = tracker.process(FramePacket(1, .04, patterned_scene(globally_shifted=3)))
    # Global translation is not lateral athlete movement.
    if shifted.displacement.value is not None:
        assert abs(shifted.displacement.value.x) < 1.5
    assert shifted.candidate is None


def test_bounded_queue_evicts_oldest() -> None:
    from queue import Queue
    q = Queue(maxsize=2)
    assert offer_latest(q, 1) == 0
    assert offer_latest(q, 2) == 0
    assert offer_latest(q, 3) == 1
    assert q.get_nowait() == 2 and q.get_nowait() == 3


def test_demo_source_and_engine_model_free_smoke() -> None:
    pytest.importorskip("cv2")
    stop = Event()
    frames = demo_frames(stop, fps=60)
    first = next(frames)
    assert first.image.shape == (360, 640, 3)
    stop.set()
    frames.close()
    engine = LiveEngine("demo", fps=40, width=256)
    engine.start()
    try:
        seen = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and seen is None:
            seen = engine.latest()
            time.sleep(.03)
        assert seen is not None, list(engine.errors)
        assert isinstance(seen.packet, FramePacket)
        assert seen.sample.intensity.source == "live.lk_flow"
        assert engine.raw.maxsize == engine.results.maxsize == 2
    finally:
        engine.close()


def test_offscreen_qt_preview_and_editable_roi(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from ttintel.live.gui import LiveWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["live-test"])
    engine = LiveEngine("demo")
    window = LiveWindow(engine)
    try:
        frame = synthetic_frame(.25)
        window.video.set_frame(frame, (), ())
        window.video.choose("target")
        # Directly instantiate the editable ROI as a deterministic UI smoke.
        from ttintel.live.gui import EditableROI
        item = EditableROI(ROI(10, 10, 90, 80), "#63ead3", lambda: None)
        window.video.scene().addItem(item)
        assert item.roi() == ROI(10, 10, 90, 80)
        assert window.video.shape == (640, 360)
    finally:
        window.close()
        app.processEvents()
