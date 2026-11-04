"""Pause/resume regression smoke using a real worker and synthetic source.

The live UI may not render new frames during PAUSE, and resumed screen/demo
capture must not replay a stale screenshot from before the pause.
"""
from __future__ import annotations

import time

import pytest

from ttintel.live.sources import LiveEngine


def _next_sample(engine: LiveEngine, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = engine.latest()
        if result is not None:
            return result
        time.sleep(.02)
    pytest.fail("live source did not deliver a frame within smoke-test deadline")


def test_pause_discards_pre_pause_screen_like_frame() -> None:
    pytest.importorskip("cv2")
    engine = LiveEngine("demo", fps=25, width=256)
    engine.start()
    try:
        _next_sample(engine)
        engine.pause()
        time.sleep(.20)  # allow a capture-generator yield already in flight
        resumed_at = time.monotonic()
        engine.start()
        fresh = _next_sample(engine)
        # Motion analysis may be UNKNOWN for a large inter-frame time gap;
        # the preview frame itself must still be captured after resuming.
        assert fresh.packet.timestamp >= resumed_at - .045
    finally:
        engine.close()


def test_paused_gui_never_renders_in_flight_results(monkeypatch) -> None:
    pytest.importorskip("cv2")
    qt = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from ttintel.live.gui import LiveWindow
    from ttintel.live.sources import Result
    from ttintel.live.core import MotionSample
    from ttintel.media import FramePacket
    import numpy as np

    app = qt.QApplication.instance() or qt.QApplication(["live-pause-smoke"])
    engine = LiveEngine("demo")
    window = LiveWindow(engine)
    try:
        # Even when a worker has produced a result during pause, a manually
        # invoked Qt timer callback must leave the preview and plot untouched.
        engine.pause()
        packet = FramePacket(1, time.monotonic(),
                             np.zeros((360, 640, 3), dtype=np.uint8))
        sample = engine.tracker._unknown(packet, "frame_gap")
        engine.results.put(Result(packet, sample, time.monotonic(),
                                  time.monotonic(), 0.1))
        window._poll()
        assert window.video.shape is None
        assert not window.samples
    finally:
        window.close()
        app.processEvents()


def test_unknown_tracking_loss_clears_relative_displacement_anchor() -> None:
    """An unobserved interval must not be bridged into the next trajectory."""
    import numpy as np
    from ttintel.live.core import LiveMotionTracker
    from ttintel.media import FramePacket

    tracker = LiveMotionTracker()
    tracker._position[:] = [14.0, -3.0]
    tracker._direction = 1
    tracker._leg = 6.0
    sample = tracker._unknown(
        FramePacket(6, 0.2, np.zeros((100, 140, 3), np.uint8)),
        "reference_loss",
    )
    assert sample.position.value is None
    assert np.allclose(tracker._position, [0.0, 0.0])
    assert tracker._direction == 0
    assert tracker._leg == 0.0
