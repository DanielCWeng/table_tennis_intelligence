"""Synthetic acceptance tests for opt-in human-assisted ball observations."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from ttintel.media import FramePacket
from ttintel.live.ball import BallAssistTracker
from ttintel.live.sources import LiveEngine, Result


def ball_frame(x: int | None = 65, y: int = 65, *,
               second_ball: int | None = None, width: int = 220,
               height: int = 150) -> np.ndarray:
    cv2 = pytest.importorskip("cv2")
    image = np.full((height, width, 3), 34, np.uint8)
    for cx in (x, second_ball):
        if cx is not None:
            cv2.circle(image, (cx, y), 5, (250, 247, 235), -1)
            cv2.circle(image, (cx - 1, y - 2), 1, (255, 255, 255), -1)
    return image


def packet(frame_id: int, x: int | None, *, t: float | None = None,
           second: int | None = None) -> FramePacket:
    return FramePacket(frame_id, frame_id / 30 if t is None else t,
                       ball_frame(x, second_ball=second))


def test_click_is_observed_and_successive_matches_are_only_heuristic() -> None:
    pytest.importorskip("cv2")
    tracker = BallAssistTracker()
    first = packet(0, 65)
    assert tracker.process(first).position.value is None
    human = tracker.seed(first, (65, 65))
    assert human.state == "manual"
    assert human.position.source == "live.ball_user_click"
    assert human.position.inference_type.value == "observed"
    assert tracker.process(first).state == "manual"
    for i, x in enumerate((70, 77, 85, 95), start=1):
        found = tracker.process(packet(i, x))
        assert found.state == "assisted", (i, found.state)
        assert found.position.inference_type.value == "temporally_tracked"
        assert abs(found.position.value.x - x) <= 1
        assert abs(found.position.value.y - 65) <= 1
        assert 0 < found.position.confidence <= 1


def test_disappearance_breaks_track_without_extrapolation_and_requires_click() -> None:
    pytest.importorskip("cv2")
    tracker = BallAssistTracker()
    tracker.seed(packet(0, 65), (65, 65))
    assert tracker.process(packet(1, 70)).valid
    vanished = tracker.process(packet(2, None))
    assert vanished.state == "ambiguous_or_weak_match"
    assert vanished.position.value is None
    assert tracker.process(packet(3, 90)).state == "needs_reseed"
    correction = tracker.seed(packet(3, 90), (90, 65))
    assert correction.state == "manual"
    assert tracker.process(packet(4, 95)).valid


def test_identical_distractor_causes_unknown_not_arbitrary_choice() -> None:
    pytest.importorskip("cv2")
    tracker = BallAssistTracker()
    tracker.seed(packet(0, 65), (65, 65))
    ambiguous = tracker.process(packet(1, 70, second=115))
    assert not ambiguous.valid, ambiguous
    assert ambiguous.state == "ambiguous_or_weak_match"


def test_cut_gap_resize_and_stale_queued_frame_never_claim_a_ball() -> None:
    pytest.importorskip("cv2")
    tracker = BallAssistTracker()
    tracker.seed(packet(5, 65), (65, 65))
    assert tracker.process(packet(4, 65)).state == "stale_before_seed"
    assert tracker.process(packet(6, 70), scene_cut=True).state == "scene_cut"
    assert tracker.process(packet(7, 75)).state == "needs_reseed"
    tracker.seed(packet(8, 80), (80, 65))
    assert tracker.process(packet(9, 85, t=5.0)).state == "frame_gap"
    tracker.seed(packet(10, 85), (85, 65))
    resized = FramePacket(11, 11 / 30, ball_frame(85, width=240))
    assert tracker.process(resized).state == "dimensions_changed"


def test_seed_validation_and_no_model_invocation() -> None:
    pytest.importorskip("cv2")
    tracker = BallAssistTracker()
    with pytest.raises(ValueError, match="outside"):
        tracker.seed(packet(0, 60), (-1, 60))
    with pytest.raises(ValueError, match="insufficient texture"):
        tracker.seed(packet(0, None), (30, 20))
    assert tracker.process(packet(1, 60)).state == "not_seeded"


def test_live_engine_ball_seed_and_clear_do_not_require_neural_models() -> None:
    pytest.importorskip("cv2")
    engine = LiveEngine("demo")
    try:
        first = packet(0, 65)
        assert engine.seed_ball(first, 65, 65).state == "manual"
        with engine.lock:
            found = engine.ball.process(packet(1, 70))
        assert found.valid and found.state == "assisted"
        old_epoch = engine.ball_epoch
        engine.clear_ball()
        assert engine.ball_epoch == old_epoch + 1
        with engine.lock:
            assert engine.ball.process(packet(2, 80)).state == "not_seeded"
    finally:
        engine.close()


def test_qt_preview_click_and_manual_annotation_csv(tmp_path, monkeypatch) -> None:
    pytest.importorskip("cv2")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtTest = pytest.importorskip("PySide6.QtTest")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from ttintel.live.gui import LiveWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["ball-test"])
    engine = LiveEngine("demo")
    window = LiveWindow(engine)
    try:
        first = packet(0, 65)
        sample = engine.tracker.process(first)
        result = Result(first, sample, 0.0, 0.0, 0.0)
        window.show()
        app.processEvents()
        window._render_result(result)
        window.video.choose("ball")
        screen = window.video.mapFromScene(QtCore.QPointF(65, 65))
        QtTest.QTest.mouseClick(window.video.viewport(), QtCore.Qt.MouseButton.LeftButton,
                                pos=screen)
        app.processEvents()
        assert engine.ball.template is not None
        assert window.ball_rows[0][0] == 65
        assert window.video.ball_items
        csv_path = tmp_path / "ball_tracking.csv"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                            lambda *args, **kwargs: (str(csv_path), "CSV (*.csv)"))
        window._export()
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["ball_x_px"] == "65.0"
        assert rows[0]["ball_source"] == "live.ball_user_click"
        window._clear_ball()
        assert engine.ball.template is None
        assert not window.video.ball_items
    finally:
        window.close()
        app.processEvents()
