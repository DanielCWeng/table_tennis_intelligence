import json
from pathlib import Path

from ttintel.media import FramePacket
from ttintel.pipeline import analyse_video
from ttintel.schemas import (
    BallState,
    Estimate,
    InferenceType,
    Point2D,
    Visibility,
    from_dict,
    to_dict,
)
from ttintel.storage import SessionStore


REAL_VIDEO = Path("third_party/tt3d/data/calibration_test/videos/test_00.mp4")


class FixtureBallTracker:
    """Keep storage tests independent of the optional TOTNet checkpoint."""

    info = type("Info", (), {"name": "test.fixture_ball_tracker"})()

    def estimate(self, packet: FramePacket) -> BallState:
        return BallState(
            image=Estimate(
                value=Point2D(20.0, 20.0),
                confidence=0.8,
                source=self.info.name,
                visibility=Visibility.VISIBLE,
                inference_type=InferenceType.MODEL_INFERRED,
            )
        )


def test_attempted_absence_survives_serialisation() -> None:
    attempted = Estimate(
        value=None,
        confidence=0.18,
        source="totnet.ball_tracker",
        quality_flags=["absent"],
    )
    not_attempted = Estimate.unknown("pose.unavailable", "model_unavailable")

    attempted_loaded = from_dict(to_dict(attempted), Estimate)
    not_attempted_loaded = from_dict(to_dict(not_attempted), Estimate)

    assert attempted_loaded.value is None
    assert attempted_loaded.attempted is True
    assert attempted_loaded.quality_flags == ["absent"]
    assert not_attempted_loaded.value is None
    assert not_attempted_loaded.attempted is False


def test_real_session_round_trips_as_typed_objects(tmp_path) -> None:
    result = analyse_video(
        REAL_VIDEO,
        output_root=tmp_path / "sessions",
        max_frames=12,
        ball_tracker_instance=FixtureBallTracker(),
        render=False,
    )
    assert result.session_path is not None

    loaded = SessionStore(tmp_path / "sessions").read_session(result.session_path)

    assert loaded == result.session
    assert isinstance(loaded.frames[0].ball.image, Estimate)
    assert isinstance(loaded.frames[0].ball.image.value, Point2D)
    assert isinstance(loaded.video.source_path, str)


def test_manifest_avoids_frame_and_event_duplication_and_writes_layers(tmp_path) -> None:
    result = analyse_video(
        REAL_VIDEO,
        output_root=tmp_path / "sessions",
        max_frames=4,
        ball_tracker_instance=FixtureBallTracker(),
        render=False,
    )
    assert result.session_path is not None
    session_path = result.session_path

    manifest = json.loads((session_path / "session.json").read_text(encoding="utf-8"))
    assert "frames" not in manifest
    assert "events" not in manifest
    assert (session_path / "raw" / "adapter_outputs.jsonl").is_file()
    assert (session_path / "raw" / "manifest.json").is_file()
    assert (session_path / "cleaned" / "frames.jsonl").is_file()
    assert (session_path / "cleaned" / "manifest.json").is_file()

    raw_manifest = json.loads(
        (session_path / "raw" / "manifest.json").read_text(encoding="utf-8")
    )
    cleaned_manifest = json.loads(
        (session_path / "cleaned" / "manifest.json").read_text(encoding="utf-8")
    )
    assert raw_manifest["available"] is False
    assert cleaned_manifest["record_count"] == 4

    supplied_path = tmp_path / "supplied-raw"
    SessionStore(tmp_path / "sessions").write_session(
        result.session,
        session_path=supplied_path,
        raw_records=[{"frame_id": 0, "adapter": "fixture", "discarded": []}],
    )
    supplied_manifest = json.loads(
        (supplied_path / "raw" / "manifest.json").read_text(encoding="utf-8")
    )
    assert supplied_manifest["available"] is True
    assert SessionStore.read_jsonl(supplied_path / "raw" / "adapter_outputs.jsonl") == [
        {"frame_id": 0, "adapter": "fixture", "discarded": []}
    ]
