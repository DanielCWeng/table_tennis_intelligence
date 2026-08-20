from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")

from ttintel.calibration import load_manual_corners
from ttintel.labeller import (
    BallLabel,
    FrameStore,
    LabelSet,
    create_app,
    load_labels,
    save_labels,
)
from ttintel.media import FramePacket
from ttintel.schemas import Point2D


def _store(tmp_path: Path) -> FrameStore:
    packets = [
        FramePacket(4, 0.0, np.full((18, 32, 3), 30, dtype=np.uint8)),
        FramePacket(7, 0.1, np.full((18, 32, 3), 90, dtype=np.uint8)),
    ]
    return FrameStore(tmp_path / "clip.mp4", packets=packets)


def test_label_format_round_trips_point_and_explicit_absence(tmp_path: Path) -> None:
    path = tmp_path / "clip.labels.json"
    original = LabelSet(
        "clip.mp4",
        {4: BallLabel("point", Point2D(12.5, 8.25)), 7: BallLabel("absent")},
    )

    save_labels(path, original)

    loaded = load_labels(path)
    assert loaded.video == original.video
    assert loaded.labels == original.labels
    assert loaded.labels[7].kind == "absent"


def test_server_handlers_persist_labels_and_write_loadable_corners(tmp_path: Path) -> None:
    store = _store(tmp_path)
    labels_path = tmp_path / "labels.json"
    corners_path = tmp_path / "corners.json"
    app = create_app(store.video, frame_store=store, labels_path=labels_path, corners_path=corners_path)
    client = app.test_client()

    session = client.get("/api/session")
    assert session.status_code == 200
    assert session.get_json()["frame_ids"] == [4, 7]

    point = client.post(
        "/api/labels",
        json={"frame_id": 4, "label": "point", "x": 12.5, "y": 8.25},
    )
    assert point.status_code == 200
    absent = client.post("/api/labels", json={"frame_id": 7, "label": "absent"})
    assert absent.status_code == 200
    assert client.get("/api/frame/4").get_json()["label"] == {
        "label": "point",
        "x": 12.5,
        "y": 8.25,
    }
    assert client.get("/api/frame/7").get_json()["label"] == {"label": "absent"}
    assert client.get("/api/frame/4/image").mimetype == "image/jpeg"

    corners = [
        {"x": 3, "y": 15},
        {"x": 29, "y": 15},
        {"x": 25, "y": 4},
        {"x": 7, "y": 4},
    ]
    response = client.post("/api/corners", json={"corners": corners})
    assert response.status_code == 200
    assert load_manual_corners(corners_path) == (
        Point2D(3, 15),
        Point2D(29, 15),
        Point2D(25, 4),
        Point2D(7, 4),
    )

    reloaded = create_app(store.video, frame_store=store, labels_path=labels_path, corners_path=corners_path)
    reloaded_session = reloaded.test_client().get("/api/session").get_json()
    assert reloaded_session["counts"] == {"labelled": 2, "point": 1, "absent": 1, "untouched": 0}
    assert reloaded_session["manual_corners"][0] == {"x": 3.0, "y": 15.0}
