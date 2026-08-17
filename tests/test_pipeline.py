import json

import numpy as np

from ttintel.calibration import calibrate_manual
from ttintel.pipeline import analyse_packets
from ttintel.media import FramePacket
from ttintel.schemas import Point2D


def test_pipeline_writes_structured_session_and_render(tmp_path) -> None:
    packets = [
        FramePacket(index, index / 25.0, np.zeros((120, 220, 3), dtype=np.uint8))
        for index in range(5)
    ]
    annotation_path = tmp_path / "annotations.json"
    frames = []
    for index in range(5):
        frames.append(
            {
                "frame_id": index,
                "poses": [
                    {
                        "id": "left",
                        "bbox": [20 + index, 25, 70 + index, 105],
                        "joints": {
                            "left_shoulder": [45 + index, 45],
                            "left_elbow": [55 + index, 58],
                            "left_wrist": [70 + index, 70],
                            "left_hip": [45 + index, 75],
                            "left_knee": [45 + index, 90],
                            "left_ankle": [45 + index, 104],
                        },
                    },
                    {
                        "id": "right",
                        "bbox": [150 - index, 25, 200 - index, 105],
                        "joints": {"right_wrist": [145 - index, 70]},
                    },
                ],
                "ball": {"image": [70 + index, 70], "confidence": 0.9},
            }
        )
    annotation_path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    calibration = calibrate_manual(
        (Point2D(20, 20), Point2D(200, 20), Point2D(200, 110), Point2D(20, 110))
    )

    result = analyse_packets(
        packets,
        output_root=tmp_path / "sessions",
        annotations=annotation_path,
        manual_calibration=calibration,
        render=True,
    )

    assert len(result.session.frames) == 5
    assert result.session.segments[0].table is not None
    assert result.session.segments[0].gameplay_quality_score >= 0.7
    assert result.session.frames[0].players.keys() == {"player_0", "player_1"}
    assert result.session_path is not None
    assert (result.session_path / "session.json").is_file()
    assert (result.session_path / "fused" / "frames.jsonl").is_file()
    assert result.render_paths
