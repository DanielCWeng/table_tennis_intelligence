import numpy as np

from ttintel.calibration import (
    calibrate_automatic,
    calibration_quality,
    detect_table_corners_heuristic,
    image_point_to_table,
)
from ttintel.geometry import TABLE_LENGTH_M, TABLE_WIDTH_M, make_calibration
from ttintel.schemas import CalibrationSource, Point2D


def _green_table_frame() -> np.ndarray:
    """A frame whose table-colour region is a known axis-aligned rectangle.

    Rows 20..109 and columns 20..179 are table green, so the near (bottom)
    edge sits at y=109 and the far (top) edge at y=20.
    """

    image = np.zeros((140, 200, 3), dtype=np.uint8)
    image[20:110, 20:180] = (40, 160, 60)
    return image


def test_heuristic_corners_put_the_near_edge_at_the_image_bottom() -> None:
    corners = detect_table_corners_heuristic(_green_table_frame())
    assert corners is not None
    near_left, near_right, far_right, far_left = corners

    # Image +y points down, so near corners must sit below far corners.
    assert near_left.y > far_left.y
    assert near_right.y > far_right.y
    # Left corners must sit left of right corners.
    assert near_left.x < near_right.x
    assert far_left.x < far_right.x


def test_automatic_calibration_maps_corners_to_the_documented_table_frame() -> None:
    """Guards the near/far orientation of the heuristic detector.

    Emitting raw image order (top-left first) instead of near-edge-first
    mirrors the table about its centre line: this assertion fails with the
    near_left corner landing at y=TABLE_WIDTH_M rather than y=0.
    """

    image = _green_table_frame()
    corners = detect_table_corners_heuristic(image)
    assert corners is not None
    calibration = calibrate_automatic(image)

    expected = (
        (0.0, 0.0),
        (TABLE_LENGTH_M, 0.0),
        (TABLE_LENGTH_M, TABLE_WIDTH_M),
        (0.0, TABLE_WIDTH_M),
    )
    for corner, (want_x, want_y) in zip(corners, expected):
        table = image_point_to_table(calibration, corner)
        assert abs(table.x - want_x) < 1e-6
        assert abs(table.y - want_y) < 1e-6


def test_heuristic_does_not_treat_a_blue_arena_as_the_table() -> None:
    image = np.zeros((140, 200, 3), dtype=np.uint8)
    image[:, :] = (35, 20, 110)

    assert detect_table_corners_heuristic(image) is None


def test_quality_fails_when_automatic_calibration_has_no_redundant_evidence() -> None:
    wrong = (
        Point2D(232, 671),
        Point2D(1279, 219),
        Point2D(1191, 16),
        Point2D(48, 22),
    )
    calibration = make_calibration(wrong, source=CalibrationSource.AUTOMATIC)

    quality = calibration_quality(calibration)

    assert quality["reprojection_error_px"] < 1e-6
    assert quality["quality_passed"] is False
    assert "no_redundant_observations" in quality["quality_failure_reasons"]
