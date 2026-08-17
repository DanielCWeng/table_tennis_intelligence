import numpy as np

from ttintel.geometry import (
    TABLE_LENGTH_M,
    TABLE_WIDTH_M,
    make_calibration,
    project_image_to_table,
    project_table_to_image,
)
from ttintel.schemas import Point2D


def test_table_homography_round_trips_ordered_corners() -> None:
    corners = (
        Point2D(100.0, 100.0),
        Point2D(500.0, 110.0),
        Point2D(450.0, 320.0),
        Point2D(120.0, 300.0),
    )
    calibration = make_calibration(corners)
    matrix = np.asarray(calibration.homography)
    assert calibration.reprojection_error_px < 1e-6
    for image_corner, table_corner in zip(corners, ((0, 0), (TABLE_LENGTH_M, 0), (TABLE_LENGTH_M, TABLE_WIDTH_M), (0, TABLE_WIDTH_M))):
        table = project_image_to_table(matrix, image_corner)
        assert abs(table.x - table_corner[0]) < 1e-6
        assert abs(table.y - table_corner[1]) < 1e-6
        image = project_table_to_image(matrix, table)
        assert abs(image.x - image_corner.x) < 1e-5
        assert abs(image.y - image_corner.y) < 1e-5
