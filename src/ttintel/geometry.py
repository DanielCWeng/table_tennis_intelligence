"""Regulation-table geometry and table-plane projective transforms.

The coordinate convention is explicit and intentionally modest:

* origin: near-left playable table corner;
* +x: along the 2.74 m table length toward near-right;
* +y: away from the camera-side/near edge toward far-left;
* +z: upward from the table plane.

The homography maps only the tabletop plane.  Airborne ball positions must
not be labelled as 3-D world coordinates by this module.
"""

from __future__ import annotations

from math import hypot
from typing import Iterable, Sequence

import numpy as np

from .schemas import (
    CalibrationSource,
    Point2D,
    Point3D,
    TableCalibration,
)


TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525
TABLE_HEIGHT_M = 0.76
NET_HEIGHT_M = 0.1525


def table_corners_world() -> tuple[Point3D, Point3D, Point3D, Point3D]:
    """Return corners in near-left, near-right, far-right, far-left order."""

    return (
        Point3D(0.0, 0.0, 0.0),
        Point3D(TABLE_LENGTH_M, 0.0, 0.0),
        Point3D(TABLE_LENGTH_M, TABLE_WIDTH_M, 0.0),
        Point3D(0.0, TABLE_WIDTH_M, 0.0),
    )


def _as_xy(points: Sequence[Point2D | Point3D | Sequence[float]]) -> np.ndarray:
    values = []
    for point in points:
        if isinstance(point, (Point2D, Point3D)):
            values.append((point.x, point.y))
        else:
            values.append((float(point[0]), float(point[1])))
    array = np.asarray(values, dtype=float)
    if array.shape != (4, 2):
        raise ValueError("exactly four 2-D points are required")
    return array


def polygon_area(points: Sequence[Point2D]) -> float:
    """Return the absolute area of a four-corner polygon."""

    xy = _as_xy(points)
    return float(abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))) / 2.0)


def validate_image_corners(corners: Sequence[Point2D], *, min_area_px: float = 100.0) -> None:
    if len(corners) != 4:
        raise ValueError("four table corners are required")
    if polygon_area(corners) < min_area_px:
        raise ValueError("table corners do not enclose a usable image area")
    edges = [
        hypot(corners[(i + 1) % 4].x - corners[i].x, corners[(i + 1) % 4].y - corners[i].y)
        for i in range(4)
    ]
    if min(edges) < 2.0:
        raise ValueError("table corner edges are too short")


def compute_homography(
    image_corners: Sequence[Point2D],
    world_corners: Sequence[Point3D] | None = None,
) -> np.ndarray:
    """Compute image -> tabletop XY homography with a normalized DLT solve."""

    validate_image_corners(image_corners)
    destination = world_corners or table_corners_world()
    src = _as_xy(image_corners)
    dst = _as_xy(destination)
    rows: list[list[float]] = []
    values: list[float] = []
    for (u, v), (x, y) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(float(u))
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(float(v))
    # Solve for the inverse mapping (world -> image), then invert it.  This
    # avoids depending on any OpenCV calibration APIs in the core package.
    inverse = np.linalg.solve(np.asarray(rows), np.asarray(values))
    world_to_image = np.array(
        [
            [inverse[0], inverse[1], inverse[2]],
            [inverse[3], inverse[4], inverse[5]],
            [inverse[6], inverse[7], 1.0],
        ],
        dtype=float,
    )
    return np.linalg.inv(world_to_image)


def apply_homography(homography: np.ndarray, point: Point2D) -> Point2D:
    vector = homography @ np.array([point.x, point.y, 1.0], dtype=float)
    if abs(float(vector[2])) < 1e-12:
        raise ValueError("homography maps point to infinity")
    return Point2D(float(vector[0] / vector[2]), float(vector[1] / vector[2]))


def project_image_to_table(homography: np.ndarray, point: Point2D) -> Point2D:
    """Map an image point to XY on z=0; caller must know it lies on the table."""

    return apply_homography(homography, point)


def project_table_to_image(homography: np.ndarray, point: Point2D) -> Point2D:
    return apply_homography(np.linalg.inv(homography), point)


def reprojection_error_px(
    homography: np.ndarray,
    image_corners: Sequence[Point2D],
    world_corners: Sequence[Point3D] | None = None,
) -> float:
    """Return the four-corner fit residual in pixels.

    This is a solver/arithmetic diagnostic, not a calibration-quality metric.
    A homography fitted from exactly four point correspondences has enough
    freedom to interpolate those four points, so this value will be close to
    zero even when all four image corners describe the wrong quadrilateral.
    Use redundant observations such as table-edge and net-line support to
    assess whether the calibration is correct.
    """

    destination = world_corners or table_corners_world()
    errors = []
    inverse = np.linalg.inv(homography)
    for expected, world in zip(image_corners, destination):
        projected = apply_homography(inverse, Point2D(world.x, world.y))
        errors.append(hypot(projected.x - expected.x, projected.y - expected.y))
    return float(np.mean(errors))


def make_calibration(
    image_corners: Sequence[Point2D],
    *,
    source: CalibrationSource = CalibrationSource.MANUAL,
    corner_confidences: Sequence[float] | None = None,
    quality_flags: Iterable[str] = (),
) -> TableCalibration:
    """Create a table calibration from four ordered image corners.

    ``corner_confidences`` must come from the caller's evidence.  They are not
    inflated or reduced using ``reprojection_error_px`` because that residual
    cannot distinguish a correct four-point calibration from a wrong one.
    """

    validate_image_corners(image_corners)
    world = table_corners_world()
    homography = compute_homography(image_corners, world)
    error = reprojection_error_px(homography, image_corners, world)
    confidences = tuple(float(c) for c in (corner_confidences or (0.8, 0.8, 0.8, 0.8)))
    if len(confidences) != 4 or any(not 0.0 <= c <= 1.0 for c in confidences):
        raise ValueError("corner_confidences must contain four values between 0 and 1")
    confidence = max(0.0, min(1.0, min(confidences)))
    flags = list(quality_flags)
    if source == CalibrationSource.MANUAL and "calibration_manual" not in flags:
        flags.append("calibration_manual")
    return TableCalibration(
        image_corners=tuple(image_corners),
        world_corners=world,
        homography=tuple(tuple(float(item) for item in row) for row in homography),
        reprojection_error_px=error,
        corner_confidences=confidences,
        source=source,
        confidence=confidence,
        quality_flags=flags,
    )


def calibration_matrix(calibration: TableCalibration) -> np.ndarray:
    return np.asarray(calibration.homography, dtype=float)
