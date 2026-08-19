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
from typing import Any, Iterable, Sequence

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


def _world_to_image_homography(image_corners: Sequence[Point2D]) -> np.ndarray:
    """Return the table-plane-to-image homography for four ordered corners."""

    # compute_homography validates its source argument as image corners, and
    # metre-scale table corners fail that minimum-area check.  Invert the
    # validated image-to-table matrix instead of fitting the reverse direction.
    world = table_corners_world()
    return np.linalg.inv(compute_homography(image_corners, world))


def rectangle_consistency(
    image_corners: Sequence[Point2D],
    *,
    image_size: tuple[int, int] | None = None,
    degenerate_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Test whether four corners can be a real camera's view of the table.

    A rectangle does not project to an arbitrary quadrilateral.  Writing the
    table-plane homography as ``H = K [r1 r2 t]`` with square pixels and the
    principal point at the image centre, the orthonormality of ``r1`` and
    ``r2`` gives two independent estimates of the squared focal length::

        f^2 = -(h1x h2x + h1y h2y) / (h1z h2z)
        f^2 = ((h1x^2 + h1y^2) - (h2x^2 + h2y^2)) / (h2z^2 - h1z^2)

    DO NOT use a single frame's result as a validity test.  The estimator is
    exact on noiseless corners and recovers the true focal length to the digit
    across camera geometries, but it is extremely high variance: at 0.5 px of
    corner noise -- below what any real detector achieves -- one of the two
    solutions goes negative in 22% of trials for a near lens and 54% for
    broadcast geometry, while the median estimate stays accurate to within
    0.3%.  A per-frame "is this a rectangle" boolean built on the sign is
    therefore close to a coin flip, and would repeat the reprojection_error_px
    mistake of shipping a number that looks like a quality signal and is not.

    Aggregated over many frames the median is usable.  Use
    :func:`corner_sensitivity_m_per_px` for per-frame conditioning instead.

    A fronto-parallel view leaves the focal length unconstrained and is
    reported as ``degenerate``.
    """

    corners = [Point2D(float(p.x), float(p.y)) for p in image_corners]
    if image_size is None:
        centre_x = float(np.mean([p.x for p in corners]))
        centre_y = float(np.mean([p.y for p in corners]))
    else:
        centre_x, centre_y = image_size[0] / 2.0, image_size[1] / 2.0

    homography = _world_to_image_homography(corners)
    shift = np.asarray([[1.0, 0.0, -centre_x], [0.0, 1.0, -centre_y], [0.0, 0.0, 1.0]])
    centred = shift @ homography
    h1, h2 = centred[:, 0], centred[:, 1]

    denominator_a = h1[2] * h2[2]
    denominator_b = h2[2] ** 2 - h1[2] ** 2
    degenerate = abs(denominator_a) < degenerate_tolerance and abs(denominator_b) < degenerate_tolerance

    focal_estimates: list[float] = []
    if abs(denominator_a) >= degenerate_tolerance:
        focal_estimates.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / denominator_a)
    if abs(denominator_b) >= degenerate_tolerance:
        focal_estimates.append(
            ((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / denominator_b
        )

    positive = [value for value in focal_estimates if value > 0.0]
    focal_lengths = [float(np.sqrt(value)) for value in positive]
    if len(focal_lengths) == 2:
        disagreement = abs(focal_lengths[0] - focal_lengths[1]) / max(
            1e-9, float(np.mean(focal_lengths))
        )
    else:
        disagreement = None

    return {
        "degenerate": bool(degenerate),
        "focal_estimates_px": focal_lengths,
        "focal_disagreement": disagreement,
        "negative_focal_solutions": int(len(focal_estimates) - len(positive)),
    }


def corner_sensitivity_m_per_px(
    image_corners: Sequence[Point2D], *, delta_px: float = 1.0
) -> float:
    """Return metres of table-coordinate error per pixel of corner error.

    This is the interpretable form of the homography's conditioning.  An
    oblique view of the table is well conditioned; a view down the long axis
    pushes the far corners toward a vanishing point, and a single pixel of
    corner error can then move a far-side table coordinate by a large
    distance.  The residual of the four-point fit stays ~0 throughout.
    """

    corners = [Point2D(float(p.x), float(p.y)) for p in image_corners]
    world = table_corners_world()
    baseline = compute_homography(corners, world)
    probes = list(corners) + [
        Point2D(
            float(np.mean([p.x for p in corners])), float(np.mean([p.y for p in corners]))
        )
    ]
    reference = [project_image_to_table(baseline, probe) for probe in probes]

    worst = 0.0
    for index in range(4):
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                moved = list(corners)
                point = moved[index]
                offset = sign * float(delta_px)
                moved[index] = Point2D(
                    point.x + (offset if axis == 0 else 0.0),
                    point.y + (offset if axis == 1 else 0.0),
                )
                try:
                    perturbed = compute_homography(moved, world)
                except (ValueError, np.linalg.LinAlgError):
                    return float("inf")
                for probe, base_point in zip(probes, reference):
                    shifted = project_image_to_table(perturbed, probe)
                    worst = max(worst, float(np.hypot(shifted.x - base_point.x, shifted.y - base_point.y)))
    return worst
