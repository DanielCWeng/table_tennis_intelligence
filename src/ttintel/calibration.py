"""Table detection/calibration seams and the reliable manual fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .geometry import make_calibration, project_image_to_table
from .schemas import CalibrationSource, Point2D, TableCalibration


class CalibrationError(ValueError):
    pass


CORNER_NAMES = ("near_left", "near_right", "far_right", "far_left")


def _point(value: Any) -> Point2D:
    if isinstance(value, Point2D):
        return value
    if isinstance(value, dict):
        return Point2D(float(value["x"]), float(value["y"]))
    if isinstance(value, Sequence) and len(value) == 2:
        return Point2D(float(value[0]), float(value[1]))
    raise CalibrationError(f"invalid corner point: {value!r}")


def parse_manual_corners(value: Any) -> tuple[Point2D, Point2D, Point2D, Point2D]:
    """Parse either a four-item list or a named corner object."""

    if isinstance(value, dict):
        try:
            return tuple(_point(value[name]) for name in CORNER_NAMES)  # type: ignore[return-value]
        except KeyError as exc:
            raise CalibrationError(f"missing named corner {exc.args[0]!r}") from exc
    if isinstance(value, Sequence) and len(value) == 4:
        return tuple(_point(item) for item in value)  # type: ignore[return-value]
    raise CalibrationError("manual corners must be a four-item list or named object")


def load_manual_corners(path: str | Path) -> tuple[Point2D, Point2D, Point2D, Point2D]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return parse_manual_corners(json.load(handle))


def save_manual_corners(path: str | Path, corners: Sequence[Point2D]) -> None:
    ordered = parse_manual_corners(corners)
    payload = {name: {"x": point.x, "y": point.y} for name, point in zip(CORNER_NAMES, ordered)}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def calibrate_manual(
    corners: Sequence[Point2D],
    *,
    corner_confidences: Sequence[float] | None = None,
) -> TableCalibration:
    return make_calibration(
        parse_manual_corners(corners),
        source=CalibrationSource.MANUAL,
        corner_confidences=corner_confidences,
    )


def _colour_table_mask(image: np.ndarray) -> np.ndarray:
    """Conservative green/blue table-colour mask for a diagnostic baseline."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise CalibrationError("automatic colour calibration requires an RGB image")
    rgb = array[..., :3].astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    green_table = (green > red * 1.05) & (green > blue * 0.75) & (spread > 18.0)
    blue_table = (blue > red * 1.12) & (blue > green * 0.72) & (spread > 18.0)
    return green_table | blue_table


def detect_table_corners_heuristic(image: np.ndarray) -> tuple[Point2D, Point2D, Point2D, Point2D] | None:
    """Find a rough quadrilateral from table-colour extrema.

    This is deliberately labelled a heuristic.  It is useful for synthetic
    fixtures and friendly views, but production footage should prefer TT3D or
    a manual four-corner override until this baseline is benchmarked.
    """

    mask = _colour_table_mask(image)
    y, x = np.nonzero(mask)
    if len(x) < 100:
        return None
    points = np.column_stack((x.astype(float), y.astype(float)))
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    candidates = [
        points[np.argmin(sums)],  # top-left
        points[np.argmax(diffs)],  # top-right
        points[np.argmax(sums)],  # bottom-right
        points[np.argmin(diffs)],  # bottom-left
    ]
    result = tuple(Point2D(float(item[0]), float(item[1])) for item in candidates)
    try:
        # The heuristic order is image clockwise order.  It is usable as a
        # projective plane, but its table-side orientation must be reviewed.
        return parse_manual_corners(result)
    except CalibrationError:
        return None


def calibrate_automatic(image: np.ndarray) -> TableCalibration:
    corners = detect_table_corners_heuristic(image)
    if corners is None:
        raise CalibrationError("heuristic table detector found insufficient table-colour evidence")
    return make_calibration(
        corners,
        source=CalibrationSource.AUTOMATIC,
        corner_confidences=(0.35, 0.35, 0.35, 0.35),
        quality_flags=("heuristic",),
    )


def image_point_to_table(calibration: TableCalibration, point: Point2D) -> Point2D:
    return project_image_to_table(np.asarray(calibration.homography, dtype=float), point)


def calibration_quality(calibration: TableCalibration) -> dict[str, Any]:
    return {
        "reprojection_error_px": calibration.reprojection_error_px,
        "corner_confidences": list(calibration.corner_confidences),
        "calibration_source": calibration.source.value,
        "calibration_confidence": calibration.confidence,
        "quality_flags": list(calibration.quality_flags),
    }
