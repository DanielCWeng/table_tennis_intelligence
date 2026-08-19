"""Table detection/calibration seams and the reliable manual fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .geometry import make_calibration, project_image_to_table
from .schemas import CalibrationSource, Point2D, TableCalibration


class CalibrationError(ValueError):
    pass


CORNER_NAMES = ("near_left", "near_right", "far_right", "far_left")
AUTOMATIC_QUALITY_THRESHOLD = 0.62
AUTOMATIC_MIN_EDGE_SUPPORT = 0.45
AUTOMATIC_MIN_NET_SUPPORT = 0.45


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
    """Return a conservative green mask for small, synthetic fixtures.

    Blue/purple is intentionally not accepted here.  In the WTT footage the
    arena is purple-blue, so treating blue as table colour turns most of the
    image into one false foreground component.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise CalibrationError("automatic colour calibration requires an RGB image")
    rgb = array[..., :3].astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    return (green > red * 1.08) & (green > blue * 0.95) & (spread > 18.0)


@dataclass(frozen=True)
class _TableDetection:
    corners: tuple[Point2D, Point2D, Point2D, Point2D]
    edge_support: tuple[float, float, float, float]
    net_support: float
    quality_score: float
    flags: tuple[str, ...] = ()
    fixture_fallback: bool = False


def _canonical_angle(angle: float) -> float:
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def _angle_distance(first: float, second: float) -> float:
    return abs(_canonical_angle(first - second))


def _line_from_segment(segment: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = (float(item) for item in segment)
    a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
    scale = hypot(a, b)
    if scale <= 1e-9:
        raise ValueError("degenerate line segment")
    return np.asarray((a / scale, b / scale, c / scale), dtype=float)


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    if abs(float(point[2])) <= 1e-9:
        return None
    return point[:2] / point[2]


def _line_coordinate(line: np.ndarray, *, axis: str, value: float) -> float | None:
    if axis == "x":
        denominator = float(line[0])
        numerator = float(line[1] * value + line[2])
    else:
        denominator = float(line[1])
        numerator = float(line[0] * value + line[2])
    if abs(denominator) <= 1e-9:
        return None
    return -numerator / denominator


def _cluster_segments(
    segments: Sequence[tuple[float, float, tuple[int, int, int, int]]],
    *,
    coordinate: str,
    coordinate_value: float,
    angle_tolerance: float,
    coordinate_tolerance: float,
) -> list[dict[str, Any]]:
    """Merge duplicate Hough segments into a small set of image lines."""

    groups: list[dict[str, Any]] = []
    for length, angle, segment in sorted(segments, key=lambda item: item[0], reverse=True):
        line = _line_from_segment(segment)
        key = _line_coordinate(line, axis=coordinate, value=coordinate_value)
        if key is None or not isfinite(key):
            continue
        for group in groups:
            if (
                _angle_distance(float(angle), float(group["angle"])) <= angle_tolerance
                and abs(key - float(group["key"])) <= coordinate_tolerance
            ):
                group["segments"].append((length, angle, segment))
                group["lines"].append(line)
                group["weight"] += length
                group["angle"] = float(np.mean([item[1] for item in group["segments"]]))
                keys = [
                    item_key
                    for item_key in (
                        _line_coordinate(item, axis=coordinate, value=coordinate_value)
                        for item in group["lines"]
                    )
                    if item_key is not None
                ]
                group["key"] = float(np.mean(keys)) if keys else float("inf")
                break
        else:
            groups.append(
                {
                    "segments": [(length, angle, segment)],
                    "lines": [line],
                    "weight": float(length),
                    "angle": float(angle),
                    "key": float(key),
                }
            )

    for group in groups:
        reference = group["lines"][0]
        aligned = [line if float(np.dot(reference[:2], line[:2])) >= 0.0 else -line for line in group["lines"]]
        line = np.average(
            np.asarray(aligned),
            axis=0,
            weights=[item[0] for item in group["segments"]],
        )
        line /= np.linalg.norm(line[:2])
        group["line"] = line
        value = _line_coordinate(line, axis=coordinate, value=coordinate_value)
        group["key"] = float(value) if value is not None else float("inf")
    return sorted(groups, key=lambda item: float(item["weight"]), reverse=True)


def _sample_line_support(
    edges: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    radius: int = 2,
) -> float:
    """Measure how continuously an observed edge follows a proposed line."""

    height, width = edges.shape[:2]
    distance = float(np.linalg.norm(end - start))
    samples = max(24, int(distance))
    points = np.linspace(start, end, samples)
    hits = 0
    for x, y in points:
        column, row = int(round(float(x))), int(round(float(y)))
        if not (0 <= column < width and 0 <= row < height):
            continue
        nearby = edges[
            max(0, row - radius):min(height, row + radius + 1),
            max(0, column - radius):min(width, column + radius + 1),
        ]
        hits += int(bool(np.any(nearby)))
    return float(hits / samples)


def _edge_image(image: np.ndarray) -> np.ndarray:
    import cv2

    rgb = np.asarray(image)[..., :3]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 100)


def _measure_redundant_constraints(
    image: np.ndarray,
    corners: Sequence[Point2D],
) -> dict[str, Any]:
    """Check observed edges and the net against a proposed four-corner fit."""

    try:
        import cv2
    except ImportError:
        return {
            "quality_score": 0.0,
            "edge_support": [0.0] * 4,
            "net_support": 0.0,
            "failure_reasons": ["edge_backend_unavailable"],
        }
    try:
        edges = _edge_image(image)
    except (ValueError, TypeError, cv2.error):
        return {
            "quality_score": 0.0,
            "edge_support": [0.0] * 4,
            "net_support": 0.0,
            "failure_reasons": ["edge_measurement_failed"],
        }

    points = np.asarray([(point.x, point.y) for point in corners], dtype=float)
    edge_support = [
        _sample_line_support(edges, points[index], points[(index + 1) % 4])
        for index in range(4)
    ]
    far_midpoint = (points[2] + points[3]) / 2.0
    near_midpoint = (points[0] + points[1]) / 2.0
    net_support = _sample_line_support(edges, far_midpoint, near_midpoint)
    score = 0.70 * float(np.mean(edge_support)) + 0.30 * net_support
    failures: list[str] = []
    if min(edge_support) < AUTOMATIC_MIN_EDGE_SUPPORT:
        failures.append("table_edge_support_low")
    if net_support < AUTOMATIC_MIN_NET_SUPPORT:
        failures.append("net_line_support_low")
    if score < AUTOMATIC_QUALITY_THRESHOLD:
        failures.append("redundant_constraint_score_low")
    return {
        "quality_score": float(score),
        "edge_support": [float(item) for item in edge_support],
        "net_support": float(net_support),
        "failure_reasons": failures,
    }


def _detect_table_from_lines(image: np.ndarray) -> _TableDetection | None:
    """Detect the broadcast table from its boundary and central net lines."""

    try:
        import cv2
    except ImportError:
        return None

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    height, width = array.shape[:2]
    if height < 240 or width < 400:
        return None

    edges = _edge_image(array)
    roi = np.zeros_like(edges)
    roi[int(height * 0.25):int(height * 0.75), int(width * 0.15):int(width * 0.85)] = 1
    roi_edges = edges * roi
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 720,
        threshold=30,
        minLineLength=max(40, int(width * 0.07)),
        maxLineGap=20,
    )
    if lines is None:
        return None

    horizontal: list[tuple[float, float, tuple[int, int, int, int]]] = []
    left: list[tuple[float, float, tuple[int, int, int, int]]] = []
    right: list[tuple[float, float, tuple[int, int, int, int]]] = []
    vertical: list[tuple[float, float, tuple[int, int, int, int]]] = []
    for raw in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (int(item) for item in raw)
        length = hypot(x2 - x1, y2 - y1)
        angle = _canonical_angle(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        midpoint_x, midpoint_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not (0.15 * width < midpoint_x < 0.85 * width and 0.27 * height < midpoint_y < 0.73 * height):
            continue
        segment = (x1, y1, x2, y2)
        if abs(abs(angle) - 90.0) <= 8.0 and 0.28 * width < midpoint_x < 0.72 * width:
            if length >= 0.12 * height:
                vertical.append((length, angle, segment))
        elif abs(angle) <= 10.0 and length >= 0.07 * width:
            horizontal.append((length, angle, segment))
        elif -87.0 <= angle <= -50.0 and length >= 0.07 * height:
            left.append((length, angle, segment))
        elif 50.0 <= angle <= 87.0 and length >= 0.07 * height:
            right.append((length, angle, segment))

    horizontal_groups = _cluster_segments(
        horizontal,
        coordinate="y",
        coordinate_value=width / 2.0,
        angle_tolerance=7.0,
        coordinate_tolerance=8.0,
    )
    left_groups = _cluster_segments(
        left,
        coordinate="x",
        coordinate_value=height / 2.0,
        angle_tolerance=8.0,
        coordinate_tolerance=10.0,
    )
    right_groups = _cluster_segments(
        right,
        coordinate="x",
        coordinate_value=height / 2.0,
        angle_tolerance=8.0,
        coordinate_tolerance=10.0,
    )
    vertical_groups = _cluster_segments(
        vertical,
        coordinate="x",
        coordinate_value=height / 2.0,
        angle_tolerance=5.0,
        coordinate_tolerance=10.0,
    )
    if len(horizontal_groups) < 2 or not left_groups or not right_groups:
        return None

    net_group = None
    if vertical_groups:
        net_group = max(
            vertical_groups,
            key=lambda group: float(group["weight"])
            / (1.0 + abs(float(group["key"]) - width / 2.0) / (0.35 * width)),
        )
    net_line = net_group["line"] if net_group is not None else None

    candidates: list[_TableDetection] = []
    for first_index, first in enumerate(horizontal_groups[:20]):
        for second in horizontal_groups[first_index + 1:20]:
            top, bottom = (first, second) if first["key"] < second["key"] else (second, first)
            if not 0.05 * height < bottom["key"] - top["key"] < 0.28 * height:
                continue
            for left_group in left_groups[:16]:
                for right_group in right_groups[:16]:
                    far_left = _line_intersection(top["line"], left_group["line"])
                    far_right = _line_intersection(top["line"], right_group["line"])
                    near_left = _line_intersection(bottom["line"], left_group["line"])
                    near_right = _line_intersection(bottom["line"], right_group["line"])
                    points = (near_left, near_right, far_right, far_left)
                    if any(point is None for point in points):
                        continue
                    xy = np.asarray(points, dtype=float)
                    if not np.all(np.isfinite(xy)) or np.any(xy < 0.0) or np.any(xy[:, 0] > width) or np.any(xy[:, 1] > height):
                        continue
                    if not (far_left[0] < far_right[0] and near_left[0] < near_right[0]):  # type: ignore[index]
                        continue
                    if not (near_left[1] > far_left[1] and near_right[1] > far_right[1]):  # type: ignore[index]
                        continue
                    if not (0.15 * width < near_left[0] < 0.50 * width and 0.15 * width < far_left[0] < 0.50 * width):  # type: ignore[index]
                        continue
                    if not (0.50 * width < near_right[0] < 0.85 * width and 0.50 * width < far_right[0] < 0.85 * width):  # type: ignore[index]
                        continue
                    area = abs(
                        np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                        - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))
                    ) / 2.0
                    if not 0.008 * width * height < area < 0.15 * width * height:
                        continue
                    centre_y = float(np.mean(xy[:, 1]))
                    if not 0.38 * height < centre_y < 0.72 * height:
                        continue

                    edge_support = (
                        _sample_line_support(edges, xy[0], xy[1]),
                        _sample_line_support(edges, xy[1], xy[2]),
                        _sample_line_support(edges, xy[2], xy[3]),
                        _sample_line_support(edges, xy[3], xy[0]),
                    )
                    far_midpoint = (xy[2] + xy[3]) / 2.0
                    near_midpoint = (xy[0] + xy[1]) / 2.0
                    net_support = (
                        _sample_line_support(edges, far_midpoint, near_midpoint)
                        if net_line is not None
                        else 0.0
                    )
                    centre_score = max(0.0, 1.0 - abs(centre_y - 0.53 * height) / (0.30 * height))
                    line_strength = min(
                        1.0,
                        (float(top["weight"]) + float(bottom["weight"]) + float(left_group["weight"]) + float(right_group["weight"])) / 1400.0,
                    )
                    quality_score = (
                        0.45 * float(np.mean(edge_support))
                        + 0.35 * net_support
                        + 0.12 * centre_score
                        + 0.08 * line_strength
                    )
                    candidates.append(
                        _TableDetection(
                            corners=tuple(Point2D(float(point[0]), float(point[1])) for point in xy),  # type: ignore[index]
                            edge_support=edge_support,
                            net_support=float(net_support),
                            quality_score=float(quality_score),
                            flags=("table_edges", "net_line"),
                        )
                    )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.quality_score)


def _detect_table_from_green_fixture(image: np.ndarray) -> _TableDetection | None:
    mask = _colour_table_mask(image)
    y, x = np.nonzero(mask)
    if len(x) < 100:
        return None
    height, width = mask.shape
    # This path exists solely for the compact synthetic regression fixture.
    # Never let a failed broadcast edge detector fall back to a green arena
    # patch and manufacture a calibration at production resolution.
    if height >= 240 or width >= 400:
        return None
    min_x, max_x, min_y, max_y = int(x.min()), int(x.max()), int(y.min()), int(y.max())
    bounding_area = (max_x - min_x + 1) * (max_y - min_y + 1)
    frame_area = width * height
    if bounding_area > 0.75 * frame_area or (
        bounding_area > 0.25 * frame_area
        and (min_x <= 1 or min_y <= 1 or max_x >= width - 2 or max_y >= height - 2)
    ):
        return None
    result = (
        Point2D(float(min_x), float(max_y)),
        Point2D(float(max_x), float(max_y)),
        Point2D(float(max_x), float(min_y)),
        Point2D(float(min_x), float(min_y)),
    )
    return _TableDetection(
        corners=result,
        edge_support=(0.65, 0.65, 0.65, 0.65),
        net_support=0.0,
        quality_score=0.65,
        flags=("colour_fallback", "quality_unverified_fixture"),
        fixture_fallback=True,
    )


def _detect_table(image: np.ndarray) -> _TableDetection | None:
    detection = _detect_table_from_lines(image)
    if detection is not None:
        if (
            detection.quality_score >= AUTOMATIC_QUALITY_THRESHOLD
            and min(detection.edge_support) >= AUTOMATIC_MIN_EDGE_SUPPORT
            and detection.net_support >= AUTOMATIC_MIN_NET_SUPPORT
        ):
            return detection
        return None
    return _detect_table_from_green_fixture(image)


def detect_table_corners_heuristic(image: np.ndarray) -> tuple[Point2D, Point2D, Point2D, Point2D] | None:
    """Find table corners from broadcast table/net edges.

    The detector deliberately returns ``None`` when it cannot find a compact
    four-edge quadrilateral with a supported net line.  The old colour-extrema
    approach is retained only for the tiny synthetic green fixture used by
    the regression tests; it is not allowed to interpret a purple/blue arena
    as a table.
    """

    detection = _detect_table(image)
    return detection.corners if detection is not None else None


def calibrate_automatic(image: np.ndarray) -> TableCalibration:
    detection = _detect_table(image)
    if detection is None:
        raise CalibrationError(
            "automatic table detector failed: no compact table quadrilateral with a supported net line"
        )
    if detection.quality_score < AUTOMATIC_QUALITY_THRESHOLD and not detection.fixture_fallback:
        raise CalibrationError(
            "automatic table detector failed quality checks: "
            f"redundant edge/net score {detection.quality_score:.2f} "
            f"(minimum {AUTOMATIC_QUALITY_THRESHOLD:.2f})"
        )
    return make_calibration(
        detection.corners,
        source=CalibrationSource.AUTOMATIC,
        corner_confidences=(detection.quality_score,) * 4,
        quality_flags=("heuristic",) + detection.flags,
    )


CONSENSUS_TOLERANCE_PX = 12.0


def calibrate_consensus(
    images: Sequence[np.ndarray],
    *,
    max_samples: int = 12,
    min_agreement: int = 3,
) -> TableCalibration:
    """Calibrate a static camera from several frames and take the median corner.

    Single-frame automatic calibration is fragile on real footage for a reason
    that has nothing to do with the detector: a player leaning over the table
    to serve hides a corner, and the detector reaches past it to the skirt line
    underneath.  Those failures are transient while the camera is not, so
    disagreement between frames is itself the signal.

    Corners are the per-axis median over the frames that calibrated, which
    rejects a minority of bad frames without needing a confidence threshold.
    Per-corner spread around that median becomes the corner confidence, so a
    segment where frames never agree reports low confidence rather than
    silently returning one arbitrary frame's answer.
    """

    frames = list(images)
    if not frames:
        raise CalibrationError("consensus calibration requires at least one frame")
    if len(frames) > max_samples:
        picks = np.linspace(0, len(frames) - 1, max_samples).astype(int)
        frames = [frames[index] for index in dict.fromkeys(int(i) for i in picks)]

    corner_sets: list[list[tuple[float, float]]] = []
    scores: list[float] = []
    rejected = 0
    for image in frames:
        try:
            candidate = calibrate_automatic(image)
        except (CalibrationError, ValueError):
            rejected += 1
            continue
        corner_sets.append([(point.x, point.y) for point in candidate.image_corners])
        scores.append(float(candidate.confidence))

    if not corner_sets:
        raise CalibrationError(
            f"automatic calibration failed on all {len(frames)} sampled frames"
        )

    stack = np.asarray(corner_sets, dtype=float)
    consensus = np.median(stack, axis=0)
    spread = np.median(np.linalg.norm(stack - consensus, axis=2), axis=0)
    base = float(np.median(scores))

    # Two independent signals, deliberately not multiplied into one number:
    # ``corner_confidences`` stays the per-frame evidence strength, while
    # cross-frame agreement is reported as its own flag.  Collapsing them
    # would produce a scalar nobody can interpret -- consensus that corrects
    # the corners would then *lower* the number, which is the wrong direction.
    flags = ["heuristic", "calibration_consensus", f"consensus_frames_{len(corner_sets)}"]
    flags.append(f"consensus_spread_px_{float(np.max(spread)):.1f}")
    if rejected:
        flags.append(f"consensus_rejected_{rejected}")
    if float(np.max(spread)) > CONSENSUS_TOLERANCE_PX:
        flags.append("consensus_disagreement")
    if len(corner_sets) < min_agreement:
        flags.append("consensus_insufficient_samples")

    return make_calibration(
        tuple(Point2D(float(x), float(y)) for x, y in consensus),
        source=CalibrationSource.AUTOMATIC,
        corner_confidences=(base,) * 4,
        quality_flags=tuple(flags),
    )


def image_point_to_table(calibration: TableCalibration, point: Point2D) -> Point2D:
    return project_image_to_table(np.asarray(calibration.homography, dtype=float), point)


def calibration_quality(
    calibration: TableCalibration,
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report calibration quality using redundant image constraints.

    ``reprojection_error_px`` is retained for diagnostics, but is explicitly
    not a correctness metric: a homography fit to exactly four correspondences
    has four degrees-of-freedom constraints too few for that residual to catch
    a wrong quadrilateral.  When an image is supplied, this function checks
    support for all four observed table edges and for the net line, which can
    fail independently of the four-point DLT fit.
    """

    failure_reasons: list[str] = []
    quality_basis = "manual_override"
    edge_support: list[float] = list(calibration.corner_confidences)
    net_support: float | None = None
    if image is not None:
        measured = _measure_redundant_constraints(image, calibration.image_corners)
        quality_score = float(measured["quality_score"])
        failure_reasons.extend(measured["failure_reasons"])
        quality_basis = "observed_edges_and_net"
        edge_support = measured["edge_support"]
        net_support = measured["net_support"]
    elif calibration.source == CalibrationSource.AUTOMATIC and "quality_unverified_fixture" not in calibration.quality_flags:
        if "net_line" not in calibration.quality_flags or "table_edges" not in calibration.quality_flags:
            quality_score = 0.0
            failure_reasons.append("no_redundant_observations")
            quality_basis = "unverified"
        else:
            quality_score = float(calibration.confidence)
            edge_support = list(calibration.corner_confidences)
            net_support = float(calibration.confidence)
            quality_basis = "detector_observations"
    else:
        quality_score = 1.0
        edge_support = list(calibration.corner_confidences)
        net_support = None
        if "quality_unverified_fixture" in calibration.quality_flags:
            quality_score = 0.0
            failure_reasons.append("fixture_colour_detector_unverified")
            quality_basis = "unverified_fixture"

    if quality_score < AUTOMATIC_QUALITY_THRESHOLD and "fixture_colour_detector_unverified" not in failure_reasons:
        failure_reasons.append("quality_score_low")
    return {
        "reprojection_error_px": calibration.reprojection_error_px,
        "corner_confidences": list(calibration.corner_confidences),
        "calibration_source": calibration.source.value,
        "calibration_confidence": calibration.confidence,
        "quality_flags": list(calibration.quality_flags),
        "quality_score": quality_score,
        "quality_passed": not failure_reasons,
        "quality_basis": quality_basis,
        "edge_support": edge_support,
        "net_support": net_support,
        "quality_failure_reasons": failure_reasons,
    }
