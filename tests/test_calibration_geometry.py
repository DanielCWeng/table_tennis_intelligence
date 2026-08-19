"""Guards for the two calibration quality signals that can actually fail."""

import numpy as np

from ttintel.geometry import (
    TABLE_LENGTH_M,
    TABLE_WIDTH_M,
    corner_sensitivity_m_per_px,
    rectangle_consistency,
)
from ttintel.schemas import Point2D


def _project(focal: float, distance: float, elevation_deg: float) -> list[Point2D]:
    """Project the regulation table with a known pinhole camera."""

    angle = np.radians(elevation_deg)
    rotation = np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )
    intrinsics = np.array([[focal, 0, 640], [0, focal, 360], [0, 0, 1]], dtype=float)
    centre = np.array(
        [TABLE_LENGTH_M / 2, -distance * np.cos(angle), distance * np.sin(angle)]
    )
    corners = []
    for point in ((0, 0, 0), (TABLE_LENGTH_M, 0, 0), (TABLE_LENGTH_M, TABLE_WIDTH_M, 0), (0, TABLE_WIDTH_M, 0)):
        camera = rotation @ (np.asarray(point, dtype=float) - centre)
        image = intrinsics @ camera
        corners.append(Point2D(image[0] / image[2], image[1] / image[2]))
    return corners


def test_focal_recovery_is_exact_on_noiseless_corners() -> None:
    for focal, distance, elevation in ((800, 6, 35), (2500, 25, 12)):
        result = rectangle_consistency(_project(focal, distance, elevation), image_size=(1280, 720))
        assert result["focal_estimates_px"]
        assert min(abs(value - focal) for value in result["focal_estimates_px"]) < 1.0


def test_focal_estimate_is_too_noise_sensitive_to_gate_on() -> None:
    """Documents why there is no per-frame rectangle-validity boolean.

    Half a pixel of corner noise flips a solution negative in a large share of
    trials, so a sign test would reject correct calibrations roughly as often
    as wrong ones.
    """

    rng = np.random.default_rng(7)
    base = _project(2500, 25, 12)
    negatives = 0
    for _ in range(120):
        noisy = [Point2D(p.x + rng.normal(0, 0.5), p.y + rng.normal(0, 0.5)) for p in base]
        if rectangle_consistency(noisy, image_size=(1280, 720))["negative_focal_solutions"]:
            negatives += 1
    assert negatives > 24, "sign test unexpectedly stable; revisit the gating decision"


def test_sensitivity_separates_oblique_views_from_end_on_views() -> None:
    oblique = [Point2D(300, 600), Point2D(980, 600), Point2D(880, 300), Point2D(400, 300)]
    end_on = [Point2D(100, 715), Point2D(1180, 715), Point2D(650, 300), Point2D(630, 300)]
    assert corner_sensitivity_m_per_px(oblique) < 0.05
    assert corner_sensitivity_m_per_px(end_on) > 0.05


def test_sensitivity_flags_the_london_skirt_quadrilateral() -> None:
    """The wrong quad from the London serve frame, which scored 0.88."""

    skirt = [Point2D(204.0, 216.6), Point2D(465.7, 248.7), Point2D(453.6, 209.2), Point2D(205.6, 211.3)]
    good = [Point2D(207.8, 204.6), Point2D(434.4, 204.5), Point2D(414.8, 142.9), Point2D(226.2, 142.5)]
    assert corner_sensitivity_m_per_px(skirt) > 0.05
    assert corner_sensitivity_m_per_px(good) < 0.05
