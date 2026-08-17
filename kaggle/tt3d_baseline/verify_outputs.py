"""Validate TT3D baseline artifacts and record manual visual approval.

The upstream scripts expose calibration rows and rendered videos, but they do
not export detected corners or per-frame reprojection errors.  This validator
keeps those fields explicitly marked as unavailable instead of inventing them.
Run it once for numeric/file checks, then rerun with ``--visual-ok`` only after
reviewing the rendered videos in Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def inspect_calibration_csv(path: Path) -> dict:
    """Summarize the raw CSV written by TT3D's ``save_camcal``."""

    expected_columns = {
        "Index",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "tvec_x",
        "tvec_y",
        "tvec_z",
        "f",
        "error",
    }
    if not path.exists():
        return {"exists": False, "path": str(path)}

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    focal_lengths = []
    valid_rows = 0
    error_values = []
    for row in rows:
        focal = _finite_float(row.get("f"))
        if focal is not None:
            focal_lengths.append(focal)
        if focal is not None and focal > 0:
            valid_rows += 1
        error = _finite_float(row.get("error"))
        if error is not None:
            error_values.append(error)

    summary = {
        "exists": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "columns": columns,
        "missing_expected_columns": sorted(expected_columns - set(columns)),
        "rows": len(rows),
        "valid_rows_f_gt_zero": valid_rows,
        "invalid_or_zero_rows": len(rows) - valid_rows,
        "focal_length": {},
        "error_column": {
            "numeric_values": len(error_values),
            "status": "exposed" if error_values else "not_exposed_or_empty",
        },
    }
    if focal_lengths:
        summary["focal_length"] = {
            "min": min(focal_lengths),
            "median": statistics.median(focal_lengths),
            "max": max(focal_lengths),
        }
    if error_values:
        summary["error_column"].update(
            {
                "min": min(error_values),
                "median": statistics.median(error_values),
                "max": max(error_values),
            }
        )
    return summary


def inspect_video(path: Path) -> dict:
    """Check that an OpenCV-readable rendered video has frames."""

    result = {"exists": path.exists(), "path": str(path)}
    if not path.exists():
        return result
    result["bytes"] = path.stat().st_size
    try:
        import cv2
    except Exception as exc:
        result["readable"] = None
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_read = False
        if opened:
            ok, _ = capture.read()
            first_read = bool(ok)
        result.update(
            {
                "readable": opened and first_read,
                "frames_reported": frames,
                "fps": fps,
                "width": width,
                "height": height,
                "first_frame_read": first_read,
            }
        )
    finally:
        capture.release()
    return result


def _resolve_report_path(value: str, report_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else report_path.parent / path


def validate_report(report_path: Path, visual_ok: bool, visual_note: str) -> tuple[dict, list[str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = list(report.get("failures", []))

    if not report.get("execution_success", False):
        failures.append("runner execution_success is false")

    videos = report.get("videos", {})
    for video_name, record in videos.items():
        calibration = record.get("calibration", {})
        csv_value = calibration.get("csv")
        if csv_value:
            csv_summary = inspect_calibration_csv(
                _resolve_report_path(csv_value.get("path", ""), report_path)
            )
            record["calibration"]["csv_validation"] = csv_summary
            if not csv_summary.get("exists"):
                failures.append(f"{video_name}: calibration CSV is missing")
            elif csv_summary.get("rows", 0) == 0:
                failures.append(f"{video_name}: calibration CSV has no rows")

        rendered = record.get("rendered_files", [])
        rendered_validation = []
        for value in rendered:
            rendered_validation.append(
                inspect_video(_resolve_report_path(value, report_path))
            )
        record["rendered_validation"] = rendered_validation
        for item in rendered_validation:
            if item.get("exists") and item.get("readable") is False:
                failures.append(f"{video_name}: rendered video is not readable")
        if record.get("execution_success") and (
            not rendered_validation
            or any(not item.get("exists") or item.get("readable") is not True for item in rendered_validation)
        ):
            failures.append(f"{video_name}: required rendered validation output is missing")

    # Preserve order while removing duplicate diagnostics.
    failures = list(dict.fromkeys(failures))
    if visual_ok:
        if not report.get("execution_success", False) or failures:
            report["visual_validation"] = {
                "status": "rejected",
                "note": visual_note,
            }
        else:
            report["visual_validation"] = {
                "status": "approved",
                "note": visual_note,
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
    else:
        report["visual_validation"] = {
            "status": "pending_manual_review",
            "note": "Inspect segmentation and calibration renders before approval.",
        }

    report["failures"] = failures
    report["success"] = bool(
        report.get("execution_success", False)
        and not failures
        and report["visual_validation"]["status"] == "approved"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/kaggle/working/tt3d_baseline/tt3d_baseline_report.json"),
    )
    parser.add_argument(
        "--visual-ok",
        action="store_true",
        help="Approve the geometry after manual visual inspection.",
    )
    parser.add_argument(
        "--visual-note",
        default="",
        help="Short record of the manual visual review.",
    )
    args = parser.parse_args()
    report, failures = validate_report(args.report, args.visual_ok, args.visual_note)
    print(json.dumps(report, indent=2))
    if failures:
        return 1
    if not args.visual_ok:
        return 2
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
