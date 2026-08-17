"""Run TT3D's unmodified segmentation and calibration sample on Kaggle.

The runner is an orchestration/reporting layer.  It never edits files under
the TT3D checkout.  Calibration rendering is performed on a byte-for-byte
copy of the fixture because the upstream ``--render`` option writes its MP4
beside the input video.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from environment_report import collect_environment
from verify_outputs import inspect_calibration_csv, inspect_video


EXPECTED_COMMIT = "a2ef524ea0400262d6808db6cacf4a0b90bd0ad7"
VIDEO_NAMES = ("test_00", "test_01", "test_02")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/tt3d_baseline")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _candidate_roots() -> Iterable[Path]:
    current = Path.cwd()
    script_root = Path(__file__).resolve().parents[2]
    yield Path("/kaggle/working/tt3d")
    yield Path("/kaggle/input/tt3d-source/tt3d")
    yield Path("/kaggle/input/tt3d/tt3d")
    yield current / "third_party" / "tt3d"
    yield script_root / "third_party" / "tt3d"


def locate_tt3d_root(requested: Path | None) -> Path:
    candidates = [requested] if requested is not None else list(_candidate_roots())
    for candidate in candidates:
        if candidate is None:
            continue
        root = candidate.expanduser().resolve()
        if (root / "tt3d" / "calibration" / "calibrate.py").is_file() and (
            root / "weights" / "table_segmentation.ckpt"
        ).is_file():
            return root
    rendered = "\n".join(f"  {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate a TT3D checkout/source bundle. Checked:\n" + rendered
    )


def verify_commit(root: Path) -> dict:
    """Verify Git HEAD or the marker included in the offline source bundle."""

    git_dir = root / ".git"
    # Some Windows launchers omit PATHEXT, making shutil.which("git") return
    # None even though git.exe is available.  Linux/Kaggle uses the first form.
    git_path = shutil.which("git") or shutil.which("git.exe") or "git"
    if git_dir.exists() and git_path:
        try:
            result = subprocess.run(
                [
                    git_path,
                    "-c",
                    f"safe.directory={root}",
                    "-C",
                    str(root),
                    "rev-parse",
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            observed = result.stdout.strip()
            if result.returncode == 0:
                return {
                    "verified": observed == EXPECTED_COMMIT,
                    "method": "git rev-parse HEAD",
                    "expected": EXPECTED_COMMIT,
                    "observed": observed,
                    "stderr": result.stderr.strip(),
                }
        except OSError:
            pass

    marker_candidates = (
        root / "TT3D_COMMIT.txt",
        root / "tt3d_commit.txt",
        root.parent / "TT3D_COMMIT.txt",
    )
    for marker in marker_candidates:
        if marker.is_file():
            observed = marker.read_text(encoding="utf-8").strip().splitlines()[0]
            return {
                "verified": observed == EXPECTED_COMMIT,
                "method": f"commit marker: {marker.name}",
                "expected": EXPECTED_COMMIT,
                "observed": observed,
            }

    return {
        "verified": False,
        "method": "none",
        "expected": EXPECTED_COMMIT,
        "observed": None,
        "error": "No usable Git checkout or TT3D_COMMIT.txt marker was found.",
    }


def verify_assets(root: Path) -> dict:
    checkpoint = root / "weights" / "table_segmentation.ckpt"
    videos = {
        name: root / "data" / "calibration_test" / "videos" / f"{name}.mp4"
        for name in VIDEO_NAMES
    }
    calibration_dir = root / "data" / "calibration_test"
    assets = {
        "checkpoint": {
            "path": str(checkpoint),
            "exists": checkpoint.is_file(),
        },
        "videos": {},
        "calibration_test_directory": {
            "path": str(calibration_dir),
            "exists": calibration_dir.is_dir(),
        },
    }
    if checkpoint.is_file():
        assets["checkpoint"].update(
            {"bytes": checkpoint.stat().st_size, "sha256": _sha256(checkpoint)}
        )
    for name, video in videos.items():
        item = {"path": str(video), "exists": video.is_file()}
        if video.is_file():
            item.update({"bytes": video.stat().st_size, "sha256": _sha256(video)})
        assets["videos"][name] = item
    return assets


def _run_command(
    name: str,
    command: list[str],
    cwd: Path,
    output_dir: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> dict:
    """Run one upstream command and persist its complete two-stream output."""

    command_string = shlex.join(command)
    print(f"\n===== {name} =====")
    print(f"cwd: {cwd}")
    print(f"command: {command_string}")
    started = time.perf_counter()
    started_at = _utc_now()
    stdout = ""
    stderr = ""
    exception = None
    timed_out = False
    exit_code = None
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        exception = f"TimeoutExpired after {timeout_seconds} seconds"
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - started

    stdout_path = output_dir / f"{name}.stdout.txt"
    stderr_path = output_dir / f"{name}.stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    print(f"runtime_seconds: {runtime:.3f}")
    print(f"exit_code: {exit_code}")
    if exception:
        print(f"exception: {exception}")
    print("--- stdout ---")
    print(stdout, end="" if stdout.endswith("\n") or not stdout else "\n")
    print("--- stderr ---")
    print(stderr, end="" if stderr.endswith("\n") or not stderr else "\n")
    return {
        "name": name,
        "command": command_string,
        "cwd": str(cwd),
        "started_at_utc": started_at,
        "runtime_seconds": runtime,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "exception": exception,
        "stdout_path": _relative(stdout_path, output_dir),
        "stderr_path": _relative(stderr_path, output_dir),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
    }


def _camera_parameters(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {"status": "not_available"}
    valid = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                focal = float(row["f"])
            except (KeyError, TypeError, ValueError):
                continue
            if focal <= 0:
                continue
            valid.append(
                {
                    "index": int(float(row["Index"])),
                    "rvec": [float(row[f"rvec_{axis}"]) for axis in "xyz"],
                    "tvec": [float(row[f"tvec_{axis}"]) for axis in "xyz"],
                    "f": focal,
                }
            )
    if not valid:
        return {"status": "no_valid_rows"}
    return {
        "status": "exposed_in_raw_csv",
        "valid_row_count": len(valid),
        "first_valid": valid[0],
        "last_valid": valid[-1],
    }


def _run_one_video(
    root: Path,
    source_video: Path,
    name: str,
    output_dir: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> dict:
    segment_script = root / "tt3d" / "calibration" / "segment.py"
    calibrate_script = root / "tt3d" / "calibration" / "calibrate.py"
    segmentation_output = output_dir / f"{name}_segmented.mp4"
    calibration_csv = output_dir / f"{name}_cam_cal.csv"
    calibration_input_dir = output_dir / "calibration_inputs"
    calibration_input_dir.mkdir(parents=True, exist_ok=True)
    calibration_input = calibration_input_dir / source_video.name
    shutil.copy2(source_video, calibration_input)
    calibration_render = calibration_input.with_name(
        f"{calibration_input.stem}_calibrated.mp4"
    )

    segmentation_result = _run_command(
        f"{name}_segment",
        [
            sys.executable,
            str(segment_script),
            str(source_video),
            "--render",
            "--output",
            str(segmentation_output),
        ],
        root,
        output_dir,
        environment,
        timeout_seconds,
    )
    calibration_result = _run_command(
        f"{name}_calibrate",
        [
            sys.executable,
            str(calibrate_script),
            str(calibration_input),
            "-o",
            str(calibration_csv),
            "--render",
        ],
        root,
        output_dir,
        environment,
        timeout_seconds,
    )

    csv_summary = inspect_calibration_csv(calibration_csv)
    segmentation_video = inspect_video(segmentation_output)
    calibration_video = inspect_video(calibration_render)
    calibration_valid = bool(
        csv_summary.get("exists")
        and csv_summary.get("rows", 0) > 0
        and csv_summary.get("valid_rows_f_gt_zero", 0) > 0
    )
    segmentation_valid = bool(
        segmentation_result.get("exit_code") == 0
        and segmentation_video.get("readable") is True
    )
    calibration_process_ok = calibration_result.get("exit_code") == 0
    calibration_render_valid = calibration_video.get("readable") is True
    execution_success = bool(
        segmentation_valid
        and calibration_process_ok
        and calibration_valid
        and calibration_render_valid
    )

    generated_files = [
        segmentation_result["stdout_path"],
        segmentation_result["stderr_path"],
        calibration_result["stdout_path"],
        calibration_result["stderr_path"],
    ]
    for candidate in (segmentation_output, calibration_csv, calibration_render):
        if candidate.is_file():
            generated_files.append(_relative(candidate, output_dir))

    warnings = []
    if segmentation_result.get("stderr_bytes", 0):
        warnings.append("segmentation stderr is non-empty; inspect the saved log")
    if calibration_result.get("stderr_bytes", 0):
        warnings.append("calibration stderr is non-empty; inspect the saved log")
    if csv_summary.get("invalid_or_zero_rows", 0):
        warnings.append(
            f"{csv_summary['invalid_or_zero_rows']} rows have f <= 0 or invalid values"
        )

    return {
        "source_video": str(source_video),
        "source_video_bytes": source_video.stat().st_size,
        "source_video_sha256": _sha256(source_video),
        "calibration_input_copy": _relative(calibration_input, output_dir),
        "segmentation": {
            "process": segmentation_result,
            "render": segmentation_video,
            "success": segmentation_valid,
        },
        "calibration": {
            "process": calibration_result,
            "csv": {
                "path": _relative(calibration_csv, output_dir),
                "summary": csv_summary,
            },
            "camera_parameters": _camera_parameters(calibration_csv),
            "render": calibration_video,
            "success": calibration_process_ok
            and calibration_valid
            and calibration_render_valid,
        },
        "detected_corners": {
            "status": "not_exported_by_upstream",
            "source": "TableCalibrator.process draws them only into debug_img",
        },
        "reprojection_error": {
            "status": "not_exposed_by_upstream_calibrate_command",
            "source": "calibrate.py receives er but save_camcal is called without errors",
        },
        "rendered_files": [
            _relative(path, output_dir)
            for path in (segmentation_output, calibration_render)
            if path.is_file()
        ],
        "generated_files": list(dict.fromkeys(generated_files)),
        "warnings": warnings,
        "execution_success": execution_success,
    }


def _base_report(output_dir: Path, environment: dict) -> dict:
    dependencies = environment.get("dependencies", {})
    cuda = environment.get("cuda", {})
    return {
        "tt3d_commit": EXPECTED_COMMIT,
        "platform": environment.get("platform", platform.platform()),
        "python": environment.get("python_version", platform.python_version()),
        "torch": dependencies.get("torch"),
        "torchvision": dependencies.get("torchvision"),
        "cuda": cuda.get("runtime"),
        "gpu": environment.get("gpu", []),
        "dependencies": dependencies,
        "environment_before": environment,
        "environment_after": None,
        "videos": {name: {"status": "not_run"} for name in VIDEO_NAMES},
        "success": False,
        "execution_success": False,
        "visual_validation": {
            "status": "pending_manual_review",
            "note": "Inspect rendered outputs with verify_outputs.py before approval.",
        },
        "failures": [],
        "generated_files": [],
        "output_directory": str(output_dir),
        "started_at_utc": _utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tt3d-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1200,
        help="Per-upstream-command timeout; default is 20 minutes.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment()
    print("===== environment before =====")
    print(json.dumps(environment, indent=2, sort_keys=True))
    report = _base_report(output_dir, environment)
    report_path = output_dir / "tt3d_baseline_report.json"
    environment_report_path = output_dir / "environment_report.json"
    _write_json(environment_report_path, environment)
    report["generated_files"] = [_relative(environment_report_path, output_dir)]

    try:
        root = locate_tt3d_root(args.tt3d_root)
        commit = verify_commit(root)
        report["source"] = {
            "root": str(root),
            "commit_verification": commit,
        }
        if not commit.get("verified"):
            report["failures"].append(
                f"TT3D commit verification failed: {json.dumps(commit, sort_keys=True)}"
            )
            _write_json(report_path, report)
            print(json.dumps(report, indent=2))
            return 1

        assets = verify_assets(root)
        report["assets"] = assets
        if not assets["checkpoint"]["exists"]:
            report["failures"].append("table_segmentation.ckpt is missing")
        for name, item in assets["videos"].items():
            if not item["exists"]:
                report["failures"].append(f"{name}.mp4 is missing")
        if report["failures"]:
            _write_json(report_path, report)
            print(json.dumps(report, indent=2))
            return 1

        runtime_environment = os.environ.copy()
        runtime_environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "MPLBACKEND": "Agg",
                "WANDB_MODE": "disabled",
                "WANDB_SILENT": "true",
            }
        )
        for name in VIDEO_NAMES:
            source_video = Path(assets["videos"][name]["path"])
            if name != "test_00" and not report["videos"]["test_00"].get(
                "execution_success", False
            ):
                report["videos"][name] = {
                    "status": "skipped_until_test_00_execution_success",
                }
                continue
            record = _run_one_video(
                root,
                source_video,
                name,
                output_dir,
                runtime_environment,
                args.timeout_seconds,
            )
            report["videos"][name] = record
            if not record["execution_success"]:
                report["failures"].append(
                    f"{name}: segmentation/calibration execution did not pass numeric checks"
                )
                if name == "test_00":
                    for later in VIDEO_NAMES[1:]:
                        report["videos"][later] = {
                            "status": "skipped_until_test_00_execution_success"
                        }
                    break

        after_environment = collect_environment()
        report["environment_after"] = after_environment
        environment_after_path = output_dir / "environment_after.json"
        _write_json(environment_after_path, after_environment)
        report["execution_success"] = not report["failures"] and all(
            report["videos"][name].get("execution_success", False)
            for name in VIDEO_NAMES
            if report["videos"][name].get("status") != "skipped_until_test_00_execution_success"
        )
        report["finished_at_utc"] = _utc_now()
        generated = []
        for record in report["videos"].values():
            generated.extend(record.get("generated_files", []))
            generated.extend(record.get("rendered_files", []))
        report["generated_files"] = list(
            dict.fromkeys(
                [
                    _relative(environment_report_path, output_dir),
                    _relative(environment_after_path, output_dir),
                    *generated,
                ]
            )
        )
        _write_json(report_path, report)
        print("===== baseline report =====")
        print(json.dumps(report, indent=2))
        # A successful process run is deliberately not the same as a validated
        # baseline. verify_outputs.py must approve the visual geometry later.
        return 0 if report["execution_success"] else 1
    except Exception as exc:
        report["failures"].append(f"runner exception: {type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
        report["finished_at_utc"] = _utc_now()
        _write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
