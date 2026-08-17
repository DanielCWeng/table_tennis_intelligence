"""Environment diagnostics for the optional heavyweight backends."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PYTHON_PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "PIL",
    "cv2",
    "av",
    "rtmlib",
    "onnxruntime",
    "onnxruntime_gpu",
    "mmpose",
    "mmengine",
    "mmcv",
    "mmdet",
    "duckdb",
    "pyarrow",
)


NATIVE_TOOLS = ("ffmpeg", "ffprobe", "git", "gh", "nvidia-smi", "nvcc")


def _package_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    result: dict[str, Any] = {"installed": spec is not None}
    if spec is None:
        return result
    try:
        module = __import__(name)
        result["version"] = getattr(module, "__version__", "present")
    except Exception as exc:  # pragma: no cover - package-specific import failures
        result["version"] = None
        result["import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _tool_status(name: str) -> dict[str, Any]:
    executable = shutil.which(name)
    result: dict[str, Any] = {"installed": executable is not None, "path": executable}
    if executable and name in {"nvidia-smi", "nvcc"}:
        try:
            completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10)
            result["version_output"] = (completed.stdout or completed.stderr).splitlines()[-1:]
        except (OSError, subprocess.SubprocessError) as exc:
            result["version_error"] = str(exc)
    return result


def collect_report(project_root: str | Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": {"version": sys.version, "executable": sys.executable},
        "packages": {name: _package_status(name) for name in PYTHON_PACKAGES},
        "native_tools": {name: _tool_status(name) for name in NATIVE_TOOLS},
        "gpu": {},
    }
    try:
        import torch

        report["gpu"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        report["gpu"] = {"error": f"{type(exc).__name__}: {exc}"}
    if project_root is not None:
        root = Path(project_root)
        report["project"] = {
            "root": str(root.resolve()),
            "third_party": sorted(path.name for path in (root / "third_party").glob("*") if path.is_dir()) if (root / "third_party").is_dir() else [],
            "checkpoint_dirs": [str(path) for path in (root / "checkpoints").glob("*")] if (root / "checkpoints").is_dir() else [],
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report optional TT intelligence dependencies and GPU state")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    report = collect_report(args.project_root)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"Python: {report['python']['executable']}")
    print(f"CUDA: {report['gpu'].get('cuda_available', False)} ({report['gpu'].get('torch_cuda')})")
    print("Packages:")
    for name, status in report["packages"].items():
        label = status.get("version", "missing") if status["installed"] else "missing"
        print(f"  {name}: {label}")
    print("Native tools:")
    for name, status in report["native_tools"].items():
        print(f"  {name}: {status['path'] or 'missing'}")


if __name__ == "__main__":
    main()
