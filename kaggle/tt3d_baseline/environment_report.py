"""Print the runtime information needed to reproduce the TT3D baseline.

This module intentionally has no project-specific dependencies.  Missing
optional packages are reported as ``unavailable`` so it can be run before the
Kaggle dependency installation decision is made.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
from typing import Iterable


def _distribution_version(names: Iterable[str]) -> str | None:
    """Return the first installed distribution version from *names*."""

    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            return f"error: {type(exc).__name__}: {exc}"
    return None


def _display_version(value: str | None) -> str:
    return value if value is not None else "unavailable"


def collect_environment() -> dict:
    """Collect package, CUDA, GPU, Python, and platform information."""

    dependencies = {
        "torch": _distribution_version(("torch",)),
        "torchvision": _distribution_version(("torchvision",)),
        "numpy": _distribution_version(("numpy",)),
        "scipy": _distribution_version(("scipy",)),
        "opencv": _distribution_version(("opencv-python", "opencv-python-headless")),
        "lightning": _distribution_version(("lightning",)),
        "segmentation_models_pytorch": _distribution_version(
            ("segmentation-models-pytorch",)
        ),
        "timm": _distribution_version(("timm",)),
        "casadi": _distribution_version(("casadi",)),
        "albumentations": _distribution_version(("albumentations",)),
        "pandas": _distribution_version(("pandas",)),
        "matplotlib": _distribution_version(("matplotlib",)),
        "pillow": _distribution_version(("pillow", "Pillow")),
        "tqdm": _distribution_version(("tqdm",)),
        "wandb": _distribution_version(("wandb",)),
        "pyyaml": _distribution_version(("PyYAML", "pyyaml")),
        "sympy": _distribution_version(("sympy",)),
    }

    cuda = {
        "available": False,
        "runtime": None,
        "device_count": 0,
        "devices": [],
        "memory": [],
    }
    torch_import_error = None
    try:
        import torch

        cuda["runtime"] = torch.version.cuda
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["device_count"] = int(torch.cuda.device_count())
        for index in range(cuda["device_count"]):
            name = torch.cuda.get_device_name(index)
            device_info = {"index": index, "name": name}
            cuda["devices"].append(device_info)
            try:
                free, total = torch.cuda.mem_get_info(index)
                cuda["memory"].append(
                    {
                        "index": index,
                        "free_bytes": int(free),
                        "total_bytes": int(total),
                    }
                )
            except Exception as exc:  # pragma: no cover - driver dependent
                cuda["memory"].append(
                    {"index": index, "error": f"{type(exc).__name__}: {exc}"}
                )
    except Exception as exc:  # diagnostic output must still be usable
        torch_import_error = f"{type(exc).__name__}: {exc}"

    report = {
        "python_version": platform.python_version(),
        "python_full_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": dependencies,
        "cuda": cuda,
        "gpu": [device["name"] for device in cuda["devices"]],
    }
    if torch_import_error is not None:
        report["torch_import_error"] = torch_import_error
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; stdout is always emitted.",
    )
    args = parser.parse_args()

    report = collect_environment()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
