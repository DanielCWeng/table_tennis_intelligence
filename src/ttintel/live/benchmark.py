"""Repeatable CPU-only synthetic analysis benchmark (not a GUI FPS claim).

Run: python -m ttintel.live.benchmark --frames 240 [--json result.json]
Input is constructed imagery, not actual WTT footage or the capture/render path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

from ttintel.media import FramePacket
from .core import LiveMotionTracker, ROI
from .sources import synthetic_frame


def benchmark(frames: int = 180, width: int = 384) -> dict:
    if frames < 2:
        raise ValueError("at least two frames required")
    tracker = LiveMotionTracker(width=width)
    tracker.set_regions(ROI(78, 208, 240, 135), ROI(430, 90, 100, 120))
    timings: list[float] = []
    valid = 0
    moving = 0
    max_lateral_step = 0.0
    states: dict[str, int] = {}
    started = time.perf_counter()
    for i in range(frames):
        image = synthetic_frame(i / 30.)
        now = time.perf_counter()
        sample = tracker.process(FramePacket(i, i / 30., image))
        timings.append((time.perf_counter() - now) * 1000)
        valid += sample.position.value is not None
        delta = sample.displacement.value
        if delta is not None:
            moving += abs(delta.x) > 0.30
            max_lateral_step = max(max_lateral_step, abs(delta.x))
        states[sample.reason] = states.get(sample.reason, 0) + 1
    elapsed = time.perf_counter() - started
    return {
        "environment": f"{platform.system()} {platform.machine()} Python {platform.python_version()}",
        "source": "synthetic constructed frames; analysis only",
        "frames": frames,
        "width_px": width,
        "analysis_frames_per_second": round(frames / (sum(timings) / 1000), 2),
        "full_synthetic_loop_fps": round(frames / elapsed, 2),
        "wall_seconds": round(elapsed, 3),
        "median_analysis_ms": round(statistics.median(timings), 3),
        "p95_analysis_ms": round(sorted(timings)[int((len(timings) - 1) * .95)], 3),
        "valid_position_frames": valid,
        "frames_with_lateral_motion_over_0_3px": moving,
        "max_abs_lateral_step_px": round(max_lateral_step, 3),
        "reasons": states,
        "capture_fps": None,
        "display_latency_ms": None,
        "true_player_detection_accuracy": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = benchmark(args.frames, args.width)
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
