"""Command-line entry point for ``analyse video.mp4``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .calibration import calibrate_manual, load_manual_corners
from .adapters.totnet import TotnetUnavailable
from .media import MediaBackendUnavailable
from .pipeline import analyse_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyse",
        description="Convert a table-tennis video into confidence-aware session data.",
    )
    parser.add_argument("video", type=Path, help="input video")
    parser.add_argument("--output", type=Path, default=Path("outputs/sessions"), help="session output root")
    parser.add_argument("--annotations", type=Path, help="optional local JSON frame annotations")
    parser.add_argument(
        "--ball-tracker",
        choices=("totnet", "blob", "none"),
        default="totnet",
        help="ball tracker (default: totnet; blob is the diagnostic baseline)",
    )
    parser.add_argument("--pose-backend", choices=("none", "rtmlib"), default="none", help="optional pose backend")
    parser.add_argument("--pose-device", default="cuda", help="rtmlib device, e.g. cuda or cpu")
    parser.add_argument("--pose-mode", choices=("performance", "balanced", "lightweight"), default="balanced")
    parser.add_argument("--manual-corners", type=Path, help="JSON corners in near-left/right, far-right/left order")
    parser.add_argument("--max-frames", type=int, help="limit decoded frames for a smoke run")
    parser.add_argument("--no-render", action="store_true", help="skip overlay frame rendering")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manual = None
    if args.manual_corners:
        manual = calibrate_manual(load_manual_corners(args.manual_corners))
    pose_estimator = None
    if args.pose_backend == "rtmlib":
        try:
            from .adapters.rtmlib import RtmlibWholebodyEstimator

            pose_estimator = RtmlibWholebodyEstimator(mode=args.pose_mode, device=args.pose_device)
        except RuntimeError as exc:
            print(f"analyse: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    try:
        result = analyse_video(
            args.video,
            output_root=args.output,
            max_frames=args.max_frames,
            annotations=args.annotations,
            manual_calibration=manual,
            pose_estimator=pose_estimator,
            ball_tracker=args.ball_tracker,
            render=not args.no_render,
        )
    except (FileNotFoundError, MediaBackendUnavailable, TotnetUnavailable, ValueError) as exc:
        print(f"analyse: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    summary = result.session.derived.get("summary", {})
    print(
        json.dumps(
            {
                "session_id": result.session.session_id,
                "session_path": str(result.session_path) if result.session_path else None,
                "segments": len(result.session.segments),
                "frames": len(result.session.frames),
                "events": len(result.session.events),
                "rendered_frames": len(result.render_paths),
                "rendered_video": str(result.render_video_path) if result.render_video_path else None,
                "ball_tracker": result.session.metadata.get("ball_tracker"),
                "summary": summary,
                "warnings": result.warnings,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
