"""Launch the optional Qt Live Lab without importing heavy offline adapters."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .core import Region


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ttintel live",
        description="Local screen/file diagnostics; no automatic recording or model downloads.",
    )
    sources = p.add_mutually_exclusive_group()
    sources.add_argument("--file", type=Path, help="play a local video (timestamped)")
    sources.add_argument("--demo", action="store_true", help="synthetic no-video smoke source")
    sources.add_argument("--region", type=Region.parse,
                         help="screen pixels left,top,width,height; bypass drag selector")
    p.add_argument("--monitor", type=int, default=1, help="MSS monitor index (default 1)")
    p.add_argument("--fps", type=int, default=30, help="screen/demo capture target FPS")
    p.add_argument("--analysis-width", type=int, default=384,
                   help="optical-flow processing width in pixels (minimum 128)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 1 <= args.fps <= 120:
        raise SystemExit("live: --fps must be 1..120")
    if args.analysis_width < 128:
        raise SystemExit("live: --analysis-width must be >=128")
    # Windows per-monitor DPI awareness must be requested before loading Qt
    # or MSS. A pre-existing awareness setting may already be installed.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (OSError, AttributeError):
            pass
    try:
        from PySide6 import QtWidgets
        import pyqtgraph  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Live UI unavailable: install python -m pip install -e '.[live]'"
        ) from exc

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["ttintel-live"])
    mode = "file" if args.file else "demo" if args.demo else "screen"
    region = args.region
    if mode == "screen" and region is None:
        if sys.platform != "win32":
            raise SystemExit("Direct desktop selection requires Windows; use --region on this OS")
        from .select import select_screen_region
        try:
            region = select_screen_region(app, monitor=args.monitor)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"live: {exc}") from exc
        if region is None:
            return

    from .sources import LiveEngine
    from .gui import LiveWindow

    try:
        engine = LiveEngine(mode, region=region, file=args.file,
                            fps=args.fps, width=args.analysis_width)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"live: {exc}") from exc
    window = LiveWindow(engine)
    window.show()
    app.exec()
