"""Media inspection and timestamp-preserving frame access.

PyAV, OpenCV, and FFmpeg are optional because research environments often
provide them outside the application environment.  The core never silently
fabricates timestamps: the OpenCV fallback marks timestamps as estimated when
the backend does not expose presentation timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator

from .schemas import VideoMetadata


class MediaBackendUnavailable(RuntimeError):
    """Raised when no installed backend can inspect or decode the video."""


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    image: Any
    timestamp_estimated: bool = False


def _video_id(path: Path) -> str:
    # A stable, local identifier without reading the whole file.
    stat = path.stat()
    return f"{path.stem}-{stat.st_size:x}-{stat.st_mtime_ns:x}"


def _probe_with_pyav(path: Path) -> VideoMetadata | None:
    try:
        import av  # type: ignore
    except ImportError:
        return None
    container = av.open(str(path))
    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"no video stream found in {path}")
        rate = stream.average_rate
        fps = float(rate) if rate else None
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / 1_000_000.0)
        return VideoMetadata(
            video_id=_video_id(path),
            source_path=str(path),
            width=int(stream.width) if stream.width else None,
            height=int(stream.height) if stream.height else None,
            duration=duration,
            nominal_fps=fps,
            is_vfr=None,
            codec=str(stream.codec_context.name) if stream.codec_context else None,
            creation_metadata={"backend": "pyav"},
        )
    finally:
        container.close()


def _probe_with_ffprobe(path: Path) -> VideoMetadata | None:
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,r_frame_rate,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout).get("streams", [{}])[0]

    def parse_rate(value: str | None) -> float | None:
        if not value or value in {"0/0", "N/A"}:
            return None
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)

    nominal = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    return VideoMetadata(
        video_id=_video_id(path),
        source_path=str(path),
        width=int(stream["width"]) if stream.get("width") else None,
        height=int(stream["height"]) if stream.get("height") else None,
        duration=float(stream["duration"]) if stream.get("duration") else None,
        nominal_fps=nominal,
        is_vfr=(
            parse_rate(stream.get("r_frame_rate")) != parse_rate(stream.get("avg_frame_rate"))
            if stream.get("r_frame_rate") and stream.get("avg_frame_rate")
            else None
        ),
        codec=stream.get("codec_name"),
        creation_metadata={"backend": "ffprobe"},
    )


def probe_video(path: str | Path) -> VideoMetadata:
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    metadata = _probe_with_pyav(video_path) or _probe_with_ffprobe(video_path)
    if metadata is None:
        raise MediaBackendUnavailable(
            "No media probe backend is installed. Install 'av' or provide FFmpeg/ffprobe."
        )
    return metadata


def _iter_pyav(path: Path, *, start_time: float = 0.0) -> Iterator[FramePacket]:
    import av  # type: ignore

    container = av.open(str(path))
    try:
        stream = next(item for item in container.streams if item.type == "video")
        if start_time > 0 and stream.time_base is not None:
            container.seek(int(start_time / float(stream.time_base)), stream=stream)
        frame_id = 0
        for frame in container.decode(stream):
            timestamp = frame.time
            if timestamp is None:
                timestamp = float(frame_id / float(stream.average_rate or 1.0))
                estimated = True
            else:
                estimated = False
            yield FramePacket(frame_id=frame_id, timestamp=float(timestamp), image=frame.to_ndarray(format="rgb24"), timestamp_estimated=estimated)
            frame_id += 1
    finally:
        container.close()


def _iter_opencv(path: Path, *, start_time: float = 0.0) -> Iterator[FramePacket]:
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MediaBackendUnavailable(f"OpenCV could not open {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if start_time > 0:
            capture.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)
        frame_id = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            pos_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            estimated = pos_ms <= 0.0
            timestamp = pos_ms / 1000.0 if not estimated else (frame_id / fps if fps else 0.0)
            # OpenCV returns BGR; normalise the package boundary to RGB.
            yield FramePacket(frame_id=frame_id, timestamp=timestamp, image=image[:, :, ::-1], timestamp_estimated=estimated)
            frame_id += 1
    finally:
        capture.release()


def iter_video_frames(path: str | Path, *, start_time: float = 0.0) -> Iterator[FramePacket]:
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    try:
        import av  # noqa: F401
    except ImportError:
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise MediaBackendUnavailable(
                "No frame decoder is installed. Install 'av' or 'opencv-python'."
            ) from exc
        yield from _iter_opencv(video_path, start_time=start_time)
    else:
        yield from _iter_pyav(video_path, start_time=start_time)


def read_frame_packets(path: str | Path, *, max_frames: int | None = None) -> list[FramePacket]:
    packets: list[FramePacket] = []
    for packet in iter_video_frames(path):
        packets.append(packet)
        if max_frames is not None and len(packets) >= max_frames:
            break
    return packets
