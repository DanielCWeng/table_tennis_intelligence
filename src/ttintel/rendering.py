"""Human-readable overlays used as the primary perception debugging output."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .geometry import project_table_to_image
from .schemas import CameraSegment, Event, EventType, FrameState, InferenceType, Point2D


SKELETON_EDGES = (
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "left_shoulder"),
    ("right_hip", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)


def _colour_for_inference(inference_type: InferenceType) -> tuple[int, int, int]:
    if inference_type in {InferenceType.OBSERVED, InferenceType.DERIVED}:
        return (50, 220, 80)
    if inference_type == InferenceType.TEMPORALLY_TRACKED:
        return (50, 180, 240)
    return (240, 180, 40)


def _event_label(event: Event) -> str:
    return f"{event.event_type.value} {event.confidence:.2f}"


def render_frame(
    image: object,
    frame: FrameState,
    *,
    segment: CameraSegment | None = None,
    trail: Sequence[Point2D] = (),
    events: Iterable[Event] = (),
) -> object:
    """Draw an RGB overlay and return a Pillow image.

    Pillow is loaded lazily so schema/geometry code remains usable in model
    environments that do not need rendering.
    """

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Pillow is required for rendering") from exc

    if isinstance(image, Image.Image):
        canvas = image.convert("RGB")
    else:
        canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    if segment and segment.table:
        corners = segment.table.image_corners
        draw.line([(point.x, point.y) for point in (*corners, corners[0])], fill=(30, 220, 255), width=3)
        try:
            homography = np.asarray(segment.table.homography, dtype=float)
            origin = project_table_to_image(homography, Point2D(0.0, 0.0))
            x_axis = project_table_to_image(homography, Point2D(0.5, 0.0))
            y_axis = project_table_to_image(homography, Point2D(0.0, 0.5))
            draw.line([(origin.x, origin.y), (x_axis.x, x_axis.y)], fill=(255, 80, 80), width=3)
            draw.line([(origin.x, origin.y), (y_axis.x, y_axis.y)], fill=(80, 255, 80), width=3)
        except (ValueError, TypeError):
            pass

    for point in trail:
        draw.ellipse((point.x - 2, point.y - 2, point.x + 2, point.y + 2), fill=(245, 220, 30))
    if frame.ball and frame.ball.image.value:
        point = frame.ball.image.value
        confidence = frame.ball.image.confidence
        radius = 5 + int(5 * confidence)
        draw.ellipse((point.x - radius, point.y - radius, point.x + radius, point.y + radius), outline=(255, 235, 30), width=3)
        draw.text((point.x + radius + 2, point.y - radius), f"ball {confidence:.2f}", fill=(255, 235, 30))

    for player_id, player in frame.players.items():
        colour = (80, 150, 255) if player_id.endswith("0") else (255, 100, 170)
        if player.bbox and player.bbox.value:
            box = player.bbox.value
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=colour, width=2)
            draw.text((box.x1, max(0.0, box.y1 - 16)), player_id, fill=colour)
        for first, second in SKELETON_EDGES:
            left = player.joint(first)
            right = player.joint(second)
            if left and right and left.image.value and right.image.value:
                draw.line(
                    [(left.image.value.x, left.image.value.y), (right.image.value.x, right.image.value.y)],
                    fill=_colour_for_inference(left.image.inference_type),
                    width=3,
                )
        for joint in player.joints.values():
            if joint.image.value:
                point = joint.image.value
                draw.ellipse((point.x - 3, point.y - 3, point.x + 3, point.y + 3), fill=_colour_for_inference(joint.image.inference_type))

    y = 6
    for event in events:
        draw.text((6, y), _event_label(event), fill=(255, 180, 60))
        y += 16
    draw.text((6, canvas.height - 18), f"t={frame.timestamp:.3f}s frame={frame.frame_id} {' '.join(frame.quality_flags)}", fill=(245, 245, 245))
    return canvas


def render_frame_sequence(
    packets: Sequence[object],
    frames: Sequence[FrameState],
    output_dir: str | Path,
    *,
    segments: Sequence[CameraSegment] = (),
    events: Sequence[Event] = (),
) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    trail: list[Point2D] = []
    by_id = {event.frame_id: [] for event in events}
    for event in events:
        by_id.setdefault(event.frame_id, []).append(event)
    segment_by_id = {segment.segment_id: segment for segment in segments}
    for packet, frame in zip(packets, frames):
        if frame.ball and frame.ball.image.value:
            trail.append(frame.ball.image.value)
            trail = trail[-32:]
        rendered = render_frame(
            packet.image,
            frame,
            segment=segment_by_id.get(frame.segment_id),
            trail=trail,
            events=by_id.get(frame.frame_id, []),
        )
        path = destination / f"frame-{frame.frame_id:06d}.png"
        rendered.save(path)
        paths.append(path)
    return paths


def write_render_manifest(output_dir: str | Path, paths: Sequence[Path], *, video_backend: str = "frames_only") -> Path:
    import json

    destination = Path(output_dir)
    manifest = destination / "render_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "backend": video_backend,
                "frame_count": len(paths),
                "frames": [path.name for path in paths],
                "note": "A video encoder is optional; PNG frames remain the canonical debugging artifact.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def encode_mp4(
    frame_paths: Sequence[Path],
    output_path: str | Path,
    *,
    fps: float = 25.0,
) -> Path | None:
    """Encode rendered PNGs when an optional local video backend is present."""

    if not frame_paths:
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore
        from PIL import Image

        first = Image.open(frame_paths[0]).convert("RGB")
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, float(fps)),
            (first.width, first.height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open an MP4 writer")
        try:
            for path in frame_paths:
                array = np.asarray(Image.open(path).convert("RGB"))
                writer.write(array[:, :, ::-1])
        finally:
            writer.release()
        return destination
    except (ImportError, RuntimeError):
        pass

    try:
        import av  # type: ignore
        from PIL import Image

        container = av.open(str(destination), mode="w")
        stream = container.add_stream("mpeg4", rate=max(1, int(round(fps))))
        first = Image.open(frame_paths[0]).convert("RGB")
        stream.width = first.width
        stream.height = first.height
        stream.pix_fmt = "yuv420p"
        try:
            for path in frame_paths:
                array = np.asarray(Image.open(path).convert("RGB"))
                video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        return destination
    except (ImportError, RuntimeError, OSError):
        return None
