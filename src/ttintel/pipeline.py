"""Runnable end-to-end MVP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import uuid

from .analytics import compute_analytics
from .adapters.totnet import (
    DEFAULT_CHECKPOINT,
    TOTNetBallTracker,
    TotnetUnavailable,
)
from .calibration import CalibrationError, calibrate_consensus
from .events import add_rally_boundaries, infer_events
from .fusion import TwoPlayerIdentityAssigner, fuse_frame
from .media import FramePacket, probe_video, read_frame_packets
from .perception import (
    AnnotationProvider,
    BrightBlobBallTracker,
    DefaultPerceptionProvider,
    FrameDetections,
)
from .rendering import encode_mp4, render_frame_sequence, write_render_manifest
from .scene import GameplayEvidence, build_segments, classify_segment, detect_cuts, segment_for_timestamp
from .schemas import CameraSegment, Session, TableCalibration, VideoMetadata
from .storage import SessionStore


@dataclass
class AnalysisResult:
    session: Session
    session_path: Path | None
    render_paths: list[Path]
    render_video_path: Path | None
    warnings: list[str]


_USE_PACKET_DEFAULT = object()


def _totnet_unavailable(exc: TotnetUnavailable) -> TotnetUnavailable:
    """Add an actionable escape hatch to an adapter initialisation failure.

    The adapter owns the model-loading details and intentionally remains
    unchanged.  The pipeline owns the user-facing choice, so this boundary
    explains how to restore the missing runtime and makes clear that selecting
    ``blob`` or ``none`` is deliberate rather than an automatic downgrade.
    """

    detail = str(exc)
    lowered = detail.lower()
    if "checkpoint" in lowered and ("not found" in lowered or "missing" in lowered):
        action = f"restore the checkpoint at {DEFAULT_CHECKPOINT}"
    elif "cuda" in lowered:
        action = "install/use a CUDA-enabled PyTorch runtime"
    else:
        action = "install the TOTNet runtime and checkpoint"
    return TotnetUnavailable(
        f"TOTNet ball tracker is unavailable: {detail}. "
        f"To use the default tracker, {action}; otherwise explicitly choose "
        "--ball-tracker blob or --ball-tracker none."
    )


def _select_ball_tracker(mode: str) -> Any | None:
    """Build the explicitly selected video ball tracker without fallback."""

    if mode == "none":
        return None
    if mode == "blob":
        return BrightBlobBallTracker()
    if mode != "totnet":
        raise ValueError(f"unknown ball tracker {mode!r}; choose totnet, blob, or none")
    try:
        return TOTNetBallTracker()
    except TotnetUnavailable as exc:
        raise _totnet_unavailable(exc) from exc


def metadata_from_packets(packets: Sequence[FramePacket], source_path: str = "<packets>") -> VideoMetadata:
    if packets and getattr(packets[0].image, "shape", None) is not None:
        height, width = packets[0].image.shape[:2]
    else:
        width = height = None
    duration = packets[-1].timestamp if packets else 0.0
    return VideoMetadata(
        video_id=f"packets-{uuid.uuid4().hex[:12]}",
        source_path=source_path,
        width=int(width) if width is not None else None,
        height=int(height) if height is not None else None,
        duration=float(duration),
        nominal_fps=None,
        is_vfr=None,
        codec=None,
        creation_metadata={"backend": "frame_packets"},
    )


def _segment_packets(packets: Sequence[FramePacket], segment: CameraSegment) -> list[FramePacket]:
    return [packet for packet in packets if segment.start_time <= packet.timestamp <= segment.end_time]


def _calibrate_segments(
    segments: Sequence[CameraSegment],
    packets: Sequence[FramePacket],
    *,
    manual_calibration: TableCalibration | None = None,
) -> list[str]:
    warnings: list[str] = []
    for segment in segments:
        if manual_calibration is not None:
            segment.table = manual_calibration
            continue
        candidates = _segment_packets(packets, segment)
        if not candidates:
            segment.quality_flags.append("calibration_no_frames")
            continue
        try:
            segment.table = calibrate_consensus([packet.image for packet in candidates])
        except (CalibrationError, ValueError) as exc:
            segment.quality_flags.extend(["calibration_failed", str(exc)])
            warnings.append(f"{segment.segment_id}: automatic calibration unavailable ({exc})")
    return warnings


def _classify_segments(
    segments: Sequence[CameraSegment],
    packets: Sequence[FramePacket],
    detections: dict[int, FrameDetections],
) -> None:
    for segment in segments:
        segment_detections = [detections[packet.frame_id] for packet in _segment_packets(packets, segment)]
        max_players = max((len(item.poses) for item in segment_detections), default=0)
        ball_count = sum(1 for item in segment_detections if item.ball and item.ball.image.value is not None)
        evidence = GameplayEvidence(
            table_detected=segment.table is not None,
            calibration_succeeded=segment.table is not None,
            sufficient_table_visible=segment.table is not None,
            plausible_player_tracks=max_players,
            camera_static=True,
            ball_evidence=ball_count > 0,
            moving_camera=False,
        )
        classify_segment(segment, evidence)


def analyse_packets(
    packets: Sequence[FramePacket],
    *,
    video: VideoMetadata | None = None,
    output_root: str | Path | None = None,
    annotations: str | Path | None = None,
    manual_calibration: TableCalibration | None = None,
    pose_estimator: object | None = None,
    racket_estimator: object | None = None,
    ball_tracker: Any = _USE_PACKET_DEFAULT,
    render: bool = True,
) -> AnalysisResult:
    """Analyse already-decoded packets with an optional injected tracker.

    Packet callers retain the lightweight blob default so fixtures and unit
    tests do not load model weights.  The video entry point selects TOTNet by
    default and passes that tracker here explicitly.
    """

    if not packets:
        raise ValueError("at least one frame packet is required")
    metadata = video or metadata_from_packets(packets)
    cuts = detect_cuts(packets)
    segments = build_segments(metadata, packets, cuts)
    selected_tracker = (
        BrightBlobBallTracker()
        if ball_tracker is _USE_PACKET_DEFAULT
        else ball_tracker
    )
    provider = DefaultPerceptionProvider(
        AnnotationProvider(annotations) if annotations else None,
        ball_tracker=selected_tracker,
        pose_estimator=pose_estimator,
        racket_estimator=racket_estimator,
        use_bright_blob=False,
    )
    detections: dict[int, FrameDetections] = {}
    for packet in packets:
        detections[packet.frame_id] = provider.infer(packet)

    warnings = _calibrate_segments(segments, packets, manual_calibration=manual_calibration)
    _classify_segments(segments, packets, detections)

    frames = []
    current_segment_id: str | None = None
    assigner = TwoPlayerIdentityAssigner()
    for packet in packets:
        segment = segment_for_timestamp(segments, packet.timestamp) or segments[-1]
        if segment.segment_id != current_segment_id:
            assigner = TwoPlayerIdentityAssigner()
            current_segment_id = segment.segment_id
        frames.append(fuse_frame(packet, segment, detections[packet.frame_id], assigner))

    events = []
    for segment in segments:
        segment_frames = [frame for frame in frames if frame.segment_id == segment.segment_id]
        events.extend(infer_events(segment_frames, calibration=segment.table))
    events = add_rally_boundaries(events)
    session = Session(
        session_id=metadata.video_id,
        video=metadata,
        segments=list(segments),
        frames=frames,
        events=events,
        derived=compute_analytics(frames, events),
        metadata={
            "pipeline": "ttintel-mvp",
            "ball_tracker": (
                provider.ball_tracker.info.name if provider.ball_tracker is not None else "none"
            ),
            "cut_candidates": [
                {"frame_id": cut.frame_id, "timestamp": cut.timestamp, "difference_score": cut.difference_score}
                for cut in cuts
            ],
            "warnings": warnings,
        },
    )

    session_path: Path | None = None
    render_paths: list[Path] = []
    render_video_path: Path | None = None
    if output_root is not None:
        store = SessionStore(output_root)
        session_path = store.write_session(session)
        if render:
            render_dir = session_path / "renders" / "frames"
            render_paths = render_frame_sequence(packets, frames, render_dir, segments=segments, events=events)
            render_video_path = encode_mp4(
                render_paths,
                session_path / "renders" / "annotated.mp4",
                fps=metadata.nominal_fps or 25.0,
            )
            write_render_manifest(
                render_dir,
                render_paths,
                video_backend="mp4" if render_video_path else "frames_only",
            )
    return AnalysisResult(
        session=session,
        session_path=session_path,
        render_paths=render_paths,
        render_video_path=render_video_path,
        warnings=warnings,
    )


def analyse_video(
    source: str | Path,
    *,
    output_root: str | Path,
    max_frames: int | None = None,
    annotations: str | Path | None = None,
    manual_calibration: TableCalibration | None = None,
    pose_estimator: object | None = None,
    racket_estimator: object | None = None,
    ball_tracker: str = "totnet",
    ball_tracker_instance: Any | None = None,
    render: bool = True,
) -> AnalysisResult:
    """Analyse a video with TOTNet as the deliberate default ball tracker.

    Offline trajectory linking is not selected here: it needs all candidate
    frames and calibration before fusion, while this pipeline currently emits
    frame states during one streaming perception pass.  Keeping that separate
    avoids exposing a partially linked trajectory as if it were streaming
    output.
    """

    metadata = probe_video(source)
    packets = read_frame_packets(source, max_frames=max_frames)
    selected_tracker = (
        ball_tracker_instance
        if ball_tracker_instance is not None
        else _select_ball_tracker(ball_tracker)
    )
    return analyse_packets(
        packets,
        video=metadata,
        output_root=output_root,
        annotations=annotations,
        manual_calibration=manual_calibration,
        pose_estimator=pose_estimator,
        racket_estimator=racket_estimator,
        ball_tracker=selected_tracker,
        render=render,
    )
