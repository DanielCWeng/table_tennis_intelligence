"""Runnable end-to-end MVP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import uuid

from .analytics import compute_analytics
from .calibration import CalibrationError, calibrate_automatic
from .events import add_rally_boundaries, infer_events
from .fusion import TwoPlayerIdentityAssigner, fuse_frame
from .media import FramePacket, probe_video, read_frame_packets
from .perception import AnnotationProvider, DefaultPerceptionProvider, FrameDetections
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
            segment.table = calibrate_automatic(candidates[len(candidates) // 2].image)
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
    render: bool = True,
) -> AnalysisResult:
    if not packets:
        raise ValueError("at least one frame packet is required")
    metadata = video or metadata_from_packets(packets)
    cuts = detect_cuts(packets)
    segments = build_segments(metadata, packets, cuts)
    provider = DefaultPerceptionProvider(
        AnnotationProvider(annotations) if annotations else None,
        pose_estimator=pose_estimator,
        racket_estimator=racket_estimator,
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
    render: bool = True,
) -> AnalysisResult:
    metadata = probe_video(source)
    packets = read_frame_packets(source, max_frames=max_frames)
    return analyse_packets(
        packets,
        video=metadata,
        output_root=output_root,
        annotations=annotations,
        manual_calibration=manual_calibration,
        pose_estimator=pose_estimator,
        racket_estimator=racket_estimator,
        render=render,
    )
