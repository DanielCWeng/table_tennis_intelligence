"""Strongly typed, serialisable internal data model.

All model-derived values are wrapped in :class:`Estimate`.  This is the
important boundary between an observation and a value inferred, interpolated,
or supplied by a physics/model repair step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
from typing import Any, Generic, Mapping, TypeVar


SCHEMA_VERSION = "0.1"


class StrEnum(str, Enum):
    """A string enum that serialises naturally in JSON."""

    def __str__(self) -> str:
        return self.value


class InferenceType(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    TEMPORALLY_TRACKED = "temporally_tracked"
    MODEL_INFERRED = "model_inferred"
    PHYSICS_INFERRED = "physics_inferred"
    INTERPOLATED = "interpolated"
    OUT_OF_FRAME_INFERRED = "out_of_frame_inferred"
    UNKNOWN = "unknown"


class Visibility(StrEnum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"
    UNKNOWN = "unknown"


class SegmentType(StrEnum):
    VALID_GAMEPLAY = "valid_gameplay"
    REPLAY = "replay"
    PLAYER_CLOSEUP = "player_closeup"
    AUDIENCE = "audience"
    SCOREBOARD = "scoreboard"
    WIDE_ARENA = "wide_arena"
    INVALID_TABLE_VIEW = "invalid_table_view"
    MOVING_CAMERA = "moving_camera"
    UNKNOWN = "unknown"


class EventType(StrEnum):
    PLAYER_CONTACT = "player_contact"
    TABLE_BOUNCE = "table_bounce"
    NET_CONTACT = "net_contact"
    SERVE_CONTACT = "serve_contact"
    RALLY_START = "rally_start"
    RALLY_END = "rally_end"
    POINT_END = "point_end"


class CalibrationSource(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    TT3D = "tt3d"
    UNKNOWN = "unknown"


class QualityFlag(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    MODEL_UNAVAILABLE = "model_unavailable"
    OCCLUDED = "occluded"
    INTERPOLATED = "interpolated"
    OUT_OF_FRAME = "out_of_frame"
    CALIBRATION_FAILED = "calibration_failed"
    CALIBRATION_MANUAL = "calibration_manual"
    TIMESTAMP_ESTIMATED = "timestamp_estimated"
    HEURISTIC = "heuristic"
    DISAGREEMENT = "model_disagreement"
    PHYSICS_WARNING = "physics_warning"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space box using left, top, right, bottom coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def centre(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


T = TypeVar("T")


@dataclass
class Estimate(Generic[T]):
    """A value plus the evidence needed to interpret it safely."""

    value: T | None
    confidence: float
    source: str
    visibility: Visibility = Visibility.UNKNOWN
    inference_type: InferenceType = InferenceType.UNKNOWN
    quality_flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = float(self.confidence)
        self.quality_flags = [str(flag) for flag in self.quality_flags]

    @classmethod
    def unknown(cls, source: str = "unavailable", *flags: str) -> "Estimate[Any]":
        return cls(
            value=None,
            confidence=0.0,
            source=source,
            visibility=Visibility.UNKNOWN,
            inference_type=InferenceType.UNKNOWN,
            quality_flags=list(flags),
        )

    @classmethod
    def observed(
        cls,
        value: T,
        confidence: float,
        source: str,
        visibility: Visibility = Visibility.VISIBLE,
        *flags: str,
    ) -> "Estimate[T]":
        return cls(
            value=value,
            confidence=confidence,
            source=source,
            visibility=visibility,
            inference_type=InferenceType.OBSERVED,
            quality_flags=list(flags),
        )


@dataclass
class JointState:
    joint_name: str
    image: Estimate[Point2D]
    world: Estimate[Point3D] | None = None


@dataclass
class PoseObservation:
    observation_id: str
    bbox: Estimate[BoundingBox]
    joints: dict[str, JointState] = field(default_factory=dict)
    detector_track_id: str | None = None
    source_model: str = "unknown"


@dataclass
class PlayerState:
    player_id: str
    bbox: Estimate[BoundingBox] | None = None
    joints: dict[str, JointState] = field(default_factory=dict)
    table_side: str | None = None
    source_model: str = "unknown"
    quality_flags: list[str] = field(default_factory=list)

    def joint(self, name: str) -> JointState | None:
        return self.joints.get(name)

    def centre_image(self) -> Point2D | None:
        if self.bbox and self.bbox.value:
            return self.bbox.value.centre
        pelvis = self.joint("pelvis") or self.joint("mid_hip")
        return pelvis.image.value if pelvis and pelvis.image.value else None


@dataclass
class BallState:
    image: Estimate[Point2D]
    world: Estimate[Point3D] | None = None
    table_xy: Estimate[Point2D] | None = None
    velocity: Estimate[Point3D] | None = None
    blur_length: Estimate[float] | None = None
    blur_direction: Estimate[float] | None = None


@dataclass
class RacketState:
    bbox: Estimate[BoundingBox] | None = None
    keypoints: dict[str, Estimate[Point2D]] = field(default_factory=dict)
    centre: Estimate[Point2D] | None = None
    orientation_proxy: Estimate[float] | None = None
    velocity_proxy: Estimate[float] | None = None


@dataclass
class CameraState:
    intrinsics: Estimate[dict[str, float]] | None = None
    rotation: Estimate[list[list[float]]] | None = None
    translation: Estimate[list[float]] | None = None
    approximately_static: Estimate[bool] | None = None


@dataclass
class TableCalibration:
    image_corners: tuple[Point2D, Point2D, Point2D, Point2D]
    world_corners: tuple[Point3D, Point3D, Point3D, Point3D]
    homography: tuple[tuple[float, float, float], ...]
    reprojection_error_px: float
    corner_confidences: tuple[float, float, float, float]
    source: CalibrationSource
    confidence: float
    focal_length: float | None = None
    camera_rotation: list[list[float]] | None = None
    camera_translation: list[float] | None = None
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class VideoMetadata:
    video_id: str
    source_path: str
    width: int | None
    height: int | None
    duration: float | None
    nominal_fps: float | None
    is_vfr: bool | None
    codec: str | None
    creation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraSegment:
    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    segment_type: SegmentType = SegmentType.UNKNOWN
    gameplay_quality_score: float = 0.0
    camera: CameraState | None = None
    table: TableCalibration | None = None
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class FrameState:
    frame_id: int
    timestamp: float
    segment_id: str
    camera_state: CameraState | None = None
    players: dict[str, PlayerState] = field(default_factory=dict)
    ball: BallState | None = None
    rackets: dict[str, RacketState] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class Event:
    event_id: str
    timestamp: float
    frame_id: int
    event_type: EventType
    actor_player_id: str | None = None
    ball_before: BallState | None = None
    ball_after: BallState | None = None
    player_state: PlayerState | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class MorphologyProfile:
    player_id: str
    estimated_height: Estimate[float] | None = None
    arm_span: Estimate[float] | None = None
    shoulder_width: Estimate[float] | None = None
    torso_length: Estimate[float] | None = None
    segment_lengths: dict[str, Estimate[float]] = field(default_factory=dict)
    dominant_hand: str | None = None
    shape_model: str | None = None
    shape_parameters: list[float] = field(default_factory=list)
    scale_method: str | None = None
    frames_used: list[int] = field(default_factory=list)
    variance_across_frames: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    video: VideoMetadata
    segments: list[CameraSegment] = field(default_factory=list)
    frames: list[FrameState] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    player_profiles: list[MorphologyProfile] = field(default_factory=list)
    derived: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def to_dict(value: Any) -> Any:
    """Convert a schema object (or nested value) into JSON-compatible data."""

    return _jsonable(value)


def to_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(to_dict(value), indent=indent, sort_keys=False)
