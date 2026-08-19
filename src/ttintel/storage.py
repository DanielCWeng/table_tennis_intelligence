"""Session-directory persistence with explicit provenance layers.

The manifest is deliberately small. Frame and event records live only in
their JSONL sidecars; ``session.json`` does not repeat those records.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterable

from .schemas import (
    CameraSegment,
    Event,
    FrameState,
    MorphologyProfile,
    Session,
    VideoMetadata,
    from_dict,
    to_dict,
)


SESSION_SUBDIRECTORIES = (
    "video",
    "segments",
    "calibration",
    "poses2d",
    "poses3d",
    "morphology",
    "ball",
    "racket",
    "events",
    "derived",
    "renders",
    "raw",
    "cleaned",
    "fused",
)

STORAGE_FORMAT = "ttintel-session"
STORAGE_VERSION = "0.2"
RAW_RECORDS_PATH = "raw/adapter_outputs.jsonl"
CLEANED_RECORDS_PATH = "cleaned/frames.jsonl"


class SessionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def create(self, session_id: str) -> Path:
        session_path = self.root / session_id
        for name in SESSION_SUBDIRECTORIES:
            (session_path / name).mkdir(parents=True, exist_ok=True)
        return session_path

    @staticmethod
    def _write_json(path: Path, value: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_dict(value), indent=2) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _write_jsonl(path: Path, values: Iterable[Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(to_dict(value), separators=(",", ":")) + "\n")
        return path

    @staticmethod
    def read_json(path: str | Path) -> Any:
        """Read one JSON document without imposing a schema on it."""

        return json.loads(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def read_jsonl(cls, path: str | Path, target_type: Any | None = None) -> list[Any]:
        """Read JSONL, optionally rebuilding each row as ``target_type``."""

        rows: list[Any] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    rows.append(from_dict(payload, target_type) if target_type else payload)
                except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"invalid JSONL row {line_number} in {path}") from exc
        return rows

    @staticmethod
    def _cleaned_index(session: Session) -> list[dict[str, Any]]:
        """Describe validated values without copying the fused frame payload.

        The current pipeline hands storage only post-fusion ``FrameState``
        objects. Those are the canonical validated/normalised records, so the
        cleaned layer indexes them instead of copying every joint a second
        time. A future caller with a real pre-fusion cleaned stream can pass
        it through ``cleaned_records``.
        """

        return [
            {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "segment_id": frame.segment_id,
                "status": "validated_normalised",
                "fused_record": {
                    "path": "../fused/frames.jsonl",
                    "line": index,
                },
                "quality_flags": list(frame.quality_flags),
            }
            for index, frame in enumerate(session.frames)
        ]

    @staticmethod
    def _raw_manifest(*, record_count: int, available: bool) -> dict[str, Any]:
        return {
            "layer": "raw",
            "records": RAW_RECORDS_PATH,
            "record_count": record_count,
            "available": available,
            "immutable": True,
            "source_contracts": {
                "pose": "PoseEstimator.estimate -> Sequence[PoseObservation]",
                "ball": "BallTracker.estimate -> BallState | None",
                "racket": "RacketEstimator.estimate -> dict[str, RacketState]",
            },
            "note": (
                "Records are adapter-returned values before fusion when supplied. "
                "The current pipeline discards FrameDetections before storage, "
                "so an empty layer is an explicit capture gap, not a copy of fused data."
                if not available
                else "Records are adapter-returned values before fusion."
            ),
        }

    @staticmethod
    def _cleaned_manifest(*, record_count: int, supplied: bool) -> dict[str, Any]:
        return {
            "layer": "cleaned",
            "records": CLEANED_RECORDS_PATH,
            "record_count": record_count,
            "materialized": supplied,
            "validated": True,
            "normalised": True,
            "note": (
                "Caller-supplied validated/normalised records."
                if supplied
                else "Compact index into canonical fused FrameState records; values are not copied."
            ),
        }

    def write_session(
        self,
        session: Session,
        *,
        session_path: Path | None = None,
        raw_records: Iterable[Any] | None = None,
        cleaned_records: Iterable[Any] | None = None,
    ) -> Path:
        """Persist a session and return its directory.

        ``raw_records`` is intentionally explicit: storage cannot recreate
        adapter output after fusion has discarded it. Each item should be the
        untouched result of one or more adapter contract calls. ``cleaned`` is
        indexed by default to keep the canonical frame payload single-copy;
        callers that retain a distinct cleaned stream may provide it.
        """

        destination = (
            Path(session_path)
            if session_path is not None
            else self.create(session.session_id)
        )
        destination.mkdir(parents=True, exist_ok=True)

        raw_values = list(raw_records) if raw_records is not None else []
        if cleaned_records is None:
            cleaned_values: list[Any] = self._cleaned_index(session)
            cleaned_supplied = False
        else:
            cleaned_values = list(cleaned_records)
            cleaned_supplied = True

        self._write_json(destination / "video" / "metadata.json", session.video)
        self._write_jsonl(destination / "segments" / "segments.jsonl", session.segments)
        self._write_jsonl(destination / "fused" / "frames.jsonl", session.frames)
        self._write_jsonl(destination / "events" / "events.jsonl", session.events)
        self._write_json(destination / "derived" / "analytics.json", session.derived)
        self._write_jsonl(destination / RAW_RECORDS_PATH, raw_values)
        self._write_jsonl(destination / CLEANED_RECORDS_PATH, cleaned_values)
        self._write_json(
            destination / "raw" / "manifest.json",
            self._raw_manifest(record_count=len(raw_values), available=bool(raw_values)),
        )
        self._write_json(
            destination / "cleaned" / "manifest.json",
            self._cleaned_manifest(
                record_count=len(cleaned_values),
                supplied=cleaned_supplied,
            ),
        )

        profiles_path: str | None = None
        if session.player_profiles:
            profiles_path = "morphology/profiles.jsonl"
            self._write_jsonl(destination / profiles_path, session.player_profiles)

        # This is a manifest, not a second serialisation of the session. The
        # large frame/event collections are available only through sidecars.
        manifest = {
            "format": STORAGE_FORMAT,
            "storage_version": STORAGE_VERSION,
            "session_id": session.session_id,
            "schema_version": session.schema_version,
            "metadata": session.metadata,
            "records": {
                "video": "video/metadata.json",
                "segments": "segments/segments.jsonl",
                "frames": "fused/frames.jsonl",
                "events": "events/events.jsonl",
                "derived": "derived/analytics.json",
                "profiles": profiles_path,
                "raw": "raw/manifest.json",
                "cleaned": "cleaned/manifest.json",
            },
        }
        self._write_json(destination / "session.json", manifest)
        return destination

    @staticmethod
    def _session_directory(path: str | Path) -> Path:
        candidate = Path(path)
        return (
            candidate.parent
            if candidate.is_file() or candidate.name == "session.json"
            else candidate
        )

    @staticmethod
    def _record_path(session_path: Path, relative_path: str) -> Path:
        root = session_path.resolve()
        candidate = (session_path / relative_path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"session record path escapes session directory: {relative_path!r}")
        return candidate

    def read_session(self, session_path: str | Path) -> Session:
        """Load a persisted session back into its typed schema objects."""

        destination = self._session_directory(session_path)
        manifest_path = destination / "session.json"
        manifest = self.read_json(manifest_path)
        if manifest.get("format") != STORAGE_FORMAT:
            raise ValueError(f"unsupported session format in {manifest_path}")
        records = manifest.get("records", {})

        def record(name: str) -> Path:
            relative_path = records.get(name)
            if not relative_path:
                raise ValueError(f"session manifest has no {name!r} record")
            return self._record_path(destination, relative_path)

        profiles = []
        if records.get("profiles"):
            profiles = self.read_jsonl(record("profiles"), MorphologyProfile)

        return Session(
            session_id=str(manifest["session_id"]),
            video=from_dict(self.read_json(record("video")), VideoMetadata),
            segments=self.read_jsonl(record("segments"), CameraSegment),
            frames=self.read_jsonl(record("frames"), FrameState),
            events=self.read_jsonl(record("events"), Event),
            player_profiles=profiles,
            derived=self.read_json(record("derived")),
            metadata=dict(manifest.get("metadata", {})),
            schema_version=str(manifest.get("schema_version", "unknown")),
        )


def read_session(path: str | Path) -> Session:
    """Convenience wrapper for loading a session without constructing a store."""

    session_path = Path(path)
    root = session_path.parent if session_path.is_file() else session_path
    return SessionStore(root).read_session(session_path)
