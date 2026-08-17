"""Session-directory persistence with raw/cleaned/fused/derived separation."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterable

from .schemas import Session, to_dict


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

    def write_session(self, session: Session, *, session_path: Path | None = None) -> Path:
        destination = session_path or self.create(session.session_id)
        self._write_json(destination / "session.json", session)
        self._write_json(destination / "video" / "metadata.json", session.video)
        self._write_jsonl(destination / "segments" / "segments.jsonl", session.segments)
        self._write_jsonl(destination / "fused" / "frames.jsonl", session.frames)
        self._write_jsonl(destination / "events" / "events.jsonl", session.events)
        self._write_json(destination / "derived" / "analytics.json", session.derived)
        if session.player_profiles:
            self._write_jsonl(destination / "morphology" / "profiles.jsonl", session.player_profiles)
        return destination

    @staticmethod
    def read_json(path: str | Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))
