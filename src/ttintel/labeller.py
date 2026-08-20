"""Local browser labeller for calibration corners and sparse ball ground truth.

The labeller is deliberately a thin application boundary.  It indexes the
video's existing ``FramePacket`` stream, stores human labels in the repository
ground-truth directory, and calls the same calibration and tracking seams used
by the pipeline.  The web framework is imported only when the application is
created so importing ``ttintel`` remains useful in the minimal core environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping, Sequence
import webbrowser

import numpy as np

from .calibration import (
    calibrate_manual,
    calibration_quality,
    detect_table_corners_heuristic,
    load_manual_corners,
    parse_manual_corners,
    save_manual_corners,
)
from .media import FramePacket, _video_id, iter_video_frames
from .schemas import InferenceType, Point2D


LABEL_FORMAT_VERSION = 2
DEFAULT_LABEL_SUFFIX = ".labels.json"
DEFAULT_CORNER_SUFFIX = ".corners.json"


def _repository_labels_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "labels"


def _stable_video_id(video: Path) -> str:
    try:
        return _video_id(video)
    except OSError:
        # FrameStore can be constructed from synthetic packets in tests before
        # a real media file exists. Real videos always use media._video_id.
        return f"{video.stem}-unmaterialized"


def _default_storage_path(video_id: str, suffix: str) -> Path:
    return _repository_labels_dir() / f"{video_id}{suffix}"


@dataclass(frozen=True)
class BallLabel:
    """One intentional human decision for one exact video frame."""

    kind: str
    point: Point2D | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"point", "absent"}:
            raise ValueError("ball label kind must be 'point' or 'absent'")
        if self.kind == "point" and self.point is None:
            raise ValueError("a point label needs x and y")
        if self.kind == "absent" and self.point is not None:
            raise ValueError("an absent label cannot contain a point")

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "absent":
            return {"label": "absent"}
        assert self.point is not None
        return {"label": "point", "x": self.point.x, "y": self.point.y}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BallLabel":
        kind = value.get("label", value.get("kind"))
        if kind == "absent":
            return cls("absent")
        if kind == "point":
            try:
                return cls("point", Point2D(float(value["x"]), float(value["y"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("point label requires numeric x and y") from exc
        raise ValueError("ball label must be 'point' or 'absent'")


@dataclass
class LabelSet:
    """Versioned sparse labels; omission means untouched, never no-ball."""

    video: str
    labels: dict[int, BallLabel] = field(default_factory=dict)
    version: int = LABEL_FORMAT_VERSION
    video_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        source_filename = Path(self.video).name
        return {
            "version": LABEL_FORMAT_VERSION,
            "video_id": self.video_id,
            "source_filename": source_filename,
            # Keep the old human-readable field, but never write a machine-
            # specific absolute path into committed ground truth.
            "video": source_filename,
            "frames": {str(frame_id): label.to_dict() for frame_id, label in sorted(self.labels.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LabelSet":
        version = int(payload.get("version", 0))
        if version not in {1, LABEL_FORMAT_VERSION}:
            raise ValueError(f"unsupported label format version: {version}")
        raw_frames = payload.get("frames", {})
        if not isinstance(raw_frames, Mapping):
            raise ValueError("label frames must be an object keyed by frame id")
        labels: dict[int, BallLabel] = {}
        for raw_id, raw_label in raw_frames.items():
            try:
                frame_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid frame id in labels: {raw_id!r}") from exc
            if not isinstance(raw_label, Mapping):
                raise ValueError(f"label for frame {frame_id} must be an object")
            labels[frame_id] = BallLabel.from_dict(raw_label)
        source = payload.get("source_filename", payload.get("video", ""))
        return cls(
            video=Path(str(source)).name,
            labels=labels,
            version=LABEL_FORMAT_VERSION,
            video_id=str(payload.get("video_id", "")),
        )


def load_labels(path: str | Path) -> LabelSet:
    """Load sparse labels, keeping an explicit absent decision distinct from omission."""

    source = Path(path)
    return LabelSet.from_dict(json.loads(source.read_text(encoding="utf-8")))


def save_labels(path: str | Path, label_set: LabelSet) -> None:
    """Write labels atomically so a browser refresh cannot leave a half-file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(label_set.to_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp: float
    width: int
    height: int
    jpeg: bytes


def _encode_jpeg(image: Any) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional app dependency
        raise RuntimeError("labeller needs Pillow; install the 'label' extra") from exc
    array = np.asarray(image)
    output = BytesIO()
    Image.fromarray(np.ascontiguousarray(array[..., :3])).save(
        output, format="JPEG", quality=92, optimize=True
    )
    return output.getvalue()


def _decode_jpeg(jpeg: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional app dependency
        raise RuntimeError("labeller needs Pillow; install the 'label' extra") from exc
    with Image.open(BytesIO(jpeg)) as image:
        return np.asarray(image.convert("RGB"))


class FrameStore:
    """Indexed JPEG frames make random browser navigation independent of codec seeking."""

    def __init__(self, video: str | Path, packets: Iterable[FramePacket] | None = None) -> None:
        self.video = Path(video).expanduser().resolve()
        source = packets if packets is not None else iter_video_frames(self.video)
        self.records = tuple(
            FrameRecord(
                frame_id=packet.frame_id,
                timestamp=float(packet.timestamp),
                width=int(np.asarray(packet.image).shape[1]),
                height=int(np.asarray(packet.image).shape[0]),
                jpeg=_encode_jpeg(packet.image),
            )
            for packet in source
        )
        if not self.records:
            raise ValueError(f"video contains no frames: {self.video}")
        self._by_id = {record.frame_id: record for record in self.records}

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(record.frame_id for record in self.records)

    @property
    def width(self) -> int:
        return self.records[0].width

    @property
    def height(self) -> int:
        return self.records[0].height

    def get(self, frame_id: int) -> FrameRecord:
        try:
            return self._by_id[int(frame_id)]
        except KeyError as exc:
            raise KeyError(f"unknown frame id: {frame_id}") from exc

    def image(self, frame_id: int) -> np.ndarray:
        return _decode_jpeg(self.get(frame_id).jpeg)


def _point_dict(point: Point2D) -> dict[str, float]:
    return {"x": float(point.x), "y": float(point.y)}


def _corners_dict(corners: Sequence[Point2D] | None) -> list[dict[str, float]] | None:
    return [_point_dict(point) for point in corners] if corners is not None else None


def _estimate_payload(estimate: Any) -> dict[str, Any]:
    point = getattr(estimate, "value", None)
    return {
        "point": _point_dict(point) if point is not None else None,
        "confidence": float(getattr(estimate, "confidence", 0.0)),
        "source": str(getattr(estimate, "source", "unknown")),
        "visibility": str(getattr(estimate, "visibility", "unknown")),
        "inference_type": str(getattr(estimate, "inference_type", InferenceType.UNKNOWN)),
        "attempted": bool(getattr(estimate, "attempted", False)),
    }


class LabellerState:
    """Server-side state shared by the JSON handlers and the canvas client."""

    def __init__(
        self,
        store: FrameStore,
        *,
        labels_path: str | Path | None = None,
        corners_path: str | Path | None = None,
    ) -> None:
        self.store = store
        self.video_id = _stable_video_id(store.video)
        self.labels_path = Path(labels_path) if labels_path is not None else _default_storage_path(self.video_id, DEFAULT_LABEL_SUFFIX)
        self.corners_path = Path(corners_path) if corners_path is not None else _default_storage_path(self.video_id, DEFAULT_CORNER_SUFFIX)
        if self.labels_path.is_file():
            loaded = load_labels(self.labels_path)
            if loaded.video_id and loaded.video_id != self.video_id:
                self.labels = {}
            else:
                self.labels = {
                    frame_id: label
                    for frame_id, label in loaded.labels.items()
                    if frame_id in set(store.frame_ids)
                }
        else:
            self.labels = {}
        self._label_history: list[tuple[int, BallLabel | None]] = []
        self._tracker: dict[int, Any] = {}
        self._tracker_status = "off"
        self._tracker_message = "Tracker overlay is off until you request a sequential pass."
        self._lock = threading.RLock()

    @property
    def manual_corners(self) -> tuple[Point2D, ...] | None:
        if not self.corners_path.is_file():
            return None
        try:
            return load_manual_corners(self.corners_path)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def session_payload(self) -> dict[str, Any]:
        with self._lock:
            point_count = sum(label.kind == "point" for label in self.labels.values())
            absent_count = sum(label.kind == "absent" for label in self.labels.values())
            return {
                "video": self.store.video.name,
                "video_id": self.video_id,
                "frame_ids": list(self.store.frame_ids),
                "width": self.store.width,
                "height": self.store.height,
                "labels": {str(frame_id): label.to_dict() for frame_id, label in self.labels.items()},
                "counts": {
                    "labelled": len(self.labels),
                    "point": point_count,
                    "absent": absent_count,
                    "untouched": len(self.store.records) - len(self.labels),
                },
                "labels_path": str(self.labels_path),
                "corners_path": str(self.corners_path),
                "manual_corners": _corners_dict(self.manual_corners),
                "tracker": {
                    "status": self._tracker_status,
                    "message": self._tracker_message,
                    "count": len(self._tracker),
                },
            }

    def record_label(self, frame_id: int, label: BallLabel) -> None:
        with self._lock:
            record = self.store.get(frame_id)
            if label.kind == "point":
                assert label.point is not None
                if not (math.isfinite(label.point.x) and math.isfinite(label.point.y)):
                    raise ValueError("point coordinates must be finite")
                if not (0.0 <= label.point.x <= record.width and 0.0 <= label.point.y <= record.height):
                    raise ValueError("point must be inside the frame")
            previous = self.labels.get(record.frame_id)
            self.labels[record.frame_id] = label
            save_labels(
                self.labels_path,
                LabelSet(self.store.video.name, dict(self.labels), video_id=self.video_id),
            )
            self._label_history.append((record.frame_id, previous))

    def clear_label(self, frame_id: int) -> bool:
        """Clear one frame's decision, returning it to the untouched state."""

        with self._lock:
            record = self.store.get(frame_id)
            previous = self.labels.get(record.frame_id)
            if previous is None:
                return False
            del self.labels[record.frame_id]
            save_labels(
                self.labels_path,
                LabelSet(self.store.video.name, dict(self.labels), video_id=self.video_id),
            )
            self._label_history.append((record.frame_id, previous))
            return True

    def undo_last_label(self) -> int | None:
        """Undo the most recent label mutation and persist the restored state."""

        with self._lock:
            if not self._label_history:
                return None
            frame_id, previous = self._label_history.pop()
            current = self.labels.get(frame_id)
            if previous is None:
                self.labels.pop(frame_id, None)
            else:
                self.labels[frame_id] = previous
            try:
                save_labels(
                    self.labels_path,
                    LabelSet(self.store.video.name, dict(self.labels), video_id=self.video_id),
                )
            except Exception:
                # Keep the in-memory state and history aligned if persistence fails.
                if current is None:
                    self.labels.pop(frame_id, None)
                else:
                    self.labels[frame_id] = current
                self._label_history.append((frame_id, previous))
                raise
            return frame_id

    def save_corners(self, corners: Sequence[Point2D]) -> None:
        # This call is intentionally the producer for the CLI's existing
        # --manual-corners consumer; duplicating its JSON format here would
        # make the two entry points drift.
        ordered = parse_manual_corners(corners)
        save_manual_corners(self.corners_path, ordered)
        # The calibration parser ignores these descriptive fields, so the
        # committed corner file can identify its source without changing the
        # four named points consumed by calibration.
        payload = json.loads(self.corners_path.read_text(encoding="utf-8"))
        payload["video_id"] = self.video_id
        payload["source_filename"] = self.store.video.name
        self.corners_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def frame_payload(self, frame_id: int, *, include_calibration: bool = False) -> dict[str, Any]:
        record = self.store.get(frame_id)
        label = self.labels.get(record.frame_id)
        auto = None
        auto_sensitivity = None
        if include_calibration:
            detected = detect_table_corners_heuristic(self.store.image(record.frame_id))
            auto = _corners_dict(detected)
            if detected is not None:
                try:
                    auto_sensitivity = float(
                        calibration_quality(calibrate_manual(detected))["corner_sensitivity_m_per_px"]
                    )
                except (ValueError, TypeError, np.linalg.LinAlgError):
                    auto_sensitivity = None
        tracker = self._tracker.get(record.frame_id)
        return {
            "frame_id": record.frame_id,
            "timestamp": record.timestamp,
            "width": record.width,
            "height": record.height,
            "label": label.to_dict() if label else None,
            "auto_corners": auto,
            "auto_corner_sensitivity_m_per_px": auto_sensitivity,
            "manual_corners": _corners_dict(self.manual_corners),
            "tracker": _estimate_payload(tracker.image) if tracker is not None else None,
            "tracker_status": self._tracker_status,
            "tracker_message": self._tracker_message,
        }

    def compute_tracker_overlay(self) -> dict[str, Any]:
        with self._lock:
            if self._tracker_status == "ready":
                return {"status": "ready", "count": len(self._tracker)}
            self._tracker_status = "running"
            self._tracker_message = "Running the five-frame tracker sequentially over the clip…"
        try:
            from .adapters.totnet import TOTNetBallTracker, TotnetUnavailable
            from .tracking import track_offline, trajectory_ball_states

            packets = list(iter_video_frames(self.store.video))
            calibration = None
            corners = self.manual_corners
            if corners is not None:
                calibration = calibrate_manual(corners)
            tracker = TOTNetBallTracker()
            trajectory = track_offline(packets, tracker, calibration=calibration)
            states = trajectory_ball_states(trajectory)
            with self._lock:
                self._tracker = {packet.frame_id: state for packet, state in zip(packets, states)}
                self._tracker_status = "ready"
                self._tracker_message = "Sequential tracker pass cached."
                return {"status": "ready", "count": len(self._tracker)}
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            message = str(exc)
            with self._lock:
                self._tracker_status = "unavailable"
                self._tracker_message = message
            return {"status": "unavailable", "message": message, "count": 0}


def create_app(
    video: str | Path,
    *,
    labels_path: str | Path | None = None,
    corners_path: str | Path | None = None,
    frame_store: FrameStore | None = None,
) -> Any:
    """Create the Flask app without importing Flask during core package import."""

    try:
        from flask import Flask, jsonify, render_template, request, send_file
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError("labeller needs Flask; install the 'label' extra") from exc

    store = frame_store or FrameStore(video)
    state = LabellerState(store, labels_path=labels_path, corners_path=corners_path)
    ui_root = Path(__file__).with_name("labeller_ui")
    app = Flask(
        __name__,
        static_folder=str(ui_root),
        static_url_path="/static",
        template_folder=str(ui_root),
    )
    app.config["labeller_state"] = state

    @app.get("/")
    def index() -> Any:
        return render_template("labeller.html")

    @app.get("/api/session")
    def session() -> Any:
        return jsonify(state.session_payload())

    @app.get("/api/frame/<int:frame_id>")
    def frame(frame_id: int) -> Any:
        include_calibration = request.args.get("calibration", "0").lower() in {"1", "true", "yes"}
        try:
            return jsonify(state.frame_payload(frame_id, include_calibration=include_calibration))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.get("/api/frame/<int:frame_id>/image")
    def frame_image(frame_id: int) -> Any:
        try:
            record = state.store.get(frame_id)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        return send_file(BytesIO(record.jpeg), mimetype="image/jpeg", max_age=0)

    @app.post("/api/labels")
    def labels() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            frame_id = int(payload["frame_id"])
            kind = str(payload["label"])
            if kind == "point":
                label = BallLabel(kind, Point2D(float(payload["x"]), float(payload["y"])))
            else:
                label = BallLabel(kind)
            state.record_label(frame_id, label)
            return jsonify({"ok": True, "frame": state.frame_payload(frame_id)})
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/labels/clear")
    def clear_label() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            frame_id = int(payload["frame_id"])
            cleared = state.clear_label(frame_id)
            return jsonify({"ok": True, "cleared": cleared, "frame": state.frame_payload(frame_id)})
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/labels/undo")
    def undo_label() -> Any:
        try:
            frame_id = state.undo_last_label()
            if frame_id is None:
                return jsonify({"error": "no labelling action to undo"}), 409
            return jsonify({"ok": True, "frame_id": frame_id, "frame": state.frame_payload(frame_id)})
        except (OSError, RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/corners")
    def corners() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            raw = payload["corners"]
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError("corners must be a four-item list")
            points = tuple(Point2D(float(item["x"]), float(item["y"])) for item in raw)
            state.save_corners(points)
            return jsonify({"ok": True, "corners": _corners_dict(state.manual_corners)})
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/tracker-overlay")
    def tracker_overlay() -> Any:
        result = state.compute_tracker_overlay()
        return jsonify(result), (200 if result["status"] == "ready" else 503)

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m ttintel.labeller", description="Label table corners and ball positions in a local browser.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--labels", type=Path, help="label sidecar (default: data/labels/<video_id>.labels.json)")
    parser.add_argument("--corners", type=Path, help="manual corners file (default: data/labels/<video_id>.corners.json)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; localhost is the safe default")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args(argv)
    app = create_app(args.video, labels_path=args.labels, corners_path=args.corners)
    url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{url_host}:{args.port}/"
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"WARNING: --host {args.host} exposes the labeller beyond loopback.", flush=True)
    print(f"ttintel labeller listening at {url}", flush=True)
    if not args.no_open_browser:
        webbrowser.open(url)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
