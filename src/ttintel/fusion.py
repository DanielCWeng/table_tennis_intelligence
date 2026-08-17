"""Frame-level fusion and two-player identity assignment."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from .media import FramePacket
from .perception import FrameDetections
from .schemas import CameraSegment, FrameState, PlayerState, PoseObservation


@dataclass
class _TrackMemory:
    player_id: str
    centre_x: float
    centre_y: float
    detector_track_id: str | None


def _centre(observation: PoseObservation) -> tuple[float, float]:
    if observation.bbox.value is None:
        return (0.0, 0.0)
    point = observation.bbox.value.centre
    return point.x, point.y


class TwoPlayerIdentityAssigner:
    """Small, explainable tracker for the two competitors in a rally."""

    def __init__(self) -> None:
        self._tracks: dict[str, _TrackMemory] = {}

    def assign(self, observations: Iterable[PoseObservation]) -> dict[str, PlayerState]:
        items = list(observations)
        if not items:
            return {}
        assignments: dict[str, PoseObservation] = {}
        remaining = set(range(len(items)))

        # Prefer detector-provided IDs when they remain unique.
        for player_id, memory in self._tracks.items():
            matches = [
                index for index, observation in enumerate(items)
                if index in remaining and observation.detector_track_id is not None
                and observation.detector_track_id == memory.detector_track_id
            ]
            if matches:
                index = matches[0]
                assignments[player_id] = items[index]
                remaining.remove(index)

        # With no prior mapping, left-to-right assignment gives reproducible
        # IDs.  With prior tracks, nearest-centre matching preserves identity.
        if not self._tracks:
            ordered = sorted(remaining, key=lambda index: _centre(items[index]))
            for slot, index in enumerate(ordered[:2]):
                assignments[f"player_{slot}"] = items[index]
                remaining.remove(index)
        else:
            for player_id, memory in sorted(self._tracks.items()):
                if player_id in assignments or not remaining:
                    continue
                index = min(
                    remaining,
                    key=lambda candidate: hypot(
                        _centre(items[candidate])[0] - memory.centre_x,
                        _centre(items[candidate])[1] - memory.centre_y,
                    ),
                )
                assignments[player_id] = items[index]
                remaining.remove(index)
            next_slot = 0
            while remaining and len(assignments) < 2:
                player_id = f"player_{next_slot}"
                next_slot += 1
                if player_id in assignments:
                    continue
                index = remaining.pop()
                assignments[player_id] = items[index]

        result: dict[str, PlayerState] = {}
        new_tracks: dict[str, _TrackMemory] = {}
        for player_id, observation in sorted(assignments.items()):
            x, y = _centre(observation)
            result[player_id] = PlayerState(
                player_id=player_id,
                bbox=observation.bbox,
                joints=observation.joints,
                table_side="left" if player_id == "player_0" else "right",
                source_model=observation.source_model,
            )
            new_tracks[player_id] = _TrackMemory(
                player_id=player_id,
                centre_x=x,
                centre_y=y,
                detector_track_id=observation.detector_track_id,
            )
        self._tracks = new_tracks
        return result


def fuse_frame(
    packet: FramePacket,
    segment: CameraSegment,
    detections: FrameDetections,
    assigner: TwoPlayerIdentityAssigner,
) -> FrameState:
    quality_flags = list(detections.diagnostics)
    if packet.timestamp_estimated:
        quality_flags.append("timestamp_estimated")
    players = assigner.assign(detections.poses)
    if len(players) < 2:
        quality_flags.append("fewer_than_two_player_tracks")
    return FrameState(
        frame_id=packet.frame_id,
        timestamp=packet.timestamp,
        segment_id=segment.segment_id,
        camera_state=segment.camera,
        players=players,
        ball=detections.ball,
        rackets=detections.rackets,
        quality_flags=quality_flags,
    )
