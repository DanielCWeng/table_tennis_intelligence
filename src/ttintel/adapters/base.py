"""Stable contracts for third-party research models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..media import FramePacket
from ..schemas import BallState, PoseObservation, RacketState


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    role: str
    version_or_commit: str | None = None
    weights: str | None = None
    environment: str | None = None
    license_status: str = "not_verified"


class PoseEstimator(Protocol):
    info: AdapterInfo

    def estimate(self, packet: FramePacket) -> Sequence[PoseObservation]: ...


class BallTracker(Protocol):
    info: AdapterInfo

    def estimate(self, packet: FramePacket) -> BallState | None: ...


class RacketEstimator(Protocol):
    info: AdapterInfo

    def estimate(self, packet: FramePacket) -> dict[str, RacketState]: ...


class PerceptionProvider(Protocol):
    info: AdapterInfo

    def infer(self, packet: FramePacket) -> Any: ...
