"""Screen/file/synthetic sources and bounded, nonblocking live processing."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import math
import time
from typing import Iterator

import numpy as np

from ttintel.media import FramePacket, iter_video_frames
from .core import LiveMotionTracker, MotionSample, ROI, Region
from .ball import BallAssistTracker, BallObservation


@dataclass(frozen=True)
class Result:
    packet: FramePacket
    sample: MotionSample
    grabbed_at: float
    analysed_at: float
    analysis_ms: float
    ball: BallObservation | None = None
    ball_epoch: int = 0


def synthetic_frame(seconds: float, width: int = 640, height: int = 360) -> np.ndarray:
    """Textured moving subjects and static reference (clearly synthetic RGB)."""
    import cv2
    frame = np.full((height, width, 3), (20, 27, 42), np.uint8)
    # Deterministic texture supports measurable camera-reference features.
    for y in range(14, height, 24):
        for x in range(12, width, 24):
            cv2.circle(frame, (x, y), 2, (47, 58, 75), -1)
    cv2.rectangle(frame, (100, 88), (width - 100, height - 88), (47, 85, 96), -1)
    for y in range(101, height - 88, 23):
        for x in range(113, width - 100, 23):
            cv2.circle(frame, (x, y), 2, (65, 106, 116), -1)
    cv2.rectangle(frame, (100, 88), (width - 100, height - 88), (140, 198, 207), 2)
    cv2.line(frame, (width // 2, 88), (width // 2, height - 88), (200, 205, 218), 2)
    x = int(width * .30 + width * .13 * math.sin(2 * math.pi * .8 * seconds))
    other = int(width * .70 + width * .1 * math.sin(2 * math.pi * .55 * seconds))
    for cx, cy, color in ((x, height - 62, (57, 221, 203)),
                           (other, 63, (248, 177, 116))):
        cv2.rectangle(frame, (cx - 22, cy - 30), (cx + 22, cy + 30), color, -1)
        for xx in (-12, 0, 12):
            for yy in (-20, -8, 7, 19):
                cv2.circle(frame, (cx + xx, cy + yy), 3, (31, 45, 52), -1)
        cv2.line(frame, (cx - 20, cy - 12), (cx + 20, cy - 12), (30, 40, 45), 2)
    return frame

def screen_frames(region: Region, stop: Event, fps: int = 30) -> Iterator[FramePacket]:
    import mss
    # Instantiate inside capture thread; no capture history is persisted.
    with mss.MSS() as grabber:
        frame_id = 0
        interval = 1 / fps
        while not stop.is_set():
            started = time.monotonic()
            shot = grabber.grab(region.as_mss())
            bgr = np.asarray(shot)[:, :, :3]
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            yield FramePacket(frame_id, time.monotonic(), rgb)
            frame_id += 1
            stop.wait(max(0.0, interval - (time.monotonic() - started)))


def demo_frames(stop: Event, fps: int = 30) -> Iterator[FramePacket]:
    started = time.monotonic()
    frame_id = 0
    interval = 1 / fps
    while not stop.is_set():
        tick = time.monotonic()
        yield FramePacket(frame_id, tick, synthetic_frame(tick - started))
        frame_id += 1
        # Per-frame pacing does not burst old frames after a pause.
        stop.wait(max(0.0, interval - (time.monotonic() - tick)))


def file_frames(path: Path, stop: Event, pause: Event | None = None
                ) -> Iterator[FramePacket]:
    """Paced media playback; pausing moves wall-clock origin, not file PTS."""
    started = time.monotonic()
    first_timestamp: float | None = None
    for packet in iter_video_frames(path):
        if stop.is_set():
            return
        if first_timestamp is None:
            first_timestamp = packet.timestamp
        elapsed = max(0., packet.timestamp - first_timestamp)
        while not stop.is_set():
            if pause is not None and pause.is_set():
                paused_at = time.monotonic()
                while pause.is_set() and not stop.wait(.025):
                    pass
                started += time.monotonic() - paused_at
                continue
            remaining = started + elapsed - time.monotonic()
            if remaining <= 0:
                break
            if stop.wait(min(.025, remaining)):
                return
        if stop.is_set():
            return
        yield packet

def offer_latest(queue: Queue, value: object) -> int:
    """Drop a stale item rather than let analysis/render latency accumulate."""
    try:
        queue.put_nowait(value)
        return 0
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put_nowait(value)
        return 1


class LiveEngine:
    """Two workers: capture/decode and lightweight flow. Qt only polls results.

    Heavy pose/ball inference deliberately NOT enabled; it cannot stall this
    capture -> analysis -> render lane. Separate worker support is future work.
    """
    def __init__(self, source: str, region: Region | None = None,
                 file: Path | None = None, fps: int = 30, width: int = 384):
        if source not in ("screen", "file", "demo"):
            raise ValueError("source must be screen, file or demo")
        if source == "screen" and region is None:
            raise ValueError("screen capture needs a region")
        if source == "file" and (file is None or not file.is_file()):
            raise FileNotFoundError(file)
        if fps < 1 or fps > 120:
            raise ValueError("fps must be 1..120")
        self.source, self.region, self.file, self.fps = source, region, file, fps
        self.raw: Queue[tuple[FramePacket, float]] = Queue(maxsize=2)
        self.results: Queue[Result] = Queue(maxsize=2)
        self.tracker = LiveMotionTracker(width=width)
        self.ball = BallAssistTracker()
        self.ball_epoch = 0
        self.lock = Lock()
        self.stop_event = Event()
        self.pause_event = Event()
        self.threads: list[Thread] = []
        self.errors: deque[str] = deque(maxlen=4)
        self.capture_times: deque[float] = deque(maxlen=90)
        self.processing_times: deque[float] = deque(maxlen=90)
        self.dropped_capture = 0
        self.dropped_display = 0
        self.finished = False

    @property
    def running(self) -> bool:
        return bool(self.threads) and not self.stop_event.is_set() and not self.finished

    def start(self) -> None:
        if self.threads:
            self._drain(self.raw)
            self._drain(self.results)
            self.pause_event.clear()
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.finished = False
        self.threads = [
            Thread(target=self._capture, name="ttintel-live-capture", daemon=True),
            Thread(target=self._analyse, name="ttintel-live-analysis", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def pause(self) -> None:
        self.pause_event.set()
        # No queued frame may appear on screen after the user hits PAUSE.
        # The analysis worker can complete one in-flight frame; the GUI
        # ignores results until resume, when old results are discarded.
        self._drain(self.raw)
        self._drain(self.results)

    def reset(self) -> None:
        with self.lock:
            self.tracker.reset()
            self.ball.reset()
            self.ball_epoch += 1
        self.capture_times.clear()
        self.processing_times.clear()
        self._drain(self.raw)
        self._drain(self.results)
        self.dropped_capture = self.dropped_display = 0

    def set_regions(self, roi: ROI | None, reference: ROI | None) -> None:
        with self.lock:
            self.tracker.set_regions(roi, reference)

    def seed_ball(self, packet: FramePacket, x: float, y: float) -> BallObservation:
        """User annotation uses the exact preview frame, never a newer hidden frame."""
        with self.lock:
            observation = self.ball.seed(packet, (x, y))
            self.ball_epoch += 1
        self._drain(self.results)
        return observation

    def clear_ball(self) -> None:
        with self.lock:
            self.ball.reset()
            self.ball_epoch += 1
        self._drain(self.results)

    def close(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.5)
        self.threads = []

    @staticmethod
    def _drain(q: Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except Empty:
                break

    @staticmethod
    def rate(timestamps: deque[float]) -> float:
        if len(timestamps) < 2:
            return 0.
        delta = timestamps[-1] - timestamps[0]
        return (len(timestamps) - 1) / delta if delta > 0 else 0.

    def _capture(self) -> None:
        try:
            if self.source == "screen":
                source = screen_frames(self.region, self.stop_event, self.fps)
            elif self.source == "demo":
                source = demo_frames(self.stop_event, self.fps)
            else:
                source = file_frames(self.file, self.stop_event, self.pause_event)
            for packet in source:
                if self.stop_event.is_set():
                    break
                # File mode handles pause inside its paced generator to avoid
                # accelerated catch-up of old frames after resume.
                if self.source != "file" and self.pause_event.is_set():
                    # Generator produced a screenshot before PAUSE. Do not
                    # enqueue that stale pre-pause image on resume.
                    while self.pause_event.is_set() and not self.stop_event.wait(.025):
                        pass
                    continue
                if self.stop_event.is_set():
                    break
                now = time.monotonic()
                self.capture_times.append(now)
                self.dropped_capture += offer_latest(self.raw, (packet, now))
        except Exception as exc:
            self.errors.append(f"capture: {exc}")
        finally:
            self.finished = True

    def _analyse(self) -> None:
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                self.stop_event.wait(.025)
                continue
            try:
                packet, grabbed_at = self.raw.get(timeout=.08)
            except Empty:
                continue
            started = time.perf_counter()
            try:
                with self.lock:
                    sample = self.tracker.process(packet)
                    ball = self.ball.process(packet, scene_cut=sample.reason == "camera_cut")
                    epoch = self.ball_epoch
            except Exception as exc:
                self.errors.append(f"analysis: {exc}")
                continue
            ended = time.monotonic()
            self.processing_times.append(ended)
            self.dropped_display += offer_latest(
                self.results, Result(packet, sample, grabbed_at, ended,
                                     (time.perf_counter() - started) * 1000,
                                     ball, epoch)
            )

    def latest(self) -> Result | None:
        value = None
        while True:
            try:
                candidate = self.results.get_nowait()
            except Empty:
                return value
            if candidate.ball_epoch == self.ball_epoch and (candidate.ball is None or candidate.ball.state != "stale_before_seed"):
                value = candidate
