"""Generate a synthetic file ONLY for pre-existing storage tests in CI.

This is not real match footage and must not be represented as such.
Usage: python scripts/create_ci_video_fixture.py
"""
from pathlib import Path

import cv2
import numpy as np

out = Path("third_party/tt3d/data/calibration_test/videos/test_00.mp4")
out.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
if not writer.isOpened():
    raise RuntimeError("CI video encoder unavailable")
for n in range(48):
    frame = np.full((360, 640, 3), (43, 56, 78), dtype=np.uint8)
    cv2.rectangle(frame, (160, 145), (480, 270), (160, 100, 85), -1)
    cv2.line(frame, (320, 145), (320, 270), (215, 225, 220), 2)
    cv2.circle(frame, (200 + n * 4, 190), 5, (255, 255, 255), -1)
    writer.write(frame)
writer.release()
print(f"Constructed synthetic CI fixture: {out} ({out.stat().st_size} bytes)")
