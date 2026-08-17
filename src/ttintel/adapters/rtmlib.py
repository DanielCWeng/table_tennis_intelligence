"""Optional RTMLib/RTMPose whole-body adapter.

The import is lazy so the main project remains runnable without OpenCV,
ONNXRuntime, or RTMLib.  RTMLib returns COCO-WholeBody keypoints; this adapter
normalises them into :class:`PoseObservation` and preserves model confidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..adapters.base import AdapterInfo
from ..media import FramePacket
from ..schemas import BoundingBox, Estimate, InferenceType, JointState, Point2D, PoseObservation, Visibility


WHOLEBODY_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
    "right_small_toe", "right_heel",
) + tuple(f"face_{index:02d}" for index in range(68)) + tuple(
    f"left_hand_{index:02d}" for index in range(21)
) + tuple(f"right_hand_{index:02d}" for index in range(21))


class RtmlibUnavailable(RuntimeError):
    pass


class RtmlibWholebodyEstimator:
    info = AdapterInfo(
        name="rtmlib.rtmpose.wholebody",
        role="2-D whole-body pose",
        environment="rtmlib + opencv + onnxruntime",
        license_status="verify model/checkpoint terms",
    )

    def __init__(
        self,
        *,
        mode: str = "balanced",
        backend: str = "onnxruntime",
        device: str = "cuda",
        score_threshold: float = 0.25,
    ) -> None:
        try:
            from rtmlib import Wholebody
        except ImportError as exc:
            raise RtmlibUnavailable(
                "Install rtmlib, opencv-contrib-python, and onnxruntime before using this adapter."
            ) from exc
        self._estimator = Wholebody(mode=mode, backend=backend, device=device)
        self.score_threshold = score_threshold

    def estimate(self, packet: FramePacket) -> list[PoseObservation]:
        image = np.asarray(packet.image)
        # RTMLib's examples use OpenCV BGR.  The project media boundary is RGB.
        bgr = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] >= 3 else image
        keypoints, scores = self._estimator(bgr)
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
        if scores.ndim == 1:
            scores = scores[None, ...]
        observations: list[PoseObservation] = []
        for person_index, (person_points, person_scores) in enumerate(zip(keypoints, scores)):
            visible = [
                (float(point[0]), float(point[1]), float(score))
                for point, score in zip(person_points, person_scores)
                if float(score) >= self.score_threshold
            ]
            if not visible:
                continue
            x_values = [item[0] for item in visible]
            y_values = [item[1] for item in visible]
            confidence = min(1.0, max(0.0, sum(item[2] for item in visible) / len(visible)))
            joints: dict[str, JointState] = {}
            for index, (point, score) in enumerate(zip(person_points, person_scores)):
                if index >= len(WHOLEBODY_NAMES):
                    break
                score = float(score)
                if score < self.score_threshold:
                    continue
                joints[WHOLEBODY_NAMES[index]] = JointState(
                    joint_name=WHOLEBODY_NAMES[index],
                    image=Estimate(
                        value=Point2D(float(point[0]), float(point[1])),
                        confidence=max(0.0, min(1.0, score)),
                        source=self.info.name,
                        visibility=Visibility.VISIBLE,
                        inference_type=InferenceType.MODEL_INFERRED,
                    ),
                )
            observations.append(
                PoseObservation(
                    observation_id=f"rtmlib-{packet.frame_id}-{person_index}",
                    bbox=Estimate(
                        value=BoundingBox(min(x_values), min(y_values), max(x_values), max(y_values)),
                        confidence=confidence,
                        source=self.info.name,
                        visibility=Visibility.VISIBLE,
                        inference_type=InferenceType.MODEL_INFERRED,
                    ),
                    joints=joints,
                    detector_track_id=None,
                    source_model=self.info.name,
                )
            )
        return observations
