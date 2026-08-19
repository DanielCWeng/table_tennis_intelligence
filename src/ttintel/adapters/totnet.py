"""TOTNet ball-tracking adapter.

TOTNet's light model consumes five normalised RGB frames at 288x512 and
returns a spatial probability heatmap for the last frame in that window.  The
adapter keeps the temporal state needed by the :class:`BallTracker` contract,
and maps the heatmap coordinates back to the packet's image coordinates.

The vendored repository lists ``einops`` and ``easydict`` as training-time
dependencies, but neither is needed for inference here.  Small compatibility
paths keep this adapter usable in the project's minimal environment without
adding a dependency or a custom CUDA operation.
"""

from __future__ import annotations

from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Sequence
import sys

import numpy as np

from .base import AdapterInfo
from ..media import FramePacket
from ..schemas import BallState, Estimate, InferenceType, Point2D, Visibility


INPUT_HEIGHT = 288
INPUT_WIDTH = 512
NUM_FRAMES = 5
_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    _REPO_ROOT
    / "third_party"
    / "totnet"
    / "weights"
    / "TOTNet_TTA_(5)_(288,512)_30epochs_Occl(0.25)_WBCE[1,2,3,3]_bs8_ch64"
    / "TOTNet_TTA_(5)_(288,512)_30epochs_Occl(0.25)_WBCE[1,2,3,3]_bs8_ch64_best.pth"
)


class TotnetUnavailable(RuntimeError):
    """Raised when the optional TOTNet runtime or checkpoint is unavailable."""


def _pure_torch_rearrange(tensor: Any, pattern: str, **axes: int) -> Any:
    """Implement the four rearrangements used by the vendored TOTNet model."""

    pattern = " ".join(pattern.split())
    if pattern == "b c n h w -> (b n) c h w":
        return tensor.permute(0, 2, 1, 3, 4).reshape(
            -1, tensor.shape[1], tensor.shape[3], tensor.shape[4]
        )
    if pattern == "b n c h w -> (b n) c h w":
        return tensor.reshape(-1, tensor.shape[2], tensor.shape[3], tensor.shape[4])
    if pattern == "(b n) c h w -> b c n h w":
        batch = int(axes["b"])
        frames = int(axes["n"])
        _, channels, height, width = tensor.shape
        return tensor.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    raise ValueError(f"Unsupported fallback einops pattern: {pattern!r}")


def _install_einops_fallback() -> None:
    try:
        import einops  # noqa: F401
    except ImportError:
        fallback = ModuleType("einops")
        fallback.rearrange = _pure_torch_rearrange  # type: ignore[attr-defined]
        sys.modules["einops"] = fallback


def _install_easydict_compat(torch: Any) -> None:
    """Make PyTorch 2.6's safe checkpoint loader understand EasyDict metadata."""

    try:
        import easydict  # noqa: F401
    except ImportError:
        module = ModuleType("easydict")

        class EasyDict(dict):
            __module__ = "easydict"
            __qualname__ = "EasyDict"

            def __getattr__(self, name: str) -> Any:
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

        module.EasyDict = EasyDict  # type: ignore[attr-defined]
        sys.modules["easydict"] = module
    try:
        torch.serialization.add_safe_globals([sys.modules["easydict"].EasyDict])
    except (AttributeError, TypeError):
        # Older PyTorch versions either do not expose this API or do not need
        # it because weights_only=False was their default.
        pass


def _load_python_module(path: Path, name: str) -> ModuleType:
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load vendored module {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_totnet_components() -> tuple[Any, ModuleType, ModuleType]:
    import torch

    _install_einops_fallback()
    _install_easydict_compat(torch)
    model_dir = _REPO_ROOT / "third_party" / "totnet" / "src" / "model"
    totnet = _load_python_module(model_dir / "TOTNet.py", "_ttintel_vendored_totnet")
    model_utils = _load_python_module(model_dir / "model_utils.py", "_ttintel_vendored_totnet_utils")
    return torch, totnet, model_utils


def _pad_temporal_window(frames: Sequence[Any], length: int = NUM_FRAMES) -> list[Any]:
    """Left-pad a causal frame history, preserving exactly ``length`` items."""

    if not frames:
        raise ValueError("at least one frame is required")
    if len(frames) >= length:
        return list(frames[-length:])
    return [frames[0]] * (length - len(frames)) + list(frames)


def _decode_heatmap(
    heatmap: Any, *, input_height: int = INPUT_HEIGHT, input_width: int = INPUT_WIDTH
) -> tuple[int, int, float]:
    """Return argmax x/y and the model heatmap probability at that location."""

    if hasattr(heatmap, "detach"):
        array = heatmap.detach().cpu().numpy()
    else:
        array = np.asarray(heatmap)
    if array.ndim == 2 and array.shape == (input_height, input_width):
        flat = array.reshape(-1)
    elif array.ndim == 2:
        if array.shape[0] != 1:
            raise ValueError("single-sample heatmap expected")
        flat = array[0]
    elif array.ndim == 3 and array.shape[0] == 1:
        flat = array[0].reshape(-1)
    elif array.ndim == 1 and array.size == input_height * input_width:
        flat = array
    else:
        raise ValueError(f"expected flattened or 2-D heatmap, got {array.shape}")
    index = int(np.argmax(flat))
    return index % input_width, index // input_width, float(flat[index])


class TOTNetBallTracker:
    """Track the ball with the vendored table-tennis TOTNet checkpoint.

    The supplied checkpoint/demo uses a causal five-frame window and predicts
    the last frame.  The first frame is therefore left-padded with itself;
    every subsequent frame has a complete causal window, including the final
    frame of a clip.  Inference is deliberately batch size one: the 6 GB GPU
    path holds one model and one five-frame clip at a time.
    """

    info = AdapterInfo(
        name="totnet.ball_tracker",
        role="model-based 2-D ball tracking",
        version_or_commit="vendored third_party/totnet",
        weights=str(DEFAULT_CHECKPOINT),
        environment="PyTorch TOTNet light model (pure PyTorch fallback for einops)",
        license_status="see third_party/totnet/LICENSE and TTA_Tracking_Dataset_Access_Agreement.pdf",
    )

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str | Any | None = None,
        confidence_threshold: float = 0.01,
    ) -> None:
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT).expanduser().resolve()
        if not checkpoint.is_file():
            raise TotnetUnavailable(f"TOTNet checkpoint not found: {checkpoint}")

        try:
            torch, totnet, model_utils = _load_totnet_components()
            requested = (
                str(device)
                if device is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            if requested.startswith("cuda") and not torch.cuda.is_available():
                requested = "cpu"
            self.device = torch.device(requested)
            args = SimpleNamespace(
                device=self.device,
                num_channels=64,
                num_frames=NUM_FRAMES,
            )
            self._model = totnet.build_motion_model_light(args)
            gpu_idx = self.device.index if self.device.type == "cuda" else None
            self._model = model_utils.load_pretrained_model(self._model, str(checkpoint), gpu_idx)
            self._model.eval()
        except Exception as exc:
            raise TotnetUnavailable(f"could not initialise TOTNet: {exc}") from exc

        self._torch = torch
        self.confidence_threshold = float(confidence_threshold)
        self._frames: deque[Any] = deque(maxlen=NUM_FRAMES)
        self._last_frame_id: int | None = None

    def reset(self) -> None:
        """Clear temporal state before starting a new clip or discontinuous stream."""

        self._frames.clear()
        self._last_frame_id = None

    @staticmethod
    def _preprocess(image: Any, torch: Any) -> Any:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("TOTNet expects an RGB image with three channels")
        array = np.ascontiguousarray(array[..., :3])
        try:
            import cv2

            resized = cv2.resize(array, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
            resized = (
                torch.nn.functional.interpolate(
                    tensor, size=(INPUT_HEIGHT, INPUT_WIDTH), mode="bilinear", align_corners=False
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .numpy()
            )
        normalised = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        return torch.from_numpy(np.ascontiguousarray(normalised)).permute(2, 0, 1)

    @staticmethod
    def _model_heatmap(output: Any, torch: Any) -> Any:
        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise ValueError("TOTNet output must be one spatial heatmap")
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError("TOTNet output is not a tensor")
        if output.ndim == 4 and output.shape[1] == 1:
            output = output[:, 0]
        if output.ndim == 3:
            output = output.reshape(output.shape[0], -1)
        if output.ndim != 2 or output.shape[1] != INPUT_HEIGHT * INPUT_WIDTH:
            raise ValueError(f"unexpected TOTNet output shape: {tuple(output.shape)}")
        # TOTNet's forward applies softmax.  Keep the model's probabilities;
        # the fallback also makes this safe if a compatible checkpoint returns
        # logits instead.
        sums = output.sum(dim=1)
        if (
            bool(torch.any(output < 0))
            or bool(torch.any(output > 1))
            or not bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-3))
        ):
            output = torch.softmax(output, dim=1)
        return output

    def _state_from_prediction(self, x: int, y: int, confidence: float, image: Any) -> BallState:
        height, width = np.asarray(image).shape[:2]
        if confidence < self.confidence_threshold:
            estimate = Estimate(
                value=None,
                confidence=confidence,
                source=self.info.name,
                visibility=Visibility.UNKNOWN,
                inference_type=InferenceType.MODEL_INFERRED,
                quality_flags=["absent"],
            )
        else:
            estimate = Estimate(
                value=Point2D(
                    float(x) * float(width) / INPUT_WIDTH,
                    float(y) * float(height) / INPUT_HEIGHT,
                ),
                confidence=confidence,
                source=self.info.name,
                visibility=Visibility.VISIBLE,
                inference_type=InferenceType.MODEL_INFERRED,
            )
        return BallState(image=estimate)

    def estimate(self, packet: FramePacket) -> BallState:
        if self._last_frame_id is not None and packet.frame_id != self._last_frame_id + 1:
            self.reset()
        current = self._preprocess(packet.image, self._torch)
        self._frames.append(current)
        self._last_frame_id = packet.frame_id
        window = _pad_temporal_window(list(self._frames))
        batch = self._torch.stack(window, dim=0).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            output = self._model(batch)
            heatmap = self._model_heatmap(output, self._torch)
            confidence, index = self._torch.max(heatmap[0], dim=0)
        index_value = int(index.item())
        return self._state_from_prediction(
            index_value % INPUT_WIDTH,
            index_value // INPUT_WIDTH,
            float(confidence.item()),
            packet.image,
        )


# Both spellings are useful to callers: the repository uses ``TOTNet`` while
# the existing RTMLib adapter uses sentence-case acronym names.
TotnetBallTracker = TOTNetBallTracker


__all__ = [
    "DEFAULT_CHECKPOINT",
    "TOTNetBallTracker",
    "TotnetBallTracker",
    "TotnetUnavailable",
]
