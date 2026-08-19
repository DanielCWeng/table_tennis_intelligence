# Table Tennis Video Intelligence

Offline, confidence-aware table-tennis video perception. The first pass turns
decoded frames into timestamped camera segments, table calibration, persistent
player IDs, ball candidates, conservative event candidates, structured JSONL,
and rendered debugging frames.

The project deliberately separates the application from research-model
dependencies. No checkpoint, dataset, or large binary is stored in this
repository.

## Current MVP

The runnable core provides:

- regulation-table geometry and table-plane homography;
- manual four-corner calibration plus a clearly labelled colour heuristic;
- timestamp-preserving PyAV/OpenCV media seams;
- deterministic scene-cut and gameplay-quality baselines;
- JSON annotation input for repeatable local fixtures;
- confidence/provenance-aware frame, joint, ball, event, and morphology schemas;
- two-player identity assignment;
- conservative bounce/contact candidates;
- deterministic image-space measurements and benchmark metrics;
- session storage under `sessions/<session-id>/` with raw/cleaned/fused/derived areas;
- Pillow overlays showing table, axes, player IDs, skeletons, ball confidence, and events.

Research adapters are intentionally optional, and the default pipeline never
fabricates model observations when one is absent.

Environment as verified on 2026-08-18 (`python -m ttintel.doctor --json`):
PyAV 18, OpenCV 5.0, Pillow 11.3, NumPy 2.2, rtmlib, onnxruntime 1.28,
duckdb, and pyarrow are all installed, as is Torch 2.6.0+cu124 with CUDA
available on an RTX 3060 Laptop (6 GB). MMPose/MMEngine/MMCV/MMDet are not.
TOTNet's 12 checkpoints and TT3D's `table_segmentation.ckpt` are present under
`third_party/`. Earlier revisions of this file claimed none of this existed;
docs written against that assumption understate what is runnable today. See
[`ARCHITECTURE.md`](ARCHITECTURE.md), [`docs/MODEL_MATRIX.md`](docs/MODEL_MATRIX.md),
[`docs/IMPLEMENTATION_CHECKLIST.md`](docs/IMPLEMENTATION_CHECKLIST.md), and
[`docs/INSTALL_HEAVY.md`](docs/INSTALL_HEAVY.md).

## Install

Python 3.10+ and NumPy are required: geometry, perception, scene, events, and
rendering all import NumPy at module load. For development:

```powershell
python -m pip install -e ".[dev]"
```

For local video decoding/rendering, install the optional media stack in the
environment that owns those dependencies:

```powershell
python -m pip install -e ".[media,render]"
```

FFmpeg/FFprobe may also be installed system-wide. Large research checkpoints
remain a separate, explicit decision and are not downloaded by this project.

## Run

The intended command is:

```powershell
analyse video.mp4 --output outputs/sessions
```

Useful development options:

```powershell
analyse video.mp4 `
  --manual-corners calibration/corners.json `
  --annotations fixtures/annotations.json `
  --max-frames 300
```

Manual corners use this order: `near_left`, `near_right`, `far_right`,
`far_left`. A frame annotation file has this small shape:

```json
{
  "frames": [
    {
      "frame_id": 0,
      "poses": [
        {"id": "p0", "bbox": [10, 20, 80, 110], "joints": {"pelvis": [45, 70]}}
      ],
      "ball": {"image": [120, 60], "confidence": 0.9}
    }
  ]
}
```

Each run writes a session directory containing `session.json`, segment and
frame/event JSONL, analytics, and PNG debugging overlays. PNG frames are the
fallback when no MP4 encoder is available; the render manifest records that
fact rather than claiming a video was produced.

## Tests

```powershell
pytest
```

The tests use tiny in-memory frames and do not download videos, checkpoints,
or datasets.

To inspect the heavyweight environment after installing optional packages:

```powershell
ttintel-doctor --json
```
