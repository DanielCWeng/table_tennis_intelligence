# Actual Architecture

## Purpose

This repository implements the first perception MVP from the master brief. It
is an offline pipeline, not a real-time coach. The core is intentionally usable
with local frame annotations while heavyweight research models are installed in
their own environments.

## Flow

```text
video
  -> media probe / timestamped frames
  -> cut candidates
  -> camera segments
  -> table calibration + gameplay score
  -> pose / ball / racket adapter outputs
  -> two-player identity assignment
  -> FrameState
  -> conservative bounce/contact candidates
  -> deterministic measurements
  -> session JSON + JSONL + overlay frames
```

`ttintel.pipeline.analyse_packets` is the dependency-light orchestration
boundary. `analyse_video` adds optional media decoding. Every external model
must implement the contracts in `ttintel.adapters.base` and return the internal
schemas rather than leaking research-repository objects into analytics.

## Coordinate convention

The table origin is the near-left playable corner. `+x` runs along the 2.74 m
length, `+y` runs across the 1.525 m width toward the far side, and `+z` is
upward. The homography is image-to-table-plane only. The code does not treat an
airborne ball projected through that homography as a 3-D world point.

## Provenance boundary

`schemas.Estimate` is required for model-derived quantities. It records:

```text
value
confidence
source
visibility
inference_type
quality_flags
```

Raw observations are not overwritten. The session writer creates:

```text
raw/       adapter-native material when an adapter provides it
cleaned/   validated/normalised values
fused/     FrameState JSONL
derived/   measurements and summaries
```

Unknown values remain `value: null` with an explicit source and flag. A
heuristic candidate is not serialised as an observed fact.

## Module map

| Module | Responsibility |
| --- | --- |
| `schemas.py` | Typed session, frame, joint, ball, event, and morphology objects |
| `geometry.py` | Regulation geometry, DLT homography, reprojection, plane transforms |
| `media.py` | PyAV/OpenCV/FFprobe seams with timestamp flags |
| `scene.py` | Thumbnail cut baseline, segment boundaries, deterministic quality score |
| `calibration.py` | Manual JSON corners and explicitly heuristic colour fallback |
| `perception.py` | Annotation provider, unavailable-model boundary, bright-blob ball baseline |
| `fusion.py` | Explainable two-player ID continuity and frame fusion |
| `events.py` | Candidate bounce/contact evidence and rally boundaries |
| `analytics.py` | Image-space measurements and summary statistics |
| `rendering.py` | Pillow overlays and render manifest |
| `storage.py` | Session directory and JSON/JSONL persistence |
| `benchmark.py` | Gold-set ball/event metrics |
| `adapters/totnet.py` | TOTNet ball tracking against the vendored TTA checkpoint |
| `cli.py` | `analyse video.mp4` command |

## What is deliberately not claimed yet

See [`docs/STATE.md`](docs/STATE.md) for what actually runs today. This
section describes the original MVP scope and is no longer a current inventory.

MP4 decoding and TOTNet ball tracking are now real and verified on broadcast
footage. RTMPose, TT3D, BlurBall, TrackNetV3, 3-D human mesh and racket
inference remain unclaimed, and the pipeline still records those components as
unavailable rather than fabricating their output.

## Extension order

Superseded by the ordering in [`docs/STATE.md`](docs/STATE.md), which was
revised against real footage. Kept for provenance.

1. Add a PyAV/FFmpeg environment and verify timestamps on a small local clip.
2. Wrap TT3D calibration and MotionBERT artifacts without changing schemas.
3. Add RTMPose/rtmlib adapter output to `PoseObservation`.
4. Add one ball tracker at a time behind `BallTracker`; store all outputs.
5. Benchmark against a labelled gold set before selecting a primary tracker.
6. Add 3-D, morphology, racket, and reference analytics only after the overlay
   demonstrates stable table/player/ball state.
