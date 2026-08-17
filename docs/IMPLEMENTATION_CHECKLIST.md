# Implementation Checklist

Checked items are implemented and verified in this repository. Deferred items
are explicit because they require external model environments, checkpoints, or
labelled footage rather than more application scaffolding.

## Foundation

- [x] Read the complete master brief.
- [x] Audit the empty repository and local runtime.
- [x] Add Python packaging and a console entry point.
- [x] Add `.gitignore` coverage for `.tmp/`, Python caches, virtual environments, and generated outputs.

## Perception MVP

- [x] Define typed video, segment, frame, joint, ball, racket, event, and morphology schemas.
- [x] Preserve confidence, visibility, source, inference type, and quality flags.
- [x] Preserve raw/cleaned/fused/derived session boundaries.
- [x] Add optional PyAV/OpenCV/FFprobe media seams with explicit timestamp fallback flags.
- [x] Add a deterministic frame-difference cut baseline.
- [x] Add deterministic gameplay evidence scoring and segment classification.
- [x] Add regulation table geometry and image/table homography.
- [x] Add manual four-corner calibration JSON support.
- [x] Add an explicitly heuristic table-colour calibration fallback.
- [x] Add stable two-player identity assignment.
- [x] Add local annotation and unavailable-model adapters.
- [x] Add a diagnostic bright-blob ball baseline without pretending it is a research tracker.
- [x] Add conservative bounce/contact candidates with evidence and confidence.
- [x] Add image-space movement/angle measurements and summaries.
- [x] Add structured JSON/JSONL session output.
- [x] Add Pillow debugging overlays and a render manifest.
- [x] Add a gold-set benchmark harness for ball and event metrics.
- [x] Add heavyweight-environment diagnostics.
- [x] Add an optional RTMLib/RTMPose whole-body adapter.

## Verification/documentation

- [x] Add geometry, schema, scene, event, and end-to-end fixture tests.
- [x] Add `ARCHITECTURE.md` with actual module interfaces and limitations.
- [x] Add `docs/MODEL_MATRIX.md` with external-repository audit status.
- [x] Add this implementation checklist.
- [x] Run the post-bootstrap project doctor and verify the dedicated `tti-tt3d` environment exists.
- [ ] Run a real video through PyAV/FFmpeg in an environment with a decoder.
- [ ] Install and smoke-test the main media + RTMLib path.
- [x] Clone and pin the small source repositories in isolated environments (source audit complete; dependency installs remain deferred).
- [ ] Run TT3D end to end on supplied/sample footage (native `tti-tt3d` install blocked by package-network and environment-write restrictions; no TT3D output claimed).
- [ ] Reproduce TT3D's raw calibration/segmentation sample in Linux/Kaggle using commit `a2ef524ea0400262d6808db6cacf4a0b90bd0ad7` and the bundled checkpoint (dataset mounted in kernel version 2; wrapper ZIP-layout bug fixed for version 3; TT3D execution and validation remain pending).
- [ ] Run RTMPose/RTMW WholeBody on a labelled clip.
- [ ] Benchmark BlurBall, TrackNetV3, TOTNet, and TTNet on the same gold clips.
- [ ] Produce a trustworthy annotated MP4 from real footage.
- [ ] Benchmark 3-D human, morphology, racket, and temporal point adapters.
- [ ] Build self-comparison and repeated-error analytics from enough labelled rallies.

## Guardrails

- [x] Do not download or vendor large checkpoints.
- [x] Do not silently convert missing or inferred values into observations.
- [x] Do not use a table-plane homography as airborne 3-D reconstruction.
- [x] Keep model disagreement and missing backend state visible.
- [x] Keep coaching interpretation out of the first MVP.
