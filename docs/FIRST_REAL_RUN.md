# First run on real footage

Date: 2026-08-18. Before this, the pipeline had only ever run on synthetic
fixtures. This records what it does on broadcast video, unedited.

**This is a historical record and is deliberately not updated.** The
calibration and ball-tracking failures below were both fixed later the same
day; see [`STATE.md`](STATE.md) for current status. The findings about
`reprojection_error_px` and about conservative thresholds still hold.

## Setup

```powershell
python -m pip install -e ".[dev]"
```

Input: `third_party/tt3d/data/calibration_test/videos/test_00.mp4` — a WTT
Frankfurt broadcast clip, 1280x720, h264, 25 fps, 12.88 s. First 100 frames,
`render=True`, no manual calibration, no annotations, default providers.

## Result

| Stage | Outcome |
| --- | --- |
| Decode (PyAV) | Worked. Correct dimensions, fps, duration, codec. |
| Cut detection | 1 segment, no false cuts. Correct for this clip. |
| Calibration | Reported success. Geometrically wrong — see below. |
| Ball | "Detected" in 100/100 frames. All false positives. |
| Pose | 0 players, correctly reported unavailable. |
| Events | 0 emitted. |
| Throughput | 4.2 fps (23.6 s for 100 frames). |

## Findings

### 1. Calibration locked onto the arena, not the table

`detect_table_corners_heuristic` returned `near_left (232, 671)`,
`near_right (1279, 219)`, `far_right (1191, 16)`, `far_left (48, 22)` — a
quadrilateral covering nearly the whole frame. The table occupies roughly
x 420-860, y 320-440.

Cause: `_colour_table_mask` accepts green *or* blue pixels, and this venue is
lit purple-blue throughout. The mask therefore selects most of the arena, and
because corners are taken as frame-wide extrema of `x+y` and `x-y`, the result
is the bounding extremes of the lighting, not of the table.

Every table coordinate derived from this calibration is meaningless.

### 2. `reprojection_error_px` cannot detect a bad calibration

The wrong calibration above reported `reprojection_error_px = 1.6e-13` and was
accepted. With exactly four point correspondences the DLT homography fits those
four points exactly, so the residual is always ~0 regardless of whether the
corners are correct. The metric measures solver arithmetic, not calibration
quality, yet `calibration_quality()` surfaces it as a quality signal.

A real check needs redundant constraints — net line, table edges, or known
markings beyond the four corners.

### 3. The ball tracker latched onto static signage at 0.75 confidence

`BrightBlobBallTracker` reported a ball in all 100 frames, at (487, 303),
moving a median of 0.5 px per frame. That location is the white lettering of a
floor advertisement. The real ball is elsewhere and moves far faster.

Two compounding causes:

- the brightness/low-saturation test matches white text on dark flooring as
  readily as it matches a ball;
- `_choose` breaks ties by proximity to the previous detection, so once the
  tracker latches onto a static bright object it can never leave it. There is
  no motion prior, no velocity gate, and no way to report "no ball this frame".

Confidence peaked at 0.75 while being wrong in every frame. This is the clearest
available evidence that the confidence constants encode no information.

### 4. Segment classification inherited the bad calibration

`_classify_segments` derives `table_detected`, `calibration_succeeded`, and
`sufficient_table_visible` from `segment.table is not None`. Because the wrong
calibration is still a `TableCalibration`, all three read true and the segment
classified as gameplay. It also hardcodes `camera_static=True` and
`moving_camera=False`, so no camera evidence is consulted at all.

## What worked

The conservative event thresholds. Given 100 false ball positions, the bounce
detector still emitted nothing, because a static blob never produces the
required vertical reversal. The system's refusal to guess is doing real work —
it converted a garbage input into silence rather than into fabricated bounces.

Provenance labelling also held: the calibration carried `source=automatic`,
`confidence=0.35`, `quality_flags=['heuristic']`, and pose was correctly
recorded as unavailable rather than empty-but-successful.

## Implications

Ordering that follows from this, replacing "Extension order" in ARCHITECTURE.md:

1. A calibration that works, or manual four-corner override as the default
   path. Nothing downstream means anything without it.
2. A real calibration quality metric, since the current one cannot fail.
3. A ball tracker with a motion model, and the ability to return "absent".
   Detection rate is not the metric; a tracker that abstains beats one that
   latches.
4. Only then pose, 3-D, and analytics.

Labelled ground truth remains the precondition for all of it. There is no way
to tell finding 3 from a working tracker without frames where the true ball
position is known.
