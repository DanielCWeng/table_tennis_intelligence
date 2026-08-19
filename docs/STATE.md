# Current state

Last updated: 2026-08-18, at commit `bf19a32`.

This is the single "where things actually are" file. Where it disagrees with
another document, this one is right and the other is stale. Every claim below
is backed by a measurement taken against real footage, not an intention.

## Environment (verified, not assumed)

Python 3.13.5 on native Windows 11. Torch 2.6.0+cu124 with CUDA available on
an RTX 3060 Laptop (6 GB). PyAV 18, OpenCV 5.0, Pillow 11.3, NumPy 2.2.6,
rtmlib, onnxruntime 1.28, duckdb 1.5.2, pyarrow 21. MMPose/MMEngine/MMCV/MMDet
are absent. TOTNet's 12 checkpoints (2.0 GB) and TT3D's
`table_segmentation.ckpt` are present under `third_party/`.

Documents written before 2026-08-18 assume none of this exists and therefore
understate what is runnable.

## Test footage

All under `third_party/tt3d/data/calibration_test/videos/` (gitignored).

| File | Source | Size | Character |
| --- | --- | --- | --- |
| `test_00.mp4` | WTT Frankfurt, shipped with TT3D | 1280x720, 25 fps, 12.9 s | Broadcast, purple table, purple-lit arena |
| `yt_-Jf5QgOIpIU.mp4` | WTT London | 640x360, 25 fps, 4 s | Broadcast, wide shot, bright blue floor |
| `yt_sOacXMsyEW8.mp4` | Club match (TTE.TV) | 640x360, 30 fps, 12 s | Sports hall, low corner angle, heavy clutter |

## What works

**Video decode.** PyAV handles all three clips: correct dimensions, fps,
duration, codec.

**Table calibration.** Detects the table from its edges and net line. Colour
thresholding was removed: the Frankfurt table surface is (114, 100, 110)
against a (107, 95, 104) floor, and the old green-or-blue mask hit 0.0% of the
true table region while selecting 3.7% of the frame as arena noise.

Calibration is taken from a consensus of up to 12 frames per segment, not one.
A single frame is unreliable for a reason unrelated to the detector: a player
leaning over the table to serve hides a corner, and the detector reaches past
it to the skirt line beneath. On London, 7 of 10 sampled frames are correct
and stable to 2 px while 3 are wrong; the median rejects the minority. On
Frankfurt, where the single frame was already right, consensus moves the
corners by under 0.5 px.

**Ball tracking on broadcast footage.** The TOTNet adapter loads the vendored
TTA table-tennis checkpoint and tracks the real ball. On Frankfurt: 60/60
frames, median displacement 38.8 px/frame, one jump over 300 px in 59
transitions, visually confirmed following the ball across a rally. This
replaced a bright-blob baseline that reported a ball in 100/100 frames at 0.75
confidence while locked onto static floor advertising.

**Session persistence.** Sessions round-trip to typed objects. `raw/` and
`cleaned/` carry data instead of being empty directories. `session.json` is a
manifest rather than a duplicate of every sidecar.

## What does not work

**Ball tracking outside broadcast conditions.** On the club clip: 120/120
frames "detected", never once abstaining, median confidence 0.064 against
0.10-0.11 on broadcast, and 35 of 119 transitions jumping over 120 px. It is
largely tracking white trainers. A light ball on a bright wooden floor in a
hall full of small white round objects is far harder single-frame than a white
ball on dark flooring, whatever a human viewer perceives.

**Distractors inside the real-ball confidence band.** On London, measured:

| What the model found | Frame | Confidence |
| --- | --- | --- |
| Player's socks | 36 | 0.019 |
| Player's hands, ball just in front | 72 | 0.059 |
| White circular logo on a shirt back | 63 | 0.109 |
| Genuine ball | 25 | 0.164 |

The shirt logo sits inside the band of real detections (0.13-0.16). No
threshold separates them, so raising `confidence_threshold` trades false
positives for false negatives without adding information. The separating
signal is motion, which the single-frame argmax discards.

**Abstention is accidental, not reasoned.** TOTNet has no "no ball" class. On
London it abstained on 17 frames, but 16 were one contiguous run at the clip
start pinned at a constant 0.0015 confidence — the model producing no signal,
with the threshold happening to sit above its floor. Only frame 67 was a
genuine mid-rally dropout.

**Calibration on cluttered venues.** The club clip fails on all 12 sampled
frames. This is the correct outcome — that floor carries badminton and
basketball markings and three more tables sit in the background — but it means
no table geometry is available for that footage.

**Cut detection.** `detect_cuts` is a 64x36 greyscale mean-absolute-difference
threshold. It catches hard cuts and misses dissolves, fades, and graphic
wipes. Replays are not detected at all: `_classify_segments` hardcodes
`camera_static=True` and never populates `replay_evidence`, so a slow-motion
replay showing a table classifies as gameplay. A missed cut carries player
identities across a camera change, because the identity assigner resets only
on detected segment boundaries.

**Events.** `events.py` still declares a bounce when image-space `y` reverses,
which fires on perspective, camera motion, and tracking noise. `_inside_table`
projects an airborne ball through the tabletop homography and labels the
result `PHYSICS_INFERRED` despite no physics being involved, contradicting the
coordinate rules in ARCHITECTURE.md.

**No ground truth.** `benchmark.py` has nothing to score against. Every
assessment above is a measurement plus visual inspection, not an error rate.

**Pipeline default.** `analyse_video` still uses the bright-blob tracker.
TOTNet is injectable via `DefaultPerceptionProvider(ball_tracker=...)` but is
not the default path.

## Quality signals: what survived testing

Two candidate calibration-quality metrics were tested. One works.

**`corner_sensitivity_m_per_px` works.** It perturbs each corner by a pixel and
measures the resulting table-coordinate error, catching both bad corners and
bad camera geometry with one number.

| Case | Sensitivity |
| --- | --- |
| Frankfurt consensus (correct) | 1.78 cm/px |
| London consensus (correct) | 3.07 cm/px |
| Synthetic oblique elevated | 0.75 cm/px |
| London frame 0, the skirt quad | 36.73 cm/px |
| Synthetic down-the-table | 7.63 cm/px |
| Synthetic extreme end-on | 31.47 cm/px |

Flagged above 5 cm/px, the point at which one pixel of corner error exceeds
the precision an edge call needs on a 1.525 m table.

**The rectangle sign test does not work and was rejected.** Focal recovery from
the table homography is exact on noiseless corners — 800, 2500 and 4000 px
recovered to the digit across geometries. But at 0.5 px of corner noise, below
what any real detector achieves, one of the two focal solutions goes negative
in 22% of near-lens trials and 54% of broadcast-geometry trials, while the
median estimate stays accurate to 0.3%. A per-frame validity boolean built on
the sign would reject correct calibrations about as often as wrong ones.
`tests/test_calibration_geometry.py` enforces this negative result.

**`reprojection_error_px` is not a quality signal and never was.** A four-point
DLT fits its four inputs exactly, so the residual is ~0 for a correct
calibration and for one spanning the entire arena alike. It is retained for
diagnostics only.

The recurring lesson, hit three times in one day: single frames lie, medians
do not. It holds for calibration corners, for focal estimates, and for ball
detections.

## Structural limits

The homography maps the table plane only. Confirmed bounce detection, impact
height, 3-D velocity and physical stroke analytics are not reachable without
3-D ball reconstruction, regardless of how good the 2-D tracking becomes.

The pipeline is committed to an oblique elevated view. Down the long axis the
far corners converge, the homography becomes ill-conditioned, and a pixel of
corner error costs 8-30 cm on the table. Broadcast main cameras are always
oblique, so the practical risk is replay segments inside a normal match rather
than a whole match shot end-on.

## Ordering

1. Multi-hypothesis trajectory linking (in flight, see below).
2. Ground truth. Four clicks per clip gives calibration truth; sparse
   per-frame ball labels give tracker error rates. Nothing above becomes an
   error rate until this exists.
3. Wire TOTNet as the pipeline default once linking is proven.
4. Cut and replay detection, before any full-match footage is trusted.
5. Pose, via rtmlib — the adapter exists and has never been exercised.

## In flight

A Codex job is building `src/ttintel/tracking.py`: top-k heatmap candidates,
motion-gated multi-hypothesis linking solved over the whole clip offline, and
piecewise fitting whose breakpoints become bounce and contact candidates. It
also touches `adapters/totnet.py` and `events.py`. Not yet reviewed, verified,
or committed at the time of writing.

## Known defects recorded but not fixed

- `Estimate.unknown()` infers `attempted` by pattern-matching the source string
  for `.unavailable`. An adapter named outside that convention is silently
  recorded as having run.
- The pipeline discards `FrameDetections` before storage, so keypoints dropped
  by `adapters/rtmlib.py` cannot reach `raw/`.
- `TwoPlayerIdentityAssigner` has no gating, occlusion memory, or velocity
  model, and no test covers identity swaps.
