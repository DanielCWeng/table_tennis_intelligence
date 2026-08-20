# Current state

Last updated: 2026-08-20, at commit `27603a8` on `main`, plus the branch
`feat/labeller-gui` at `fde84c7` where noted.

This is the single "where things actually are" file. Where it disagrees with
another document, this one is right and the other is stale. Every claim below
is backed by a measurement taken against real footage, not an intention.

## Environment (verified, not assumed)

Python 3.13.5 on native Windows 11. Torch 2.6.0+cu124 with CUDA available on
an RTX 3060 Laptop (6 GB). PyAV 18, OpenCV 5.0, Pillow 11.3, NumPy 2.2.6,
rtmlib, onnxruntime 1.28, duckdb 1.5.2, pyarrow 21. **The onnxruntime build is
CPU-only** — its providers are Azure and CPU, with no `CUDAExecutionProvider`.
Torch having CUDA does not mean ONNX does; pose would run on CPU, and the CLI's
`--pose-device cuda` default cannot be honoured. MMPose/MMEngine/MMCV/MMDet
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

**TOTNet is the pipeline default.** `analyse_video` now builds
`TOTNetBallTracker`; `--ball-tracker {totnet,blob,none}` is an explicit opt-out
and there is no automatic fallback, because silently downgrading to a detector
documented as broken is worse than not running. A missing checkpoint exits 2
naming the path. Measured on London's first 40 frames: the default reports
`totnet.ball_tracker` with ball availability 0.60, abstaining across the
16-frame dead run at the clip start, while `--ball-tracker blob` reports 1.00 —
a ball in every frame, which is the baseline's documented pathology.

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

1. Fix the labeller's overlay calibration bug (see below), then merge
   `feat/labeller-gui`.
2. Ground truth, using that tool. Four clicks per clip gives calibration truth;
   sparse per-frame ball labels give tracker error rates. Nothing in this
   document becomes an error rate until this exists, including the entropy
   signal above, which currently rests on hand-labelled frames from one dead
   run.
3. Frame 36. It is wrong under every variant tried and wrong on the argmax path
   too. Ground truth decides whether it is a 1-in-100 event or a 1-in-10 one,
   which decides whether it is worth chasing at all.
4. Cut and replay detection, before any full-match footage is trusted.
5. Pose, via rtmlib — never executed. Note the CPU-only onnxruntime above: on
   this hardware a whole-body model runs per frame on CPU, which probably
   settles the question before it is asked.

## Offline trajectory linking: measured, inspected, merged

`src/ttintel/tracking.py` keeps the
top-k heatmap candidates and chooses a path over the whole clip with a
second-order dynamic program. It was run against real footage and inspected
frame by frame in annotated montages, not just scored, before merging.

Comparing the linked path with the rank-0 argmax path that `main` uses today,
over 200 frames of Frankfurt, all 100 of London and 200 of the club clip:

| | argmax | linked |
| --- | --- | --- |
| Frankfurt, jumps over 120 px | 60 / 199 | 4 / 190 |
| Frankfurt, median displacement | 63.7 px | 42.5 px |
| London, jumps over 120 px | 15 / 99 | 3 / 76 |

**It works on broadcast footage.** On Frankfurt the marker sits on the
motion-blurred ball through an entire rally; the frames it abstains on are the
ones where the ball is genuinely out of shot. On London it rejects the shirt
logo at frame 63 — the distractor that sits inside the real-ball confidence band
and that no confidence threshold can remove — and picks the ball at the racket
instead. That is the specific win the whole approach was for.

**It does not fix the club clip and is not expected to.** That failure is a
detector problem, not a linking problem: 24 consecutive frames park on the
sports-hall balcony under both the argmax and the linked path. The club clip is
deliberately not an acceptance criterion for this work.

**It could invent a ball where the detector had none, and now mostly does not.**
TOTNet's head is a softmax over the whole heatmap with no null class, so a frame
containing no ball still produces an argmax — on London, 17 of 100 frames are
pure noise at ~0.0015. Linking originally bridged eight of those into a smooth,
entirely fictional trajectory across the floor, emitted as tracked positions.
Gating interpolation on anchor evidence removed the bridge; frames 8 and 10
survive as isolated stray points that no longer connect into a path. Of the 17
no-signal frames, 3 still carry a position, down from 9.

Linear interpolation between two selected points is now reported as
`InferenceType.INTERPOLATED` rather than `PHYSICS_INFERRED`. No physics is
involved and the old label contradicted the coordinate rules in
ARCHITECTURE.md.

### What was tried here and removed

An absolute-confidence floor in the emission term, estimated as a per-clip 10th
percentile of the argmax confidence. It bought two London frames and nothing on
Frankfurt, in exchange for hand-fitted sigmoids — slopes of 16 and 32 in
log-ratio space, a band, an offset — which are step functions with soft edges,
the confidence-space form of the fixed detector gate `tracking.py` disclaims in
its own docstring.

It also only worked by accident. The band fires when a frame's top-1 sits within
about 10% of the clip's own 10th percentile, which catches London because its
dead run is pinned at a constant 0.0015 that then *becomes* that percentile.
That is one clip's pathology, not a test for absence. `tests/test_tracking.py`
keeps the case as a strict `xfail` so the missing capability stays visible.

The percentile is retained where it was measured to be safe: gating
interpolation anchors only. Replacing that gate's log-ratio margins with a plain
"anchor must reach the floor" test costs Frankfurt 18 positioned frames, so the
margins are load-bearing.

### The open failure, and why confidence cannot close it

London frame 36 selects a rank-7 candidate at 0.00063 on empty table, under
every variant tried. Absolute confidence cannot separate it from a genuine weak
detection, because the scale is not comparable between clips: London's detector
floor is 0.0015 and Frankfurt's is 0.027, and London's real ball sits at
0.13-0.16 while Frankfurt's sits at 0.026.

A candidate signal was measured: the **entropy of the top-k confidences within
one frame**, on the theory that a no-signal frame yields a diffuse softmax while
a real detection yields a dominant peak even when its absolute value is low.

| Group | n | entropy range | top-1 range |
| --- | --- | --- | --- |
| Diffuse noise (London 0-15) | 16 | 1.759 - 1.808 | 0.0015 - 0.0016 |
| Weak real ball (Frankfurt) | 8 | 0.907 - 1.537 | 0.0122 - 0.0350 |
| Distractor lock (Frankfurt) | 16 | 1.277 - 1.880 | 0.0098 - 0.0341 |

Entropy separates diffuse noise from a weak real ball cleanly, and unlike
confidence its scale looks comparable across the two clips. It does **not**
separate a distractor lock from a real ball, and cannot: a logo produces a
genuinely concentrated peak. So concentration is a candidate abstention signal
for "there is nothing here", and is no help at all for frames 36 and 63, which
need motion — which is what the linker already supplies.

This is a lead, not a result, and it is not in the code. The three groups were
hand-labelled from one montage over two clips, and the 16 noise frames are a
single contiguous dead run pinned at one confidence value, so they are closer to
one observation than to sixteen. Confirming it needs the ground truth in
"Ordering" below, not another eyeball pass.

## Ground-truth labeller: on a branch, one blocking bug

`src/ttintel/labeller.py` on `feat/labeller-gui` (head `fde84c7`) is a local
browser tool for producing the ground truth item 2 of Ordering needs. It exists
because `calibration.save_manual_corners` had no callers: the CLI accepts
`--manual-corners` and `load_manual_corners` reads the file, but nothing could
write one except a human editing JSON. That matters most for the club clip,
where automatic calibration correctly fails on all 12 sampled frames.

It scrubs frames by `FramePacket.frame_id`, zooms, records a ball position or an
explicit "no ball visible" — absence is a first-class label, because a tracker
inventing a ball where there is none cannot be scored otherwise — and writes
corners through `save_manual_corners` so the result is loadable by the CLI by
construction. Labels live in `data/labels/<video_id>.*.json`, committed and keyed
by the stable id from `media._video_id`, not by an absolute path.

Verified by driving the running server: clearing a label returns a frame to
untouched rather than to absent (they are different states), undo works, counts
stay correct, `--host` is honoured with a warning past loopback, and the tracker
overlay distinguishes off, running, ready, and unavailable.

**The blocking bug.** `compute_tracker_overlay` passes `calibration=None` unless
manual corners already exist. It never calls `calibrate_consensus`, so on first
open the overlay runs the linker with no table prior and shows a materially
worse path while attributing it to `totnet.ball_tracker.viterbi`:

| | overlay, no calibration | pipeline, consensus |
| --- | --- | --- |
| positions | 99/100 | 77/100 |
| frame 0 | (250.0, 286.2) at 0.00067 | none |
| frame 63 | (176.2, 155.0) rank 0 at 0.109 | (213.8, 170.0) rank 2 at 0.0024 |

Frame 63's (176, 155) is the shirt-back logo — the exact distractor the linking
work was built to reject. Without the table prior nothing penalises a candidate
sitting off the table, so the logo wins on confidence. In a tool whose purpose is
establishing what is true, displaying a degraded path under authoritative
provenance is the worst available failure. Fix before merging.

Two smaller notes: the tool holds the model resident on the GPU after a pass, so
a 6 GB card cannot host a second consumer at the same time; and the header reads
`Frame 26` while the footer reads `27 / 100`, which is frame id versus 1-based
position.

## Known defects recorded but not fixed

- `Estimate.unknown()` infers `attempted` by pattern-matching the source string
  for `.unavailable`. An adapter named outside that convention is silently
  recorded as having run.
- The pipeline discards `FrameDetections` before storage, so keypoints dropped
  by `adapters/rtmlib.py` cannot reach `raw/`.
- `TwoPlayerIdentityAssigner` has no gating, occlusion memory, or velocity
  model, and no test covers identity swaps.
