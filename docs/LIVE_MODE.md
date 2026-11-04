# Live Mode — model-free movement lab

## Installation and launch

Python 3.10+ on Windows 10/11 (other OSes can supply a region manually):

    python -m pip install -e ".[dev,live]"
    ttintel live

Open a table-tennis match in a browser, **start playback**, run the command,
then **drag directly over the playing video** on monitor 1. The translucent
desktop picker closes; the diagnostic window opens. Press **START / RESUME**.
Move that window away from the captured browser region to prevent feedback.
No browser extension or download is required; the source is local screen pixels.

Use a different display, a manually specified region, a local clip, or a
self-contained synthetic demo:

    ttintel live --monitor 2
    ttintel live --region "120,180,900,510"
    ttintel live --file clips/match.mp4
    ttintel live --demo
    python -m ttintel.live.benchmark --frames 180

The existing offline CLI remains unchanged:

    analyse clips/match.mp4 --ball-tracker none --no-render
    ttintel clips/match.mp4 --ball-tracker none --no-render

Desktop selector: click-drag on the chosen monitor; reverse drags are valid.
Esc/right-click cancels. Qt GUI coordinates are converted to physical MSS
pixels using actual monitor/selector dimensions; the selector is per-monitor.
**Mixed-DPI / rotated-monitor configurations need physical Windows testing.**
If it selects the wrong monitor use the manual --region escape hatch.
DRM-protected videos or hardware-accelerated browser surfaces may capture
black; disable browser hardware acceleration or use a local file where permitted.

## Interaction

- **Select player ROI**: drag a region *inside the preview* around one player.
  The label refers to the ROI you chose, not an independently detected human.
- **Background reference**: optionally drag a visually static patch outside the
  player box. Otherwise, the rest of the image supplies background motion.
- **Move/resize** either box by dragging or by using its bottom-right grip.
  ROIs use original image pixels, unaffected by preview zoom.
- **START / RESUME**, **PAUSE**, **RESET**, **EXIT** are explicit actions.
  Reset clears the optical-flow state and plots; no stale positions are retained.
- **EXPORT CSV** writes only selected motion diagnostics to a file you choose.
  Frames are **never** recorded, saved, sent, or uploaded by the live app.

## Evidence semantics

The first release uses sparse pyramidal Lucas–Kanade optical flow with
forward/backward validation, background-motion subtraction, timestamps,
cut/gap detection and bounded processing. It can **observe image-space point
motion inside a chosen region**, not identify a player or measure metres.
Relative X/Y displacement and speed are **processed-image pixels**, and the
quality indicator describes heuristic consistency of tracks, not a calibrated
probability of any sporting action. Only direction-change *candidates* are
shown; there is **no shot, bounce, rally or recovery verification**. The
low-motion interval is a proxy, not an anatomical recovery measurement.

Missing, cut, untextured or inconsistent frames are displayed as **UNKNOWN**;
a stable false zero or extrapolated position is never substituted. Movement
trails are only instantaneous tracked feature dots, not persistent player IDs.

Table calibration, RTMLib pose, TOTNet ball and shot inference are
**explicitly disabled in Live Mode** for this first version. The existing offline
implementations remain available. There is no bright-blob fallback that claims
to be the genuine TOTNet model, and no automatic checkpoint downloads.

## Architecture / performance

The GUI timer only displays results. A capture/decode thread produces
timestamped FramePacket objects into a **two-frame queue**. A separate
analysis thread runs low-resolution flow into a **two-result queue**. Full
RGB frames are retained transiently only for preview. Old frames are
discarded rather than building latency. Capture, processing, and display-age
metrics come from monotonic runtime clocks. GUI drawing runs on the Qt main
thread and never calls a neural detector.

CPU synthetic-analysis benchmark:

    python -m ttintel.live.benchmark --frames 180 --json live-benchmark.json

This reports measured throughput, median/p95 flow cost, reasons for invalid
tracking, and machine identification. It is **not** screen-grab FPS or actual
GUI latency. The Live Lab measures those separately during use. Neither a
target 30 FPS nor correct athlete tracking is asserted without measured
desktop footage.

### Measured CI baseline (commit 2128e7b, 4 November 2026)

The constructed 180-frame benchmark completed on the hosted runners:

| Metric | Ubuntu 3.11 runner | Windows 3.11 runner |
| --- | ---: | ---: |
| Optical-flow analysis FPS (excludes synthetic frame creation) | 249.79 | 205.02 |
| Full synthetic loop FPS (includes frame generation; no GUI/screen capture) | 189.30 | 164.67 |
| Median analysis time | 3.728 ms | 4.828 ms |
| p95 analysis time | 5.052 ms | 5.680 ms |
| Valid image-space position frames | 179/180 | 179/180 |
| Frames with lateral step above 0.3 processed pixels | 169/180 | 169/180 |
| Largest measured absolute lateral step | 8.431 px | 8.431 px |

[CI benchmark and log evidence](https://github.com/DanielCWeng/table_tennis_intelligence/actions/runs/41936172775).
The fixture is deliberately constructed and easy to track. **These numbers
are not real-match FPS, player-detection accuracy, CPU/GPU usage, memory
requirements, desktop capture FPS or end-to-end display latency**. Those
remain unmeasured until an actual Windows/browser test and representative
table-tennis footage are available. There are no current GPU requirements
for the lightweight optical-flow lane.

For a Windows desktop acceptance test: open a permitted browser match; select
a monitor with the drag overlay; run 2 minutes with one textured player ROI;
move and resize it; pause/resume; introduce a camera cut; confirm UNKNOWN
after tracking loss and cut; report actual FPS/latency and CPU/RAM usage.
A normal CI runner cannot certify actual desktop capture.

## Provenance: MotionSync -> Table Tennis Intelligence

Source reviewed at DanielCWeng/MotionSybc commit
27703a77cf93542bbe60db43de2a580e762c9868 (private, no
repository-level license found). This project **does not copy private source
verbatim**. Engineering concepts were reimplemented inside the public
destination, with explicit evidence semantics:

| MotionSync inspiration | Live adaptation |
| --- | --- |
| temporary desktop selector | Qt per-monitor desktop overlay + pixel transform |
| in-memory screen capture and timestamps | MSS capture + existing FramePacket |
| ROI and background reference | independent movable/resizable Qt scene items |
| sparse and dense directional flow | lightweight sparse LK residual tracking; dense intentionally deferred |
| reversal/cycle estimators | lateral direction-change *candidate* only; no invented shot/CPM |
| OpenCV dashboard + benchmark | Qt + PyQtGraph live diagnostics and independent CPU benchmark |
| controllers, device profiles, Bluetooth, actuation | **excluded entirely** |

Limitations inherited from research: camera pan cancellation can suppress
legitimate co-moving players; tracking may fail on low-texture clothing,
occlusion and camera cuts. Hard-cut detection does not recognise every dissolve.
2D image displacement cannot support real-world player speed or confirmed
stroke events. TOTNet offline performance on club footage is documented as
unreliable in docs/STATE.md; that is not silently promoted to Live Mode.

## Validation

Run all prior offline tests and new synthetic tests:

    python -m pytest -q

GitHub Actions runs the suite on headless Ubuntu and Windows; those runs validate
code pathways, NOT physical desktop interaction. See the PR for the actual CI
status and benchmark artifact. No local pretrained model is necessary.
