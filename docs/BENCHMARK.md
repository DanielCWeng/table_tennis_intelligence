# Benchmark Harness

The benchmark API lives in `ttintel.benchmark`. It is designed around a small
manually labelled gold set rather than synthetic confidence claims.

## Gold-set shape

Start with 5–10 short clips covering:

- more than one camera position;
- broadcast and training footage;
- different video quality;
- temporary player/ball occlusion;
- selected frame-level ball points and event timestamps.

Do not store the videos or model weights in this repository. Keep local gold
labels under an ignored session/fixture directory.

## Metrics currently available

| Target | Metrics |
| --- | --- |
| Ball | visible recall, mean pixel error, false positives, longest missed sequence |
| Events | precision, recall, mean timing error |
| Players/table | schema supports availability/confidence; add ID-switch and reprojection labels next |

The evaluator uses tolerances explicitly and reports candidate metrics, not
ground truth beyond the supplied labels. Model outputs should be evaluated
separately before arbitration; disagreement must not be hidden.
