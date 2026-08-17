# TT3D Kaggle baseline

This package runs the original TT3D calibration/segmentation scripts without
editing `third_party/tt3d`. It is pinned to:

```text
TT3D commit: a2ef524ea0400262d6808db6cacf4a0b90bd0ad7
branch: main
```

The baseline is not considered successful from process exit codes alone. The
runner records numeric/file success first; `verify_outputs.py --visual-ok`
must be run only after the rendered videos show correct table masks, corners,
projected table geometry, and a plausible camera pose.

## Upstream entry points

From the TT3D repository root, the README's calibration path is:

```bash
python tt3d/calibration/segment.py ./data/calibration_test/videos/test_00.mp4 --display
python tt3d/calibration/calibrate.py ./data/calibration_test/videos/test_00.mp4 -o cam_cal.csv
```

The Kaggle runner uses the same unmodified scripts with headless-safe options:

- `segment.py --render --output <working-output>.mp4`
- `calibrate.py <byte-for-byte fixture copy> -o <working-output>.csv --render`

The calibration input copy is necessary because upstream `calibrate.py` writes
its `--render` MP4 beside the input video. The TT3D checkout itself is never
written.

## Required assets

The source bundle must contain only the canonical TT3D checkout and this
baseline package. The checkout already contains everything needed for this
phase:

```text
tt3d/weights/table_segmentation.ckpt
tt3d/data/calibration_test/videos/test_00.mp4
tt3d/data/calibration_test/videos/test_01.mp4
tt3d/data/calibration_test/videos/test_02.mp4
tt3d/data/calibration_test/...
tt3d/tt3d/calibration/...
```

No RTMPose, MotionBERT, BlurBall, human model, rally model, dataset, or new
checkpoint is needed.

## Dependencies

Run the environment report before installing anything:

```bash
python /kaggle/working/tt3d_baseline/environment_report.py \
  --output /kaggle/working/tt3d_baseline/environment_before.json
```

The minimum calibration imports are Torch/torchvision, Lightning,
segmentation-models-pytorch, timm, Albumentations, NumPy, OpenCV, Pillow,
SciPy, pandas, Matplotlib, tqdm, SymPy, PyYAML, and W&B. The exact non-platform
pins are in `requirements_kaggle.txt`. CasADi, pykalman, pose dependencies,
rally dependencies, and human reconstruction packages are deliberately absent.

Do not downgrade a compatible Kaggle Torch/CUDA stack. Install only entries
that are missing or incompatible after reviewing `environment_before.json`.
Torch fallback pins, only if the Kaggle runtime lacks a compatible pair, are:

```text
torch==2.4.1
torchvision==0.19.1
```

If installation is needed, keep the complete pip output in the Kaggle notebook
and save the post-install report:

```bash
python /kaggle/working/tt3d_baseline/environment_report.py \
  --output /kaggle/working/tt3d_baseline/environment_after_install.json
```

## Mode A: Kaggle internet enabled

Clone only the canonical TT3D repository and check out the exact commit:

```bash
git clone https://github.com/cogsys-tuebingen/tt3d.git /kaggle/working/tt3d
git -C /kaggle/working/tt3d checkout a2ef524ea0400262d6808db6cacf4a0b90bd0ad7
```

Make `kaggle/tt3d_baseline` available at
`/kaggle/working/tt3d_baseline`, then run:

```bash
python /kaggle/working/tt3d_baseline/run_tt3d_baseline.py \
  --tt3d-root /kaggle/working/tt3d \
  --output-dir /kaggle/working/tt3d_baseline
```

The runner automatically starts `test_01` and `test_02` only when `test_00`
passes its execution and numeric checks.

## Mode B: offline source bundle

Upload the archive produced by the documented local command below as a Kaggle
dataset. Extract it into `/kaggle/working` so it provides:

```text
/kaggle/working/tt3d/
/kaggle/working/tt3d_baseline/
/kaggle/working/TT3D_COMMIT.txt
```

Then run the same `run_tt3d_baseline.py` command above. The commit marker is
used when the upload bundle does not contain `.git` metadata.

### Exact local archive command

Run from the project root in PowerShell. It creates a self-contained zip with
only the TT3D checkout, the baseline scripts, and the commit marker; it does
not alter `third_party/tt3d`:

```powershell
$stage = Join-Path (Resolve-Path .tmp).Path 'tt3d_baseline_upload_stage'
$repo = (Resolve-Path third_party\tt3d).Path
$inner = Join-Path (Resolve-Path .tmp).Path 'tt3d-source-a2ef524ea0400262d6808db6cacf4a0b90bd0ad7.zip'
$archive = Join-Path (Resolve-Path .tmp).Path 'tt3d-baseline-a2ef524ea0400262d6808db6cacf4a0b90bd0ad7.zip'
$marker = Join-Path (Resolve-Path .tmp).Path 'TT3D_COMMIT.txt'
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
git -c "safe.directory=$repo" -C third_party\tt3d archive --format=zip --output=$inner a2ef524ea0400262d6808db6cacf4a0b90bd0ad7
Expand-Archive -LiteralPath $inner -DestinationPath (Join-Path $stage 'tt3d') -Force
if (Test-Path -LiteralPath (Join-Path $stage 'tt3d\.git')) { throw 'Unexpected .git directory in source bundle' }
New-Item -ItemType Directory -Force -Path (Join-Path $stage 'tt3d_baseline') | Out-Null
$packageFiles = @('environment_report.py','README.md','requirements_kaggle.txt','run_tt3d_baseline.py','verify_outputs.py')
foreach ($name in $packageFiles) { Copy-Item -LiteralPath (Join-Path (Resolve-Path kaggle\tt3d_baseline).Path $name) -Destination (Join-Path $stage 'tt3d_baseline') -Force }
Set-Content -LiteralPath $marker -Value 'a2ef524ea0400262d6808db6cacf4a0b90bd0ad7' -NoNewline
Compress-Archive -Path (Join-Path $stage 'tt3d'), (Join-Path $stage 'tt3d_baseline'), $marker -DestinationPath $archive -Force
Get-Item -LiteralPath $archive | Select-Object FullName,Length
```

The archive size is recorded in `docs/KAGGLE_TT3D_BASELINE.md` after local
creation. Do not add unrelated `third_party` repositories to it.

## Outputs

All output goes under `/kaggle/working/tt3d_baseline/`:

```text
tt3d_baseline_report.json
environment_report.json
environment_after.json
test_00_segmented.mp4
test_00_cam_cal.csv
calibration_inputs/test_00_calibrated.mp4
test_00_segment.stdout.txt
test_00_segment.stderr.txt
test_00_calibrate.stdout.txt
test_00_calibrate.stderr.txt
```

If `test_00` passes, the corresponding `test_01` and `test_02` artifacts are
added. The raw CSV exposes focal length, rotation vector, and translation
vector rows. TT3D's upstream `calibrate.py` does not pass its internal `er`
value to `save_camcal`, so the report marks reprojection error as unavailable
unless the upstream output itself exposes it.

## Visual approval

Review the rendered MP4s in Kaggle. Check table segmentation, corner alignment,
projected table geometry, and camera pose. Then record the human decision:

```bash
python /kaggle/working/tt3d_baseline/verify_outputs.py \
  --report /kaggle/working/tt3d_baseline/tt3d_baseline_report.json \
  --visual-ok \
  --visual-note "Reviewed renders; geometry is plausible."
```

Before that command, `success` remains `false` with
`visual_validation.status = pending_manual_review` by design.
