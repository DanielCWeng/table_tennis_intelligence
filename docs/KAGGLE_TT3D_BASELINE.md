# Kaggle TT3D baseline

Status: dataset upload succeeded externally and is ready at version 1. Kernel
version 1 missed the mount, and version 2 found the expanded mount but failed
on a wrapper ZIP-layout assumption. Version 3 was successfully submitted with
the corrected bundle discovery logic.

This is a Linux/GPU reference-run package for the canonical TT3D checkout. It
does not classify TT3D as Windows-incompatible. The earlier Windows result was
limited to the Codex runner's PyPI `WinError 10013` and read-only Conda
environment; native host Windows remains a separate possibility.

## 1. Exact upstream entry point

The TT3D README at commit
`a2ef524ea0400262d6808db6cacf4a0b90bd0ad7` identifies, from the repository
root:

```bash
python tt3d/calibration/segment.py ./data/calibration_test/videos/test_00.mp4 --display
python tt3d/calibration/calibrate.py ./data/calibration_test/videos/test_00.mp4 -o cam_cal.csv
```

The local runner invokes those same unmodified scripts with `--render` and
headless-safe environment variables. It copies each fixture only for the
calibration render because upstream writes that MP4 beside its input. It never
writes inside `third_party/tt3d`.

## 2. Required assets

The existing checkout already contains sufficient calibration-phase assets:

- `weights/table_segmentation.ckpt` (tracked upstream checkpoint)
- `data/calibration_test/videos/test_00.mp4`
- `data/calibration_test/videos/test_01.mp4`
- `data/calibration_test/videos/test_02.mp4`
- `data/calibration_test/` calibration/evaluation fixtures
- `tt3d/calibration/` source modules

No new checkpoint, dataset, LFS blob, RTMPose, MotionBERT, BlurBall, MMPose,
TOTNet, or human reconstruction package is included.

## 3. Minimum dependency set

The calibration import graph requires:

```text
torch (conditional Kaggle platform dependency)
torchvision (conditional Kaggle platform dependency)
albumentations==1.4.18
lightning==2.3.3
matplotlib==3.7.5
numpy==1.24.4
opencv-python==4.8.1.78
pandas==2.0.3
pillow==10.4.0
PyYAML==6.0.2
scikit-image==0.21.0
scipy==1.10.1
segmentation-models-pytorch==0.3.3
sympy==1.13.3
timm==0.9.2
tqdm==4.67.1
wandb==0.19.10
```

Some encoder/model and W&B packages are normal transitive dependencies. The
upstream `requirements.txt` entries for CasADi, pose/rally/model stacks, and
the optional Kalman filter are not needed by `segment.py` plus raw
`calibrate.py`. The exact conditional requirements file is
`kaggle/tt3d_baseline/requirements_kaggle.txt`.

## 4. Kaggle preflight and installation

The exact Kaggle image versions cannot be established from this Windows runner.
Run the included report before installation:

```bash
python /kaggle/working/tt3d_baseline/environment_report.py \
  --output /kaggle/working/tt3d_baseline/environment_before.json
```

Reuse a compatible preinstalled Torch/torchvision CUDA stack. Do not downgrade
it merely to match the upstream requirements. Install only missing or
incompatible entries from `requirements_kaggle.txt`, then save a second report:

```bash
python /kaggle/working/tt3d_baseline/environment_report.py \
  --output /kaggle/working/tt3d_baseline/environment_after_install.json
```

If Torch is genuinely absent or incompatible, the conditional fallback is
`torch==2.4.1` and `torchvision==0.19.1`; record the CUDA build actually
installed. No Kaggle package result is claimed yet.

## 5. Source transfer

Mode A, with Kaggle internet enabled:

```bash
git clone https://github.com/cogsys-tuebingen/tt3d.git /kaggle/working/tt3d
git -C /kaggle/working/tt3d checkout a2ef524ea0400262d6808db6cacf4a0b90bd0ad7
```

Mode B, with Kaggle internet disabled: upload the local archive created by the
PowerShell command in `kaggle/tt3d_baseline/README.md`. The archive contains
only `tt3d/`, `tt3d_baseline/`, and `TT3D_COMMIT.txt`.

Local TT3D working-tree footprint before compression: `76,780,547` bytes.
The archive recipe uses `git archive` at the pinned commit, so it excludes the
checkout's `.git` history and generated Python caches. The measured upload
archive is `69,512,346` bytes (`66.29 MiB`), SHA-256
`4A5E2F08ED544C8AD8527548BD23B9FF48DE2CAF465B9BBC28CCAB34276FA44B`. It contains the three test
videos, the tracked segmentation checkpoint, calibration/evaluation data, the
TT3D source, the five baseline files, and `TT3D_COMMIT.txt`.

## 6. Exact Kaggle run

After Mode A or Mode B places the source at `/kaggle/working/tt3d` and the
runner at `/kaggle/working/tt3d_baseline`:

```bash
python /kaggle/working/tt3d_baseline/run_tt3d_baseline.py \
  --tt3d-root /kaggle/working/tt3d \
  --output-dir /kaggle/working/tt3d_baseline
```

The runner prints Python/platform/Torch/CUDA/GPU/NumPy/SciPy/OpenCV/Lightning/
SMP/timm/CasADi information, verifies the expected commit and assets, captures
both streams for every upstream command, records runtime/exit code, and saves
the machine-readable report at:

```text
/kaggle/working/tt3d_baseline/tt3d_baseline_report.json
```

`test_01` and `test_02` run only if `test_00` passes the execution and numeric
checks. A process-only pass leaves report `success` false until visual review.

## 7. Expected outputs

For each executed video, expect:

- a segmentation render, `<video>_segmented.mp4`;
- a raw calibration CSV, `<video>_cam_cal.csv`;
- a calibration debug render, `calibration_inputs/<video>_calibrated.mp4`;
- complete `.stdout.txt` and `.stderr.txt` files;
- `environment_report.json` and `environment_after.json`;
- CSV focal length, rotation-vector, and translation-vector summaries.

The upstream raw CSV does not contain the internal `er` reprojection value
because `calibrate.py` calls `save_camcal` without passing `errors`. Detected
corners are drawn into debug frames but are not exported as machine-readable
coordinates. The report records both limitations explicitly.

## 8. Visual validation and success

Inspect the rendered files in Kaggle for stable table segmentation, correct
corners, projected table alignment, and physically plausible camera pose. Then:

```bash
python /kaggle/working/tt3d_baseline/verify_outputs.py \
  --report /kaggle/working/tt3d_baseline/tt3d_baseline_report.json \
  --visual-ok \
  --visual-note "Record the actual visual findings here."
```

Only that explicit visual approval can change the report's top-level `success`
to `true`. No Kaggle baseline success is claimed in this local preparation
pass.

## 9. Submitted dataset and kernel package

The prepared archive was uploaded successfully from an external PowerShell
session after the Codex runner's API socket restriction was bypassed. Kaggle
reported:

```text
Upload successful: tt3d-baseline-a2ef524ea0400262d6808db6cacf4a0b90bd0ad7.zip (66MB)
Private dataset: polyleviathan/tt3d-baseline-a2ef524
```

The invalid custom-tag warning was non-fatal. The dataset was still being
created when reported; its final version/status was not yet captured.

The local kernel package is:

```text
.tmp/kaggle_kernel_tt3d_baseline_20260808/
```

It contains `kernel-metadata.json` and `tt3d_baseline_kernel.py`. The metadata
  attaches `polyleviathan/tt3d-baseline-a2ef524`, enables GPU and internet, and
  submits kernel `polyleviathan/tt3d-baseline-reproduction-a2ef524`. Kaggle
  derived this title-based slug when version 1 was pushed; the initial metadata
  id ending in `-run` was not the actual notebook slug.

Submit it from the external Kaggle-enabled terminal:

```powershell
kaggle kernels push -p .tmp/kaggle_kernel_tt3d_baseline_20260808
kaggle kernels status polyleviathan/tt3d-baseline-reproduction-a2ef524
kaggle kernels logs polyleviathan/tt3d-baseline-reproduction-a2ef524
```

After completion, retrieve outputs into a timestamped directory:

```powershell
$run = '.tmp/kaggle_runs/tt3d_baseline_20260808'
New-Item -ItemType Directory -Force -Path $run | Out-Null
kaggle kernels output polyleviathan/tt3d-baseline-reproduction-a2ef524 -p $run
```

The wrapper verifies the archive hash and commit, writes the pre-install
environment report, installs only missing/import-failing non-Torch calibration
packages one at a time, records each pip stream, preserves the post-install
report, runs the prepared baseline runner, and runs automated verification
without `--visual-ok`.

## 10. Kernel version 1 preflight result

Kernel version 1 reached Kaggle Linux/Python 3.12.13 but ended with
`KernelWorkerStatus.ERROR` before environment inspection. The wrapper found no
ZIP file under `/kaggle/input`:

```text
FileNotFoundError: Could not uniquely locate
tt3d-baseline-a2ef524ea0400262d6808db6cacf4a0b90bd0ad7.zip under /kaggle/input;
candidates=[]
```

The complete retrieved log is
`.tmp/kaggle_runs/tt3d_baseline_20260808/tt3d-baseline-reproduction-a2ef524.log`.
No TT3D command, dependency installation, environment report, segmentation,
or calibration ran. The dataset is now confirmed ready at version 1. The
separate file-list request returned HTTP 403, but that listing is not required
because the expected ZIP name and SHA are already known. The kernel metadata
now pins the attachment explicitly to:

```text
polyleviathan/tt3d-baseline-a2ef524/1
```

Kernel version 2 reached the dataset mount, but failed before environment
inspection because Kaggle expanded the uploaded ZIP into a directory tree. The
wrapper only searched for a `.zip` file. Its input listing proved the bundle
was present at:

```text
/kaggle/input/datasets/polyleviathan/tt3d-baseline-a2ef524/
```

The exact failure was `candidates=[]`, despite the extracted
`TT3D_COMMIT.txt`, `tt3d/`, calibration videos, and evaluation data being
present. This is a wrapper packaging bug, not a TT3D failure. No environment
inspection, dependency installation, segmentation, or calibration ran.

The retrieved artifacts are:

```text
.tmp/kaggle_runs/tt3d_baseline_20260808_v2/tt3d-baseline-reproduction-a2ef524.log
.tmp/kaggle_runs/tt3d_baseline_20260808_v2/tt3d_baseline/kernel_wrapper_failure.json
```

The wrapper is now corrected to accept either a ZIP or Kaggle's expanded
dataset layout. Kernel version 3 was successfully pushed. Monitor it with:

```powershell
kaggle kernels status polyleviathan/tt3d-baseline-reproduction-a2ef524
kaggle kernels logs polyleviathan/tt3d-baseline-reproduction-a2ef524
```

After completion, retrieve outputs into a fresh directory:

```powershell
$run = '.tmp/kaggle_runs/tt3d_baseline_20260808_v3'
New-Item -ItemType Directory -Force -Path $run | Out-Null
kaggle kernels output polyleviathan/tt3d-baseline-reproduction-a2ef524 -p $run
```

The wrapper now writes `kernel_wrapper_failure.json` and lists all
`/kaggle/input` entries if another preflight failure occurs.

## 11. Actual Kaggle API attempts from the Codex runner

The installed CLI was `Kaggle CLI 2.2.2`. Authentication was refreshed
successfully as `polyleviathan`. The prepared upload directory
contained only `dataset-metadata.json` and the verified archive. The intended
dataset slug was `polyleviathan/tt3d-baseline-a2ef524`.

The upload command was attempted twice, including once after authentication
was refreshed:

```powershell
kaggle datasets create -p .tmp/kaggle_upload_tt3d_baseline -r skip
```

Both attempts exited `1` before dataset creation. Raw CLI output:

```text
HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url:
/v1/datasets.DatasetApiService/GetDatasetStatus (Caused by NewConnectionError(...:
Failed to establish a new connection: [WinError 10013] An attempt was made to
access a socket in a way forbidden by its access permissions'))
```

Full outputs are captured at
`.tmp/logs/kaggle_dataset_create_20260808.log` and
`.tmp/logs/kaggle_dataset_create_retry_20260808.log`. No dataset version,
kernel identifier, Kaggle environment, dependency installation, baseline
output, or downloaded artifact was available to the Codex runner from those
attempts. The later external upload and kernel version 1 submission succeeded;
the actual kernel status/log/output retrieval is still pending. This is not
evidence that TT3D is Windows-incompatible or that TT3D itself failed.

## 12. Remaining uncertainties

- Kaggle's live Torch/torchvision/CUDA versions and available GPU memory.
- Whether the bundled Lightning checkpoint loads without compatibility warnings
  on Kaggle's preinstalled Torch version.
- Runtime per video and whether OpenCV's MP4 writer is available in the image.
- The visual quality of the unmodified segmentation and calibration renders.
- Per-frame reprojection error and corner coordinates remain unavailable from
  the upstream CLI unless the upstream output itself exposes them.
- Final Kaggle dataset status/version, live package versions, GPU, runtimes,
  and all baseline outputs remain unknown until kernel version 2 runs with the
  dataset mounted.
