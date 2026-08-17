# Third-party source audit

Audit date: 2026-08-08.

Scope: clone and inspect the 27 public repositories listed in
.tmp/FollowUp.txt. This pass intentionally did not install dependencies,
create environments, fetch submodules, run model download scripts, download
datasets, or fetch Git LFS blobs.

## 1. Clone result

All requested repositories are present and have clean tracked worktrees.

- Successful new clones: 26.
- Already-valid clone reused: TT3D.
- Clone failures: 0.
- Redirected, dead, or substituted URLs: none observed.
- New clones used depth 1 with GIT_LFS_SKIP_SMUDGE=1.
- TT3D was cloned earlier with full history and is not shallow.
- Full commit, branch, license, and per-repository findings are in
  docs/MODEL_MATRIX.md.

Successful targets:

| Component | Local target |
| --- | --- |
| TT3D | third_party/tt3d |
| TTNet | third_party/ttnet |
| TrackNetV3 | third_party/tracknetv3 |
| TOTNet | third_party/totnet |
| RacketVision | third_party/racketvision |
| MMPose | third_party/mmpose |
| RTMLib | third_party/rtmlib |
| MotionBERT | third_party/motionbert |
| 4DHumans | third_party/4dhumans |
| WHAM | third_party/wham |
| WHAC | third_party/whac |
| SMPLest-X | third_party/smplest-x |
| SAM 3D Body | third_party/sam-3d-body |
| MHR | third_party/mhr |
| SMPL-X | third_party/smplx |
| SMPL-Anthropometry | third_party/smpl-anthropometry |
| CoTracker | third_party/co-tracker |
| TAPNet/TAPIR/TAPNext | third_party/tapnet |
| SAM 2 | third_party/sam2 |
| Grounded SAM 2 | third_party/grounded-sam-2 |
| Cutie | third_party/cutie |
| XMem | third_party/xmem |
| FastReID | third_party/fast-reid |
| CLIP-ReID | third_party/clip-reid |
| TransNetV2 | third_party/transnetv2 |
| PySceneDetect | third_party/pyscenedetect |
| ST-GCN | third_party/st-gcn |

## 2. Storage and asset observations

The current third_party tree occupies 3,251,123,670 bytes, approximately
3.03 GiB, including Git metadata. TOTNet accounts for approximately 1.97 GiB
including its Git metadata.

No checkpoint or dataset downloader was invoked. Two ordinary-Git asset cases
need to be kept visible:

- TT3D was already cloned before this audit and contains its upstream-tracked
  table_segmentation.ckpt, about 65 MiB.
- TOTNet's requested shallow commit contains upstream-tracked .pth files,
  about 1.0 GiB in the working tree. GIT_LFS_SKIP_SMUDGE cannot omit ordinary
  Git blobs. Do not use those weights until their terms and the storage policy
  are approved.

TransNetV2 is the only repository with Git LFS markers. Its three tracked model
files remain pointer files because smudge was disabled. No Git LFS blob was
fetched.

## 3. Submodules

Submodules were inspected but not fetched.

WHAM declares:

- third-party/DPVO from https://github.com/princeton-vl/DPVO.git
- third-party/ViTPose from https://github.com/ViTAE-Transformer/ViTPose.git

WHAC declares:

- third_party/DPVO from https://github.com/princeton-vl/DPVO.git
- third_party/SMPLest-X from https://github.com/wqyin/SMPLest-X.git

The requested WHAC URL is MotrixLab/WHAC and the requested standalone SMPLest-X
URL is MotrixLab/SMPLest-X. The WHAC submodule points at a different
wqyin/SMPLest-X repository; this was recorded and not silently substituted.

## 4. Repositories likely runnable natively on Windows

Best native candidates:

- RTMLib: NumPy, OpenCV, and ONNX Runtime; CPU inference is possible.
- PySceneDetect: explicitly supports Windows and has Windows packaging.
- TrackNetV3: source is simple enough for a legacy isolated environment,
  although its documented stack is old and its checkpoint is external.
- SMPL-X and SMPL-Anthropometry: source-level utilities can run on Windows,
  but licensed SMPL-family model files are still required.
- CoTracker and Cutie: mostly Python/PyTorch, with GPU and asset caveats.
- TransNetV2 PyTorch inference: possible after acquiring the LFS checkpoint and
  resolving FFmpeg/runtime details.

TT3D is source-level plausible on Windows, but its 2026-08-08 sample attempt
could not install even the Torch pair in `tti-tt3d`: this runner cannot reach
PyPI and has read-only access to the Conda environment. Keep it isolated and
use Linux/Kaggle for the first reproducible baseline unless a writable native
Windows execution context is provided.

MMPose should remain isolated because of its pinned OpenMMLab/native-extension
and external model dependencies.

## 5. Repositories better under WSL/Linux

WHAM, WHAC, SMPLest-X, 4DHumans, SAM 3D Body, MHR, SAM 2, Grounded SAM 2,
FastReID, CLIP-ReID, XMem, TAPNet, and ST-GCN are Linux/WSL-first in their
documentation or build behavior. The main reasons are CUDA extensions,
Detectron2, PyTorch3D, DPVO, OSMesa/OpenGL, JAX/TensorFlow CUDA support, FAISS,
old shell scripts, or Linux package-manager commands.

TTNet and TOTNet are also safer under WSL/Linux for their documented
PyTurboJPEG, NCCL, Cython/native-op, and multi-GPU paths.

## 6. Obvious dependency conflicts

| Cluster | Observed requirements | Consequence |
| --- | --- | --- |
| Core TT3D/TOTNet | Torch 2.4.1; TOTNet requests CUDA 11.8 wheels and native build tools | Keep the existing tti-tt3d and tti-totnet environments separate until tested. |
| Direct pose | MMPose current OpenMMLab stack versus RTMLib ONNX Runtime | Prefer RTMLib for the light Windows path; use tti-mmpose for RTMPose/MMPose. |
| Legacy pose | MotionBERT documents Python 3.7 and CUDA 11.6 | Do not mix it into a current MMPose environment without a pinned compatibility test. |
| Mesh recovery | 4DHumans Python 3.10/CUDA 11.8; WHAM Python 3.9/CUDA 11.3; WHAC and SMPLest-X Python 3.8/CUDA 11.3 | Keep each research stack isolated; do not combine them with SAM 2. |
| New segmentation | SAM 2 requires Python >=3.10 and Torch >=2.5.1, with optional nvcc extension | It conflicts with the older motion and mesh environments. |
| Point tracking | TAPNet is primarily JAX/TensorFlow; CoTracker is PyTorch | Use separate environments if both are evaluated. |
| Legacy action/ReID | ST-GCN uses Torch 0.4; CLIP-ReID uses Torch 1.8/CUDA 10.2; FastReID uses Torch >=1.6 | These are reference-only until a dedicated legacy environment is justified. |

## 7. Licensing and restricted assets

- SMPL-X is explicitly restricted to non-commercial scientific research,
  non-commercial education, or non-commercial artistic projects. Commercial,
  military, surveillance, and redistribution uses are prohibited by the
  checked-in license without separate permission.
- SMPLest-X and WHAC use S-Lab License 1.0, which is non-commercial and asks
  users to contact contributors for commercial use.
- CoTracker's checked-in license is CC BY-NC 4.0. It cannot be treated as a
  commercially permissive dependency.
- SAM 3D Body uses a custom Meta SAM License and its checkpoints require
  Hugging Face access approval. MHR model assets are separate from the
  Apache-2.0 source license.
- TOTNet's TTA dataset agreement says research-only use and prohibits
  commercial use and redistribution. The source license does not override
  those dataset terms.
- RacketVision, MMPose model-zoo entries, Grounded SAM 2 components, CLIP
  weights, ReID datasets, VOS datasets, and hosted checkpoints all need
  asset-level review even where the source repository is MIT or Apache-2.0.
- TT3D and TTNet have no license file in the checked-out source. Treat them as
  uncleared for redistribution or commercial integration.

## 8. Recommended environment-to-repository mapping

Use the existing environments without installing into them during this pass:

- tti-mmpose: MMPose and the RTMPose path; consider RacketVision's RacketPose
  only after its model and dataset terms are cleared.
- tti-tt3d: TT3D source and its adapter boundary. MotionBERT should be added
  only after its legacy version requirements are separately validated.
- tti-totnet: TOTNet only. TrackNetV3 is a different Torch generation and
  should not be installed into this environment by assumption.
- tti-4dhumans: 4DHumans and compatible SMPL-X API code, without downloading
  body models or HMR checkpoints.

## TT3D calibration baseline status

The TT3D checkout is present at commit
`a2ef524ea0400262d6808db6cacf4a0b90bd0ad7` on `main`, including its tracked
table segmentation checkpoint and `test_00.mp4` through `test_02.mp4` fixtures.
The upstream README commands are:

```powershell
python tt3d/calibration/segment.py ./data/calibration_test/videos/test_00.mp4 --display
python tt3d/calibration/calibrate.py ./data/calibration_test/videos/test_00.mp4 -o cam_cal.csv
```

The raw calibration subset consists of Torch/torchvision, the Lightning/SMP
segmentation stack, Albumentations, NumPy, OpenCV, Pillow, SciPy, pandas,
Matplotlib, tqdm, SymPy, PyYAML, and the normal resolver dependencies. The
full TT3D requirements additionally include CasADi, the optional filter's
`pykalman`, and research pose/rally/model dependencies that are not imported by
this raw path.

On 2026-08-08, `conda run -n tti-tt3d python --version` verified Python 3.10.20.
The first verbose, binary-only install stage for `torch==2.4.1` and
`torchvision==0.19.1` failed with `WinError 10013` opening PyPI. Pip also
reported that the environment site-packages was not writable and proposed a
user install; that fallback was deliberately not used. No package, checkpoint,
dataset, source file, or other Conda environment was changed. No TT3D sample
was run, so there are no calibration values, error metrics, masks, CSVs, YAMLs,
or rendered videos to validate. See the combined log at
`.tmp/logs/tt3d_install_torch_20260808.log`.

Minimal Linux/Kaggle reproduction plan: create Python 3.10 with the pinned
calibration subset, disable W&B and GUI display (`WANDB_MODE=disabled`,
`MPLBACKEND=Agg`), check out the exact TT3D commit, use only the already-present
checkpoint and fixtures, run the two upstream commands from the repository root,
and save outputs under the platform working directory. Run `test_01.mp4` and
`test_02.mp4` only after `test_00.mp4` produces a valid upstream CSV.

Later, if justified, create isolated environments for:

- native video and ONNX utilities: RTMLib, PySceneDetect, and possibly
  TransNetV2 PyTorch inference;
- segmentation: SAM 2 and Grounded SAM 2;
- Linux mesh research: WHAM, WHAC, SMPLest-X, or SAM 3D Body;
- legacy comparisons: TTNet, MotionBERT, ST-GCN, FastReID, or CLIP-ReID.

## 9. Recommended first five smoke tests

These are recommendations for the next pass, not tests run here:

1. RTMLib source import and CPU/API smoke test, after selecting an ONNX model.
2. PySceneDetect on a small existing video fixture.
3. TT3D source/config smoke test in tti-tt3d using its already-present assets.
4. MMPose RTMPose configuration/import smoke test in tti-mmpose.
5. TrackNetV3 source/config smoke test after separately approving its external
   checkpoint and shuttlecock dataset assumptions.

No environment or model smoke test was run during this audit.
