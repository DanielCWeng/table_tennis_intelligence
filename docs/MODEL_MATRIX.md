# Third-party model matrix

Audit date: 2026-08-08.

This matrix covers the 27 canonical repositories requested in .tmp/FollowUp.txt.
All new clones were made with depth 1 and GIT_LFS_SKIP_SMUDGE=1. TT3D was already
present and is a full checkout. No dependency was installed, no environment was
changed, no submodule was fetched, and no checkpoint or dataset download script
was run.

The license column describes the checked-in source license only. Checkpoints,
datasets, SMPL-family files, and hosted model assets can have separate terms.
Windows ratings describe likely integration feasibility, not a completed test.

| Component | Repository | Local path | Commit SHA | Branch | License | Language/runtime | Environment files | Python expectation | PyTorch expectation | CUDA expectation | Windows feasibility | WSL recommended? | Checkpoints required? | Datasets required? | SMPL/external licence required? | Submodules? | LFS? | Priority | Notes / blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TT3D | [cogsys-tuebingen/tt3d](https://github.com/cogsys-tuebingen/tt3d) | third_party/tt3d | a2ef524ea0400262d6808db6cacf4a0b90bd0ad7 | main | No license file found | Python, PyTorch, OpenCV, CasADi, Lightning | requirements.txt | Python 3.10.20 verified in `tti-tt3d` | Torch 2.4.1, torchvision 0.19.1 | GPU useful; no explicit CUDA extension | Medium upstream; blocked in this runner | Optional | Yes: tracked table segmentation checkpoint; external RTMPose, MotionBERT, BlurBall | Included calibration/evaluation fixtures; rally video and pose/ball inputs needed | RTMPose, MotionBERT, BlurBall terms | No | No | P0 | Core monocular table/camera/3D reconstruction. The 2026-08-08 calibration-only install was not completed: PyPI socket access was denied and the environment is read-only to this runner. No TT3D command or output is claimed. |
| TTNet | [maudzung/TTNet](https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch) | third_party/ttnet | a7b8430e9f3da69bbb7cde9fccea77800c9ceb00 | master | No license file found | Python, legacy PyTorch, OpenCV, PyTurboJPEG | requirement.txt | Not pinned; legacy stack | Torch 1.5.0, torchvision 0.6.0 | GPU and NCCL for multi-GPU training | Low | Yes | Optional pretrained_path; no upstream checkpoint found | TT table-tennis dataset required for training/evaluation | PyTurboJPEG and dataset terms | No | No | P1 | README asks for libturbojpeg via apt and NCCL for distributed training; source is useful as a dataset/event reference. |
| TrackNetV3 | [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) | third_party/tracknetv3 | 77c123ad4dd449b7d275f16cc43f316ba5b54042 | master | MIT | Python, PyTorch, OpenCV, Plotly | requirements.txt | 3.8.7 in the documented environment | Torch 1.10.0 | GPU optional; no obvious custom extension | Medium | Optional | Yes: external Google Drive checkpoints | Shuttlecock Trajectory Dataset; not table-tennis native | Dataset and checkpoint terms | No | No | P1 | Strong temporal ball-tracking reference, but its published task is shuttlecock tracking and its environment is old. |
| TOTNet | [AugustRushG/TOTNet](https://github.com/AugustRushG/TOTNet) | third_party/totnet | 8a757f63391b262c14d18b4095486336852dbeef | main | MIT source; TTA agreement separate | Python, PyTorch, OpenCV, Cython, custom ops | requirements.txt, src/model/ops/setup.py | 3.10 | Torch 2.4.1 with cu118 index | CUDA 11.8, NCCL commands, Cython/ninja build | Low | Yes | Yes: 12 upstream .pth files are present, about 1.0 GiB total | Tennis, badminton, TT, and TTA datasets | TTA is research-only; commercial use and redistribution prohibited | No | No | P1 | Table-tennis-capable tracker, but upstream-tracked weights were included by ordinary Git; this is the largest checkout at about 1.97 GiB including .git. |
| RacketVision | [OrcustD/RacketVision](https://github.com/OrcustD/RacketVision) | third_party/racketvision | c44af2a08524d3cb54d818f19686f4cdea4d2793 | main | MIT source; hosted data/model terms separate | Python, PyTorch, MMEngine, MMDetection, MMPose | README setup only; download_checkpoints.py | 3.10 | Torch 2.1.2, torchvision 0.16.2 | CUDA 12.1 in documented environment | Medium | Optional | Yes: Hugging Face models | Yes: Hugging Face RacketVision dataset | Hugging Face dataset/model terms; MMPose/MMCV terms | No | No | P1 | Useful ball, racket-pose, and trajectory benchmark. Dataset and checkpoints are deliberately not downloaded. |
| MMPose | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | third_party/mmpose | 759b39c13fea6ba094afc1fa932f51dc1b11cbf9 | main | Apache-2.0; special algorithms listed in LICENSES.md | Python, PyTorch, OpenMMLab stack | requirements.txt, requirements tree, setup.py, setup.cfg, Dockerfiles | Current branch is not pinned in checkout docs | Torch >=1.8 plus MMEngine/MMCV ecosystem | GPU optional; MMCV/native ops may need compiler/CUDA | Medium | Recommended for full stack | Yes: model-zoo checkpoints for inference | Only for training/evaluation | Model-zoo and special-algorithm terms; EDPose is separately listed | No | No | P0 | Direct RTMPose dependency for the core path. Use the existing tti-mmpose environment; avoid mixing with legacy stacks. |
| RTMLib | [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) | third_party/rtmlib | 03a1693e59e4f7cd84582c0fb30459b3bf18ad42 | main | Apache-2.0 | Python, NumPy, OpenCV, ONNX Runtime | requirements.txt, pyproject.toml | >=3.10 | None required; ONNX models | CPU works; ONNX Runtime GPU/OpenVINO/TensorRT optional | High | No | Yes: external ONNX model archives from the model zoo | No for inference | External ONNX model terms | No | No | P0 | Smallest practical pose adapter and the best native-Windows-first smoke candidate. |
| MotionBERT | [Walter0807/MotionBERT](https://github.com/Walter0807/MotionBERT) | third_party/motionbert | 705d3a95354db8bdb696b3492e47a3b5537174ff | main | Apache-2.0 | Python, PyTorch, NumPy, SMPL-X optional | requirements.txt | 3.7 in documented setup | PyTorch installed for CUDA 11.6 | CUDA 11.6 in documented setup | Low | Recommended | Yes: pretrained MotionBERT weights | 2D pose input for inference; H36M/AMASS and others for training | Checkpoint and optional SMPL-X/data terms | No | No | P0 | TT3D uses it for 2D-to-3D lifting. Python and CUDA age make a dedicated environment necessary. |
| 4DHumans | [shubham-goel/4D-Humans](https://github.com/shubham-goel/4D-Humans) | third_party/4dhumans | efe18deff163b29dff87ddbd575fa29b716a356c | main | MIT source; SMPL assets separate | Python, PyTorch, Detectron2, SMPL-X, HMR2 | environment.yml, setup.py | 3.10 | PyTorch plus Lightning and Detectron2 | CUDA 11.8 in environment.yml | Low | Yes | Yes: HMR2 and detector checkpoints | Input images/video; training/evaluation datasets optional | SMPL/SMPL-X model files and related terms | No | No | P1 | Good heavier alternative to MotionBERT. Detectron2, rendering, and SMPL assets are the main Windows blockers. |
| WHAM | [yohanshin/WHAM](https://github.com/yohanshin/WHAM) | third_party/wham | 2b54f7797391c94876848b905ed875b154c4a295 | main | MIT source; SMPL assets separate | Python, PyTorch, PyTorch3D, DPVO, ViTPose | requirements.txt, docs/INSTALL.md, Docker guidance | 3.9 on Ubuntu 20.04/22.04 | Torch 1.11.0, torchvision 0.12.0 | CUDA 11.3; DPVO and PyTorch3D native builds | Low | Yes | Yes: fetch_demo_data.sh and evaluation checkpoints | Videos; AMASS, 3DPW, RICH, EMDB for training/evaluation | SMPL/SMPLify registration and external model terms | Yes: DPVO and ViTPose, not fetched | No | P3 | Linux-first world-grounded motion stack with multiple native extensions and registered body models. |
| WHAC | [MotrixLab/WHAC](https://github.com/MotrixLab/WHAC) | third_party/whac | ef18d127477d77678544c11c40e98e1d11eace3d | main | S-Lab License 1.0, non-commercial | Python, PyTorch, PyTorch3D, DPVO, SMPL-X | requirements.txt, scripts/install.sh, .gitmodules | 3.8 in install.sh | Torch 1.12.0, torchvision 0.13.0 | CUDA 11.3, DPVO/PyTorch3D, apt/OSMesa | Low | Yes | Yes: WHAC checkpoint plus DPVO and SMPLest-X assets | WHAC-A-Mole and demo inputs | S-Lab non-commercial; SMPLest-X/SMPL-X/DPVO terms | Yes: DPVO and SMPLest-X, not fetched | No | P3 | Strong but restricted and Linux-only in practice. The requested repo and its SMPLest-X submodule URL differ; no submodule was fetched. |
| SMPLest-X | [MotrixLab/SMPLest-X](https://github.com/MotrixLab/SMPLest-X) | third_party/smplest-x | fdebd887a317f9004b435c57812d1a8936295360 | main | S-Lab License 1.0, non-commercial | Python, PyTorch, SMPL-X, custom mesh pipeline | requirements.txt, scripts/install.sh | 3.8 in install.sh | Torch 1.12.0, torchvision 0.13.0 | CUDA 11.3; OSMesa/OpenGL support | Low | Yes | Yes: 8.2G Hugging Face model plus ViTPose/YOLO assets | HumanData for training; input video for inference | S-Lab non-commercial; SMPL/SMPL-X terms | No | No | P3 | Large expressive model; do not request its checkpoint or human-model assets until licensing is approved. |
| SAM 3D Body | [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body) | third_party/sam-3d-body | b5c765a0d89d789985e186d396315e7590887b94 | main | Custom Meta SAM License | Python, PyTorch, Detectron2, Hydra, MHR | INSTALL.md only; data setup README | 3.11 for inference; data guide uses 3.9 | PyTorch required; data guide uses 2.4.0 | GPU/CUDA expected for practical inference and Detectron2 | Low | Yes | Yes: access-requested Hugging Face checkpoints | Images for inference; many datasets for training/data prep | Custom SAM License and MHR assets; review before commercial use | No | No | P2 | Single-image mesh recovery, not temporal replacement. Checkpoint access requires Hugging Face approval; nothing was requested. |
| MHR | [facebookresearch/MHR](https://github.com/facebookresearch/MHR) | third_party/mhr | 5deda6a69f30019915ded6299c7aab74acaca57f | main | Apache-2.0 source; model assets separate | Python, PyTorch, PyMomentum, trimesh | pyproject.toml with Pixi config | >=3.11 | PyTorch and PyMomentum >=0.1.90 | GPU optional; model assets required | Low | Yes | Yes: mhr-download-assets | No training dataset for basic use | MHR model asset terms; conversion assets included in source | No | No | P2 | Pixi declares linux-64 and osx-arm64, not Windows. Use only after asset licensing is clear. |
| SMPL-X | [vchoutas/smplx](https://github.com/vchoutas/smplx) | third_party/smplx | 1265df7ba545e8b00f72e7c557c766e15c71632f | main | Custom non-commercial scientific-research license | Python, PyTorch, SMPL family | requirements.txt, setup.py, transfer_model/requirements.txt | Not pinned | Torch >=1.0.1 | No explicit CUDA requirement | Medium | Optional | No neural checkpoint; model files are required | Optional sample/transfer data | Yes: SMPL/SMPL-X license is non-commercial and prohibits commercial, military, surveillance, and redistribution uses | No | No | P1 | Foundational morphology dependency, but its license is a hard blocker for commercial product use. |
| SMPL-Anthropometry | [DavidBoja/SMPL-Anthropometry](https://github.com/DavidBoja/SMPL-Anthropometry) | third_party/smpl-anthropometry | d8f43bca7a2ae6263f43ebb0efe74136e37e25eb | master | MIT source; body models separate | Python, PyTorch, NumPy, trimesh, Plotly | docker/requirements.txt, Dockerfile | Not pinned; legacy Docker stack | Torch 1.6.0 | No explicit CUDA requirement | Medium via Docker/WSL | Optional | No | No dataset; requires supplied SMPL or SMPL-X model files | Yes: SMPL/SMPL-X model license | No | No | P2 | Useful measurement utility, but it cannot run meaningfully without licensed body model files. |
| CoTracker | [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) | third_party/co-tracker | 82e02e8029753ad4ef13cf06be7f4fc5facdda4d | main | CC BY-NC 4.0 | Python, PyTorch, TorchVision | setup.py, gradio_demo/requirements.txt | Not pinned | PyTorch and TorchVision required | GPU strongly recommended; CPU possible | Medium | Optional | Yes: Torch Hub/pretrained model | TAP-Vid/Kubric for evaluation/training; input video for use | Checkpoint terms plus source is non-commercial | No | No | P2 | Good point-repair candidate, but CC BY-NC prevents assuming commercial integration. |
| TAPNet/TAPIR/TAPNext | [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet) | third_party/tapnet | c2cbab81cc06092b5f05bfe2da7bfec54e2079c9 | main | Apache-2.0 source | Python, JAX, TensorFlow, Haiku; optional PyTorch | requirements.txt, requirements_inference.txt, pyproject.toml | Not pinned | Optional torch/torchvision; primary stack is JAX/TensorFlow | CUDA depends on JAX/CUDA compatibility | Low | Yes | Yes: GCS TAPIR/TAPNext checkpoints | TAP-Vid, RoboTAP, TAPVid-3D, Kubric for benchmarks/training | Dataset and checkpoint terms | No | No | P3 | Powerful but a separate JAX/TensorFlow stack; keep isolated from PyTorch environments. |
| SAM 2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | third_party/sam2 | 2b90b9f5ceec907a1c18123530e92e794ad901a4 | main | Apache-2.0 with BSD and dataset notices | Python, PyTorch, TorchVision | pyproject.toml, setup.py, INSTALL.md | >=3.10 | Torch >=2.5.1, torchvision >=0.20.1 | CUDA toolkit/nvcc for optional custom kernel | Low | Yes | Yes: SAM 2.1 checkpoints | Video/image input; SA-V only for training | Meta checkpoint/data terms and bundled notices | No | No | P2 | README explicitly recommends WSL on Windows; CUDA extension and torch version conflict with legacy environments. |
| Grounded SAM 2 | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | third_party/grounded-sam-2 | b7a9c29f196edff0eb54dbe14588d7ae5e3dde28 | main | Apache-2.0 composite with SAM2/GroundingDINO/BSD notices | Python, PyTorch, SAM2, GroundingDINO, supervision | pyproject.toml, Dockerfile, grounding_dino/environment.yaml and requirements | Not pinned; follows component docs | PyTorch/SAM2 plus GroundingDINO | GPU/CUDA expected for practical demos and custom ops | Low | Yes | Yes: SAM2 and GroundingDINO or hosted API models | Optional demo/training datasets; video input | Composite component and model/API terms | No | No | P2 | Heavy open-world fallback; treat each bundled component and hosted model as separately licensed. |
| Cutie | [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie) | third_party/cutie | ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a | main | MIT source; model/data terms separate | Python, PyTorch, OpenCV, Cython, Gradio | pyproject.toml | >=3.8 | PyTorch required but not pinned in root metadata | GPU useful; no obvious required CUDA extension | Medium | Optional | Yes: download_models.py | VOS datasets for training; input video for inference | Checkpoint and VOS dataset terms | No | No | P2 | Practical long-term segmentation candidate; source is Windows-friendly but inference assets remain external. |
| XMem | [hkchengrex/XMem](https://github.com/hkchengrex/XMem) | third_party/xmem | f3b841d50df058910bbf690229ddc15fb1aef7d6 | main | MIT source; model/data terms separate | Python, PyTorch, Cython, fBRS | requirements.txt, requirements_demo.txt | Not pinned; older stack | PyTorch required | GPU useful; fBRS/Cython build complicates Windows | Low | Recommended | Yes: shell download scripts | VOS datasets for training/evaluation; input video for inference | Checkpoint and VOS dataset terms | No | No | P3 | Older VOS reference with shell-centric setup and optional interactive Cython components. |
| FastReID | [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid) | third_party/fast-reid | c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d | master | Apache-2.0 | Python, PyTorch, Detectron-style re-ID toolkit | docs/requirements.txt, Dockerfiles, INSTALL.md | >=3.6; documented conda env 3.7 | Torch >=1.6 | GPU optional; FAISS GPU/TensorRT paths add CUDA | Low | Yes | Yes: model-zoo checkpoints | Person re-ID datasets for training/evaluation | Dataset and checkpoint terms | No | No | P3 | INSTALL says Linux or macOS; not a first Windows target. |
| CLIP-ReID | [Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) | third_party/clip-reid | eb1898b72c882875f478bebfc6d41644eece0a5d | master | MIT source; CLIP/data/model terms separate | Python, PyTorch, CLIP, timm | README setup only | 3.8 | Torch 1.8.0, torchvision 0.9.0 | CUDA 10.2 in documented setup | Low | Yes | Yes: trained ReID weights and CLIP weights | Market-1501, MSMT17, Duke, VeRi, VehicleID and others | CLIP, checkpoint, and dataset terms | No | No | P3 | Legacy CUDA 10.2 stack and nontrivial dataset terms make it a comparison-only reference. |
| TransNetV2 | [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) | third_party/transnetv2 | 85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed | master | MIT | Python, TensorFlow or PyTorch inference, FFmpeg/Docker | setup.py, inference/training Dockerfiles | Not pinned | TensorFlow for original inference; PyTorch implementation also present | CUDA optional for inference; Docker is documented | Medium | Optional | Yes: LFS checkpoint pointers are present but blobs were not smudged | Input video; training datasets are tens to hundreds of GB | Checkpoint and dataset terms | No | Yes: three tracked weight patterns, pointer files only | P1 | Useful shot-boundary reference. Do not run its checkpoint or dataset download steps during this pass. |
| PySceneDetect | [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | third_party/pyscenedetect | bba97f59ff082875cf1c41b8ce2cb52a34ed2020 | main | BSD-3-Clause with bundled third-party notices | Python, OpenCV, PyAV/MoviePy, FFmpeg | pyproject.toml, Dockerfile, packaging/windows/requirements.txt | >=3.10 | None | None; FFmpeg/mkvmerge for splitting | High | No | No model checkpoint | Input video; benchmark datasets optional | FFmpeg/mkvmerge and bundled dependency notices | No | No | P0 | Best native-Windows scene/cut smoke test. The checkout includes a Windows installer archive, not a model asset. |
| ST-GCN | [lxtGH/st-gcn](https://github.com/lxtGH/st-gcn) | third_party/st-gcn | e7024ac16714d6d6ac911f7cfb2910aea1940b15 | master | BSD-3-Clause | Python, very old PyTorch, torchlight, OpenPose | requirements.txt, torchlight/setup.py | >=3.5 in README | Torch 0.4.0 | GPU optional; OpenPose/FFmpeg for demo | Low | Yes | Yes: tools/get_models.sh | NTU RGB+D and Kinetics-skeleton | OpenPose, model, and dataset terms | No | No | P3 | Action-recognition reference only; obsolete PyTorch and shell tooling make it unsuitable for the first implementation path. |

## Audit interpretation

- P0 means directly useful to the core path or safe native-Windows infrastructure.
- P1 means a plausible near-term alternative or supporting component.
- P2 means useful after the baseline works, but asset or platform cost is material.
- P3 means a comparison/reference path or a restricted/heavy research stack.
- "No" in the checkpoint column means no neural checkpoint is needed for the source-level utility, not that every possible application is asset-free.

No inference or benchmark was run in this pass. The next pass should pin exact
checkpoint identifiers and asset licenses before any download or environment
installation.

## TT3D calibration-only execution audit

Audit date: 2026-08-08. The checkout was independently verified at commit
`a2ef524ea0400262d6808db6cacf4a0b90bd0ad7` on `main`, with the tracked
`weights/table_segmentation.ckpt` and all three calibration test videos present.

The upstream README identifies these commands, run from the TT3D repository
root, as the sample path:

```powershell
python tt3d/calibration/segment.py ./data/calibration_test/videos/test_00.mp4 --display
python tt3d/calibration/calibrate.py ./data/calibration_test/videos/test_00.mp4 -o cam_cal.csv
```

The calibration command imports `TableCalibrator`, whose path uses the bundled
segmentation checkpoint, image preprocessing, OpenCV line/corner extraction,
SciPy focal-length/pose optimization, and CSV writing. The direct runtime
subset is:

```text
albumentations==1.4.18
lightning==2.3.3
matplotlib==3.7.5
numpy==1.24.4
opencv-python==4.8.1.78
pandas==2.0.3
pillow==10.4.0
scipy==1.10.1
segmentation-models-pytorch==0.3.3
timm==0.9.2
torch==2.4.1
torchvision==0.19.1
tqdm==4.67.1
wandb==0.19.10
PyYAML
sympy
```

`scikit-image` and the encoder/model support packages may be resolved by the
specified Albumentations/SMP versions. `casadi`, pose dependencies, rally
dependencies, MotionBERT/RTMPose/BlurBall, and `pykalman` for the optional
post-calibration filtering command are not required by the raw calibration
command. `pytorch-lightning`, `efficientnet-pytorch`, and `requests` are not
direct TT3D calibration imports; they may appear as resolver dependencies and
must be recorded if a target environment installs them.

The isolated environment verification succeeded:

```text
Python 3.10.20
C:\Users\Danie\miniconda3\envs\tti-tt3d\python.exe
```

No package was installed. The staged command attempted only the binary Torch
pair first:

```powershell
conda run --no-capture-output -n tti-tt3d python -m pip install -v --progress-bar on --disable-pip-version-check --retries 1 --timeout 60 --only-binary=:all: torch==2.4.1 torchvision==0.19.1
```

It failed before downloading or installing `torch`: the process received
`WinError 10013` while opening `pypi.org`, and pip reported that the Conda
environment site-packages was not writable, so it would have fallen back to a
user installation. The fallback was not used. The full combined log is
`.tmp/logs/tt3d_install_torch_20260808.log`; the environment still contains
only its original bootstrap packages (`packaging`, `pip`, `setuptools`, and
`wheel`).

Consequently, no segmentation or calibration command was run, no GPU/CUDA
execution occurred, no CSV/YAML/MP4/debug output was generated, and no
calibration values or reprojection metric were produced. Native Windows is
not established as incompatible; the Codex runner lacks both package-network
access and a writable isolated environment. A Kaggle dataset upload was then
attempted twice with CLI 2.2.2, including once after successful authentication
refresh, but both attempts exited `1` before creation because the sandbox
blocked the Kaggle API socket with `WinError 10013` at
`/v1/datasets.DatasetApiService/GetDatasetStatus`. The raw outputs are
`.tmp/logs/kaggle_dataset_create_20260808.log` and
`.tmp/logs/kaggle_dataset_create_retry_20260808.log`. The user then reran the
same upload externally; Kaggle accepted the 66.3 MiB private dataset
`polyleviathan/tt3d-baseline-a2ef524`. Kernel version 1 failed before input
mounting; version 2 confirmed the extracted dataset was mounted but failed in
the wrapper because it assumed a ZIP file. This is not a TT3D result. Version 3
contains the expanded-dataset fix and TT3D outputs remain pending.

The Linux/Kaggle reproduction therefore remains pending. It requires the
prepared 69,512,346-byte archive, `MPLBACKEND=Agg`, `WANDB_MODE=disabled`, the
existing checkpoint/video assets, and the two unmodified upstream commands
run from `third_party/tt3d`; use `--render -o <working-output>.mp4` for the
segmentation output and add `--render` to calibration only if the generated
video is intentionally written beside the fixture.
