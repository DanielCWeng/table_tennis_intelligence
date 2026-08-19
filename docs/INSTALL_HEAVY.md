# Heavy Integration Install Plan

This project should not put every research repository into one Python
environment. The upstream projects pin incompatible Python, PyTorch, CUDA, and
OpenCV versions. Use the main environment for orchestration and small adapters;
use isolated environments for TT3D, MMPose, ball trackers, and 4DHumans.

## Local audit (2026-08-07)

Already present:

```text
Python 3.13.5
PyTorch 2.6.0+cu124
TorchVision 0.21.0+cu124
CUDA runtime reported by PyTorch: 12.4
CUDA toolkit: 11.8
GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB
NVIDIA driver: 610.47
NumPy, SciPy, scikit-learn, Pillow, DuckDB, PyArrow, pytest
```

Missing from the main environment:

```text
FFmpeg / FFprobe
PyAV
OpenCV
RTMLib / RTMPose ONNX runtime
MMPose / MMEngine / MMCV / optional MMDetection
```

Do not replace the existing PyTorch installation yet. It is useful for the
core and can remain the orchestration environment.

## Install now: smallest useful real-video path

Install FFmpeg and FFprobe system-wide and put their `bin` directory on
`PATH`. Verify in a new PowerShell:

```powershell
ffmpeg -version
ffprobe -version
```

Then install the main-project media and lightweight pose path:

```powershell
python -m pip install -e ".[dev,render]"
python -m pip install av opencv-contrib-python rtmlib onnxruntime
ttintel-doctor --json
```

`rtmlib` uses NumPy, OpenCV, and ONNX Runtime and supports RTMPose/RTMW
whole-body models. The adapter is now in `src/ttintel/adapters/rtmlib.py`.
Model `.onnx`/`.zip` files are separate assets; do not let a library silently
download them. Put approved local files under an ignored `checkpoints/` folder
and record their licence and checksum in `docs/MODEL_MATRIX.md`.

If ONNX Runtime GPU is required after the CPU smoke test, use a separate
environment and install the GPU wheel only after checking its CUDA/cuDNN
compatibility with that environment. Do not install both CPU and GPU runtime
wheels into the same environment.

## Environment 1: MMPose / RTMPose reference stack

Use this when we need the full OpenMMLab path or TT3D's exact pose workflow.
The upstream installation requires MMEngine and MMCV; some demos also require
MMDetection. Use Python 3.10 and CUDA 11.8-compatible PyTorch rather than the
main Python 3.13 environment:

```powershell
conda create -n tti-mmpose python=3.10 -y
conda run -n tti-mmpose python -m pip install `
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 `
  --index-url https://download.pytorch.org/whl/cu118
conda run -n tti-mmpose python -m pip install -U openmim
conda run -n tti-mmpose mim install mmengine
conda run -n tti-mmpose mim install "mmcv>=2.0.1"
conda run -n tti-mmpose mim install "mmpose>=1.3.0"
conda run -n tti-mmpose mim install "mmdet>=3.1.0"
```

`mmdet` is optional for some APIs but required by common top-down demos. Keep
this environment separate: MMCV extensions are tightly coupled to the
PyTorch/CUDA pair.

## Environment 2: TT3D

TT3D's current requirements pin NumPy 1.24.4, OpenCV 4.8.1, PyTorch 2.4.1,
TorchVision 0.19.1, Lightning, segmentation-models-pytorch, timm, CasADi,
SciPy, and related packages. Clone only source first, then install into a
Python 3.10/CUDA 11.8 environment:

```powershell
conda create -n tti-tt3d python=3.10 -y
conda run -n tti-tt3d python -m pip install `
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 `
  --index-url https://download.pytorch.org/whl/cu118
git clone --depth 1 https://github.com/cogsys-tuebingen/tt3d.git third_party/tt3d
conda run -n tti-tt3d python -m pip install -r third_party/tt3d/requirements.txt
```

TT3D also expects RTMPose, MotionBERT, and a ball detector's artifacts. Start
with its calibration path and record intermediate files; do not pretend the
full ball/3-D path is installed until those artifacts exist.

## Environment 3: TOTNet ball tracker

TOTNet currently recommends Python 3.10 and pins the PyTorch 2.4.1/cu118
family plus OpenCV, scikit-learn, Cython, pycocotools, SciPy, Ninja, TensorBoard,
and ptflops:

```powershell
conda create -n tti-totnet python=3.10 -y
conda run -n tti-totnet python -m pip install `
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 `
  --index-url https://download.pytorch.org/whl/cu118
git clone --depth 1 https://github.com/AugustRushG/TOTNet.git third_party/TOTNet
conda run -n tti-totnet python -m pip install -r third_party/TOTNet/requirements.txt
```

The source repository includes weights, but dataset access is separately
restricted. We need only an approved inference checkpoint for the adapter; do
not download the TTA dataset unless its agreement is accepted.

## Environment 4: TrackNetV3 legacy baseline

TrackNetV3 is an older stack: its upstream development environment is Python
3.8.7, Torch 1.10.0, NumPy 1.22.4, and OpenCV 4.4. This is not compatible with
the current environment and should be isolated or containerised:

```powershell
conda create -n tti-tracknetv3 python=3.8 -y
git clone --depth 1 --branch master https://github.com/qaz812345/TrackNetV3.git third_party/TrackNetV3
conda run -n tti-tracknetv3 python -m pip install -r third_party/TrackNetV3/requirements.txt
```

The inference checkpoints are separate downloads. TrackNetV3 was developed for
shuttlecock footage, so it must be benchmarked rather than assumed to win on
table tennis.

## Environment 5: 4DHumans / HMR2

4DHumans upstream provides a Python 3.10 environment with PyTorch CUDA 11.8,
SMPL-X, pyrender, OpenCV, YACS, scikit-image, einops, timm, dill, pandas,
Detectron2, and chumpy. It also needs a registered SMPL neutral model file and
downloads HMR checkpoints on first use. On Windows, use WSL2/Ubuntu or a Linux
machine for this environment; Detectron2/PHALP is the likely native-Windows
failure point.

```bash
git clone --depth 1 https://github.com/shubham-goel/4D-Humans.git third_party/4D-Humans
conda env create -f third_party/4D-Humans/environment.yml
conda activate 4D-humans
pip install git+https://github.com/brjathu/PHALP.git
```

Then separately obtain the licensed SMPL neutral model and an approved HMR
checkpoint. Do not download training/evaluation datasets for the MVP.

## Later, not now

- SAM 3D Body: requires its own install instructions, checkpoint access, and
  MHR assets; the released checkpoints are heavyweight and license-gated.
- MHR: Python 3.11+, PyTorch, PyMomentum, and downloaded model assets; Pixi is
  upstream's recommended environment manager.
- CoTracker3, SAM2, Cutie, WHAM, WHAC, SMPLest-X, ViTPose, PaddleOCR,
  FastReID, and CLIP-ReID: add only after the P0 path has a labelled failure
  case that justifies each dependency.

## What I need from you now

1. Install FFmpeg/FFprobe and run `ffmpeg -version`.
2. Install `av`, `opencv-contrib-python`, `rtmlib`, and `onnxruntime` in the
   current environment.
3. Install Miniconda/Anaconda if it is not already available, then create the
   `tti-mmpose` and `tti-tt3d` environments above.
4. If you want 4DHumans, enable WSL2/Ubuntu and place the licensed SMPL asset
   somewhere outside Git; tell me its path, not the file itself.
5. Run `ttintel-doctor --json` and send the output.

Do not install all five environments into one interpreter and do not download
the large checkpoints or datasets yet. Once the doctor report shows the media
and MMPose paths, I can wire and smoke-test them one at a time.
