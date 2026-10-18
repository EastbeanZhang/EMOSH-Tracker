# Installation

Tracking requires Linux and an NVIDIA CUDA GPU. Python 3.10 is the intended runtime. The models are initialized together. The tested examples peaked at **8.23 GiB allocated / 8.50 GiB reserved** in the PyTorch CUDA allocator; allow additional headroom for the CUDA runtime and larger inputs. Use a separate environment and start with a short clip.

```bash
git clone https://github.com/EastbeanZhang/EMOSH-Tracker.git
cd EMOSH-Tracker
conda create -n emosh-tracker python=3.10 -y
conda activate emosh-tracker
python -m pip install pip==23.0.1 setuptools==80.10.2 wheel==0.48.0
```

The tested environment uses Python 3.10.18, PyTorch 2.9.1 with CUDA 13.0, and PyTorch3D 0.7.9. A CUDA 13.0 toolkit/compiler and compatible NVIDIA driver are required for this tested CUDA setup. The complete tested dependency list is in [tested-environment.txt](tested-environment.txt).

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt -c docs/tested-environment.txt
python -m pip install --no-build-isolation chumpy==0.70
python -m pip install ninja
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export MAX_JOBS=4
export NVCC_FLAGS='--device-entity-has-hidden-visibility=false -static-global-template-stub=false'
python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'
```

PyTorch3D builds for the visible GPU by default. Set `TORCH_CUDA_ARCH_LIST` only when cross-compiling for a different supported device.

For a different CUDA/PyTorch combination, build PyTorch3D against the matching toolkit in the same environment. See [PyTorch compatibility](https://pytorch.org/get-started/previous-versions/).

MediaPipe uses `opencv-contrib-python`; do not install a second OpenCV wheel into the same environment. On a minimal Linux host, install the system runtime libraries `libgl1`, `libglib2.0-0`, and a C++ build toolchain if they are absent. `imageio-ffmpeg` supplies the FFmpeg executable used for previews.

