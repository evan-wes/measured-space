# Sparse4D environment (`measured-space-sparse4d`)

A separate conda environment for the Sparse4D labeller, so the main environment is untouched. It runs
`scripts/extract_smartspaces_frames.py`, `scripts/run_sparse4d.py` and `scripts/eval_sparse4d.py`. `measured_space`
is not installed in it and none of these scripts imports it. Its output is converted in the main environment by
`python -m measured_space.sparse4d convert`.

It needs three source checkouts next to the packages:

| Code | Where | Commit | License | Used for |
|---|---|---|---|---|
| SparseQID | https://github.com/playbox-dev/SparseQID | `7aea61d` | Apache-2.0 | detector and frame loader (`--sparseqid-src <SparseQID>/src`; not pip-installed) |
| pytorch3d | https://github.com/facebookresearch/pytorch3d | `33824be` (0.7.9) | BSD-3-Clause | `box3d_overlap` (3D IoU) in NVIDIA's evaluator and `eval_sparse4d.py hota --sim 3d` |
| video-search-and-summarization | https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization | `82a5011` | Apache-2.0 (bundled TrackEval: MIT) | `spatialai_data_utils` (NVIDIA's `evaluate_aicity_mtmc`, TrackEval HOTA), on `PYTHONPATH` |

The model weights are the NGC checkpoints `sparse4d_rn50` `trainable_v2.2` and `sparse4d_rn101` `trainable_v1.1`
(NVIDIA Open Model License): `bash scripts/download_sparse4d_weights.sh <listing-dir> <models>/sparse4d <log>`,
which checks every file's size and SHA-256 against NGC's listing.

## Build

Linux, conda, a C++20 compiler (g++), git. The results were produced with this environment on an RTX 5090 (driver
supporting CUDA 13.0); the evaluation runs on the CPU.

```bash
conda create -n measured-space-sparse4d python=3.12 pip
conda activate measured-space-sparse4d

# 1. PyTorch (CUDA 13.0 build) from PyTorch's index
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1 torchvision==0.29.1

# 2. everything else, pinned (numpy 1.26.4: the evaluator's nuscenes-devkit needs numpy < 2)
pip install -r envs/sparse4d/requirements.txt

# 3. pytorch3d from source, CPU-only extension, C++ standard raised from 17 to 20
git clone https://github.com/facebookresearch/pytorch3d.git <build>/pytorch3d_src
git -C <build>/pytorch3d_src checkout 33824be
git -C <build>/pytorch3d_src apply "$PWD/envs/sparse4d/pytorch3d_cxx20.patch"
PYTORCH3D_FORCE_NO_CUDA=1 pip install --no-build-isolation <build>/pytorch3d_src

# 4. the two other checkouts (not installed)
git clone https://github.com/playbox-dev/SparseQID.git <third_party>/SparseQID
git -C <third_party>/SparseQID checkout 7aea61d
git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git \
    <third_party>/video-search-and-summarization
git -C <third_party>/video-search-and-summarization checkout 82a5011
```

**Why the pytorch3d patch.** pytorch3d 0.7.9's `setup.py` compiles its C++ extension with `-std=c++17`, and the
headers of torch 2.14.1 refuse anything below C++20 (`#error C++20 or later compatible compiler is required to use
PyTorch`). `pytorch3d_cxx20.patch` changes that one flag. The extension was built for the CPU only (the build
machine had no CUDA compiler; `PYTORCH3D_FORCE_NO_CUDA=1` gives the same build on a machine that has one):
`box3d_overlap` runs on CPU tensors in both evaluators.

**Use.** The scripts are run with this environment's Python from the repository root, e.g.
`$CONDA_PREFIX/bin/python -I scripts/run_sparse4d.py ...`. The evaluator needs
`PYTHONPATH=<third_party>/video-search-and-summarization/libs/analytics/spatialai-data-utils` (and so cannot run
with `-I`); see the module docstrings and `REPRODUCE.md` for the full sequence.

**Tests.** `tests/test_run_sparse4d.py`, `tests/test_eval_sparse4d.py` and `tests/test_extract_smartspaces_frames.py`
run in the main environment and skip what needs this one. Run in this environment, with the evaluator on
`PYTHONPATH` and `SPARSEQID_SRC=<third_party>/SparseQID/src`, the HOTA, 3D-IoU and decoder checks run too.

`requirements.txt` is `pip freeze` of the environment that produced the results, grouped into direct requirements
and what they pulled in; only the local pytorch3d build (step 3) is left out.
