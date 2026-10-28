# Third-party notices

measured-space is licensed under the PolyForm Noncommercial License 1.0.0 ([LICENSE](LICENSE)). The files below
contain code from other projects and stay under those projects' licences, whose texts are in
[LICENSES/](LICENSES).

## Code in this repository under another licence

### NVIDIA TAO PyTorch — Apache-2.0

- **`scripts/run_sparse4d.py`**, function `tao_decode`, is adapted from NVIDIA TAO PyTorch
  (<https://github.com/NVIDIA/tao_pytorch_backend>, commit `d99ff772`),
  `nvidia_tao_pytorch/cv/sparse4d/model/detection3d/decoder.py` (`SparseBox3DDecoder.decode`). Copyright (c) 2025
  NVIDIA CORPORATION & AFFILIATES. Licensed under the Apache License 2.0
  ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)); the notice and the list of changes are kept in the file.

### PyTorch3D (Meta) — BSD-3-Clause

- **`envs/sparse4d/pytorch3d_cxx20.patch`** changes one line of PyTorch3D's `setup.py`
  (<https://github.com/facebookresearch/pytorch3d>, commit `33824be`); its context lines come from that file.
  Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the BSD License
  ([LICENSES/BSD-3-Clause-pytorch3d.txt](LICENSES/BSD-3-Clause-pytorch3d.txt)).

## Code this project uses but does not include

These are installed or checked out separately (`REPRODUCE.md`, `envs/sparse4d/README.md`) and keep their own
licences:

- **SparseQID** (<https://github.com/playbox-dev/SparseQID>, Apache-2.0): the Sparse4D detector and frame loader,
  imported by `scripts/run_sparse4d.py`.
- **NVIDIA video-search-and-summarization** (<https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization>,
  Apache-2.0; its bundled TrackEval code is MIT): the official 3D-IoU tracking evaluator, imported by
  `scripts/eval_sparse4d.py`.
- **Cubify Anything** (<https://github.com/apple/ml-cubifyanything>): Apple's code licence, and the Apple Machine
  Learning Research Model License (research purposes only) for its weights; run by `scripts/run_cubify.py`.
- The Python packages in `environment.yml`, `requirements-nodeps.txt` and `envs/sparse4d/requirements.txt`.

## Datasets

No dataset files are included. `results/` holds aggregate tables, figures and two spot-check exports (pair
identifiers and verdicts, no images or geometry) derived from:

- **ARKitScenes** (Apple, <https://github.com/apple/ARKitScenes>): under the licence in that repository. Its
  application to derived data is not stated, so only aggregates and identifiers are committed.
- **NVIDIA PhysicalAI-SmartSpaces** (<https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces>), licensed
  under CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>). The warehouse results in `results/` are
  aggregate statistics computed from its `MTMC_Tracking_2026` scenes.

The datasets you download for Tier 2 of `REPRODUCE.md` have their own terms, which you accept by using them;
`docs/DATASETS.md` summarises them.

## Model weights

No weights are included. Those the pipeline uses are downloaded separately and keep their own terms: SAM 3,
MoGe-2, Cubify Anything (Apple Machine Learning Research Model License, research purposes only), CLIP ViT-H-14
(laion2B), the NVIDIA TAO Sparse4D NGC checkpoints (NVIDIA Open Model License), Qwen3-VL-8B and
Cosmos-Reason2-8B. See each model card before use.
