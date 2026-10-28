# Measured Space

**Calibrated, physically validated uncertainty for metric 3D grounding.**

Metric 3D perception from images (object positions, extents, distances between objects, lifted from depth
and segmentation) cannot tell when it is wrong. Its errors are structured: partly visible objects come out
too small, and a single scale error is shared by everything in a capture. Those errors flow silently into the
spatial question–answer labels used to train and evaluate vision-language models. This repository is a
feasibility prototype for a six-month research plan. It:
1. generates the same metric spatial questions from exact ground truth and from a perception pipeline;
2. estimates each object's 3D state and uncertainty by fusing it across frames;
3. checks whether that uncertainty is calibrated against independent measurements, including laser scans;
4. measures what unaudited labels do to conclusions about VLMs.

**Status:** bootstrap. No pipeline code or results yet. Every number this README eventually reports will be
produced by a script in this repository, with the command to rerun it.

- Datasets, licences and download sizes: [`docs/DATASETS.md`](docs/DATASETS.md)
- Decisions and dead ends: [`docs/LOG.md`](docs/LOG.md)
- Checking the key claims (`python claims.py`) and regenerating the results: [`REPRODUCE.md`](REPRODUCE.md)

## Environment

```bash
conda env create -f environment.yml      # Python 3.12 from conda-forge; PyTorch via pip (CUDA 13.0 wheels)
conda activate metric-grounding-audit
pip install --no-deps -r requirements-nodeps.txt   # MoGe-2 (monocular metric depth), pinned commits
```

Data and model weights live outside the repository (their location is configured, not hard-coded). Raw
dataset files are never committed.

## Licence

Not chosen yet. Dataset licences and attribution are in [`docs/DATASETS.md`](docs/DATASETS.md).
