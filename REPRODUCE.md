# Reproducing the results

There are two tiers.

- **Tier 1** checks every key claim from the committed result tables. It takes about a minute and needs no
  datasets, models or GPU.
- **Tier 2** regenerates those tables from the raw data. It needs the datasets, seven sets of model weights, a GPU
  and many hours. **Tier 2 has not been re-run end to end from scratch**: the steps below are the ones the results
  were produced with, taken from the scripts' own documentation and the run summaries, but nobody has yet followed
  them on a clean machine.

## Tier 1: check the claims (minutes, CPU only)

```bash
./setup_env.sh claims            # conda env "measured-space-claims"; refuses if it already exists
```

This creates a light conda environment (Python 3.12 from conda-forge; numpy, scipy, pandas, matplotlib, pyyaml,
pillow, h5py, opencv-python-headless and pytest at the versions pinned in `environment.yml`; no PyTorch), installs
this package without dependencies, and runs `claims.py`. Tested from scratch on Linux: about 30 s with a fast
connection, about 0.76 GB on disk.

In any environment with numpy and pandas:

```bash
python claims.py                     # every claim: stated value, recomputed value, PASS / FAIL
python claims.py --block core        # leave out the decision-level block
python claims.py --report claims.md  # also write a markdown report
```

`claims.py` lists each claim once, grouped by the project's aims (A1–A7, listed in `claims.py`), with the value as stated,
the tables it is read from (under `results/`), the script that wrote those tables, and the lines that recompute
it. A value passes when it rounds to the stated one (half a unit of the last written digit). The decision-level
claims (`DEC.*`) are a separate block, read from `results/decisions_exact/` (the exact certificate, from the
audits' saved calibration predictions; `results/decisions/` holds the earlier approximate run).

### What is committed under `results/`

Aggregate tables only (per-item tables of ARKitScenes are not redistributed; see `docs/DATASETS.md`). Each folder's
`summary.json` records the inputs, the settings and the git commit of the run.

| Folder | Contents | Written by |
|---|---|---|
| `audit_ark106_{2d,cutr_sam3,cutr_clip}[_noamb]/` | Handheld audits (106 videos): errors, coverage, gates, rankings; `_noamb` without ambiguous pairs | `scripts/run_audit.py` |
| `audit_ss12_{2d,sparse4d_v22,sparse4d_v11}[_noamb]/` | Fixed-camera audits (12 synthetic warehouses) | `scripts/run_audit.py` |
| `audit_ark106_2d_lidar/` | The v0 baseline audited on its iPad LiDAR answers only | `scripts/run_audit.py --depth-sources lidar` |
| `audit_ss12_{2d,sparse4d_v22,sparse4d_v11}_bylayout/` | Fixed-camera audits cross-fitted with one fold per building layout | `scripts/run_audit.py --fold-groups configs/smartspaces_layout_groups.json` |
| `misses/<run>/` | Recall, miss causes, effects on answers, miss flags | `scripts/run_miss_audit.py` |
| `recall_control/<run>/` | Conformal risk control of the miss rate | `scripts/run_recall_control.py` |
| `references/<run>/` | Per-site reference recalibration | `scripts/run_reference_experiment.py` |
| `fusion/fusion_ark106_{2d,cutr_sam3}/` | NIS / NEES of single-frame and fused estimates | `scripts/run_fusion_eval.py` |
| `vlm_results/<labeller>/` | VLM evaluation re-scored per labeller | `scripts/run_vlm_eval.py score` |
| `sparse4d_eval/` | Sparse4D tracking scores (stage 1, scene 022; stage 2, 12 scenes) | `scripts/eval_sparse4d.py aggregate` (stage 2) and `hota --no-raw-arrays` (stage 1); Tier 2, step 5 |
| `decisions/` | Decision-level gates, threshold sweep, margins | `scripts/run_decision_analysis.py` |
| `review_effort/` | Review needed for a stated quality; operating points; miss-flag review cost | `scripts/run_review_effort.py` |
| `spotcheck/` | The author's manual pairing spot check | `scripts/make_viewer.py` |

The older folders (`audit/`, `association/`, `fusion/*.csv`, `references/*.csv`, `figures/`, ...) are the earlier
6-video and design runs, kept for reference; `claims.py` does not read them.

## Tier 2: regenerate the tables from raw data

### 1. Data (read each license first; see `docs/DATASETS.md`)

- **ARKitScenes** (Apple; custom license, redistribution of derived data unclear): the 3DOD annotations and the
  "upsampling" archives (RGB, iPad LiDAR depth, laser-rendered depth) of the validation videos. 106 videos are used;
  video 48458647 is excluded (portrait images with landscape intrinsics). Copy `configs/paths.example.yaml`
  to `configs/paths.local.yaml` (git-ignored) and set the data folders there.
- **Physical AI Smart Spaces, 2026 split** (CC-BY-4.0): `MTMC_Tracking_2026` warehouse scenes 000, 002, 004, 006,
  010–016 and 020 (12 scenes); 008 and 009 are excluded because their layout matches a 2025 scene. The per-camera
  selection is in `configs/` (`scripts/select_smartspaces_cameras.py`).

### 2. Model weights (each under its own license; see the model cards)

SAM 3 (text-prompted segmentation), MoGe-2 (monocular metric depth), Cubify Anything `cutr_rgbd.pth` (research-only
license), CLIP ViT-H-14 laion2B (naming Cubify's boxes), NVIDIA TAO Sparse4D NGC checkpoints `rn50_trainable_v2.2`
and `rn101_trainable_v1.1` (NVIDIA Open Model License), Qwen3-VL-8B and Cosmos-Reason2-8B (VLM evaluation).

### 3. Environments

- **Main** (`./setup_env.sh full`): `environment.yml` (Python 3.12; PyTorch 2.14.1, whose default Linux wheel is a
  CUDA 13.0 build, driver ≥ 580) plus `requirements-nodeps.txt` (MoGe-2 at a pinned commit). Used for everything
  except steps 4 and 5.
- **Cubify Anything** (`mga-cubify`, separate so the main environment is untouched): Python 3.10, torch 2.14.1
  (CUDA 13.0 build), torchvision 0.29.1, timm 1.0.30, numpy, scipy, Pillow; `apple/ml-cubifyanything` at commit
  `00e9cb1` on `PYTHONPATH`. `measured_space` is not installed there (see `scripts/run_cubify.py`).
- **Sparse4D** (`mga-sparse4d`): Python 3.12, numpy 1.26.4, torch 2.14.1 / torchvision 0.29.1 (CUDA 13.0 builds),
  timm 1.0.30, opencv-python-headless 4.11, and pytorch3d 0.7.9 (commit `33824be`) built from source for the CPU
  with its C++ standard raised from C++17 to C++20 (`envs/sparse4d/pytorch3d_cxx20.patch`; needed with this
  toolchain); two source checkouts that are not installed: SparseQID's pure-PyTorch runtime at commit `7aea61d`
  and NVIDIA's `spatialai_data_utils` evaluator (video-search-and-summarization at commit `82a5011`, on
  `PYTHONPATH` for scoring). Build steps: `envs/sparse4d/README.md`; pinned packages:
  `envs/sparse4d/requirements.txt`. `measured_space` is not installed there and the Sparse4D scripts
  (`scripts/extract_smartspaces_frames.py`, `scripts/run_sparse4d.py`, `scripts/eval_sparse4d.py`) do not import
  it; their output is converted in the main environment by `src/measured_space/sparse4d.py`. Weights:
  `bash scripts/download_sparse4d_weights.sh <listing-dir> <models>/sparse4d <log>` (size and SHA-256 checked
  against NGC's listing).

### 4. Handheld labels (106 ARKitScenes videos)

1. `python scripts/run_labels.py` (GPU): SAM 3 masks per view, lifted through laser, iPad LiDAR and MoGe-2 depth;
   oracle visibility, questions and paired answers. Then re-associate with the 2D pairing:
   `python scripts/run_labels.py --rematch-from <labels> --association 2d --out-dir <data>/outputs/labels_ark106_2d`.
2. Cubify Anything: export frames (`python -m measured_space.cubify export ...`), then in `mga-cubify`
   `PYTHONPATH=<ml-cubifyanything> python scripts/run_cubify.py <frames> --out <predictions> --checkpoint cutr_rgbd.pth`
   (GPU; once at the published 0.25 threshold and once at 0.05 for recall control).
3. `python scripts/run_box_source.py --variant cutr_sam3 ...` and `--variant cutr_clip ...` (CLIP on the GPU) build
   `labels_ark106_cutr_{sam3,clip}` from the v0 label folder and the predictions.

### 5. Fixed-camera labels (12 warehouses)

1. `python scripts/run_labels.py` with the Smart Spaces options (`--ss-*`; MoGe-2 depth) builds `labels_ss12_2d`.
2. Sparse4D, frozen, both checkpoints, every camera of a scene at once, in `mga-sparse4d` (details in each
   script's docstring; `<s>` = 000, 002, 004, 006, 010–016, 020; split `val` for 020, `train` otherwise):
   1. Frame cache (CPU): `python -I scripts/extract_smartspaces_frames.py --data-root <data>/MTMC_Tracking_2026
      --split <split> --scene Warehouse_<s> --cache-root <out>/frame_cache --start 0 --end 9000 --step 6`.
   2. Inference (GPU): `python -I scripts/run_sparse4d.py --sparseqid-src <SparseQID>/src --model-dir
      <models>/sparse4d/rn50_trainable_v2.2 --ckpt sparse4d_warehouse_v2.2_r50.pth --tag rn50_trainable_v2.2
      --data-root <data>/MTMC_Tracking_2026 --cache-root <out>/frame_cache --split <split> --scene Warehouse_<s>
      --scene-id <s as integer> --out-dir <out> --num-workers 6`; for the second checkpoint `--model-dir
      <models>/sparse4d/rn101_trainable_v1.1 --ckpt sparse4d_tracking_aic25v0.5_warehousev3_classes_iter_17400_v1.1.pth
      --backbone-if-missing resnet_101 --tag rn101_trainable_v1.1`. The defaults are the protocol: world frame
      re-centred on the mean camera centre (xy) and shifted back, RGB input, the decoder's own scores and
      instance-bank track ids, class indices remapped to the 2026 challenge ids, decoder threshold 0, 11-column rows
      with score ≥ 0.2, JSONL boxes with score ≥ 0.05.
   3. Scores (CPU, `PYTHONPATH=<video-search-and-summarization>/libs/analytics/spatialai-data-utils`): per scene
      `python scripts/eval_sparse4d.py gt --scene-dir <data>/MTMC_Tracking_2026/<split>/Warehouse_<s> --scene-id <s>
      --start 0 --end 9000 --step 6 --out <out>/gt/Warehouse_<s>_f0-9000s6.txt`; per scene and tag `hota --sim bev
      --gt ... --pred <out>/<tag>/Warehouse_<s>.txt --out <out>/eval/<tag>/Warehouse_<s>.bev/hota_bev.json` (the
      challenge's footprint-IoU HOTA, as the EVA/GaugeAlign team computes it) and `score --gt ... --pred ...
      --scene-ids <json> --out-dir <out>/eval/<tag>/Warehouse_<s>` (NVIDIA's official 3D-IoU evaluator; the JSON maps
      all 12 ids in scene order, `{"0": "Warehouse_000", "2": "Warehouse_002", …, "20": "Warehouse_020"}`); then
      `aggregate --out-root <out> --tags rn50_trainable_v2.2 rn101_trainable_v1.1 --summary
      <out>/eval/summary_stage2.json`, which is `results/sparse4d_eval/summary_stage2.json`.
      `results/sparse4d_eval/stage1_w022_rn50_v2.2_rgb_hota_bev.json` is `hota --sim bev --no-raw-arrays` on scene
      022 (val), frames 0–894 at step 6 (`gt` and the cache with `--end 900 --step 6`), of a CPU run of the rn50
      checkpoint (`--device cpu`, tag `cpu_rn50_trainable_v2.2_rgb`).
3. `python scripts/run_box_source.py --variant external_named --source-name sparse4d --min-score 0.2
   --labels-from <data>/outputs/labels_ss12_2d --predictions <sparse4d output>/<tag> --out-dir ...` per checkpoint
   builds `labels_ss12_sparse4d_{v22,v11}`.

### 6. Analyses (CPU unless stated), per label folder

| Step | Command (see each script's docstring and `--help`) | Writes |
|---|---|---|
| Audit | `python scripts/run_audit.py --labels <labels> --results <...>` (and `--exclude-ambiguous`) | `audit_*` |
| Audit, one depth source | the same with `--depth-sources lidar` (v0 handheld labels) | `audit_ark106_2d_lidar` |
| Audit, layout held out | `python scripts/group_smartspaces_layouts.py` (writes `configs/smartspaces_layout_groups.json`), then the same with `--fold-groups configs/smartspaces_layout_groups.json` | `audit_ss12_*_bylayout` |
| Misses and flags | `python scripts/run_miss_audit.py --labels NAME=<labels> ... --results <...>` | `misses/` |
| Recall control | `python scripts/run_recall_control.py --labels <labels> --predictions <...> --results <...>` | `recall_control/` |
| References | `python scripts/run_reference_experiment.py --labels <labels> --results <...>` | `references/` |
| Fusion | `python scripts/run_fusion_eval.py --labels-dir <labels>/arkitscenes --results-dir <...>` | `fusion/` |
| VLM evaluation | `python scripts/run_vlm_eval.py sample`, `predict --model ...` (GPU), `score ...` | `vlm_results/` |
| Spot check | `python scripts/make_viewer.py ... --spotcheck-source lidar`, then `--summarize-answers` | `spotcheck/` |
| Decisions | `python scripts/run_decision_analysis.py --results <...>` | `decisions/` |
| Review effort | `python scripts/run_review_effort.py --results <...>` | `review_effort/` |

The exact inputs and settings of every committed run are in its `summary.json` (`inputs`, `config` or
`settings`, and the git commit). Audits cross-fit by scene and group scenes into at most 20 folds above 20 scenes
(`--max-folds`); with the same inputs and seeds they are deterministic.
