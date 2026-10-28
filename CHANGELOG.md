# Changelog

All notable changes to this project are documented here. Format is a
loose variant of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with 4-digit `MAJOR.MINOR.PATCH.MICRO` version numbers.

## [0.2.0.0] - 2026-10-27

The first complete prototype: the label audit, its certificates and the miss audit, run on 106 ARKitScenes videos
and 12 synthetic PhysicalAI-SmartSpaces warehouse scenes with four labellers.

### Added

- **Label generation** (`qa.py`, `pipeline.py`, `scripts/run_labels.py`): nine metric question types answered from
  ground truth and from a labeller's 3D boxes; ARKitScenes and Smart Spaces loaders with visibility; a baseline
  labeller that lifts SAM 3 masks through laser, iPad LiDAR or MoGe-2 depth; 2D pairing of labels with reference
  objects, with an ambiguity flag (`matching.py`).
- **External labellers:** Cubify Anything, named by SAM 3 or CLIP (`cubify.py`, `box_sources.py`,
  `scripts/run_cubify.py`); NVIDIA TAO Sparse4D, with its frame extractor, runner and tracking scorers
  (`sparse4d.py`, `scripts/run_sparse4d.py`, `scripts/eval_sparse4d.py`, `envs/sparse4d/`).
- **The audit** (`audit.py`, `scripts/run_audit.py`): a per-answer error model cross-fitted leaving scenes out,
  split conformal intervals, Learn-then-Test gates, scene-cluster bootstrap; options to restrict depth sources and
  to cross-fit by explicit scene groups (`scripts/group_smartspaces_layouts.py`).
- **Decision-level certificates** with a second-opinion rule (`decisions.py`, `scripts/run_decision_analysis.py`).
- **Misses:** a false-negative audit with ground-truth-free flags (`misses.py`) and recall bounded by conformal
  risk control (`recall_control.py`).
- **Per-site references** (`references.py`), **review effort** (`review_effort.py`), **multi-frame fusion**
  checks (`estimation.py`, `fusion_eval.py`), **VLM evaluation** under label noise (`eval_vlm.py`, `scoring.py`).
- **A pairing viewer and a blind spot-check page** (`viewer.py`, `scripts/make_viewer.py`), with the author's two
  spot-check exports in `results/spotcheck/`.
- **`claims.py`**: recomputes every reported number from the committed tables in `results/` (190 quantities in 38
  claims); `setup_env.sh` builds the environments; `REPRODUCE.md` documents both tiers.
- `LICENSE` (PolyForm Noncommercial 1.0.0), `LICENSES/`, `THIRD_PARTY_NOTICES.md`.

### Changed

- The README describes the prototype, its results and its limitations.

## [0.1.0.0] - 2026-10-26

### Added

- **Repository bootstrap.** Adds:
  - a README stating the problem and the prototype's scope;
  - `environment.yml` (Python 3.12 from conda-forge, `torch==2.14.1` from PyPI);
  - a `.gitignore` that keeps data, weights and run outputs out of git;
  - the version-tag workflows;
  - `docs/DATASETS.md` from the planning phase: the dataset survey, licences and measured download sizes.

  No pipeline code and no results yet.
