# Changelog

All notable changes to this project are documented here. Format is a
loose variant of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
with 4-digit `MAJOR.MINOR.PATCH.MICRO` version numbers.

## [0.1.0.0] - 2026-10-26

### Added

- **Repository bootstrap.** Adds:
  - a README stating the problem and the prototype's scope;
  - `environment.yml` (Python 3.12 from conda-forge, `torch==2.14.1` from PyPI);
  - a `.gitignore` that keeps data, weights and run outputs out of git;
  - the version-tag workflows;
  - `docs/DATASETS.md` from the planning phase: the dataset survey, licences and measured download sizes.

  No pipeline code and no results yet.
