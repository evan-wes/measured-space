"""Where data and model weights live. Nothing in the repository names a folder on any machine.

The data root comes from the ``MEASURED_SPACE_DATA`` environment variable, or else from ``data_root`` in
``configs/paths.local.yaml``, which is git-ignored (copy ``configs/paths.example.yaml``). The root must
contain the layout described in ``docs/DATASETS.md`` (``data/arkitscenes``, ``data/smartspaces``, ...).
A missing root is an error, never a silent fallback, so a disconnected external drive cannot cause writes
elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Path: the repository's root folder."""

LOCAL_CONFIG = REPO_ROOT / "configs" / "paths.local.yaml"
"""Path: the machine-specific, git-ignored path configuration."""


def _from_local_config(key: str) -> str | None:
    """Read one value from the local path configuration.

    Parameters
    ----------
    key : str
        Key to look up in ``configs/paths.local.yaml``.

    Returns
    -------
    str or None
        The value as text, or ``None`` if the file or the key is missing or empty.
    """
    if LOCAL_CONFIG.exists():
        value = (yaml.safe_load(LOCAL_CONFIG.read_text()) or {}).get(key)
        return str(value) if value else None
    return None


def data_root() -> Path:
    """Return the data root directory.

    Returns
    -------
    pathlib.Path
        The configured data root.

    Raises
    ------
    RuntimeError
        If no data root is configured, or the configured directory does not exist.
    """
    value = os.environ.get("MEASURED_SPACE_DATA") or _from_local_config("data_root")
    if not value:
        raise RuntimeError("Data root not configured: set MEASURED_SPACE_DATA or data_root in "
                           f"{LOCAL_CONFIG.relative_to(REPO_ROOT)}")
    root = Path(value).expanduser()
    if not root.is_dir():
        raise RuntimeError(f"Data root {root} does not exist (is the external drive mounted?)")
    return root


def dataset_dir(name: str) -> Path:
    """Return a dataset's folder, ``<data root>/data/<name>``.

    Parameters
    ----------
    name : str
        Dataset folder name, e.g. ``"arkitscenes"``.

    Returns
    -------
    pathlib.Path
        The dataset folder.

    Raises
    ------
    RuntimeError
        If the data root is unavailable or the dataset folder does not exist.
    """
    path = data_root() / "data" / name
    if not path.is_dir():
        raise RuntimeError(f"Dataset folder {path} not found; see docs/DATASETS.md for the download recipe")
    return path
