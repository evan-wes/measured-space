"""Dataset loaders. Each turns a dataset's files into :mod:`measured_space.scene` types.

:func:`load_rgb` is the one entry point for "the image a person or model sees" for any view. Use it instead
of reading ``FrameView.image_ref`` directly: ARKitScenes stores its frames sideways, and its loader returns
them turned upright to match the view's camera.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from measured_space.scene import FrameView


def load_rgb(view: FrameView) -> NDArray[np.uint8]:
    """Load a view's colour image in the orientation its camera describes.

    Parameters
    ----------
    view : FrameView
        A view from any dataset loader in this package.

    Returns
    -------
    numpy.ndarray, shape (H, W, 3), uint8
        RGB image matching ``view.camera`` (upright for ARKitScenes views).

    Raises
    ------
    ValueError
        If the view's dataset has no loader.
    """
    if view.dataset == "arkitscenes":
        from measured_space.datasets import arkitscenes

        return arkitscenes.load_rgb(view)
    if view.dataset == "smartspaces":
        from measured_space.datasets import smartspaces

        return smartspaces.load_rgb(view)
    raise ValueError(f"no image loader for dataset {view.dataset!r}")
