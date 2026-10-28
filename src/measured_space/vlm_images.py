"""The image a question was asked about, rebuilt for the VLM evaluation.

A :class:`~measured_space.qa_schema.QuestionSpec` names its view (dataset, scene, camera, frame) and the
stored file (``image_ref``), but not the camera model that its answer was computed in. A model must see that
camera's image. ARKitScenes is the case where the two can differ: its frames are stored sideways, and the
questions are posed on the upright views. So the image is never read from ``image_ref`` directly. Instead:

1. the dataset's own ``list_frames`` rebuilds the :class:`~measured_space.scene.FrameView`, with the same
   orientation flag the label pipeline used (ARKitScenes ``"wide"`` is upright, ``"wide_sensor"`` is not);
2. the rebuilt view must carry the spec's ``image_ref``, so a renamed or re-indexed frame fails loudly
   instead of feeding the wrong picture;
3. :func:`measured_space.datasets.load_rgb` reads it, and its shape must match the view's camera.

Rebuilt views and a few decoded images are cached. Questions about one image are usually asked back to back,
so a small image cache is enough when the caller groups questions by image.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

from measured_space import datasets
from measured_space.datasets import arkitscenes, smartspaces
from measured_space.qa_schema import QuestionSpec
from measured_space.scene import FrameView

ViewKey = tuple[str, str, str, str]
"""``(dataset, scene, camera_id, frame)``: identifies one image."""


def view_key(spec: QuestionSpec) -> ViewKey:
    """The image a question is about, as a hashable key.

    Parameters
    ----------
    spec : QuestionSpec
        A question.

    Returns
    -------
    tuple of str
        ``(dataset, scene, camera_id, frame)``.
    """
    return (spec.dataset, spec.scene, spec.camera_id, spec.frame)


class SpecImageLoader:
    """Load the RGB image of a question, through a rebuilt :class:`FrameView`, with small caches.

    Calling the loader on a spec returns the image; it can be passed as ``load_image`` to
    :func:`measured_space.eval_vlm.run_predictions`.

    Parameters
    ----------
    max_images : int, default 4
        Decoded images kept in memory, least recently used dropped first (a 1920 x 1440 RGB frame is
        about 8 MB).

    Attributes
    ----------
    n_listed : int
        Calls made to a dataset's ``list_frames``.
    n_decoded : int
        Images decoded (cache misses).

    Raises
    ------
    ValueError
        If ``max_images`` is below 1.
    """

    def __init__(self, max_images: int = 4) -> None:
        """Create an empty loader.

        Parameters
        ----------
        max_images : int, default 4
            Decoded images kept in memory.

        Raises
        ------
        ValueError
            If ``max_images`` is below 1.
        """
        if max_images < 1:
            raise ValueError(f"max_images must be at least 1, got {max_images}")
        self.max_images = max_images
        self.n_listed = 0
        self.n_decoded = 0
        self._arkit_views: dict[tuple[str, str], dict[str, FrameView]] = {}
        self._views: dict[ViewKey, FrameView] = {}
        self._images: OrderedDict[ViewKey, NDArray[np.uint8]] = OrderedDict()

    def view(self, spec: QuestionSpec) -> FrameView:
        """Rebuild the view a question was posed on.

        Parameters
        ----------
        spec : QuestionSpec
            A question from the label pipeline.

        Returns
        -------
        FrameView
            The view, from the dataset's ``list_frames``, with ``image_ref`` equal to the spec's.

        Raises
        ------
        ValueError
            If the dataset or camera is not supported, or the rebuilt view's ``image_ref`` differs from the
            spec's.
        LookupError
            If the dataset lists no view (or more than one) for the spec's frame.
        """
        key = view_key(spec)
        if key not in self._views:
            view = self._rebuild(spec)
            if view.image_ref != spec.image_ref:
                raise ValueError(f"Question {spec.id}: rebuilt view has image_ref {view.image_ref!r}, "
                                 f"the question was asked about {spec.image_ref!r}")
            self._views[key] = view
        return self._views[key]

    def __call__(self, spec: QuestionSpec) -> NDArray[np.uint8]:
        """The RGB image of a question, as the question's camera describes it.

        Parameters
        ----------
        spec : QuestionSpec
            A question from the label pipeline.

        Returns
        -------
        numpy.ndarray, shape (H, W, 3), uint8
            The image from :func:`measured_space.datasets.load_rgb` (upright for ARKitScenes ``"wide"``).

        Raises
        ------
        ValueError
            If the view cannot be rebuilt (see :meth:`view`) or the image's size differs from its camera's.
        LookupError
            If the dataset lists no single view for the spec's frame.
        """
        key = view_key(spec)
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        view = self.view(spec)
        image = datasets.load_rgb(view)
        expected = (view.camera.height, view.camera.width, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(f"Question {spec.id}: image is {image.shape} {image.dtype}, its camera expects "
                             f"{expected} uint8")
        self.n_decoded += 1
        self._images[key] = image
        while len(self._images) > self.max_images:
            self._images.popitem(last=False)
        return image

    def _rebuild(self, spec: QuestionSpec) -> FrameView:
        """List the dataset's views for the spec's camera and pick the spec's frame.

        Parameters
        ----------
        spec : QuestionSpec
            A question.

        Returns
        -------
        FrameView
            The one view whose camera and frame match the spec.

        Raises
        ------
        ValueError
            If the dataset or the ARKitScenes camera id is not supported.
        LookupError
            If no view, or more than one, matches.
        """
        if spec.dataset == arkitscenes.DATASET:
            if spec.camera_id not in (arkitscenes.CAMERA_ID, arkitscenes.SENSOR_CAMERA_ID):
                raise ValueError(f"Question {spec.id}: unknown ARKitScenes camera {spec.camera_id!r}")
            listed_key = (spec.scene, spec.camera_id)
            if listed_key not in self._arkit_views:
                views = arkitscenes.list_frames(spec.scene, upright=spec.camera_id == arkitscenes.CAMERA_ID)
                self.n_listed += 1
                self._arkit_views[listed_key] = {}
                for v in views:
                    if v.camera_id == spec.camera_id:
                        if v.frame in self._arkit_views[listed_key]:
                            raise LookupError(f"ARKitScenes video {spec.scene} lists frame {v.frame} twice")
                        self._arkit_views[listed_key][v.frame] = v
            matches = [self._arkit_views[listed_key][spec.frame]] if spec.frame in self._arkit_views[listed_key] \
                else []
        elif spec.dataset == smartspaces.DATASET:
            views = smartspaces.list_frames(spec.scene, spec.camera_id, [int(spec.frame)])
            self.n_listed += 1
            matches = [v for v in views if v.camera_id == spec.camera_id and v.frame == spec.frame]
        else:
            raise ValueError(f"Question {spec.id}: no view loader for dataset {spec.dataset!r}")
        if len(matches) != 1:
            raise LookupError(f"Question {spec.id}: {len(matches)} views match {view_key(spec)}")
        return matches[0]
