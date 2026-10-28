"""Tests for the VLM evaluation inputs: the image loader's view rebuilding and the stratified question sample.

Both run on synthetic data (no dataset files, no model), so they are fast.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from measured_space import datasets
from measured_space.datasets import arkitscenes, smartspaces
from measured_space.geometry import Camera
from measured_space.qa_schema import QuestionSpec
from measured_space.scene import FrameView
from measured_space.vlm_images import SpecImageLoader

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_vlm_eval.py"


def _script() -> ModuleType:
    """Import ``scripts/run_vlm_eval.py`` as a module (it is not part of the package).

    Returns
    -------
    types.ModuleType
        The script's module, registered in ``sys.modules`` (its dataclasses need that); importing it has no
        other side effects.
    """
    if "run_vlm_eval" in sys.modules:
        return sys.modules["run_vlm_eval"]
    spec = importlib.util.spec_from_file_location("run_vlm_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_vlm_eval"] = module
    spec.loader.exec_module(module)
    return module


def _spec(dataset: str, scene: str, camera_id: str, frame: str, image_ref: str, qtype: str = "count",
          obj: str = "o1") -> QuestionSpec:
    """A minimal count question about one image.

    Parameters
    ----------
    dataset, scene, camera_id, frame : str
        The image asked about.
    image_ref : str
        Stored image location.
    qtype : str
        Question type.
    obj : str
        Counted class; varying it gives distinct ids for the same image.

    Returns
    -------
    QuestionSpec
        The question.
    """
    return QuestionSpec(f"{qtype}_v1", qtype, dataset, scene, camera_id, frame, image_ref, (),
                        f"How many {obj}?", "count", params={"label": obj})


def _view(dataset: str, scene: str, camera_id: str, frame: str, image_ref: str, width: int,
          height: int) -> FrameView:
    """A synthetic view with an identity pose.

    Parameters
    ----------
    dataset, scene, camera_id, frame : str
        Its identity.
    image_ref : str
        Stored image location.
    width, height : int
        Camera image size, pixels.

    Returns
    -------
    FrameView
        The view.
    """
    return FrameView(dataset, scene, camera_id, frame, Camera(10.0, 10.0, width / 2, height / 2, width, height),
                     np.eye(4), image_ref)


def test_loader_rebuilds_the_spec_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader picks the spec's frame from the right listing, checks it, reads it once and caches it.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the dataset listing and image reading with synthetic ones.
    """
    listed: list[dict[str, Any]] = []
    wide = [_view("arkitscenes", "v1", "wide", f, f"up/v1.zip::v1/wide/v1_{f}.png", 3, 4) for f in ("1.0", "2.0")]

    def list_arkit(video_id: str, *, upright: bool = True, **_: Any) -> list[FrameView]:
        """Synthetic ARKitScenes listing that records how it was called.

        Parameters
        ----------
        video_id : str
            Video id.
        upright : bool
            Orientation flag.
        **_ : Any
            Ignored.

        Returns
        -------
        list of FrameView
            Two upright frames.
        """
        listed.append({"video": video_id, "upright": upright})
        return wide

    def list_smart(scene: str, camera_id: str, frames: Any) -> list[FrameView]:
        """Synthetic Smart Spaces listing.

        Parameters
        ----------
        scene, camera_id : str
            Scene and camera.
        frames : iterable of int
            Frame indices.

        Returns
        -------
        list of FrameView
            One view per frame.
        """
        return [_view("smartspaces", scene, camera_id, str(f), f"{scene}/videos/{camera_id}.mp4::{f}", 5, 2)
                for f in frames]

    reads: list[str] = []

    def read(view: FrameView) -> np.ndarray:
        """Synthetic image of the view's camera size, filled with its frame number.

        Parameters
        ----------
        view : FrameView
            The view.

        Returns
        -------
        numpy.ndarray
            ``(height, width, 3)`` uint8.
        """
        reads.append(view.image_ref)
        value = int(float(view.frame)) % 256
        return np.full((view.camera.height, view.camera.width, 3), value, np.uint8)

    monkeypatch.setattr(arkitscenes, "list_frames", list_arkit)
    monkeypatch.setattr(smartspaces, "list_frames", list_smart)
    monkeypatch.setattr(datasets, "load_rgb", read)

    loader = SpecImageLoader(max_images=2)
    q1 = _spec("arkitscenes", "v1", "wide", "2.0", "up/v1.zip::v1/wide/v1_2.0.png", obj="chair")
    q2 = _spec("arkitscenes", "v1", "wide", "2.0", "up/v1.zip::v1/wide/v1_2.0.png", obj="table")
    image = loader(q1)
    assert image.shape == (4, 3, 3) and int(image[0, 0, 0]) == 2
    assert loader(q2) is image                                     # same image: decoded once
    assert loader.view(q1) is wide[1]
    assert listed == [{"video": "v1", "upright": True}]             # "wide" questions are posed on upright views
    assert reads == ["up/v1.zip::v1/wide/v1_2.0.png"]

    q3 = _spec("smartspaces", "W/018", "Camera_0002", "60", "W/018/videos/Camera_0002.mp4::60")
    assert loader(q3).shape == (2, 5, 3) and loader.view(q3).frame == "60"

    stale = _spec("arkitscenes", "v1", "wide", "1.0", "moved.zip::v1/wide/v1_1.0.png")
    with pytest.raises(ValueError, match="image_ref"):
        loader(stale)
    with pytest.raises(LookupError):
        loader(_spec("arkitscenes", "v1", "wide", "9.0", "up/v1.zip::v1/wide/v1_9.0.png"))
    assert loader.n_listed == 2                                     # one ARKitScenes listing, one Smart Spaces


def test_stratified_sample_is_deterministic_and_capped() -> None:
    """The sample keeps at most ``per_cell`` per dataset and type, regardless of input order and duplicates."""
    run_vlm_eval = _script()
    specs = [_spec(d, "s", "c", str(f), "img", qtype=q, obj=f"o{k}")
             for d, q, n in [("arkitscenes", "count", 7), ("arkitscenes", "left_right", 2), ("smartspaces", "count", 9)]
             for k in range(n) for f in (k % 3,)]
    first = run_vlm_eval.stratified_sample(specs, per_cell=4, seed=0)
    shuffled = [specs[i] for i in np.random.default_rng(1).permutation(len(specs))] + specs[:3]
    assert run_vlm_eval.stratified_sample(shuffled, per_cell=4, seed=0) == first
    assert list(first) == [("arkitscenes", "count"), ("arkitscenes", "left_right"), ("smartspaces", "count")]
    assert [len(v) for v in first.values()] == [4, 2, 4]
    by_cell = {(s.dataset, s.qtype): set() for s in specs}
    for s in specs:
        by_cell[(s.dataset, s.qtype)].add(s.id)
    assert all(set(ids) <= by_cell[cell] for cell, ids in first.items())
    other = run_vlm_eval.stratified_sample(specs, per_cell=4, seed=1)
    assert other != first                                           # the seed matters
    assert run_vlm_eval.stratified_sample(specs, per_cell=100, seed=0)[("smartspaces", "count")] == sorted(
        by_cell[("smartspaces", "count")])


def test_draws_are_stable_when_candidates_are_added() -> None:
    """A later draw over more candidates keeps every earlier question whose key still ranks, and only appends."""
    run_vlm_eval = _script()
    old = [_spec("arkitscenes", f"v{k % 2}", "wide", "0", "img", qtype="count", obj=f"o{k}") for k in range(6)]
    new = [_spec("arkitscenes", "v9", "wide", "0", "img", qtype="count", obj=f"n{k}") for k in range(30)]
    first = run_vlm_eval.draw_sample(old, per_cell=3, seed=0)
    assert set(first) == set(run_vlm_eval.stratified_sample(old, per_cell=3, seed=0)[("arkitscenes", "count")])
    second = run_vlm_eval.draw_sample(old + new, per_cell=3, seed=0)
    rank = {s.id: run_vlm_eval.sample_rank(0, s.id) for s in old + new}
    cutoff = max(rank[i] for i in second)
    assert {i for i in first if rank[i] <= cutoff} <= set(second)       # earlier picks that still rank stay
    assert run_vlm_eval.draw_sample(list(reversed(old + new)), per_cell=3, seed=0) == second
    bigger = run_vlm_eval.draw_sample(old + new, per_cell=8, seed=0)
    assert set(second) <= set(bigger) and len(bigger) == 8              # a larger cell only appends
