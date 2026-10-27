"""Tests for zero-shot box naming: vocabulary, crops and margins, the "other" rule, and the CLIP wrapper.

Naming runs on a fake model that reads the colour at the centre of each crop, so no weights are needed. The
:class:`~measured_space.clip_naming.ClipScorer` loader is exercised on a tiny, randomly initialised CLIP that is
written into a temporary Hugging Face cache (skipped without torch and transformers).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from measured_space.clip_naming import (
    PAD_RGB,
    PROMPT_TEMPLATE,
    UNNAMED_LOW_PROB,
    UNNAMED_OTHER,
    ClipScorer,
    NamingConfig,
    crop_box,
    crop_window,
    name_boxes,
    softmax,
    vocabulary,
)
from measured_space.perception import prompt_phrases

COLOUR_PHRASE = {0: "chair", 1: "table", 2: "wall"}
"""dict of int to str: the fake model's reading of a crop's dominant channel (red, green, blue)."""


class ColourModel:
    """A fake CLIP: each crop's centre pixel picks a phrase by its brightest channel.

    Attributes
    ----------
    calls : list of tuple
        ``(n_images, texts)`` per call.
    """

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.calls: list[tuple[int, list[str]]] = []

    def __call__(self, images: Sequence[Image.Image], texts: Sequence[str]) -> NDArray[np.float64]:
        """Logits of 8 for the phrase the centre colour names, 0 elsewhere.

        Parameters
        ----------
        images : sequence of PIL.Image.Image
            Crops.
        texts : sequence of str
            Prompts, ``PROMPT_TEMPLATE`` formatted.

        Returns
        -------
        numpy.ndarray, shape (len(images), len(texts))
            The logits.
        """
        self.calls.append((len(images), list(texts)))
        phrases = [t.removeprefix(PROMPT_TEMPLATE.format("")) for t in texts]
        out = np.zeros((len(images), len(texts)))
        for i, image in enumerate(images):
            a = np.asarray(image)
            centre = a[a.shape[0] // 2, a.shape[1] // 2]
            out[i, phrases.index(COLOUR_PHRASE[int(np.argmax(centre))])] = 8.0
        return out


def coloured_image() -> tuple[NDArray[np.uint8], list[tuple[float, float, float, float]]]:
    """A black 120 x 200 image with a red, a green and a blue rectangle, and their boxes.

    Returns
    -------
    image : numpy.ndarray, shape (120, 200, 3), uint8
        The image.
    boxes : list of tuple of float
        The rectangles' boxes (red, green, blue), pixel edges at integer coordinates.
    """
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    boxes = [(10.0, 10.0, 60.0, 50.0), (80.0, 20.0, 190.0, 40.0), (20.0, 70.0, 40.0, 115.0)]
    for channel, (x0, y0, x1, y1) in enumerate(boxes):
        image[int(y0):int(y1), int(x0):int(x1), channel] = 255
    return image, boxes


def test_vocabulary_puts_dataset_phrases_first_and_drops_their_duplicates() -> None:
    """ "shelf" is an ARKitScenes class, so it is an "other" phrase only for Smart Spaces."""
    named, other = vocabulary("arkitscenes")
    assert list(named) == prompt_phrases("arkitscenes") and "shelf" in named
    assert "shelf" not in other and "wall" in other and "small object" in other
    named_ss, other_ss = vocabulary("smartspaces")
    assert list(named_ss) == prompt_phrases("smartspaces") and "shelf" in other_ss
    assert vocabulary("arkitscenes", ["wall", "wall", "chair"])[1] == ("wall",)


def test_crop_window_adds_the_margin_and_stays_in_the_image() -> None:
    """A 10 % margin per side, clipped to the image; a degenerate box still gets one pixel."""
    assert crop_window((10, 20, 30, 60), 100, 100, 0.1) == (8, 16, 32, 64)
    assert crop_window((10, 20, 30, 60), 100, 100, 0.0) == (10, 20, 30, 60)
    assert crop_window((0, 0, 50, 50), 100, 100, 0.1) == (0, 0, 55, 55)
    assert crop_window((95, 95, 120, 120), 100, 100, 0.1) == (92, 92, 100, 100)
    assert crop_window((50, 50, 50, 50), 100, 100, 0.1) == (50, 50, 51, 51)
    with pytest.raises(ValueError):
        crop_window((30, 20, 10, 60), 100, 100)
    with pytest.raises(ValueError):
        crop_window((np.nan, 20, 10, 60), 100, 100)


def test_crop_box_pads_an_elongated_crop_to_a_square() -> None:
    """The window (box plus margin) is centred on a square canvas of CLIP's mean colour."""
    image, boxes = coloured_image()
    crop = np.asarray(crop_box(image, boxes[1], 0.1))         # (80, 20, 190, 40) -> window (69, 18, 200, 42)
    assert crop.shape == (131, 131, 3)
    oy = (131 - 24) // 2
    assert tuple(crop[0, 0]) == PAD_RGB and tuple(crop[oy - 1, 60]) == PAD_RGB
    assert tuple(crop[oy, 0]) == (0, 0, 0)                     # margin row above the rectangle: context
    assert tuple(crop[65, 65]) == (0, 255, 0)
    with pytest.raises(ValueError):
        crop_box(image[..., 0], boxes[0])


def test_name_boxes_suppresses_names_of_other_phrases() -> None:
    """A box CLIP calls "wall" has no name but keeps its best phrase; prompts follow the vocabulary."""
    image, boxes = coloured_image()
    model = ColourModel()
    names = name_boxes(image, boxes, "arkitscenes", model)
    named, other = vocabulary("arkitscenes")
    assert model.calls == [(3, [PROMPT_TEMPLATE.format(p) for p in named + other])]
    assert [n.phrase for n in names] == ["chair", "table", None]
    assert [n.best_phrase for n in names] == ["chair", "table", "wall"]
    assert names[2].unnamed_reason == UNNAMED_OTHER and names[0].unnamed_reason is None
    for n in names:
        assert len(n.probs) == len(named + other) and sum(n.probs) == pytest.approx(1.0)
        assert n.prob == pytest.approx(max(n.probs)) and n.prob > 0.9


def test_min_prob_is_off_by_default_and_withholds_uncertain_names() -> None:
    """A near tie between two classes is named by default and unnamed above a probability threshold."""
    image, boxes = coloured_image()

    def tie(images: Sequence[Image.Image], texts: Sequence[str]) -> NDArray[np.float64]:
        """Logits 1.0 for "chair" and 0.9 for "table".

        Parameters
        ----------
        images : sequence of PIL.Image.Image
            Crops.
        texts : sequence of str
            Prompts.

        Returns
        -------
        numpy.ndarray
            The logits.
        """
        out = np.zeros((len(images), len(texts)))
        out[:, texts.index(PROMPT_TEMPLATE.format("chair"))] = 1.0
        out[:, texts.index(PROMPT_TEMPLATE.format("table"))] = 0.9
        return out

    assert NamingConfig().min_prob is None
    default = name_boxes(image, boxes[:1], "arkitscenes", tie)[0]
    assert default.phrase == "chair" and default.prob < 0.5
    strict = name_boxes(image, boxes[:1], "arkitscenes", tie, NamingConfig(min_prob=0.5))[0]
    assert strict.phrase is None and strict.best_phrase == "chair" and strict.unnamed_reason == UNNAMED_LOW_PROB


def test_name_boxes_edge_cases() -> None:
    """No boxes means no model call; a model with the wrong output shape is an error; bad settings fail."""
    image, boxes = coloured_image()
    model = ColourModel()
    assert name_boxes(image, [], "arkitscenes", model) == [] and model.calls == []
    with pytest.raises(ValueError):
        name_boxes(image, boxes, "arkitscenes", lambda images, texts: np.zeros((len(images), 2)))
    for bad in ({"margin": -0.1}, {"template": "a photo"}, {"min_prob": 1.5}):
        with pytest.raises(ValueError):
            NamingConfig(**bad)
    assert json.loads(json.dumps(NamingConfig().to_json()))["other_phrases"][0] == "wall"
    assert softmax(np.array([[1000.0, 1000.0]])).tolist() == [[0.5, 0.5]]


def _tiny_clip_cache(root: Path, revision: str) -> tuple[object, object, object]:
    """Write a tiny random CLIP, its tokenizer and a PIL image processor into a Hugging Face cache layout.

    Parameters
    ----------
    root : pathlib.Path
        The cache folder to create.
    revision : str
        Snapshot name (also written as ``refs/main``).

    Returns
    -------
    tuple
        ``(model, tokenizer, image_processor)`` as written.
    """
    import torch
    from transformers import CLIPConfig, CLIPImageProcessorPil, CLIPModel, CLIPTokenizer

    words = root / "words"
    words.mkdir(parents=True)
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for c in "abcdefghijklmnopqrstuvwxyz":
        vocab[c] = len(vocab)
        vocab[c + "</w>"] = len(vocab)
    (words / "vocab.json").write_text(json.dumps(vocab))
    (words / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(vocab=str(words / "vocab.json"), merges=str(words / "merges.txt"))
    config = CLIPConfig(
        text_config={"vocab_size": len(vocab), "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "max_position_embeddings": 77, "eos_token_id": 1,
                     "bos_token_id": 0, "pad_token_id": 1},
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                       "image_size": 32, "patch_size": 16},
        projection_dim=8)
    torch.manual_seed(0)
    model = CLIPModel(config).eval()
    processor = CLIPImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    repo = root / "cache" / "models--org--tiny-clip"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    model.save_pretrained(snapshot)
    tokenizer.save_pretrained(snapshot)
    processor.save_pretrained(snapshot)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision)
    return model, tokenizer, processor


def test_clip_scorer_loads_offline_at_a_pinned_revision_and_batches_consistently(tmp_path: Path) -> None:
    """The loader reads only the local cache at the given snapshot; logits do not depend on the batch size.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder for the cache.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    _tiny_clip_cache(tmp_path, "abc123")
    cache = tmp_path / "cache"
    scorer = ClipScorer.load("org/tiny-clip", revision="abc123", device="cpu", cache_dir=cache, batch_size=2)
    assert scorer.info["snapshot"] == "abc123" and scorer.info["dtype"] == "float32"
    assert scorer.info["image_processor"] == "CLIPImageProcessorPil"
    image, boxes = coloured_image()
    crops = [crop_box(image, b) for b in boxes]
    texts = ["a photo of a chair", "a photo of a wall"]
    logits = scorer(crops, texts)
    assert logits.shape == (3, 2) and np.all(np.isfinite(logits))
    whole = ClipScorer(scorer.model, scorer.tokenizer, scorer.image_processor, batch_size=8)(crops, texts)
    np.testing.assert_allclose(logits, whole, rtol=1e-5, atol=1e-5)
    tokens = scorer.tokenizer(texts, padding=True, return_tensors="pt")
    pixels = scorer.image_processor(images=crops, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        direct = scorer.model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
                              pixel_values=pixels).logits_per_image.numpy()
    np.testing.assert_allclose(logits, direct, rtol=1e-5, atol=1e-5)
    assert scorer(crops[:0], texts).shape == (0, 2)
    names = name_boxes(image, boxes, "smartspaces", scorer)
    assert len(names) == 3 and all(sum(n.probs) == pytest.approx(1.0) for n in names)
    with pytest.raises(FileNotFoundError):
        ClipScorer.load("org/tiny-clip", revision="deadbeef", device="cpu", cache_dir=cache)
    with pytest.raises(ValueError):
        ClipScorer(scorer.model, scorer.tokenizer, scorer.image_processor, batch_size=0)
