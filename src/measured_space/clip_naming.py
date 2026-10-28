"""Zero-shot naming of 2D boxes with CLIP, for class-agnostic detectors.

A class-agnostic 3D detector (CuTR, :mod:`measured_space.box_sources`) finds objects but does not say what
they are. Each box is named by asking CLIP which phrase of a fixed vocabulary best describes the image region
under its 2D box. The design follows BoxFusion, the published pipeline that names CuTR boxes with CLIP, and the
choices below are fixed in advance, never tuned on the evaluation data.

**Vocabulary.** The dataset's prompt phrases (:func:`measured_space.perception.prompt_phrases`, the same phrases
SAM 3 is prompted with, so both automatic sources speak the same class language), followed by the "other"
phrases of :data:`OTHER_PHRASES`, minus any phrase the dataset already uses ("shelf" is an ARKitScenes class,
so it is an "other" phrase for Smart Spaces only). The "other" phrases give CLIP somewhere to put a box that is
none of the dataset's classes. Without them a softmax over the dataset phrases alone would force every box into
a class: a pillow would become a "sofa" and be counted as one. The list is short on purpose:

- **room structure** (``wall``, ``floor``, ``door``, ``window``): the 2D box of a large or truncated object
  can hold mostly structure;
- **the commonest objects outside the classes**: ``pillow``, ``lamp`` and ``picture`` in homes; ``shelf``
  (storage racks), ``pallet`` and ``box`` in warehouses;
- **a catch-all**, ``small object``: the class-agnostic detector also boxes small items.

**Naming rule.** The probabilities are a softmax over the whole vocabulary. A box whose most probable phrase is
an "other" phrase gets **no name**. The caller keeps it in the records, but it takes part in no question and no
count. Otherwise its name is that phrase and ``prob`` is its probability. :attr:`NamingConfig.min_prob` can
also withhold names below a probability. It defaults to ``None`` (off): a threshold chosen without held-out data
would be a tuned knob.

**Prompts.** ``"a photo of a {phrase}"`` (:data:`PROMPT_TEMPLATE`), one per phrase.

**Crops** (:func:`crop_box`). The box (pixel edges at integer coordinates, the convention of
:attr:`measured_space.perception.Instance.box2d`) is widened by :data:`CROP_MARGIN` of its width and height on
each side, for a little context, and clipped to the image. The crop is then padded to a square with CLIP's mean
colour (:data:`PAD_RGB`). CLIP's image processor resizes the shorter side and centre-crops a square, so without
the padding an elongated box (a sofa, a counter) would lose its ends.

**Model** (:class:`ClipScorer`). The default is :data:`CLIP_MODEL_ID`, the CLIP that BoxFusion uses, at the
pinned snapshot :data:`CLIP_REVISION`. It is loaded with ``transformers.CLIPModel`` and the PIL-backend
image processor (torchvision is not installed; see :mod:`measured_space.eval_vlm`), from the project's Hugging Face cache (``<data root>/hf_cache``, the layout
:func:`measured_space.eval_vlm.hf_cache_dir` describes: ``models--<org>--<name>`` directly inside it) with
``local_files_only=True``, so nothing is downloaded. Everything else takes the model as a plain function
``logits_fn(images, texts) -> logits`` (:data:`LogitsFn`), so tests use a fake.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from measured_space.perception import prompt_phrases

CLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
"""str: the default CLIP (OpenCLIP ViT-H/14, LAION-2B, MIT license), as used by BoxFusion."""

CLIP_REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
"""str: the pinned snapshot (commit) of :data:`CLIP_MODEL_ID`; only its transformers files are cached."""

PROMPT_TEMPLATE = "a photo of a {}"
"""str: the text prompt, formatted with each phrase."""

CROP_MARGIN = 0.1
"""float: context added on each side of a box before cropping, as a fraction of the box's width (height)."""

PAD_RGB: tuple[int, int, int] = (123, 117, 104)
"""tuple of int: padding colour of the square crops, CLIP's mean pixel (0.481, 0.458, 0.408) in 8-bit RGB."""

OTHER_PHRASES: tuple[str, ...] = ("wall", "floor", "door", "window", "shelf", "box", "pallet", "pillow", "lamp",
                                  "picture", "small object")
"""tuple of str: phrases for things that are none of a dataset's classes (see the module docstring)."""

UNNAMED_OTHER = "other"
"""str: :attr:`BoxName.unnamed_reason` when the most probable phrase is an "other" phrase."""

UNNAMED_LOW_PROB = "below_min_prob"
"""str: :attr:`BoxName.unnamed_reason` when the best phrase's probability is below ``min_prob``."""

LogitsFn = Callable[[Sequence[Image.Image], Sequence[str]], NDArray[np.floating]]
"""Callable: ``logits_fn(images, texts)`` -> image-text logits, shape ``(len(images), len(texts))``."""


@dataclass(frozen=True)
class NamingConfig:
    """Every setting of the naming step.

    Parameters
    ----------
    margin : float, default CROP_MARGIN
        Context on each side of a box, as a fraction of its width (height); at least 0.
    template : str, default PROMPT_TEMPLATE
        Prompt with one ``{}`` for the phrase.
    other_phrases : tuple of str, default OTHER_PHRASES
        Phrases that mean "none of the dataset's classes".
    min_prob : float or None, default None
        If set, a box whose best dataset phrase has a lower probability gets no name; ``None`` disables it.
    """

    margin: float = CROP_MARGIN
    template: str = PROMPT_TEMPLATE
    other_phrases: tuple[str, ...] = OTHER_PHRASES
    min_prob: float | None = None

    def __post_init__(self) -> None:
        """Validate the margin, the template and the probability threshold.

        Raises
        ------
        ValueError
            If ``margin`` is negative, ``template`` lacks exactly one ``{}``, or ``min_prob`` is outside [0, 1].
        """
        if self.margin < 0:
            raise ValueError(f"margin must be at least 0, got {self.margin}")
        if self.template.count("{}") != 1:
            raise ValueError(f"template needs exactly one '{{}}', got {self.template!r}")
        if self.min_prob is not None and not 0.0 <= self.min_prob <= 1.0:
            raise ValueError(f"min_prob must be in [0, 1] or None, got {self.min_prob}")

    def to_json(self) -> dict[str, Any]:
        """The settings as a JSON-serialisable dict.

        Returns
        -------
        dict
            Every field (``other_phrases`` as a list).
        """
        record = asdict(self)
        record["other_phrases"] = list(self.other_phrases)
        return record


@dataclass(frozen=True)
class BoxName:
    """CLIP's verdict on one box.

    Parameters
    ----------
    phrase : str or None
        The box's name, a dataset phrase; ``None`` if it gets no name.
    best_phrase : str
        The most probable phrase of the whole vocabulary (an "other" phrase when ``unnamed_reason`` is
        :data:`UNNAMED_OTHER`).
    prob : float
        Probability of ``best_phrase``, softmax over the whole vocabulary.
    unnamed_reason : str or None
        :data:`UNNAMED_OTHER` or :data:`UNNAMED_LOW_PROB` when ``phrase`` is ``None``, else ``None``.
    probs : tuple of float
        Probability of every vocabulary phrase, in :func:`vocabulary` order.
    """

    phrase: str | None
    best_phrase: str
    prob: float
    unnamed_reason: str | None
    probs: tuple[float, ...]


def vocabulary(dataset: str, other_phrases: Sequence[str] = OTHER_PHRASES) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A dataset's naming vocabulary: its prompt phrases, then the "other" phrases it does not already use.

    Parameters
    ----------
    dataset : str
        Dataset name, a key of :data:`measured_space.perception.CLASS_PHRASES`.
    other_phrases : sequence of str, default OTHER_PHRASES
        Candidate "other" phrases.

    Returns
    -------
    named : tuple of str
        The dataset's prompt phrases, in mapping order.
    other : tuple of str
        The "other" phrases minus the dataset's phrases, each once, in input order.

    Raises
    ------
    KeyError
        If the dataset has no phrase mapping.
    """
    named = tuple(prompt_phrases(dataset))
    other = tuple(dict.fromkeys(p for p in other_phrases if p not in named))
    return named, other


def crop_window(box2d: Sequence[float], width: int, height: int, margin: float = CROP_MARGIN
                ) -> tuple[int, int, int, int]:
    """The integer pixel window cropped for a box: the box widened by a margin, clipped to the image.

    Parameters
    ----------
    box2d : sequence of float, shape (4,)
        ``(xmin, ymin, xmax, ymax)``, pixels, edges at integer coordinates.
    width, height : int
        Image size, pixels.
    margin : float, default CROP_MARGIN
        Context on each side, as a fraction of the box's width (height).

    Returns
    -------
    tuple of int
        ``(left, top, right, bottom)`` with ``image[top:bottom, left:right]`` the crop; at least one pixel.

    Raises
    ------
    ValueError
        If the box is not finite or has negative width or height.
    """
    x0, y0, x1, y1 = (float(v) for v in box2d)
    if not np.all(np.isfinite([x0, y0, x1, y1])) or x1 < x0 or y1 < y0:
        raise ValueError(f"invalid 2D box {tuple(box2d)}")
    w, h = x1 - x0, y1 - y0
    left = int(np.clip(np.floor(x0 - margin * w), 0, width - 1))
    top = int(np.clip(np.floor(y0 - margin * h), 0, height - 1))
    right = int(np.clip(np.ceil(x1 + margin * w), left + 1, width))
    bottom = int(np.clip(np.ceil(y1 + margin * h), top + 1, height))
    return left, top, right, bottom


def crop_box(image: NDArray[np.uint8], box2d: Sequence[float], margin: float = CROP_MARGIN,
             pad_rgb: tuple[int, int, int] = PAD_RGB) -> Image.Image:
    """Crop a box with context and pad it to a square, ready for CLIP's image processor.

    Parameters
    ----------
    image : numpy.ndarray, shape (H, W, 3), uint8
        RGB image.
    box2d : sequence of float, shape (4,)
        ``(xmin, ymin, xmax, ymax)``, pixels, edges at integer coordinates.
    margin : float, default CROP_MARGIN
        Context on each side, as a fraction of the box's width (height).
    pad_rgb : tuple of int, default PAD_RGB
        Padding colour.

    Returns
    -------
    PIL.Image.Image
        RGB, square, side ``max`` of the window's width and height, the window centred in it.

    Raises
    ------
    ValueError
        If the image is not ``(H, W, 3)`` uint8 or the box is invalid.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB image, got {image.shape} {image.dtype}")
    left, top, right, bottom = crop_window(box2d, image.shape[1], image.shape[0], margin)
    patch = image[top:bottom, left:right]
    h, w = patch.shape[:2]
    side = max(h, w)
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[:] = np.asarray(pad_rgb, dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = patch
    return Image.fromarray(canvas)


def softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    """Row-wise softmax, computed stably in float64.

    Parameters
    ----------
    logits : numpy.ndarray, shape (N, V)
        Scores.

    Returns
    -------
    numpy.ndarray, shape (N, V)
        Probabilities; each row sums to 1.
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def name_boxes(image: NDArray[np.uint8], boxes2d: Sequence[Sequence[float]], dataset: str, logits_fn: LogitsFn,
               config: NamingConfig | None = None) -> list[BoxName]:
    """Name each box by CLIP zero-shot classification of its crop.

    Parameters
    ----------
    image : numpy.ndarray, shape (H, W, 3), uint8
        The view's RGB image, in the orientation the boxes are given in.
    boxes2d : sequence of sequence of float, each shape (4,)
        ``(xmin, ymin, xmax, ymax)`` per box, pixels, edges at integer coordinates.
    dataset : str
        Dataset name; selects the vocabulary (:func:`vocabulary`).
    logits_fn : callable
        ``logits_fn(images, texts)`` -> logits of shape ``(len(images), len(texts))`` (:data:`LogitsFn`), e.g. a
        :class:`ClipScorer`.
    config : NamingConfig, optional
        Settings; defaults to ``NamingConfig()``.

    Returns
    -------
    list of BoxName
        One per box, in input order; empty (without calling the model) for no boxes.

    Raises
    ------
    ValueError
        If the model returns logits of the wrong shape, or as :func:`crop_box`.
    """
    config = config or NamingConfig()
    if len(boxes2d) == 0:
        return []
    named, other = vocabulary(dataset, config.other_phrases)
    phrases = named + other
    texts = [config.template.format(p) for p in phrases]
    crops = [crop_box(image, box, config.margin) for box in boxes2d]
    logits = np.asarray(logits_fn(crops, texts), dtype=np.float64)
    if logits.shape != (len(crops), len(texts)):
        raise ValueError(f"model returned logits of shape {logits.shape}, expected {(len(crops), len(texts))}")
    probs = softmax(logits)
    other_set = set(other)
    names = []
    for row in probs:
        k = int(np.argmax(row))
        best, p = phrases[k], float(row[k])
        if best in other_set:
            phrase, reason = None, UNNAMED_OTHER
        elif config.min_prob is not None and p < config.min_prob:
            phrase, reason = None, UNNAMED_LOW_PROB
        else:
            phrase, reason = best, None
        names.append(BoxName(phrase, best, p, reason, tuple(float(v) for v in row)))
    return names


class ClipScorer:
    """A CLIP model as a :data:`LogitsFn`: image-text logits for crops and prompts.

    Build it with :meth:`load` (weights from the local Hugging Face cache) or directly from loaded components.

    Parameters
    ----------
    model : transformers.CLIPModel
        The model, in eval mode, on ``device``.
    tokenizer : transformers.PreTrainedTokenizerBase
        Its tokenizer.
    image_processor : transformers.BaseImageProcessor
        Its image processor (PIL backend).
    device : str, default "cpu"
        Torch device of the model and inputs.
    batch_size : int, default 32
        Images per forward pass.
    info : mapping, optional
        Model id, snapshot and settings, recorded with the outputs.

    Raises
    ------
    ValueError
        If ``batch_size`` is below 1.
    """

    def __init__(self, model: Any, tokenizer: Any, image_processor: Any, *, device: str = "cpu",
                 batch_size: int = 32, info: Mapping[str, Any] | None = None) -> None:
        """Keep the components.

        Parameters
        ----------
        model : transformers.CLIPModel
            The model, in eval mode, on ``device``.
        tokenizer : transformers.PreTrainedTokenizerBase
            Its tokenizer.
        image_processor : transformers.BaseImageProcessor
            Its image processor.
        device : str, default "cpu"
            Torch device of the model and inputs.
        batch_size : int, default 32
            Images per forward pass.
        info : mapping, optional
            Recorded settings.

        Raises
        ------
        ValueError
            If ``batch_size`` is below 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.batch_size = batch_size
        self._info = dict(info or {})

    @classmethod
    def load(cls, model_id: str = CLIP_MODEL_ID, *, revision: str | None = None, device: str = "cuda",
             cache_dir: str | Path | None = None, batch_size: int = 32, dtype: str | None = None) -> ClipScorer:
        """Load a CLIP model, its tokenizer and its PIL image processor from the local cache, at a pinned snapshot.

        Parameters
        ----------
        model_id : str, default CLIP_MODEL_ID
            Hugging Face repository id.
        revision : str, optional
            Snapshot (commit hash) to load; defaults to :data:`CLIP_REVISION` for :data:`CLIP_MODEL_ID` and to the
            cache's ``main`` reference for any other model.
        device : str, default "cuda"
            Torch device.
        cache_dir : str or pathlib.Path, optional
            Hugging Face hub cache; defaults to :func:`measured_space.eval_vlm.hf_cache_dir`. Files are read with
            ``local_files_only=True``: nothing is downloaded.
        batch_size : int, default 32
            Images per forward pass.
        dtype : str, optional
            Torch dtype name; defaults to ``"float16"`` on CUDA and ``"float32"`` otherwise.

        Returns
        -------
        ClipScorer
            The scorer; :attr:`info` holds the model id, the snapshot loaded, dtype, device, image processor
            class, library versions and load time (s).

        Raises
        ------
        RuntimeError
            If the data root or the model cache is missing.
        FileNotFoundError
            If the model (at that revision) is not in the cache.
        OSError
            If other model files are not in the cache.
        """
        import torch
        import transformers
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel

        from measured_space.eval_vlm import hf_cache_dir

        start = time.perf_counter()
        cache = Path(cache_dir) if cache_dir is not None else hf_cache_dir()
        if revision is None and model_id == CLIP_MODEL_ID:
            revision = CLIP_REVISION
        config_path = try_to_load_from_cache(model_id, "config.json", cache_dir=str(cache), revision=revision)
        if not isinstance(config_path, str):
            raise FileNotFoundError(f"{model_id} at revision {revision or 'main'} is not in the cache at {cache}")
        snapshot = Path(config_path).parent.name
        dtype_name = dtype or ("float16" if device.startswith("cuda") else "float32")
        common = {"cache_dir": str(cache), "local_files_only": True, "revision": snapshot}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
        image_processor = AutoImageProcessor.from_pretrained(model_id, backend="pil", **common)
        model = CLIPModel.from_pretrained(model_id, dtype=getattr(torch, dtype_name), **common).to(device).eval()
        info = {"model_id": model_id, "snapshot": snapshot, "dtype": dtype_name, "device": device,
                "batch_size": batch_size, "image_processor": type(image_processor).__name__,
                "transformers": transformers.__version__, "torch": torch.__version__,
                "load_seconds": round(time.perf_counter() - start, 2)}
        return cls(model, tokenizer, image_processor, device=device, batch_size=batch_size, info=info)

    @property
    def info(self) -> dict[str, Any]:
        """dict: the model id, snapshot and settings given at construction (a copy)."""
        return dict(self._info)

    def __call__(self, images: Sequence[Image.Image], texts: Sequence[str]) -> NDArray[np.float64]:
        """Image-text logits (CLIP's scaled cosine similarities).

        Parameters
        ----------
        images : sequence of PIL.Image.Image
            RGB crops.
        texts : sequence of str
            Prompts.

        Returns
        -------
        numpy.ndarray, shape (len(images), len(texts))
            ``logits_per_image``, float64.
        """
        import torch

        if len(images) == 0:
            return np.zeros((0, len(texts)), dtype=np.float64)
        dtype = next(self.model.parameters()).dtype
        tokens = self.tokenizer(list(texts), padding=True, return_tensors="pt")
        text_inputs = {k: tokens[k].to(self.device) for k in ("input_ids", "attention_mask") if k in tokens}
        out = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = list(images[start:start + self.batch_size])
                pixels = self.image_processor(images=batch, return_tensors="pt")["pixel_values"]
                logits = self.model(**text_inputs, pixel_values=pixels.to(self.device, dtype)).logits_per_image
                out.append(logits.float().cpu().numpy().astype(np.float64))
        return np.concatenate(out, axis=0)
