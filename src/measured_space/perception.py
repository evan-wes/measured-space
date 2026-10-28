"""Automatic 3D labels from one image: text-prompted instance masks, metric depth, and robust box lifting.

The automatic ("auto") labels that the audit compares with ground truth come from three replaceable stages:

1. :class:`Sam3Segmenter` finds every instance of a noun phrase in an image (SAM 3 concept segmentation).
2. :class:`MogeDepth` predicts metric depth from the RGB image alone (MoGe-2), for settings with no depth
   sensor. Sensor depth (LiDAR, laser) can be used instead: lifting does not care where depth came from.
3. :func:`lift_instance` turns one mask and a depth map into a gravity-aligned :class:`OrientedBox`, and
   returns cheap image-evidence features (share of mask pixels near an edge, depth spread, truncation, how
   much of the footprint is observed, ...) that are later used to predict how wrong the box is likely to be.

Lifting is robust rather than exact on purpose. Mask edges straddle depth discontinuities, so edge pixels
take background or occluder depth ("flying pixels"): erosion removes the edge band and a MAD test on depth
removes what is left. Extents come from percentiles rather than min/max for the same reason, which shrinks a
uniformly covered face by about the trimmed percentage per side. A camera sees only an object's near faces,
so the box measures the *visible* surface and the extent along the viewing direction tends to be undersized
(amodal undersizing). The features are meant to expose that, not to correct it.

Models are loaded once and kept: the module-level :func:`segment_instances` and :func:`estimate_depth_moge`
share one process-wide instance of each model. Weights are read from local caches only; nothing is downloaded.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from measured_space.config import data_root
from measured_space.geometry import Array, Camera, OrientedBox, transform

if TYPE_CHECKING:
    import torch

SAM3_MODEL_ID = "facebook/sam3"
MOGE_MODEL_ID = "Ruicheng/moge-2-vitl-normal"

# Consistency factor that turns a median absolute deviation into a standard deviation for normal data.
_MAD_TO_SIGMA = 1.4826

# --------------------------------------------------------------------------------------------------------
# Dataset classes and the noun phrases used to prompt for them
# --------------------------------------------------------------------------------------------------------

ARKITSCENES_PHRASES: dict[str, str] = {
    "cabinet": "cabinet",
    "refrigerator": "refrigerator",
    "shelf": "shelf",
    "stove": "stove",
    "bed": "bed",
    "sink": "sink",
    "washer": "washer",
    "toilet": "toilet",
    "bathtub": "bathtub",
    "oven": "oven",
    "dishwasher": "dishwasher",
    "fireplace": "fireplace",
    "stool": "stool",
    "chair": "chair",
    "table": "table",
    "tv_monitor": "tv",
    "sofa": "sofa",
}
"""ARKitScenes 3DOD label -> prompt phrase. All 17 classes; only ``tv_monitor`` needs rewording."""

SMARTSPACES_PHRASES: dict[str, str] = {
    "Person": "person",
    "Forklift": "forklift",
    "PalletTruck": "pallet jack",
    "NovaCarter": "mobile robot",
    "Transporter": "mobile robot",
    "AgilityDigit": "humanoid robot",
    "FourierGR1T2": "humanoid robot",
}
"""Physical AI Smart Spaces label -> prompt phrase.

Product names mean nothing to a text encoder, so robots are prompted by kind. The mapping is therefore
many-to-one: a "mobile robot" detection may be a NovaCarter or a Transporter, and a "humanoid robot" an
AgilityDigit or a FourierGR1T2. :func:`labels_for_phrase` returns every candidate, and matching against
ground truth must accept any of them.
"""

CLASS_PHRASES: dict[str, dict[str, str]] = {"arkitscenes": ARKITSCENES_PHRASES, "smartspaces": SMARTSPACES_PHRASES}
"""Dataset name (as in ``FrameView.dataset``) -> {dataset label: prompt phrase}."""


def _class_map(dataset: str) -> dict[str, str]:
    """Look up a dataset's label-to-phrase mapping.

    Parameters
    ----------
    dataset : str
        Dataset name, a key of :data:`CLASS_PHRASES`.

    Returns
    -------
    dict of str to str
        The dataset's {label: phrase} mapping.

    Raises
    ------
    KeyError
        If the dataset has no mapping.
    """
    try:
        return CLASS_PHRASES[dataset]
    except KeyError:
        known = sorted(CLASS_PHRASES)
        raise KeyError(f"No class-phrase mapping for dataset {dataset!r}; known: {known}") from None


def prompt_phrases(dataset: str) -> list[str]:
    """The distinct prompt phrases for a dataset's classes, in mapping order.

    Parameters
    ----------
    dataset : str
        Dataset name, a key of :data:`CLASS_PHRASES`.

    Returns
    -------
    list of str
        Each phrase once (several labels may share one phrase).

    Raises
    ------
    KeyError
        If the dataset has no mapping.
    """
    return list(dict.fromkeys(_class_map(dataset).values()))


def labels_for_phrase(dataset: str, phrase: str) -> tuple[str, ...]:
    """The dataset labels a detection of ``phrase`` may correspond to.

    Parameters
    ----------
    dataset : str
        Dataset name, a key of :data:`CLASS_PHRASES`.
    phrase : str
        A prompt phrase, as stored in :attr:`Instance.phrase`.

    Returns
    -------
    tuple of str
        Candidate labels in mapping order; empty if the phrase belongs to none of the dataset's classes.

    Raises
    ------
    KeyError
        If the dataset has no mapping.
    """
    return tuple(label for label, p in _class_map(dataset).items() if p == phrase)


# --------------------------------------------------------------------------------------------------------
# Instance segmentation (SAM 3)
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One object found by a text prompt.

    Attributes
    ----------
    phrase : str
        The noun phrase that found it.
    mask : numpy.ndarray, shape (H, W), bool
        Instance mask at the input image's resolution and orientation.
    box2d : tuple of float
        ``(xmin, ymin, xmax, ymax)`` in pixels of the input image, from the model's box head; pixel edges are
        at integer coordinates (a box covering the whole image is ``(0, 0, W, H)``).
    score : float
        Detection confidence in [0, 1]: the query's probability times the image-level presence probability.
    """

    phrase: str
    mask: NDArray[np.bool_]
    box2d: tuple[float, float, float, float]
    score: float


def _check_image(image: NDArray[Any]) -> NDArray[np.uint8]:
    """Validate an RGB image.

    Parameters
    ----------
    image : numpy.ndarray, shape (H, W, 3), uint8
        RGB image.

    Returns
    -------
    numpy.ndarray, shape (H, W, 3), uint8
        The same image, C-contiguous.

    Raises
    ------
    ValueError
        If the shape or dtype is wrong.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected an RGB uint8 image of shape (H, W, 3); got {image.dtype} {image.shape}")
    return np.ascontiguousarray(image)


def rot90_points(xy: Array, k: int, shape: tuple[int, int]) -> Array:
    """Map continuous pixel coordinates through ``numpy.rot90(image, k)``.

    Pixel edges are at integer coordinates, so a point at ``(x, y)`` of an ``(H, W)`` image moves to
    ``(y, W - x)`` under one counter-clockwise quarter turn (``k = 1``).

    Parameters
    ----------
    xy : numpy.ndarray, shape (N, 2), float
        Points ``(x, y)`` in the image before rotation.
    k : int
        Number of counter-clockwise quarter turns, as in :func:`numpy.rot90`; negative turns clockwise.
    shape : tuple of int
        ``(H, W)`` of the image before rotation.

    Returns
    -------
    numpy.ndarray, shape (N, 2), float
        The points in the rotated image.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2).copy()
    h, w = shape
    for _ in range(k % 4):
        xy = np.stack([xy[:, 1], w - xy[:, 0]], axis=1)
        h, w = w, h
    return xy


def rot90_camera(camera: Camera, k: int) -> Camera:
    """The intrinsics of ``numpy.rot90(image, k)`` for an image taken with ``camera``.

    Rotating the image by quarter turns is rotating the camera about its optical axis: focal lengths swap
    on odd turns and the principal point moves with :func:`rot90_points`. Depth is unchanged.

    Parameters
    ----------
    camera : Camera
        Intrinsics of the unrotated image.
    k : int
        Number of counter-clockwise quarter turns, as in :func:`numpy.rot90`.

    Returns
    -------
    Camera
        Intrinsics of the rotated image.
    """
    centre = rot90_points([[camera.cx + 0.5, camera.cy + 0.5]], k, (camera.height, camera.width))[0] - 0.5
    if k % 2:
        return Camera(camera.fy, camera.fx, float(centre[0]), float(centre[1]), camera.height, camera.width)
    return Camera(camera.fx, camera.fy, float(centre[0]), float(centre[1]), camera.width, camera.height)


def mask_iou(masks: NDArray[np.bool_]) -> Array:
    """Pairwise intersection-over-union of binary masks.

    Parameters
    ----------
    masks : numpy.ndarray, shape (N, H, W), bool
        Masks of one image.

    Returns
    -------
    numpy.ndarray, shape (N, N), float64
        IoU matrix; 0 where both masks are empty.
    """
    flat = masks.reshape(len(masks), -1).astype(np.float32)
    inter = flat @ flat.T
    area = flat.sum(axis=1)
    union = area[:, None] + area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).astype(np.float64)


def suppress_duplicates(instances: list[Instance], iou_threshold: float) -> list[Instance]:
    """Greedy non-maximum suppression on masks, across phrases.

    Different phrases can fire on the same object (a cabinet that is also a "shelf"); one physical object
    should yield one automatic label, so the higher-scoring instance wins.

    Parameters
    ----------
    instances : list of Instance
        Instances of one image.
    iou_threshold : float
        An instance is dropped if its mask IoU with a kept, higher-scoring instance exceeds this.

    Returns
    -------
    list of Instance
        The kept instances, highest score first.
    """
    order = sorted(instances, key=lambda inst: -inst.score)
    if len(order) < 2:
        return order
    iou = mask_iou(np.stack([inst.mask for inst in order]))
    kept: list[int] = []
    for i in range(len(order)):
        if all(iou[i, j] <= iou_threshold for j in kept):
            kept.append(i)
    return [order[i] for i in kept]


def _default_device() -> str:
    """The device models run on by default.

    Returns
    -------
    str
        ``"cuda"`` if a GPU is available, else ``"cpu"``.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Sam3Segmenter:
    """SAM 3 text-prompted instance segmentation, loaded once and reused.

    Uses transformers' native ``Sam3Model`` and the checkpoint's own tokenizer. Image preprocessing
    (antialiased bilinear resize to the model's square input, rescale, normalise) and post-processing
    (score = sigmoid(query logit) x sigmoid(presence logit); mask probabilities upsampled bilinearly, then
    thresholded) follow transformers' ``Sam3ImageProcessor``, with its settings read from the snapshot's
    ``processor_config.json``. That class is not used directly because it imports torchvision, which this
    environment does not install; the only difference is that resizing is done in float rather than uint8.
    """

    def __init__(self, model_id: str = SAM3_MODEL_ID, cache_dir: str | Path | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None) -> None:
        """Load the model, tokenizer and preprocessing settings from the local cache.

        Parameters
        ----------
        model_id : str, optional
            Hugging Face repository id of the weights.
        cache_dir : str or pathlib.Path or None, optional
            Hugging Face hub cache holding the snapshot; default ``<data_root>/hf_cache``.
        device : str or None, optional
            Torch device; default CUDA when available.
        dtype : torch.dtype or None, optional
            Weight dtype; default float32.

        Raises
        ------
        OSError
            If the weights are not in the local cache (nothing is downloaded).
        """
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, Sam3Model

        cache = str(cache_dir if cache_dir is not None else data_root() / "hf_cache")
        self.device = torch.device(device or _default_device())
        self.dtype = dtype or torch.float32
        self.model = Sam3Model.from_pretrained(model_id, cache_dir=cache, local_files_only=True,
                                               dtype=self.dtype).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, local_files_only=True)
        config_path = hf_hub_download(model_id, "processor_config.json", cache_dir=cache, local_files_only=True)
        settings = json.loads(Path(config_path).read_text())["image_processor"]
        self.input_size = (int(settings["size"]["height"]), int(settings["size"]["width"]))
        self.rescale = float(settings["rescale_factor"])
        self.mean = torch.tensor(settings["image_mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(settings["image_std"], device=self.device).view(1, 3, 1, 1)
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _pixel_values(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Turn an RGB image into the model's normalised square input.

        Parameters
        ----------
        image : numpy.ndarray, shape (H, W, 3), uint8
            RGB image.

        Returns
        -------
        torch.Tensor, shape (1, 3, S, S), float
            Pixel values on the model's device and dtype.
        """
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=self.input_size, mode="bilinear", antialias=True, align_corners=False)
        x = (x * self.rescale - self.mean) / self.std
        return x.to(self.dtype)

    def _tokens(self, phrase: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a phrase once and cache it.

        Parameters
        ----------
        phrase : str
            Noun phrase.

        Returns
        -------
        tuple of torch.Tensor
            ``input_ids`` and ``attention_mask``, each of shape (1, 32), on the model's device.
        """
        if phrase not in self._text_cache:
            tokens = self.tokenizer([phrase], padding="max_length", max_length=32, return_tensors="pt")
            self._text_cache[phrase] = (tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device))
        return self._text_cache[phrase]

    def segment(self, image: NDArray[np.uint8], phrases: list[str], score_threshold: float = 0.5, *,
                mask_threshold: float = 0.5, rot90_k: int = 0,
                dedupe_iou: float | None = 0.7) -> list[Instance]:
        """Find every instance of each phrase in one image.

        The image is encoded once; the text-conditioned decoder then runs once per phrase.

        Parameters
        ----------
        image : numpy.ndarray, shape (H, W, 3), uint8
            RGB image.
        phrases : list of str
            Noun phrases to look for; duplicates are ignored.
        score_threshold : float, optional
            Minimum detection score kept.
        mask_threshold : float, optional
            Probability above which a pixel belongs to a mask.
        rot90_k : int, optional
            Segment ``numpy.rot90(image, rot90_k)`` instead, e.g. to stand a sideways handheld frame
            upright; masks and boxes are mapped back to the input orientation.
        dedupe_iou : float or None, optional
            Mask-IoU threshold for :func:`suppress_duplicates` across phrases; ``None`` keeps everything.

        Returns
        -------
        list of Instance
            Instances with non-empty masks, highest score first, in the input image's frame.

        Raises
        ------
        ValueError
            If the image is not an RGB uint8 array.
        """
        import torch
        import torch.nn.functional as F

        image = _check_image(image)
        work = np.ascontiguousarray(np.rot90(image, rot90_k)) if rot90_k % 4 else image
        h, w = work.shape[:2]
        instances: list[Instance] = []
        with torch.inference_mode():
            vision = self.model.get_vision_features(pixel_values=self._pixel_values(work))
            for phrase in dict.fromkeys(phrases):
                input_ids, attention_mask = self._tokens(phrase)
                out = self.model(vision_embeds=vision, input_ids=input_ids, attention_mask=attention_mask)
                scores = out.pred_logits[0].float().sigmoid()
                if out.presence_logits is not None:
                    scores = scores * out.presence_logits[0].float().sigmoid()
                keep = torch.nonzero(scores > score_threshold).flatten()
                if keep.numel() == 0:
                    continue
                probs = F.interpolate(out.pred_masks[0, keep][None].float().sigmoid(), size=(h, w),
                                      mode="bilinear", align_corners=False)[0]
                masks = (probs > mask_threshold).cpu().numpy()
                scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
                boxes = (out.pred_boxes[0, keep].float() * scale).cpu().numpy().astype(np.float64)
                for mask, box, score in zip(masks, boxes, scores[keep].cpu().numpy()):
                    instance = self._to_input_frame(phrase, mask, box, float(score), rot90_k, image.shape[:2])
                    if instance.mask.any():
                        instances.append(instance)
        if dedupe_iou is not None:
            return suppress_duplicates(instances, dedupe_iou)
        return sorted(instances, key=lambda inst: -inst.score)

    @staticmethod
    def _to_input_frame(phrase: str, mask: NDArray[np.bool_], box: Array, score: float, rot90_k: int,
                        shape: tuple[int, int]) -> Instance:
        """Undo the working rotation for one detection.

        Parameters
        ----------
        phrase : str
            Prompt phrase.
        mask : numpy.ndarray, shape (H', W'), bool
            Mask in the rotated working image.
        box : numpy.ndarray, shape (4,), float
            ``(xmin, ymin, xmax, ymax)`` pixels in the rotated working image.
        score : float
            Detection score.
        rot90_k : int
            Quarter turns that produced the working image from the input image.
        shape : tuple of int
            ``(H, W)`` of the input image.

        Returns
        -------
        Instance
            The detection in the input image's frame, box clipped to the image.
        """
        h, w = shape
        if rot90_k % 4:
            corners = rot90_points(np.array([[box[0], box[1]], [box[2], box[3]]]), -rot90_k, mask.shape)
            box = np.concatenate([corners.min(axis=0), corners.max(axis=0)])
            mask = np.ascontiguousarray(np.rot90(mask, -rot90_k))
        box = np.clip(box, 0.0, [w, h, w, h])
        return Instance(phrase, mask.astype(bool), (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        score)


# --------------------------------------------------------------------------------------------------------
# Monocular metric depth (MoGe-2)
# --------------------------------------------------------------------------------------------------------


def horizontal_fov_deg(camera: Camera) -> float:
    """Horizontal field of view of a pinhole camera, edge to edge.

    Parameters
    ----------
    camera : Camera
        Pinhole camera.

    Returns
    -------
    float
        Field of view in degrees; independent of the image resolution the camera is scaled to.
    """
    return float(np.degrees(2.0 * np.arctan(camera.width / (2.0 * camera.fx))))


class MogeDepth:
    """MoGe-2 metric depth (and intrinsics) from a single RGB image, loaded once and reused.

    MoGe-2 predicts an affine-invariant point map plus a metric scale. When the camera is known its
    horizontal field of view is passed in so that only the depth shift and scale are estimated. MoGe assumes
    the principal point is at the image centre; a small offset (a few pixels for phone cameras) is ignored.

    Feed it upright images (``rot90_k``): its metric scale degrades on sideways frames. On three sideways
    ARKitScenes frames the median relative depth error against the laser was 0.300, 0.115 and 0.234 as
    captured, and 0.008, 0.063 and 0.012 after standing the images upright with matching intrinsics.
    """

    def __init__(self, model_id: str = MOGE_MODEL_ID, cache_dir: str | Path | None = None,
                 device: str | None = None, num_tokens: int | None = None, use_fp16: bool = True) -> None:
        """Load the checkpoint from the local cache.

        Parameters
        ----------
        model_id : str, optional
            Hugging Face repository id of the checkpoint (a single ``model.pt``).
        cache_dir : str or pathlib.Path or None, optional
            Hugging Face hub cache holding it; default the user's default hub cache.
        device : str or None, optional
            Torch device; default CUDA when available.
        num_tokens : int or None, optional
            ViT token budget; ``None`` uses MoGe's default (the most tokens, slowest and most accurate).
        use_fp16 : bool, optional
            Run the network in mixed precision, as MoGe does by default.

        Raises
        ------
        OSError
            If the checkpoint is not in the local cache (nothing is downloaded).
        """
        import torch
        from moge.model.v2 import MoGeModel

        self.device = torch.device(device or _default_device())
        self.model = MoGeModel.from_pretrained(model_id, cache_dir=None if cache_dir is None else str(cache_dir),
                                               local_files_only=True).to(self.device).eval()
        self.num_tokens = num_tokens
        self.use_fp16 = use_fp16

    def infer(self, image: NDArray[np.uint8], camera: Camera | None = None,
              rot90_k: int = 0) -> tuple[NDArray[np.float32], Camera]:
        """Predict metric depth for one image.

        Parameters
        ----------
        image : numpy.ndarray, shape (H, W, 3), uint8
            RGB image.
        camera : Camera or None, optional
            Known intrinsics of ``image`` (any resolution; only the field of view is used). ``None`` lets
            MoGe estimate the field of view too.
        rot90_k : int, optional
            Run on ``numpy.rot90(image, rot90_k)`` (with the camera rotated to match) to present an upright
            image; depth and intrinsics are returned in the input orientation.

        Returns
        -------
        depth : numpy.ndarray, shape (H, W), float32
            Planar depth ``z`` in metres; NaN where MoGe marks the pixel invalid (e.g. sky).
        camera : Camera
            The intrinsics MoGe used, in pixels of ``image``.

        Raises
        ------
        ValueError
            If the image is not an RGB uint8 array.
        """
        import torch

        image = _check_image(image)
        if rot90_k % 4:
            image = np.ascontiguousarray(np.rot90(image, rot90_k))
            camera = None if camera is None else rot90_camera(camera, rot90_k)
        h, w = image.shape[:2]
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() / 255.0
        fov_x = None if camera is None else horizontal_fov_deg(camera)
        out = self.model.infer(x, num_tokens=self.num_tokens, fov_x=fov_x, use_fp16=self.use_fp16)
        depth = out["depth"].float().cpu().numpy().astype(np.float32)
        depth[~np.isfinite(depth)] = np.nan
        k = out["intrinsics"].float().cpu().numpy()          # normalised: image spans [0, 1] on both axes
        used = Camera(float(k[0, 0] * w), float(k[1, 1] * h), float(k[0, 2] * w - 0.5), float(k[1, 2] * h - 0.5),
                      w, h)
        if rot90_k % 4:
            depth = np.ascontiguousarray(np.rot90(depth, -rot90_k))
            used = rot90_camera(used, -rot90_k)
        return depth, used


@functools.cache
def default_segmenter() -> Sam3Segmenter:
    """The process-wide SAM 3 segmenter, created on first use.

    Returns
    -------
    Sam3Segmenter
        Shared instance with default settings.
    """
    return Sam3Segmenter()


@functools.cache
def default_moge() -> MogeDepth:
    """The process-wide MoGe-2 model, created on first use.

    Returns
    -------
    MogeDepth
        Shared instance with default settings.
    """
    return MogeDepth()


def segment_instances(image: NDArray[np.uint8], phrases: list[str], score_threshold: float = 0.5,
                      **kwargs: Any) -> list[Instance]:
    """Find every instance of each phrase in one image with the shared SAM 3 model.

    Parameters
    ----------
    image : numpy.ndarray, shape (H, W, 3), uint8
        RGB image.
    phrases : list of str
        Noun phrases, e.g. ``prompt_phrases("arkitscenes")``.
    score_threshold : float, optional
        Minimum detection score kept.
    **kwargs
        Passed to :meth:`Sam3Segmenter.segment` (``mask_threshold``, ``rot90_k``, ``dedupe_iou``).

    Returns
    -------
    list of Instance
        Instances, highest score first.
    """
    return default_segmenter().segment(image, phrases, score_threshold, **kwargs)


def estimate_depth_moge(image: NDArray[np.uint8], camera: Camera | None = None,
                        rot90_k: int = 0) -> NDArray[np.float32]:
    """Metric depth from one RGB image with the shared MoGe-2 model.

    Parameters
    ----------
    image : numpy.ndarray, shape (H, W, 3), uint8
        RGB image.
    camera : Camera or None, optional
        Known intrinsics; its field of view constrains the prediction.
    rot90_k : int, optional
        Quarter turns that stand the image upright (see :meth:`MogeDepth.infer`).

    Returns
    -------
    numpy.ndarray, shape (H, W), float32
        Planar depth in metres, in the input orientation; NaN where invalid.
    """
    return default_moge().infer(image, camera, rot90_k)[0]


# --------------------------------------------------------------------------------------------------------
# Lifting a mask to a gravity-aligned 3D box
# --------------------------------------------------------------------------------------------------------


class LiftError(ValueError):
    """Raised when a mask has too few usable depth pixels to fit a box."""


def resample_depth(depth: NDArray[np.floating], shape: tuple[int, int]) -> NDArray[np.floating]:
    """Nearest-neighbour resampling of a depth map to another resolution.

    Nearest rather than bilinear so that no depth is invented between a foreground edge and the background
    behind it. Pixel centres map as in :meth:`Camera.scaled`.

    Parameters
    ----------
    depth : numpy.ndarray, shape (h, w), float
        Depth map (any units).
    shape : tuple of int
        Target ``(H, W)``.

    Returns
    -------
    numpy.ndarray, shape (H, W)
        Resampled depth, same dtype.
    """
    hs, ws = depth.shape[:2]
    h, w = shape
    rows = np.minimum(((np.arange(h) + 0.5) * hs / h).astype(int), hs - 1)
    cols = np.minimum(((np.arange(w) + 0.5) * ws / w).astype(int), ws - 1)
    return depth[rows[:, None], cols[None, :]]


def mask_features(mask: NDArray[np.bool_], boundary_px: float = 5.0) -> dict[str, float]:
    """Image-only evidence about a mask, independent of depth.

    Parameters
    ----------
    mask : numpy.ndarray, shape (H, W), bool
        Instance mask; must not be empty.
    boundary_px : float, optional
        Width in pixels of the edge band counted by ``boundary_fraction``.

    Returns
    -------
    dict of str to float
        ``mask_area_px``: pixels in the mask. ``n_components``: 8-connected pieces (occlusion splits
        objects). ``boundary_fraction``: share of mask pixels within ``boundary_px`` of a non-mask pixel
        (the image border does not count as an edge). ``border_touch``: 1 if the mask reaches the image
        border (truncated object), else 0. ``n_border_sides``: how many of the four borders it reaches.
    """
    m8 = mask.astype(np.uint8)
    area = float(m8.sum())
    n_labels, _ = cv2.connectedComponents(m8, connectivity=8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sides = [mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()]
    return {
        "mask_area_px": area,
        "n_components": float(n_labels - 1),
        "boundary_fraction": float(np.count_nonzero(mask & (dist <= boundary_px)) / area),
        "border_touch": float(any(sides)),
        "n_border_sides": float(sum(sides)),
    }


def _unit(vector: Array) -> Array:
    """Normalise a 3-vector.

    Parameters
    ----------
    vector : numpy.ndarray, shape (3,), float
        Non-zero vector.

    Returns
    -------
    numpy.ndarray, shape (3,), float64
        Unit vector.

    Raises
    ------
    ValueError
        If the vector has zero or non-finite length.
    """
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Cannot normalise {vector}")
    return vector / norm


def _horizontal_basis(up: Array) -> tuple[Array, Array]:
    """Two unit vectors spanning the plane perpendicular to ``up``.

    Parameters
    ----------
    up : numpy.ndarray, shape (3,), float
        Unit up direction.

    Returns
    -------
    e1, e2 : numpy.ndarray, shape (3,), float64
        Orthonormal, with ``(e1, e2, up)`` right-handed.
    """
    seed = np.eye(3)[np.argmin(np.abs(up))]
    e1 = _unit(seed - (seed @ up) * up)
    return e1, np.cross(up, e1)


def _weighted_percentiles(values: Array, percentiles: tuple[float, ...], weights: Array | None) -> Array:
    """Percentiles of samples along axis 0, optionally weighted.

    With weights, each sample sits at the centre of its share of the cumulative weight and percentiles are
    interpolated linearly between samples (equal weights reproduce the midpoint rule, which differs from
    :func:`numpy.percentile`'s default only near the ends).

    Parameters
    ----------
    values : numpy.ndarray, shape (N,) or (N, T), float
        Samples along axis 0; with two dimensions each column is treated separately.
    percentiles : tuple of float
        Percentiles in [0, 100].
    weights : numpy.ndarray, shape (N,), float, or None
        Non-negative sample weights shared by all columns; ``None`` uses :func:`numpy.percentile`.

    Returns
    -------
    numpy.ndarray, shape (P,) or (P, T), float
        One row per requested percentile.
    """
    if weights is None:
        return np.percentile(values, percentiles, axis=0)
    columns = values.reshape(len(values), -1)
    if len(columns) == 1:
        return np.repeat(columns, len(percentiles), axis=0).reshape((len(percentiles),) + values.shape[1:])
    order = np.argsort(columns, axis=0)
    sorted_values = np.take_along_axis(columns, order, axis=0)
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, axis=0)
    position = (cumulative - 0.5 * sorted_weights) / cumulative[-1]
    out = []
    for q in np.asarray(percentiles, dtype=np.float64) / 100.0:
        upper = np.clip(np.count_nonzero(position < q, axis=0), 1, len(columns) - 1)[None]
        p0, p1 = np.take_along_axis(position, upper - 1, 0)[0], np.take_along_axis(position, upper, 0)[0]
        v0, v1 = np.take_along_axis(sorted_values, upper - 1, 0)[0], np.take_along_axis(sorted_values, upper, 0)[0]
        fraction = np.clip((q - p0) / np.maximum(p1 - p0, 1e-12), 0.0, 1.0)
        out.append(v0 + fraction * (v1 - v0))
    return np.stack(out).reshape((len(percentiles),) + values.shape[1:])


def _trimmed_extent(values: Array, percentiles: tuple[float, float],
                    weights: Array | None = None) -> tuple[Array, Array]:
    """Robust lower and upper bounds of samples along axis 0.

    Parameters
    ----------
    values : numpy.ndarray, shape (N,) or (N, T), float
        Samples along axis 0.
    percentiles : tuple of float
        Lower and upper percentiles, in [0, 100].
    weights : numpy.ndarray, shape (N,), float, or None, optional
        Sample weights; ``None`` weighs samples equally.

    Returns
    -------
    lo, hi : numpy.ndarray, shape () or (T,), float
        The two percentiles.
    """
    lo, hi = _weighted_percentiles(values, percentiles, weights)
    return lo, hi


def _min_area_yaw(flat: Array, percentiles: tuple[float, float], weights: Array | None = None,
                  step_deg: float = 1.0, refine_deg: float = 0.1, max_samples: int = 20000) -> float:
    """Yaw of the robust minimum-area rectangle around 2D points.

    The rectangle's sides at yaw ``theta`` are the percentile-trimmed extents of the points along
    ``(cos theta, sin theta)`` and its perpendicular; the yaw minimising the area is found on a grid over
    [0, 90) degrees and refined around the best cell. Unlike PCA, this aligns with the two legs of an L-shaped
    footprint (two visible faces) instead of its diagonal, and the trimming makes it insensitive to stray
    points, unlike the exact minimum-area rectangle of the convex hull.

    Parameters
    ----------
    flat : numpy.ndarray, shape (N, 2), float
        Horizontal coordinates of the points (metres).
    percentiles : tuple of float
        Lower and upper percentiles used for the extents.
    weights : numpy.ndarray, shape (N,), float, or None, optional
        Per-point weights for the percentiles; ``None`` weighs points equally.
    step_deg : float, optional
        Coarse grid step (degrees).
    refine_deg : float, optional
        Fine grid step (degrees) within one coarse step of the best yaw.
    max_samples : int, optional
        At most this many points (a fixed-seed random subset) are used for the search.

    Returns
    -------
    float
        Yaw in radians, in the frame of ``flat``'s two axes.
    """
    pts = flat - np.median(flat, axis=0)
    if len(pts) > max_samples:
        pick = np.random.default_rng(0).choice(len(pts), max_samples, replace=False)
        pts = pts[pick]
        weights = None if weights is None else weights[pick]

    def areas(thetas: Array) -> Array:
        """Trimmed rectangle area at each yaw.

        Parameters
        ----------
        thetas : numpy.ndarray, shape (T,), float
            Yaws in radians.

        Returns
        -------
        numpy.ndarray, shape (T,), float
            Areas (square metres).
        """
        c, s = np.cos(thetas), np.sin(thetas)
        lo_a, hi_a = _trimmed_extent(pts[:, :1] * c + pts[:, 1:] * s, percentiles, weights)
        lo_b, hi_b = _trimmed_extent(-pts[:, :1] * s + pts[:, 1:] * c, percentiles, weights)
        return (hi_a - lo_a) * (hi_b - lo_b)

    coarse = np.deg2rad(np.arange(0.0, 90.0, step_deg))
    best = coarse[np.argmin(areas(coarse))]
    fine = best + np.deg2rad(np.arange(-step_deg, step_deg + refine_deg / 2, refine_deg))
    return float(fine[np.argmin(areas(fine))])


def _pca_yaw(flat: Array, percentiles: tuple[float, float], weights: Array | None = None) -> float:
    """Yaw of the principal axis of 2D points, after trimming each axis to its percentile range.

    Parameters
    ----------
    flat : numpy.ndarray, shape (N, 2), float
        Horizontal coordinates of the points (metres).
    percentiles : tuple of float
        Points outside these percentiles on either axis are ignored.
    weights : numpy.ndarray, shape (N,), float, or None, optional
        Per-point weights for the percentiles and the covariance; ``None`` weighs points equally.

    Returns
    -------
    float
        Yaw in radians of the direction of largest variance.
    """
    lo, hi = _trimmed_extent(flat, percentiles, weights)
    inside = np.all((flat >= lo) & (flat <= hi), axis=1)
    if inside.sum() < 3:
        inside[:] = True
    _, vectors = np.linalg.eigh(np.cov(flat[inside].T, aweights=None if weights is None else weights[inside]))
    major = vectors[:, -1]
    return float(np.arctan2(major[1], major[0]))


def _fit_gravity_box(points: Array, up: Array, height_percentiles: tuple[float, float],
                     extent_percentiles: tuple[float, float], footprint: Literal["min_area", "pca"],
                     weights: Array | None = None) -> tuple[OrientedBox, dict[str, float]]:
    """Fit a yaw-only box to 3D points: robust height along ``up``, robust rectangle across it.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 3), float
        Points in the output frame (metres).
    up : numpy.ndarray, shape (3,), float
        Unit up direction in the same frame.
    height_percentiles : tuple of float
        Percentiles of the height distribution taken as the box's bottom and top.
    extent_percentiles : tuple of float
        Percentiles taken as the footprint's sides.
    footprint : {"min_area", "pca"}
        How the yaw is chosen: :func:`_min_area_yaw` or :func:`_pca_yaw`.
    weights : numpy.ndarray, shape (N,), float, or None, optional
        Per-point weights for every percentile; ``None`` weighs points equally.

    Returns
    -------
    box : OrientedBox
        Axes ``(x, y, up)`` as rotation columns, with ``size[0] >= size[1]`` (x is the longer horizontal
        side) so yaw is defined modulo 180 degrees.
    features : dict of str to float
        ``footprint_residual``: median horizontal distance (m) from the points to the rectangle's outline
        (near 0 when the visible points lie on the box's vertical faces). ``footprint_hull_fill``: area of
        the convex hull of the points inside the rectangle over the rectangle's area; low when only one face
        is visible and the depth side is barely observed.
    """
    e1, e2 = _horizontal_basis(up)
    flat = np.stack([points @ e1, points @ e2], axis=1)
    if footprint == "min_area":
        yaw = _min_area_yaw(flat, extent_percentiles, weights)
    else:
        yaw = _pca_yaw(flat, extent_percentiles, weights)
    x_axis = np.cos(yaw) * e1 + np.sin(yaw) * e2
    y_axis = np.cross(up, x_axis)
    lo_x, hi_x = _trimmed_extent(points @ x_axis, extent_percentiles, weights)
    lo_y, hi_y = _trimmed_extent(points @ y_axis, extent_percentiles, weights)
    if hi_y - lo_y > hi_x - lo_x:
        x_axis, y_axis = y_axis, -x_axis
        (lo_x, hi_x), (lo_y, hi_y) = (lo_y, hi_y), (-hi_x, -lo_x)
    lo_z, hi_z = _trimmed_extent(points @ up, height_percentiles, weights)
    cx, cy, cz = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2, (lo_z + hi_z) / 2
    box = OrientedBox(cx * x_axis + cy * y_axis + cz * up, [hi_x - lo_x, hi_y - lo_y, hi_z - lo_z],
                      np.column_stack([x_axis, y_axis, up]))

    half = box.size[:2] / 2
    local = np.stack([points @ x_axis - cx, points @ y_axis - cy], axis=1)
    excess = np.abs(local) - half
    inside = np.all(excess <= 0, axis=1)
    to_outline = np.where(inside, -excess.max(axis=1), np.linalg.norm(np.maximum(excess, 0.0), axis=1))
    rect_area = float(np.prod(box.size[:2]))
    hull_fill = 0.0
    if rect_area > 0 and inside.sum() >= 3:
        hull = cv2.convexHull(local[inside].astype(np.float32))
        hull_fill = float(cv2.contourArea(hull)) / rect_area
    return box, {"footprint_residual": float(np.median(to_outline)), "footprint_hull_fill": hull_fill}


def lift_instance(mask: NDArray[np.bool_], depth: NDArray[np.floating], camera: Camera, T_wc: Array, up: Array, *,
                  erode_px: int = 3, mad_k: float | None = 3.0, min_sigma: float = 0.15,
                  height_percentiles: tuple[float, float] = (1.0, 99.0),
                  extent_percentiles: tuple[float, float] = (1.0, 99.0),
                  footprint: Literal["min_area", "pca"] = "min_area", area_weighted: bool = True,
                  boundary_px: float = 5.0, min_points: int = 50,
                  max_depth: float | None = None) -> tuple[OrientedBox, dict[str, float]]:
    """Lift an instance mask to a gravity-aligned 3D box using a depth map.

    Steps: erode the mask by ``erode_px`` (edge pixels mix object and background depth); keep pixels with
    valid depth; drop pixels whose depth is more than ``mad_k`` robust standard deviations from the median
    (the deviation is ``1.4826 * MAD``, floored at ``min_sigma``); back-project and move to the output
    frame with ``T_wc``; take the box's bottom and top from height percentiles along ``up``, and its
    footprint from a robust minimum-area rectangle (default) or PCA, with sides at percentile extents. If
    erosion leaves fewer than ``min_points`` pixels (thin or tiny masks) the un-eroded mask is used and
    ``erosion_fallback`` is set.

    The floor matters when one flat, fronto-parallel face dominates the mask: its MAD is near zero, and
    without a floor the band would cut the object's own deeper parts. With a 0.05 m floor, a chair seen
    behind a table in the smoke test lost its seat and legs (lifted 0.26 m tall from laser depth); with
    0.15 m (a band of at least +/-0.45 m) it lifted 0.99 m tall, agreeing with the LiDAR lift.

    With ``area_weighted`` (default) every statistic weighs a pixel by ``z**2``, its footprint on a
    fronto-parallel surface. Without it, near surfaces outnumber far ones per square metre, so trimming a
    percentile of *pixels* removes more *surface* at the far end: on the synthetic test scene that undersized
    a 1.6 m box by about 0.12 m and pulled its centre towards the camera. The incidence-angle factor of the
    footprint is unknown without normals and is ignored.

    Parameters
    ----------
    mask : numpy.ndarray, shape (H, W), bool
        Instance mask.
    depth : numpy.ndarray, shape (H, W), float
        Planar depth ``z`` in metres at the mask's resolution (see :func:`resample_depth`); 0, negative,
        NaN and inf mean no depth.
    camera : Camera
        Intrinsics; rescaled with :meth:`Camera.scaled` if its resolution differs from the mask's.
    T_wc : numpy.ndarray, shape (4, 4), float
        Camera-to-output-frame pose; the identity gives a camera-frame box.
    up : numpy.ndarray, shape (3,), float
        Up direction (against gravity) in the output frame; need not be unit length.
    erode_px : int, optional
        Erosion radius in pixels (elliptical structuring element); 0 disables erosion.
    mad_k : float or None, optional
        Depth inlier band in robust standard deviations; ``None`` disables the test.
    min_sigma : float, optional
        Floor (metres) on the robust standard deviation.
    height_percentiles : tuple of float, optional
        Percentiles of point height taken as the box's bottom and top.
    extent_percentiles : tuple of float, optional
        Percentiles taken as the footprint's sides.
    footprint : {"min_area", "pca"}, optional
        Yaw estimator for the horizontal footprint.
    area_weighted : bool, optional
        Weigh pixels by ``z**2`` in the median, MAD and percentiles (see above).
    boundary_px : float, optional
        Edge band width for the ``boundary_fraction`` feature.
    min_points : int, optional
        Fewest 3D points needed to fit a box.
    max_depth : float or None, optional
        Depth beyond which pixels are ignored (metres); ``None`` keeps all.

    Returns
    -------
    box : OrientedBox
        Gravity-aligned box in the output frame: rotation columns ``(x, y, up)``, ``size[0] >= size[1]``,
        ``size[2]`` the height.
    features : dict of str to float
        Evidence that should predict the box's error; every value is a float so dicts stack into a feature
        matrix. From :func:`mask_features`: ``mask_area_px``, ``n_components``, ``boundary_fraction``,
        ``border_touch``, ``n_border_sides``. Depth and filtering: ``eroded_fraction`` (eroded / original
        mask pixels), ``erosion_fallback`` (0/1), ``valid_depth_fraction`` (pixels with depth / pixels
        considered), ``inlier_fraction`` (MAD inliers / pixels with depth), ``n_points`` (3D points used),
        ``kept_fraction`` (``n_points`` / mask pixels), ``depth_median`` (m), ``depth_mad`` (m, 1.4826 * MAD,
        before flooring), ``depth_mad_rel`` (``depth_mad / depth_median``); the median and MAD are weighted
        like the percentiles and computed before the inlier test. Geometry: ``footprint_residual``
        and ``footprint_hull_fill`` (see :func:`_fit_gravity_box`), ``view_elevation_deg`` (angle of the
        camera-to-box-centre ray below the horizontal; positive when looking down).

    Raises
    ------
    ValueError
        If shapes disagree or ``footprint`` is unknown.
    LiftError
        If the mask is empty or fewer than ``min_points`` pixels survive filtering.
    """
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth)
    if mask.ndim != 2 or depth.shape != mask.shape:
        raise ValueError(f"mask {mask.shape} and depth {depth.shape} must be 2D with the same shape; "
                         "use resample_depth() for depth at another resolution")
    if footprint not in ("min_area", "pca"):
        raise ValueError(f"Unknown footprint estimator {footprint!r}")
    if not mask.any():
        raise LiftError("Empty mask")
    h, w = mask.shape
    cam = camera if (camera.height, camera.width) == (h, w) else camera.scaled(w, h)
    up = _unit(up)
    features = mask_features(mask, boundary_px)

    core, fallback = mask, False
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        if np.count_nonzero(eroded) >= min_points:
            core = eroded
        else:
            fallback = True
    rows, cols = np.nonzero(core)
    z = depth[rows, cols].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        valid &= z <= max_depth
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if len(z) < min_points:
        raise LiftError(f"Only {len(z)} mask pixels have valid depth (need {min_points})")

    weights = z**2 if area_weighted else None
    median = float(_weighted_percentiles(z, (50.0,), weights)[0])
    sigma = _MAD_TO_SIGMA * float(_weighted_percentiles(np.abs(z - median), (50.0,), weights)[0])
    n_valid = len(z)
    if mad_k is not None:
        inlier = np.abs(z - median) <= mad_k * max(sigma, min_sigma)
        rows, cols, z = rows[inlier], cols[inlier], z[inlier]
        weights = None if weights is None else weights[inlier]
    if len(z) < min_points:
        raise LiftError(f"Only {len(z)} depth inliers (need {min_points})")

    points = transform(T_wc, cam.backproject(cols, rows, z))
    box, geometry = _fit_gravity_box(points, up, height_percentiles, extent_percentiles, footprint, weights)
    ray = box.center - T_wc[:3, 3]
    elevation = float(np.degrees(np.arcsin(np.clip(-(ray @ up) / np.linalg.norm(ray), -1.0, 1.0))))
    features.update({
        "eroded_fraction": float(np.count_nonzero(core)) / features["mask_area_px"],
        "erosion_fallback": float(fallback),
        "valid_depth_fraction": n_valid / float(np.count_nonzero(core)),
        "inlier_fraction": len(z) / float(n_valid),
        "n_points": float(len(z)),
        "kept_fraction": len(z) / features["mask_area_px"],
        "depth_median": median,
        "depth_mad": sigma,
        "depth_mad_rel": sigma / median,
        **geometry,
        "view_elevation_deg": elevation,
    })
    return box, features
