"""Ask open vision-language models the generated spatial questions, and parse their answers.

Design choices, and why:

- **One prediction per question.** Oracle and automatic labels share a :class:`QuestionSpec`, so a model
  answers each spec once and :mod:`measured_space.scoring` scores that answer against both label sources.
- **Strict JSON answers.** The prompt asks for a single JSON object, so parsing is mechanical. Regex
  fallbacks recover answers from models that ignore the format. Anything still ambiguous parses to
  ``None``, a parse failure, which scoring counts as wrong.
- **Reasoning models.** Cosmos-Reason2 writes ``<think>...</think>`` before its answer. Only the text
  after ``</think>`` is parsed, never a guess read from half-finished reasoning. If the token budget cuts
  the think block off, the block is closed (``</think>`` is appended) and a short answer is decoded after
  it. That answer is flagged as forced and gets its own parse method, so scoring can count it or not.
- **Inputs come from the tokenizer and the PIL image processor, not ``AutoProcessor``.** The Qwen3-VL
  processor in transformers 5 cannot be built without torchvision, which its video processor needs. This
  project feeds only still images. Building the inputs here also fixes the preprocessing backend, so
  results don't depend on whether torchvision is installed.
- **Everything that can change an answer is written with every prediction:** model snapshot, dtype,
  attention kernel, resolution policy, token budget, prompt version and library versions.
- ``torch`` and ``transformers`` are imported only when a model is loaded, so prompting, parsing and the
  run loop can be tested on CPU without them.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Protocol

import numpy as np

from measured_space.config import data_root
from measured_space.qa_schema import QAItem, QuestionSpec

PROMPT_VERSION = "v1"
IMAGE_PAD = "<|image_pad|>"
FORCE_CLOSE = "\n</think>\n\n"            # appended to cut-off reasoning; the answer follows "</think>\n\n"
FORCED_ANSWER_TOKENS = 48                    # budget for the answer decoded after a forced close
FORCED_METHOD = "forced_after_truncation"

# Metres per unit, for numeric answers asked in metres. Bare "in" is only inches when no word follows it,
# so "2 in length" does not become 5 cm. The ' and " marks count only in a feet-and-inches pair (5'6"): a
# quoted number such as "2.5" must not read as inches.
LENGTH_UNITS: dict[str, float] = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001, "millimetre": 0.001, "millimetres": 0.001,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
}

NUMBER_WORDS: dict[str, int] = {
    word: value for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen twenty".split())
} | {"none": 0, "no": 0}

_NUM = r"(?<![\d.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_UNIT = (r"(?:meters?|metres?|centimeters?|centimetres?|millimeters?|millimetres?|feet|foot|inch(?:es)?"
         r"|cm|mm|ft|m|in(?!\s*[A-Za-z]))")
_UNIT_WORD = rf"(?P<unit>{_UNIT})(?![A-Za-z])"
_LENGTH_NUMBER = re.compile(rf"(?P<value>{_NUM})\s*{_UNIT_WORD}", re.IGNORECASE)
_FEET_INCHES = re.compile(rf"(?P<ft>{_NUM})\s*(?:ft|feet|foot|')\s*(?P<inch>{_NUM})\s*(?:inch(?:es)?|in|\")",
                          re.IGNORECASE)
_RANGE = re.compile(rf"(?P<lo>{_NUM})\s*(?:-|–|to|and)\s*(?P<hi>{_NUM})\s*{_UNIT_WORD}", re.IGNORECASE)
_BARE_NUMBER = re.compile(_NUM)
_ANSWER_FIELD = re.compile(r"""\banswer\b["']?\s*[:=]\s*(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^,}\n]+))""",
                           re.IGNORECASE)
_ANSWER_TAG = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.IGNORECASE | re.DOTALL)
_QUOTED = re.compile(r"""["'“‘*`]+([^"'”’*`\n]+?)["'”’*`]+""")


@dataclass(frozen=True)
class Prompt:
    """A chat prompt: a system instruction and the user's question text.

    Attributes
    ----------
    system : str
        System message: the answer-format contract.
    user : str
        User message text. The image, if any, is placed before it.
    with_image : bool
        Whether an image accompanies the question (False in blind mode).
    """

    system: str
    user: str
    with_image: bool


@dataclass(frozen=True)
class ResolutionPolicy:
    """Bounds on the pixel count an image is resized to before the vision encoder.

    Qwen3-VL resizes so that both sides are multiples of 32 and the pixel count lies within the bounds,
    keeping the aspect ratio. Each 32x32 block becomes one visual token.

    Attributes
    ----------
    name : str
        Short name written with every prediction.
    min_pixels : int
        Lower bound on height x width after resizing, in pixels.
    max_pixels : int
        Upper bound on height x width after resizing, in pixels.
    """

    name: str
    min_pixels: int
    max_pixels: int


# 256 to 1280 visual tokens. A 1920x1440 frame becomes 1312x960 px: 1230 tokens.
DEFAULT_RESOLUTION = ResolutionPolicy("tok256-1280", 256 * 32 * 32, 1280 * 32 * 32)


@dataclass(frozen=True)
class ModelSpec:
    """A model to evaluate and its decoding budget.

    Attributes
    ----------
    key : str
        Short name used in outputs (``"qwen3-vl-8b"``).
    repo_id : str
        Hugging Face repository id.
    max_new_tokens : int
        Generation budget. Reasoning models need room for their think block.
    reasoning : bool
        Whether the model is prompted to reason in a ``<think>`` block before answering.
    licence : str
        Licence of the weights, for the record.
    """

    key: str
    repo_id: str
    max_new_tokens: int
    reasoning: bool
    licence: str


MODELS: dict[str, ModelSpec] = {
    "qwen3-vl-8b": ModelSpec("qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct", 128, False, "Apache-2.0"),
    "cosmos-reason2-8b": ModelSpec("cosmos-reason2-8b", "nvidia/Cosmos-Reason2-8B", 1024, True,
                                   "NVIDIA Open Model License"),
}


@dataclass(frozen=True)
class Generation:
    """One model output and its cost.

    Attributes
    ----------
    text : str
        Decoded output, special tokens removed.
    n_new_tokens : int
        Number of generated tokens.
    truncated : bool
        True if generation stopped at the token budget rather than at an end-of-turn token.
    latency_s : float
        Wall time for preprocessing, generation and decoding, in seconds.
    n_image_tokens : int
        Visual tokens fed to the language model (0 without an image).
    forced_text : str or None
        If the budget cut off an open ``<think>`` block: the answer decoded after appending
        :data:`FORCE_CLOSE`. None otherwise.
    n_forced_tokens : int
        Tokens generated in that forced phase (0 if there was none). ``n_new_tokens`` counts the first
        phase only.
    """

    text: str
    n_new_tokens: int
    truncated: bool
    latency_s: float
    n_image_tokens: int
    forced_text: str | None = None
    n_forced_tokens: int = 0


class Parsed(NamedTuple):
    """A parsed answer and how it was found.

    Attributes
    ----------
    value : float or int or str or None
        Metres for numbers, an int for counts, the matching choice (as spelled in the spec) for choices;
        None if unparseable.
    method : str
        ``"json"``, ``"answer_field"``, ``"text"``, ``"unterminated_think"``, ``"failed"``, or
        :data:`FORCED_METHOD` for an answer forced after cut-off reasoning.
    """

    value: float | int | str | None
    method: str


class Runner(Protocol):
    """What the run loop needs from a loaded model."""

    @property
    def info(self) -> dict[str, Any]:
        """Run configuration written with every prediction (model, snapshot, dtype, resolution, ...)."""
        ...

    def generate(self, prompt: Prompt, image: np.ndarray | None) -> Generation:
        """Answer a prompt about an image.

        Parameters
        ----------
        prompt : Prompt
            System and user text.
        image : numpy.ndarray or None
            ``(H, W, 3)`` uint8 RGB image, or None for a text-only prompt.

        Returns
        -------
        Generation
            The decoded output and its cost.
        """
        ...


# --------------------------------------------------------------------------------------------- prompts


def answer_format(spec: QuestionSpec) -> str:
    """The JSON object the model must reply with, for the spec's answer type.

    Parameters
    ----------
    spec : QuestionSpec
        The question.

    Returns
    -------
    str
        The answer-format instruction, including the allowed choices for choice questions.

    Raises
    ------
    ValueError
        If the answer type is unknown, or a choice question has no choices.
    """
    if spec.answer_type == "number":
        unit = spec.unit or "m"
        return f'{{"answer": <number>, "unit": "{unit}"}}, with the answer in {_unit_name(unit)}'
    if spec.answer_type == "choice":
        if not spec.choices:
            raise ValueError(f"Choice question {spec.id} has no choices")
        allowed = ", ".join(json.dumps(c) for c in spec.choices)
        return f'{{"answer": "<choice>"}}, where <choice> is exactly one of: {allowed}'
    if spec.answer_type == "count":
        return '{"answer": <integer>}'
    raise ValueError(f"Unknown answer type {spec.answer_type!r}")


def build_prompt(spec: QuestionSpec, *, blind: bool = False, reasoning: bool = False) -> Prompt:
    """The prompt for one question.

    Parameters
    ----------
    spec : QuestionSpec
        The question; its ``question`` text is asked verbatim.
    blind : bool
        Text-only baseline: no image is sent and the model is told to answer from the question alone.
    reasoning : bool
        Ask for a ``<think>`` block before the answer, in the format the reasoning model was trained on.

    Returns
    -------
    Prompt
        System instruction and user text.
    """
    system = ["You answer questions about the 3D layout of a real scene. Distances and sizes are physical "
              "quantities in the real world, not in pixels."]
    if blind:
        system.append("No image is provided. Answer from the question alone with your best guess.")
    system.append("Always give an answer, even if you are unsure.")
    final = "Your final answer must be" if reasoning else "Reply with only"
    system.append(f"{final} a single JSON object: {answer_format(spec)}.")
    user = spec.question
    if reasoning:
        user += ("\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\n"
                 "Write your final answer immediately after the </think> tag.")
    return Prompt(" ".join(system), user, with_image=not blind)


def _unit_name(unit: str) -> str:
    """Spell out a unit for the prompt.

    Parameters
    ----------
    unit : str
        Unit symbol, e.g. ``"m"``.

    Returns
    -------
    str
        ``"metres"`` for ``"m"``, otherwise the symbol unchanged.
    """
    return {"m": "metres", "cm": "centimetres", "mm": "millimetres"}.get(unit, unit)


# --------------------------------------------------------------------------------------------- parsing


def parse_answer(spec: QuestionSpec, text: str) -> Parsed:
    """Parse a model's reply into an answer of the spec's type.

    Order: drop a closed ``<think>`` block (an unclosed one is a failure); keep the inside of an
    ``<answer>`` tag if present; then try JSON objects with an ``"answer"`` key, then an ``answer: ...``
    field in free text, then the whole text. A JSON answer that cannot be read (say, in an unknown unit)
    is a failure: the text fallbacks are not tried, so they cannot replace it with a different number.

    Parameters
    ----------
    spec : QuestionSpec
        The question; its answer type, unit and choices drive parsing.
    text : str
        Raw model output.

    Returns
    -------
    Parsed
        The value (None if unparseable) and the method that found it.
    """
    body = final_answer_text(text)
    if body is None:
        return Parsed(None, "unterminated_think")
    tag = _ANSWER_TAG.search(body)
    if tag:
        body = tag.group(1)
    answered = False
    for obj in _json_objects(body):
        key = next((k for k in obj if str(k).strip().lower() == "answer"), None)
        if key is not None:
            answered = True
            value = _from_json(spec, obj[key], obj.get("unit"))
            if value is not None:
                return Parsed(value, "json")
    if answered:                                # a JSON answer we cannot read (e.g. an unknown unit) is final
        return Parsed(None, "failed")
    field = _ANSWER_FIELD.search(body)
    if field:
        raw = next(g for g in (field.group("dq"), field.group("sq"), field.group("bare")) if g is not None)
        value = _from_text(spec, raw)
        if value is not None:
            return Parsed(value, "answer_field")
    value = _from_text(spec, body)
    return Parsed(value, "text") if value is not None else Parsed(None, "failed")


def final_answer_text(text: str) -> str | None:
    """The part of a reply after any reasoning block.

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    str or None
        Text after the last ``</think>``; the whole text if there is no think block; None if a
        ``<think>`` block is opened but never closed (reasoning cut off by the token budget).
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None
    return text


def parse_number(text: str, unit: str | None = "m") -> float | None:
    """A length from free text, converted to ``unit``.

    Preference: a feet-and-inches pair; a range with a unit, such as "2-3 m" or "between 2 and 3 m" (its
    midpoint); the first number carrying a length unit; otherwise the only number in the text, taken to be
    in ``unit``. Unit conversion applies only when ``unit`` is ``"m"``; for any other unit the number is
    returned as written.

    Parameters
    ----------
    text : str
        Free text, e.g. ``"about 150 cm"``.
    unit : str or None
        The unit the question asks for.

    Returns
    -------
    float or None
        The value, or None if no number (or several numbers, none with a unit) is found.
    """
    convert = unit in (None, "m")
    if convert and (m := _FEET_INCHES.search(text)):
        return float(m["ft"]) * LENGTH_UNITS["ft"] + float(m["inch"]) * LENGTH_UNITS["in"]
    if convert and (m := _RANGE.search(text)):
        return 0.5 * (float(m["lo"]) + float(m["hi"])) * LENGTH_UNITS[m["unit"].lower()]
    if convert and (m := _LENGTH_NUMBER.search(text)):
        return float(m["value"]) * LENGTH_UNITS[m["unit"].lower()]
    numbers = _BARE_NUMBER.findall(text)
    if len(numbers) == 1:
        return float(numbers[0])
    return None


def parse_count(text: str) -> int | None:
    """A non-negative count from free text.

    Parameters
    ----------
    text : str
        Free text, e.g. ``"There are three chairs."``.

    Returns
    -------
    int or None
        The first number (digits or an English number word up to twenty); None if there is none or it is
        not a whole number.
    """
    for match in re.finditer(rf"{_NUM}|\b[A-Za-z]+\b", text):
        token = match.group(0).lower()
        if token in NUMBER_WORDS:
            if token == "no" and not re.match(r"no\s+\w", text[match.start():], re.IGNORECASE):
                continue                                     # "No." / "no," is a reply, not "no chairs"
            return NUMBER_WORDS[token]
        if token[0].isdigit() or token[0] in "+-.":
            value = float(token)
            return int(value) if value.is_integer() else None
    return None


def parse_choice(text: str, choices: tuple[str, ...]) -> str | None:
    """Match free text to one of the allowed choices, case-insensitively.

    Tried in order: the whole text is a choice; exactly one distinct choice among quoted (or bold or
    backticked) phrases; the text starts with a choice; exactly one distinct choice appears as whole
    words, where a choice inside a longer matching choice does not count separately.

    Parameters
    ----------
    text : str
        Free text, e.g. ``'The chair is on the "left".'``.
    choices : tuple of str
        Allowed answers.

    Returns
    -------
    str or None
        The matching choice as spelled in ``choices``; None if no choice or more than one matches.
    """
    by_norm = {_normalise(c): c for c in choices}
    whole = _normalise(text)
    if whole in by_norm:
        return by_norm[whole]
    quoted = {by_norm[q] for q in map(_normalise, _QUOTED.findall(text)) if q in by_norm}
    if len(quoted) == 1:
        return quoted.pop()
    for norm in sorted(by_norm, key=len, reverse=True):
        if re.match(rf"{re.escape(norm)}(?!\w)", whole):
            return by_norm[norm]
    spans: list[tuple[int, int, str]] = []
    for norm in sorted(by_norm, key=len, reverse=True):
        for m in re.finditer(rf"(?<!\w){re.escape(norm)}(?!\w)", whole):
            if not any(s <= m.start() and m.end() <= e for s, e, _ in spans):
                spans.append((m.start(), m.end(), by_norm[norm]))
    found = {choice for _, _, choice in spans}
    return found.pop() if len(found) == 1 else None


def _from_json(spec: QuestionSpec, value: Any, unit: Any) -> float | int | str | None:
    """Interpret the ``"answer"`` value of a JSON reply.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    value : Any
        The JSON value of ``"answer"``.
    unit : Any
        The JSON value of ``"unit"``, if the reply had one.

    Returns
    -------
    float or int or str or None
        The parsed answer, or None.
    """
    if isinstance(value, bool) or value is None:
        return None
    if spec.answer_type == "number" and isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        if unit is None or spec.unit not in (None, "m"):
            return float(value)
        factor = LENGTH_UNITS.get(str(unit).strip().lower())
        return float(value) * factor if factor is not None else None
    if spec.answer_type == "count" and isinstance(value, (int, float)):
        return int(value) if math.isfinite(value) and float(value).is_integer() else None
    text = str(value)
    if spec.answer_type == "number" and unit is not None and _BARE_NUMBER.fullmatch(text.strip()):
        text = f"{text} {unit}"
    return _from_text(spec, text)


def _from_text(spec: QuestionSpec, text: str) -> float | int | str | None:
    """Parse free text according to the spec's answer type.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    text : str
        Free text.

    Returns
    -------
    float or int or str or None
        The parsed answer, or None.
    """
    if spec.answer_type == "number":
        return parse_number(text, spec.unit)
    if spec.answer_type == "count":
        return parse_count(text)
    return parse_choice(text, spec.choices or ())


def _json_objects(text: str) -> Iterable[dict[str, Any]]:
    """JSON objects embedded in text, in order (fenced or bare).

    Parameters
    ----------
    text : str
        Free text possibly containing JSON objects.

    Returns
    -------
    Iterable of dict
        Each object that decodes from an opening brace; nested objects are not reported separately.
    """
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos = text.find("{", pos + 1)
            continue
        if isinstance(obj, dict):
            yield obj
        pos = text.find("{", end)


def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace and strip surrounding quotes and punctuation.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    str
        Normalised text, for comparison only.
    """
    return " ".join(text.lower().split()).strip(" \t\n.,;:!?\"'`*“”‘’()[]")


# --------------------------------------------------------------------------------------------- models


def hf_cache_dir() -> Path:
    """The Hugging Face cache holding the model weights.

    Returns
    -------
    Path
        ``<data root>/hf_cache``.

    Raises
    ------
    RuntimeError
        If the data root is not configured or the cache folder is missing.
    """
    path = data_root() / "hf_cache"
    if not path.is_dir():
        raise RuntimeError(f"Model cache {path} not found")
    return path


def snapshot_hash(repo_id: str, cache_dir: Path) -> str:
    """The commit hash of the cached snapshot of a model repository.

    Parameters
    ----------
    repo_id : str
        Hugging Face repository id.
    cache_dir : Path
        Hugging Face hub cache directory.

    Returns
    -------
    str
        The snapshot folder name (a git commit hash).

    Raises
    ------
    FileNotFoundError
        If the repository's ``config.json`` is not in the cache.
    """
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id, "config.json", cache_dir=str(cache_dir))
    if not isinstance(path, str):
        raise FileNotFoundError(f"{repo_id} is not in the cache at {cache_dir}")
    return Path(path).parent.name


class VLMRunner:
    """A Qwen3-VL-architecture model loaded for greedy single-image question answering.

    Attributes
    ----------
    load_seconds : float
        Wall time to load tokenizer, image processor and weights, in seconds.
    """

    dtype_name = "bfloat16"
    attn_implementation = "sdpa"

    def __init__(self, model: ModelSpec, resolution: ResolutionPolicy = DEFAULT_RESOLUTION, *,
                 device: str = "cuda", cache_dir: Path | None = None, max_new_tokens: int | None = None,
                 forced_answer_tokens: int = FORCED_ANSWER_TOKENS) -> None:
        """Load the tokenizer, the PIL image processor and the weights (bf16, SDPA) onto ``device``.

        Parameters
        ----------
        model : ModelSpec
            Which model, and its token budget.
        resolution : ResolutionPolicy
            Image resizing bounds.
        device : str
            Torch device for the model and inputs.
        cache_dir : Path or None
            Hugging Face cache; defaults to :func:`hf_cache_dir`. Weights are loaded with
            ``local_files_only=True``: nothing is downloaded.
        max_new_tokens : int or None
            Overrides ``model.max_new_tokens``.
        forced_answer_tokens : int
            Budget for the answer forced after cut-off reasoning; 0 disables forcing.

        Raises
        ------
        RuntimeError
            If the data root or the model cache is missing.
        OSError
            If the model files are not in the cache.
        """
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        start = time.perf_counter()
        cache = cache_dir or hf_cache_dir()
        self.spec = model
        self.resolution = resolution
        self.device = device
        self.max_new_tokens = max_new_tokens or model.max_new_tokens
        self.forced_answer_tokens = forced_answer_tokens
        common = {"cache_dir": str(cache), "local_files_only": True}
        self.tokenizer = AutoTokenizer.from_pretrained(model.repo_id, **common)
        self.image_processor = AutoImageProcessor.from_pretrained(model.repo_id, backend="pil", **common)
        # Loaded on the CPU, then moved: loading straight to the GPU (device_map) needs `accelerate`.
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model.repo_id, dtype=getattr(torch, self.dtype_name), attn_implementation=self.attn_implementation,
            **common).to(device).eval()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PAD)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self._info = {
            "model": model.key, "model_id": model.repo_id, "snapshot": snapshot_hash(model.repo_id, cache),
            "dtype": self.dtype_name, "attn_implementation": self.attn_implementation,
            "resolution": resolution.name, "min_pixels": resolution.min_pixels,
            "max_pixels": resolution.max_pixels, "image_processor": type(self.image_processor).__name__,
            "max_new_tokens": self.max_new_tokens, "forced_answer_tokens": forced_answer_tokens,
            "reasoning": model.reasoning, "decoding": "greedy",
            "prompt_version": PROMPT_VERSION, "transformers": transformers.__version__,
            "torch": torch.__version__,
        }
        self.load_seconds = time.perf_counter() - start

    @property
    def info(self) -> dict[str, Any]:
        """Run configuration written with every prediction.

        Returns
        -------
        dict
            Model key and id, snapshot hash, dtype, attention kernel, resolution policy, image processor
            class, token budget, reasoning flag, decoding, prompt version and library versions.
        """
        return dict(self._info)

    def encode(self, prompt: Prompt, image: np.ndarray | None) -> tuple[dict[str, Any], int]:
        """Model inputs for one prompt, as the Qwen3-VL processor would build them.

        The chat template places one ``<|image_pad|>`` per image; it is expanded to one token per merged
        2x2 patch, and ``mm_token_type_ids`` (1 at image tokens, else 0) drives the multimodal rotary
        position index.

        Parameters
        ----------
        prompt : Prompt
            System and user text.
        image : numpy.ndarray or None
            ``(H, W, 3)`` uint8 RGB image, or None for a text-only prompt.

        Returns
        -------
        inputs : dict
            Tensors on the model's device: ``input_ids``, ``attention_mask``, ``mm_token_type_ids`` and,
            with an image, ``pixel_values`` and ``image_grid_thw``.
        n_image_tokens : int
            Number of visual tokens in the sequence.

        Raises
        ------
        ValueError
            If an image is required but missing, given in blind mode, or not ``(H, W, 3)`` uint8.
        """
        import torch
        from PIL import Image

        if prompt.with_image != (image is not None):
            raise ValueError("Image presence does not match the prompt (blind mode takes no image)")
        user: list[dict[str, Any]] = [{"type": "image"}] if image is not None else []
        user.append({"type": "text", "text": prompt.user})
        messages = [{"role": "system", "content": [{"type": "text", "text": prompt.system}]},
                    {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs: dict[str, Any] = {}
        n_image_tokens = 0
        if image is not None:
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Expected an (H, W, 3) uint8 RGB image, got {image.shape} {image.dtype}")
            vision = self.image_processor(
                images=[Image.fromarray(image)], return_tensors="pt",
                size={"shortest_edge": self.resolution.min_pixels, "longest_edge": self.resolution.max_pixels})
            n_image_tokens = int(vision["image_grid_thw"][0].prod()) // self.image_processor.merge_size ** 2
            text = text.replace(IMAGE_PAD, IMAGE_PAD * n_image_tokens, 1)
            inputs.update(pixel_values=vision["pixel_values"], image_grid_thw=vision["image_grid_thw"])
        tokens = self.tokenizer(text, return_tensors="pt", return_token_type_ids=False)
        inputs.update(tokens)
        inputs["mm_token_type_ids"] = (tokens["input_ids"] == self.image_token_id).to(torch.int64)
        return {k: v.to(self.model.device) for k, v in inputs.items()}, n_image_tokens

    def generate(self, prompt: Prompt, image: np.ndarray | None) -> Generation:
        """Greedy-decode an answer.

        Parameters
        ----------
        prompt : Prompt
            System and user text.
        image : numpy.ndarray or None
            ``(H, W, 3)`` uint8 RGB image, or None in blind mode.

        Returns
        -------
        Generation
            Decoded text, token counts, truncation flag, forced answer (if any), latency of both phases and
            visual-token count.
        """
        start = time.perf_counter()
        inputs, n_image_tokens = self.encode(prompt, image)
        gen = decode_greedy(self.model, self.tokenizer, inputs, max_new_tokens=self.max_new_tokens,
                            eos_ids=self.eos_ids, forced_answer_tokens=self.forced_answer_tokens)
        return replace(gen, latency_s=time.perf_counter() - start, n_image_tokens=n_image_tokens)


def decode_greedy(model: Any, tokenizer: Any, inputs: dict[str, Any], *, max_new_tokens: int,
                  eos_ids: set[int], forced_answer_tokens: int = FORCED_ANSWER_TOKENS) -> Generation:
    """Greedy decoding, closing reasoning that the budget cut off and decoding a short answer after it.

    If the first phase stops at ``max_new_tokens`` inside an open ``<think>`` block, the generated tokens
    are kept exactly. The tokens of :data:`FORCE_CLOSE` are appended, and decoding continues for at most
    ``forced_answer_tokens``. Image inputs are reused. The appended and forced tokens are text
    (``mm_token_type_ids`` 0).

    Parameters
    ----------
    model : Any
        A transformers generation model; ``model.generate(**inputs, max_new_tokens=...)`` returns
        ``(1, L)`` token ids, prompt included.
    tokenizer : Any
        Its tokenizer (``__call__`` and ``decode``).
    inputs : dict
        Encoded prompt for batch size 1, as from :meth:`VLMRunner.encode`. Every tensor is on the
        model's device, and ``input_ids`` has shape ``(1, L)``.
    max_new_tokens : int
        First-phase budget.
    eos_ids : set of int
        End-of-turn token ids.
    forced_answer_tokens : int
        Forced-phase budget; 0 disables forcing.

    Returns
    -------
    Generation
        Text and token count of the first phase, its truncation flag, and the forced answer and its
        token count if there was one. ``latency_s`` and ``n_image_tokens`` are 0, for the caller to fill.
    """
    import torch

    new = _greedy(model, inputs, max_new_tokens)
    text = tokenizer.decode(new, skip_special_tokens=True)
    truncated = len(new) >= max_new_tokens and (not new or new[-1] not in eos_ids)
    if not (truncated and forced_answer_tokens > 0 and final_answer_text(text) is None):
        return Generation(text, len(new), truncated, 0.0, 0)
    prompt_ids = inputs["input_ids"]
    close = tokenizer(FORCE_CLOSE, add_special_tokens=False, return_tensors="pt")["input_ids"]
    close = close.to(prompt_ids.device)
    ids = torch.cat([prompt_ids, torch.tensor([new], dtype=prompt_ids.dtype, device=prompt_ids.device), close],
                    dim=1)
    extra = ids.shape[1] - prompt_ids.shape[1]
    forced_inputs = {**inputs, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if "mm_token_type_ids" in inputs:
        types = inputs["mm_token_type_ids"]
        forced_inputs["mm_token_type_ids"] = torch.cat([types, types.new_zeros((1, extra))], dim=1)
    forced = _greedy(model, forced_inputs, forced_answer_tokens)
    return Generation(text, len(new), truncated, 0.0, 0, tokenizer.decode(forced, skip_special_tokens=True),
                      len(forced))


def _greedy(model: Any, inputs: dict[str, Any], max_new_tokens: int) -> list[int]:
    """Greedy-decode and return only the new token ids.

    Parameters
    ----------
    model : Any
        A transformers generation model.
    inputs : dict
        Model inputs for batch size 1, with ``input_ids`` of shape ``(1, L)``.
    max_new_tokens : int
        Budget.

    Returns
    -------
    list of int
        Generated token ids, end-of-turn token included if one was produced.
    """
    import torch

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None)
    return out[0, inputs["input_ids"].shape[1]:].tolist()


def parse_generation(spec: QuestionSpec, gen: Generation) -> Parsed:
    """Parse a generation, using the forced answer if reasoning was cut off.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    gen : Generation
        The model output.

    Returns
    -------
    Parsed
        :func:`parse_answer` of the output. If an answer was forced, it is parsed instead, with method
        :data:`FORCED_METHOD` (or ``"failed"`` if even that cannot be parsed).
    """
    if gen.forced_text is None:
        return parse_answer(spec, gen.text)
    forced = parse_answer(spec, gen.forced_text)
    return Parsed(forced.value, FORCED_METHOD if forced.value is not None else "failed")


# --------------------------------------------------------------------------------------------- run loop

# Fields that must agree between an existing predictions file and the run appending to it.
RESUME_KEYS = ("model_id", "snapshot", "dtype", "attn_implementation", "resolution", "min_pixels",
               "max_pixels", "image_processor", "max_new_tokens", "forced_answer_tokens", "reasoning", "decoding",
               "prompt_version", "blind")


@dataclass(frozen=True)
class RunSummary:
    """What a call to :func:`run_predictions` did.

    Attributes
    ----------
    n_specs : int
        Distinct questions requested.
    n_skipped : int
        Questions already in the output file.
    n_written : int
        Predictions written by this call.
    """

    n_specs: int
    n_skipped: int
    n_written: int


def unique_specs(items: Iterable[QAItem | QuestionSpec]) -> list[QuestionSpec]:
    """Distinct question specs, in first-seen order.

    Parameters
    ----------
    items : iterable of QAItem or QuestionSpec
        Labelled items (oracle and auto items share specs) or bare specs.

    Returns
    -------
    list of QuestionSpec
        One spec per id.
    """
    seen: dict[str, QuestionSpec] = {}
    for item in items:
        spec = item.spec if isinstance(item, QAItem) else item
        seen.setdefault(spec.id, spec)
    return list(seen.values())


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Prediction records from a JSONL file, skipping lines that do not decode.

    A line can be cut short if a run is killed mid-write; it is skipped, so that question is re-asked
    on resume.

    Parameters
    ----------
    path : str or Path
        Predictions file; a missing file reads as empty.

    Returns
    -------
    list of dict
        One record per prediction.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def run_predictions(runner: Runner, items: Iterable[QAItem | QuestionSpec], out_path: str | Path,
                    load_image: Callable[[QuestionSpec], np.ndarray] | None, *, blind: bool = False,
                    limit: int | None = None, log: Callable[[str], None] | None = print) -> RunSummary:
    """Ask every question once and append one prediction per line to ``out_path``.

    Resume-safe: questions whose id is already in the file are skipped, and each record is flushed as it
    is written. Appending to a file written with a different configuration (see :data:`RESUME_KEYS`) is
    refused.

    Parameters
    ----------
    runner : Runner
        A loaded model (:class:`VLMRunner`) or any object with ``info`` and ``generate``.
    items : iterable of QAItem or QuestionSpec
        Questions; duplicates by id are asked once.
    out_path : str or Path
        Predictions JSONL, created if missing.
    load_image : callable or None
        ``load_image(spec) -> (H, W, 3) uint8 RGB`` array, oriented as the question assumes. Not called in
        blind mode, where it may be None.
    blind : bool
        Text-only baseline: no image is sent.
    limit : int or None
        Ask at most this many new questions.
    log : callable or None
        Receives one progress line per prediction; None for silence.

    Returns
    -------
    RunSummary
        Counts of requested, skipped and written questions.

    Raises
    ------
    ValueError
        If the output file holds predictions from a different configuration, or ``load_image`` is
        missing outside blind mode.
    """
    if load_image is None and not blind:
        raise ValueError("load_image is required unless blind=True")
    out_path = Path(out_path)
    config = {**runner.info, "blind": blind}
    existing = read_predictions(out_path)
    _check_resume(existing, config, out_path)
    done = {r["id"] for r in existing}
    specs = unique_specs(items)
    pending = [s for s in specs if s.id not in done]
    todo = pending[:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _end_partial_line(out_path)
    with out_path.open("a") as f:
        for i, spec in enumerate(todo, 1):
            image = None if blind else load_image(spec)
            prompt = build_prompt(spec, blind=blind, reasoning=bool(config.get("reasoning")))
            gen = runner.generate(prompt, image)
            parsed = parse_generation(spec, gen)
            record = {"id": spec.id, "dataset": spec.dataset, "scene": spec.scene, "qtype": spec.qtype,
                      "answer_type": spec.answer_type, **config, "image_hw": None if image is None
                      else list(image.shape[:2]), "raw_output": gen.text, "parsed": parsed.value,
                      "parse_method": parsed.method, "forced": gen.forced_text is not None,
                      **{k: v for k, v in asdict(gen).items() if k != "text"},
                      "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            f.write(json.dumps(record) + "\n")
            f.flush()
            if log is not None:
                log(f"[{i}/{len(todo)}] {spec.id} {spec.qtype}: parsed={parsed.value!r} ({parsed.method}) "
                    f"{gen.latency_s:.2f}s")
    return RunSummary(len(specs), len(specs) - len(pending), len(todo))


def _check_resume(existing: list[dict[str, Any]], config: dict[str, Any], path: Path) -> None:
    """Refuse to mix configurations in one predictions file.

    Parameters
    ----------
    existing : list of dict
        Records already in the file.
    config : dict
        The configuration of the run about to append.
    path : Path
        The file, for the error message.

    Raises
    ------
    ValueError
        If any record differs from ``config`` on a key in :data:`RESUME_KEYS`.
    """
    for record in existing:
        diff = [k for k in RESUME_KEYS if record.get(k) != config.get(k)]
        if diff:
            raise ValueError(f"{path} holds predictions from a different configuration ({', '.join(diff)} "
                             f"differ); write to a new file")


def _end_partial_line(path: Path) -> None:
    """Terminate a final line left unfinished by a killed run, so the next record starts on its own line.

    Parameters
    ----------
    path : Path
        Predictions file; nothing happens if it is missing or empty.
    """
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        if last != b"\n":
            with path.open("a") as f:
                f.write("\n")
