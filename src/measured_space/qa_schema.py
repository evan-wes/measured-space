"""Question and answer records, the shared correctness rule, and JSONL storage.

A :class:`QuestionSpec` fixes *what* is asked about *which* objects in *which* image, independent of any
answer. The same spec is answered twice, once from ground-truth boxes (``source="oracle"``) and once from
perception boxes (``source="auto"``), so oracle and automatic labels are paired by construction.
:func:`is_correct` is the single definition of a correct answer, used for labels and for models alike.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ANSWER_TYPES = ("number", "choice", "count")
"""tuple of str: the allowed values of :attr:`QuestionSpec.answer_type`."""


@dataclass(frozen=True)
class Referent:
    """An object a question refers to.

    Parameters
    ----------
    object_id : str
        The object's id within its scene.
    label : str
        The object's dataset class name.
    expression : str
        The phrase the question uses for it, e.g. ``"the leftmost chair"``.
    """

    object_id: str
    label: str
    expression: str


@dataclass(frozen=True)
class QuestionSpec:
    """A question about one image, independent of its answer.

    Parameters
    ----------
    template_id : str
        Template name with a version suffix, e.g. ``"distance_center_v1"``.
    qtype : str
        Question type, e.g. ``"distance_center"``.
    dataset, scene, camera_id, frame : str
        Which image is asked about (see :class:`measured_space.scene.FrameView`).
    image_ref : str
        Where to load the image from, relative to the dataset folder.
    referents : tuple of Referent
        The objects the question names, in template order.
    question : str
        The question text shown to a model.
    answer_type : str
        One of :data:`ANSWER_TYPES`.
    unit : str or None, default None
        Unit of a numeric answer, e.g. ``"m"``.
    choices : tuple of str or None, default None
        Allowed answers when ``answer_type == "choice"``.
    tolerance : dict, default {}
        ``{"abs": metres or None, "rel": fraction or None}`` for numeric answers; see :func:`is_correct`.
    params : dict, default {}
        Template-specific values, e.g. the class being counted.
    """

    template_id: str
    qtype: str
    dataset: str
    scene: str
    camera_id: str
    frame: str
    image_ref: str
    referents: tuple[Referent, ...]
    question: str
    answer_type: str
    unit: str | None = None
    choices: tuple[str, ...] | None = None
    tolerance: dict[str, float | None] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """str: a stable 16-character id derived from the template, image, referents and parameters."""
        key = "|".join([self.template_id, self.dataset, self.scene, self.camera_id, self.frame,
                        *[r.object_id for r in self.referents], json.dumps(self.params, sort_keys=True)])
        return hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class QAItem:
    """A question with an answer from one source.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    answer : float or int or str
        The answer from this source.
    source : str
        ``"oracle"`` (ground truth) or ``"auto"`` (perception).
    objects : tuple of dict, default ()
        For each referent (for a count, each counted object), the box used to compute the answer
        (``object_id``, ``label``, ``center``, ``size``, ``rotation`` in the world frame), so every answer
        can be recomputed.
    extra : dict, default {}
        Per-item diagnostics: decision margins, visibility, and for automatic labels the predicted
        uncertainty.
    """

    spec: QuestionSpec
    answer: float | int | str
    source: str
    objects: tuple[dict[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """str: the id of the item's question (shared by the oracle and automatic answers)."""
        return self.spec.id

    def to_json(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable record.

        Returns
        -------
        dict
            The spec's fields flattened, plus ``id``, ``answer``, ``source``, ``objects`` and ``extra``.
        """
        record = {"id": self.id, **asdict(self.spec), "answer": self.answer, "source": self.source,
                  "objects": list(self.objects), "extra": self.extra}
        record["referents"] = [asdict(r) for r in self.spec.referents]
        return record

    @staticmethod
    def from_json(record: dict[str, Any]) -> QAItem:
        """Rebuild an item from a record written by :meth:`to_json`.

        Parameters
        ----------
        record : dict
            One parsed JSONL line.

        Returns
        -------
        QAItem
            The item.
        """
        spec_fields = {k: record[k] for k in QuestionSpec.__dataclass_fields__ if k in record}
        spec_fields["referents"] = tuple(Referent(**r) for r in record["referents"])
        if spec_fields.get("choices") is not None:
            spec_fields["choices"] = tuple(spec_fields["choices"])
        return QAItem(QuestionSpec(**spec_fields), record["answer"], record["source"],
                      tuple(record.get("objects", ())), record.get("extra", {}))


def is_correct(spec: QuestionSpec, predicted: Any, reference: Any) -> bool:
    """Decide whether a predicted answer matches a reference answer.

    Numbers are correct if the absolute error is within ``tolerance["abs"]`` **or** within
    ``tolerance["rel"]`` times the reference. Either bound suffices, and a missing bound is ignored. Counts
    need an exact integer match. Choices match case- and whitespace-insensitively. ``None`` (an unparsed
    prediction) is never correct.

    Parameters
    ----------
    spec : QuestionSpec
        The question, which gives the answer type and tolerance.
    predicted : Any
        The answer being judged.
    reference : Any
        The answer it is judged against.

    Returns
    -------
    bool
        Whether ``predicted`` counts as correct.
    """
    if predicted is None:
        return False
    if spec.answer_type == "number":
        try:
            error = abs(float(predicted) - float(reference))
        except (TypeError, ValueError):
            return False
        abs_tol, rel_tol = spec.tolerance.get("abs"), spec.tolerance.get("rel")
        return bool((abs_tol is not None and error <= abs_tol)
                    or (rel_tol is not None and error <= rel_tol * abs(float(reference))))
    if spec.answer_type == "count":
        try:
            return int(predicted) == int(reference) and float(predicted) == int(predicted)
        except (TypeError, ValueError):
            return False
    return str(predicted).strip().lower() == str(reference).strip().lower()


def write_jsonl(items: Iterable[QAItem], path: str | Path) -> int:
    """Write items to a JSONL file, one JSON object per line.

    Parameters
    ----------
    items : iterable of QAItem
        Items to write.
    path : str or pathlib.Path
        Output file; its parent folders are created if needed. An existing file is replaced.

    Returns
    -------
    int
        Number of items written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item.to_json(), default=_json_default) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[QAItem]:
    """Read items from a JSONL file written by :func:`write_jsonl`.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file.

    Yields
    ------
    QAItem
        One item per non-empty line.
    """
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield QAItem.from_json(json.loads(line))


def _json_default(value: Any) -> Any:
    """Convert numpy values for ``json.dumps``.

    Parameters
    ----------
    value : Any
        A value the standard encoder could not serialise.

    Returns
    -------
    Any
        A list for numpy arrays, a Python scalar for numpy scalars.

    Raises
    ------
    TypeError
        For any other type.
    """
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
