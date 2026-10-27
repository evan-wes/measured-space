"""Question and answer records, and their JSONL storage.

A :class:`QuestionSpec` fixes *what* is asked about *which* objects in *which* image, independent of any
answer. The same spec is answered twice, once from ground-truth boxes (``source="oracle"``) and once from
perception boxes (``source="auto"``), so oracle and automatic labels are paired by construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ANSWER_TYPES = ("number", "choice", "count")


@dataclass(frozen=True)
class Referent:
    """An object a question refers to: its id in the scene, its class, and the phrase that names it."""

    object_id: str
    label: str
    expression: str


@dataclass(frozen=True)
class QuestionSpec:
    """A question about one image.

    ``tolerance`` is ``{"abs": metres or None, "rel": fraction or None}`` for numeric answers: an answer is
    correct if it is within either bound. ``choices`` lists the allowed answers for ``answer_type="choice"``.
    ``params`` holds template-specific values (e.g. the class counted).
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
        """Stable id from the template, image and referents."""
        key = "|".join([self.template_id, self.dataset, self.scene, self.camera_id, self.frame,
                        *[r.object_id for r in self.referents], json.dumps(self.params, sort_keys=True)])
        return hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class QAItem:
    """A question with an answer from one source.

    ``objects`` records, for each referent, the box used to compute the answer (``center``, ``size``,
    ``rotation`` in the world frame) so every answer can be recomputed. ``extra`` carries per-item
    diagnostics: decision margins, visibility, and for automatic labels the predicted uncertainty.
    """

    spec: QuestionSpec
    answer: float | int | str
    source: str
    objects: tuple[dict[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.spec.id

    def to_json(self) -> dict[str, Any]:
        record = {"id": self.id, **asdict(self.spec), "answer": self.answer, "source": self.source,
                  "objects": list(self.objects), "extra": self.extra}
        record["referents"] = [asdict(r) for r in self.spec.referents]
        return record

    @staticmethod
    def from_json(record: dict[str, Any]) -> "QAItem":
        spec_fields = {k: record[k] for k in QuestionSpec.__dataclass_fields__ if k in record}
        spec_fields["referents"] = tuple(Referent(**r) for r in record["referents"])
        if spec_fields.get("choices") is not None:
            spec_fields["choices"] = tuple(spec_fields["choices"])
        return QAItem(QuestionSpec(**spec_fields), record["answer"], record["source"],
                      tuple(record.get("objects", ())), record.get("extra", {}))


def write_jsonl(items: Iterable[QAItem], path: str | Path) -> int:
    """Write items, one JSON object per line; returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item.to_json(), default=_json_default) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[QAItem]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield QAItem.from_json(json.loads(line))


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
