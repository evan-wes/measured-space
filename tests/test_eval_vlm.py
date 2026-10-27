"""Tests for VLM prompting, answer parsing and the resume-safe run loop (CPU only, no model loaded)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from measured_space.eval_vlm import (Generation, Prompt, build_prompt, parse_answer, read_predictions,
                                     run_predictions)
from measured_space.qa_schema import QAItem, QuestionSpec


def _spec(answer_type: str, frame: str = "0") -> QuestionSpec:
    """A minimal question of one answer type.

    Parameters
    ----------
    answer_type : str
        ``"number"``, ``"choice"`` or ``"count"``.
    frame : str
        Frame name; varying it gives distinct ids.

    Returns
    -------
    QuestionSpec
        Numbers in metres with a 0.1 m / 10 % tolerance; choices ``("left", "right")``.
    """
    return QuestionSpec("t_v1", answer_type, "synthetic", "s0", "cam0", frame, "img.png", (), "Where?",
                        answer_type, unit="m" if answer_type == "number" else None,
                        choices=("left", "right") if answer_type == "choice" else None,
                        tolerance={"abs": 0.1, "rel": 0.1} if answer_type == "number" else {})


@pytest.mark.parametrize("answer_type, expected", [
    ("number", '{"answer": <number>, "unit": "m"}'),
    ("choice", 'exactly one of: "left", "right"'),
    ("count", '{"answer": <integer>}'),
])
def test_prompt_states_the_json_format_for_each_answer_type(answer_type: str, expected: str) -> None:
    """The system instruction carries the answer format; the user text is the question.

    Parameters
    ----------
    answer_type : str
        Answer type under test.
    expected : str
        Text the system instruction must contain.
    """
    prompt = build_prompt(_spec(answer_type))
    assert expected in prompt.system and "Reply with only a single JSON object" in prompt.system
    assert prompt.user == "Where?" and prompt.with_image


def test_blind_and_reasoning_prompts() -> None:
    """Blind prompts send no image and say so; reasoning prompts ask for a closed think block."""
    blind = build_prompt(_spec("number"), blind=True)
    assert not blind.with_image and "No image is provided" in blind.system
    reasoning = build_prompt(_spec("count"), reasoning=True)
    assert "</think>" in reasoning.user and "final answer must be" in reasoning.system


@pytest.mark.parametrize("answer_type, text, expected", [
    ("number", '{"answer": 2.5, "unit": "m"}', 2.5),
    ("number", 'Here you go:\n```json\n{"answer": 250, "unit": "cm"}\n```', 2.5),
    ("number", '{"answer": "1200 mm"}', 1.2),
    ("number", "It is about 5 ft 6 in.", 5 * 0.3048 + 6 * 0.0254),
    ("number", "Roughly 24 inches", 24 * 0.0254),
    ("number", "between 2 and 3 meters", 2.5),
    ("number", "The table is 2 in length", 2.0),                 # "in" before a word is not inches
    ("number", '{"answer": 2, "unit": "furlongs"}', None),        # unknown unit: no silent metres
    ("number", "<think>The room is 4 m wide.</think>\n{\"answer\": 1.5, \"unit\": \"m\"}", 1.5),
    ("number", "<think>The room is 4 m wide and the sofa", None),  # reasoning cut off by the budget
    ("number", "I cannot tell from this image, maybe 1 or 2", None),
    ("choice", '{"answer": "LEFT"}', "left"),
    ("choice", 'The chair is on the "right".', "right"),
    ("choice", "It is left of the table, not right.", None),       # both choices: ambiguous
    ("count", '{"answer": 3.0}', 3),
    ("count", "There are three chairs.", 3),
    ("count", '{"answer": 2.5}', None),
    ("count", "asdf qwerty", None),
])
def test_parse_answer(answer_type: str, text: str, expected: Any) -> None:
    """JSON first, then text fallbacks; unit conversion; think blocks; garbage parses to None.

    Parameters
    ----------
    answer_type : str
        Answer type of the question.
    text : str
        Raw model output.
    expected : Any
        Parsed value (metres for numbers), or None.
    """
    value = parse_answer(_spec(answer_type), text).value
    if isinstance(expected, float):
        assert value == pytest.approx(expected)
    else:
        assert value == expected


class FakeRunner:
    """A stand-in model that always answers 1.5 m and counts its calls."""

    def __init__(self, max_new_tokens: int = 128) -> None:
        """Set the reported budget and zero the call counter.

        Parameters
        ----------
        max_new_tokens : int
            Token budget reported in ``info``.
        """
        self.max_new_tokens = max_new_tokens
        self.calls = 0

    @property
    def info(self) -> dict[str, Any]:
        """Run configuration, as :class:`measured_space.eval_vlm.VLMRunner` reports it."""
        return {"model": "fake", "model_id": "test/fake", "snapshot": "abc123", "dtype": "bfloat16",
                "attn_implementation": "sdpa", "resolution": "r", "min_pixels": 1, "max_pixels": 2,
                "image_processor": "none", "max_new_tokens": self.max_new_tokens, "reasoning": False,
                "decoding": "greedy", "prompt_version": "v1"}

    def generate(self, prompt: Prompt, image: np.ndarray | None) -> Generation:
        """Return a fixed answer.

        Parameters
        ----------
        prompt : Prompt
            Ignored apart from its image flag.
        image : numpy.ndarray or None
            Must be present exactly when the prompt expects one.

        Returns
        -------
        Generation
            ``{"answer": 1.5, "unit": "m"}`` with made-up costs.
        """
        assert prompt.with_image == (image is not None)
        self.calls += 1
        return Generation('{"answer": 1.5, "unit": "m"}', 12, False, 0.01, 0 if image is None else 4)


def test_run_predictions_is_resume_safe(tmp_path: Path) -> None:
    """Questions are asked once per spec, resumed after a cut-off line, and configurations never mix.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    specs = [_spec("number", frame=str(i)) for i in range(3)]
    items = [QAItem(s, 1.0, source) for s in specs for source in ("oracle", "auto")]   # each spec twice
    out = tmp_path / "preds.jsonl"
    loads: list[str] = []

    def load_image(spec: QuestionSpec) -> np.ndarray:
        """Record the request and return a blank image.

        Parameters
        ----------
        spec : QuestionSpec
            The question whose image is requested.

        Returns
        -------
        numpy.ndarray
            ``(4, 6, 3)`` uint8 zeros.
        """
        loads.append(spec.id)
        return np.zeros((4, 6, 3), np.uint8)

    runner = FakeRunner()
    first = run_predictions(runner, items, out, load_image, limit=2, log=None)
    assert (first.n_specs, first.n_skipped, first.n_written) == (3, 0, 2)
    with out.open("a") as f:
        f.write('{"id": "cut-off mid-wri')                         # a run killed mid-write
    second = run_predictions(runner, items, out, load_image, log=None)
    assert (second.n_skipped, second.n_written) == (2, 1) and runner.calls == 3 and len(loads) == 3

    records = read_predictions(out)
    assert sorted(r["id"] for r in records) == sorted(s.id for s in specs)
    assert records[0]["parsed"] == 1.5 and records[0]["image_hw"] == [4, 6]
    assert {"model_id", "snapshot", "dtype", "resolution", "max_new_tokens", "raw_output", "latency_s",
            "blind"} <= set(records[0])

    with pytest.raises(ValueError, match="max_new_tokens"):
        run_predictions(FakeRunner(max_new_tokens=64), items, out, load_image, log=None)
    run_predictions(runner, items, tmp_path / "blind.jsonl", None, blind=True, log=None)
    assert all(r["blind"] and r["image_hw"] is None for r in read_predictions(tmp_path / "blind.jsonl"))
