"""Tests for ``claims.py``: the rounding check, and that every claim can be recomputed from the committed tables."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "claims.py"


def load_claims() -> ModuleType:
    """Import ``claims.py`` as a module.

    Returns
    -------
    types.ModuleType
        The module; importing it has no other side effects.
    """
    if "claims" in sys.modules:
        return sys.modules["claims"]
    spec = importlib.util.spec_from_file_location("claims", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["claims"] = module
    spec.loader.exec_module(module)
    return module


def test_matches_uses_the_stated_precision() -> None:
    """Numbers pass within half a unit of their last written digit; fractions and booleans match exactly."""
    m = load_claims()
    assert m.matches("33", 33.49) and not m.matches("33", 33.51)
    assert m.matches("7.0", 6.95) and not m.matches("7.0", 6.94)
    assert m.matches("0.90", 0.8969) and not m.matches("0.70", 0.7053)
    assert m.matches("-0.064", -0.0644)
    assert m.matches("1/6", 0.155) and not m.matches("1/6", 0.25)
    assert m.matches("True", 1.0) and not m.matches("True", 0.0) and m.matches("False", 0.0)
    assert not m.matches("1", float("nan"))


def test_every_claim_recomputes_from_the_committed_tables() -> None:
    """No claim errors on the committed tables (whether it passes is the script's report, not this test's)."""
    m = load_claims()
    if not (m.RESULTS / "review_effort" / "operating_points.csv").exists():
        pytest.skip("committed result tables not present")
    rows = m.evaluate(m.CLAIMS, m.RESULTS)
    errors = [r for r in rows if r["status"] == "ERROR"]
    assert not errors, errors
    assert {c.id for c in m.CLAIMS} == {r["id"] for r in rows}


def test_report_refuses_to_overwrite(tmp_path: Path) -> None:
    """``--report`` never overwrites an existing file."""
    m = load_claims()
    existing = tmp_path / "claims.md"
    existing.write_text("keep\n")
    assert m.main(["--report", str(existing)]) == 2
    assert existing.read_text() == "keep\n"
