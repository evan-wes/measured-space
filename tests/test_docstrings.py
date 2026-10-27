"""Enforce the documentation standard on the package source.

Every module, class, function and method under ``src/measured_space`` must have a docstring. Functions
and methods that take arguments (other than ``self``/``cls``) must document them in a numpy-style
``Parameters`` section, and functions that return a value must have a ``Returns`` or ``Yields`` section.
Properties are exempt from the section rules (their docstring states the type and meaning inline).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "measured_space"
SOURCES = sorted(SRC.rglob("*.py"))


def _returns_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Tell whether a function returns something other than ``None``.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        The function's syntax tree.

    Returns
    -------
    bool
        True if its annotation is not ``None``, or, unannotated, if it has a ``return <value>``.
    """
    if node.returns is not None:
        return not (isinstance(node.returns, ast.Constant) and node.returns.value is None)
    return any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node))


def _problems(path: Path) -> list[str]:
    """List documentation-standard violations in one source file.

    Parameters
    ----------
    path : pathlib.Path
        A Python source file.

    Returns
    -------
    list of str
        One message per violation, ``"<file>:<line> <name>: <problem>"``.
    """
    tree = ast.parse(path.read_text())
    rel = path.relative_to(SRC.parent)
    problems = [] if ast.get_docstring(tree) else [f"{rel}: missing module docstring"]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
            problems.append(f"{rel}:{node.lineno} {node.name}: missing class docstring")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        where = f"{rel}:{node.lineno} {node.name}"
        if not doc:
            problems.append(f"{where}: missing docstring")
            continue
        decorators = {d.id if isinstance(d, ast.Name) else getattr(d, "attr", "") for d in node.decorator_list}
        if "property" in decorators:
            continue
        args = [a.arg for a in node.args.args + node.args.kwonlyargs if a.arg not in ("self", "cls")]
        if node.args.vararg:
            args.append(node.args.vararg.arg)
        if node.args.kwarg:
            args.append(node.args.kwarg.arg)
        if args and "Parameters\n" not in doc:
            problems.append(f"{where}: no 'Parameters' section")
        if _returns_value(node) and node.name != "__post_init__" and not (
                "Returns\n" in doc or "Yields\n" in doc):
            problems.append(f"{where}: no 'Returns'/'Yields' section")
    return problems


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(SRC.parent)))
def test_source_file_meets_documentation_standard(path: Path) -> None:
    """Every docstring rule holds for this source file.

    Parameters
    ----------
    path : pathlib.Path
        The source file under test.
    """
    problems = _problems(path)
    assert not problems, "\n".join(problems)
