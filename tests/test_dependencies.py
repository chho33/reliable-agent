"""Every third-party module imported by the package must be declared in pyproject.toml.

Guards against "works in my environment" failures: a module that happens to be installed
locally but is missing from the dependency list breaks a fresh `uv sync` checkout.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {"agent", "backends", "tools", "llm", "evals", "tests", "conftest"}
# import name -> distribution name, where they differ
DIST_NAME = {"yaml": "pyyaml"}


def _declared() -> set[str]:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    specs = list(meta.get("dependencies", []))
    for extra in meta.get("optional-dependencies", {}).values():
        specs += extra
    return {re.split(r"[<>=!~\[; ]", s, maxsplit=1)[0].strip().lower() for s in specs}


def _imported() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in ROOT.rglob("*.py"):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue  # .venv, .git, ...
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in LOCAL or top == "__future__":
                    continue
                found.setdefault(DIST_NAME.get(top, top).lower(), set()).add(str(path.relative_to(ROOT)))
    return found


def test_all_third_party_imports_are_declared() -> None:
    missing = {dist: sorted(files) for dist, files in _imported().items() if dist not in _declared()}
    assert not missing, f"imported but not declared in pyproject.toml: {missing}"
