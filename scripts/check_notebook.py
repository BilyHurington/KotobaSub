"""Validate notebook JSON and Python cell syntax."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/check_notebook.py NOTEBOOK.ipynb")

    notebook_path = Path(sys.argv[1])
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    if notebook.get("nbformat") != 4:
        raise ValueError(f"Expected nbformat 4, got {notebook.get('nbformat')!r}")

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        python_source = strip_colab_shell_lines(source)
        if not python_source.strip():
            continue

        try:
            ast.parse(python_source, filename=f"{notebook_path}:cell-{index}")
        except SyntaxError as exc:
            raise SyntaxError(f"Syntax error in code cell {index}: {exc}") from exc


def strip_colab_shell_lines(source: str) -> str:
    """Replace Colab shell escape lines with blank lines for AST parsing."""

    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            indent = line[: len(line) - len(stripped)]
            lines.append(f"{indent}pass")
        else:
            lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
