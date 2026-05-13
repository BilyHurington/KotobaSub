"""Utilities for building the Colab notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any


def markdown_cell(source: str) -> dict[str, Any]:
    """Create a notebook markdown cell."""

    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split_source(source),
    }


def code_cell(source: str) -> dict[str, Any]:
    """Create a notebook code cell."""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_source(source),
    }


def build_notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a minimal Colab-compatible notebook document."""

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(notebook: dict[str, Any], output_path: str | Path) -> Path:
    """Write a notebook as formatted JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _split_source(source: str) -> list[str]:
    source = textwrap.dedent(source).strip("\n")
    if not source:
        return []
    return [line + "\n" for line in source.splitlines()]
