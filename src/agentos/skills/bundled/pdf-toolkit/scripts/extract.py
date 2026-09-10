"""Extract text and tables from a PDF using pdfplumber + pypdf metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader


def parse_coordinates(raw: str | None, label: str) -> list[float]:
    """Parse a comma-separated list of page coordinates for the explicit strategy."""
    if not raw:
        return []
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"{label} expects numbers, got {token!r}") from exc
    return values


def build_table_settings(
    tables_strategy: str | None,
    vertical_lines: list[float] | None = None,
    horizontal_lines: list[float] | None = None,
) -> dict[str, Any]:
    """Return the pdfplumber table settings for *tables_strategy*.

    A strategy names how to find the edges of a table, and pdfplumber asks for
    it once per axis. Setting only ``vertical_strategy`` left the horizontal
    axis on its ``"lines"`` default, so ``--tables-strategy text`` still needed
    ruling lines to find its rows and returned nothing on exactly the
    borderless tables it exists to read.

    ``explicit`` additionally reads the edge positions out of
    ``explicit_vertical_lines`` / ``explicit_horizontal_lines``; with neither
    key present pdfplumber dereferences ``None`` and raises ``TypeError`` from
    inside ``get_edges()``, so the caller has to supply them.
    """
    strategy = tables_strategy or "lines"
    settings: dict[str, Any] = {
        "vertical_strategy": strategy,
        "horizontal_strategy": strategy,
    }
    if strategy == "explicit":
        if not vertical_lines or not horizontal_lines:
            raise ValueError(
                "--tables-strategy explicit needs both --explicit-vertical-lines and "
                "--explicit-horizontal-lines (comma-separated page coordinates)"
            )
        settings["explicit_vertical_lines"] = list(vertical_lines)
        settings["explicit_horizontal_lines"] = list(horizontal_lines)
    return settings


def extract(
    path: Path,
    tables_strategy: str | None,
    vertical_lines: list[float] | None = None,
    horizontal_lines: list[float] | None = None,
) -> dict[str, Any]:
    reader = PdfReader(str(path))
    metadata: dict[str, Any] = {}
    if reader.metadata is not None:
        for key, value in reader.metadata.items():
            metadata[str(key).lstrip("/")] = str(value)

    pages_text: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    table_settings = build_table_settings(tables_strategy, vertical_lines, horizontal_lines)
    with pdfplumber.open(str(path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            content = page.extract_text() or ""
            pages_text.append({"page": idx, "content": content})
            for tbl in page.extract_tables(table_settings) or []:
                tables.append({"page": idx, "rows": tbl})

    return {
        "pages": len(pages_text),
        "metadata": metadata,
        "text": pages_text,
        "tables": tables,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text and tables from a PDF.")
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--tables-strategy",
        choices=("lines", "text", "explicit"),
        default=None,
        help="pdfplumber table-detection strategy, applied to both axes",
    )
    parser.add_argument(
        "--explicit-vertical-lines",
        default=None,
        help="Comma-separated x coordinates of the column edges, for --tables-strategy explicit",
    )
    parser.add_argument(
        "--explicit-horizontal-lines",
        default=None,
        help="Comma-separated y coordinates of the row edges, for --tables-strategy explicit",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Force JSON output (default)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.path.is_file():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 2
    try:
        vertical = parse_coordinates(args.explicit_vertical_lines, "--explicit-vertical-lines")
        horizontal = parse_coordinates(
            args.explicit_horizontal_lines, "--explicit-horizontal-lines"
        )
        payload = extract(args.path, args.tables_strategy, vertical, horizontal)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
