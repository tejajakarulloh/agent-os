"""Split a PDF into multiple files by page-range spec.

Each disjoint range becomes one output file: <stem>_001.pdf, _002.pdf, ...

Usage:
    split.py input.pdf --pages "1-3,5,7-9" --out out_dir/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader, PdfWriter


@dataclass
class SplitResult:
    """What a split produced, including the pages it could not honour."""

    files: list[Path] = field(default_factory=list)
    pages: list[list[int]] = field(default_factory=list)
    total_pages: int = 0
    pages_out_of_range: list[int] = field(default_factory=list)


def split_ranges(spec: str) -> list[list[int]]:
    groups: list[list[int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            groups.append(list(range(lo, hi + 1)))
        else:
            groups.append([int(token)])
    return groups


def split(input_path: Path, pages_spec: str, out_dir: Path) -> SplitResult:
    """Write one file per range, and report what each one actually holds.

    Pages past the end of the document are dropped rather than refused, so a
    range that overruns still yields its valid pages — but the caller cannot
    see that from the file list alone, which is why the dropped pages and the
    per-file page lists are part of the result.
    """
    reader = PdfReader(str(input_path))
    total = len(reader.pages)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pages_written: list[list[int]] = []
    dropped: list[int] = []
    for idx, group in enumerate(split_ranges(pages_spec), start=1):
        valid_pages = [p for p in group if 1 <= p <= total]
        dropped.extend(p for p in group if not 1 <= p <= total)
        if not valid_pages:
            continue
        writer = PdfWriter()
        for page_num in valid_pages:
            writer.add_page(reader.pages[page_num - 1])
        out_path = out_dir / f"{input_path.stem}_{idx:03d}.pdf"
        with out_path.open("wb") as fh:
            writer.write(fh)
        written.append(out_path)
        pages_written.append(valid_pages)
    return SplitResult(
        files=written,
        pages=pages_written,
        total_pages=total,
        pages_out_of_range=dropped,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a PDF by page ranges.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--pages", required=True, help="e.g. '1-3,5,7-9'")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.input.is_file():
        print(f"error: input {args.input} not found", file=sys.stderr)
        return 2
    result = split(args.input, args.pages, args.out)
    print(
        json.dumps(
            {
                "files": [str(p) for p in result.files],
                "count": len(result.files),
                "pages": result.pages,
                "total_pages": result.total_pages,
                "pages_out_of_range": result.pages_out_of_range,
            },
            ensure_ascii=False,
        )
    )
    if not result.files:
        print(
            f"error: no requested page is within {args.input} "
            f"(1-{result.total_pages}); nothing was written",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
