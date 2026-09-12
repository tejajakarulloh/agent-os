"""pdf-toolkit skill — load, eligibility, and merge→split→extract round-trip."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agentos.skills.eligibility import EligibilityContext, check_eligibility
from agentos.skills.loader import SkillLoader

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "src" / "agentos" / "skills" / "bundled"
SCRIPTS = BUNDLED / "pdf-toolkit" / "scripts"


def _spec() -> object:
    return SkillLoader(bundled_dir=BUNDLED).get_by_name("pdf-toolkit")


def test_skill_loads() -> None:
    spec = _spec()
    assert spec is not None
    assert spec.name == "pdf-toolkit"
    description = spec.description.lower()
    assert "nano-pdf" in description, (
        "description must explicitly distinguish from sibling nano-pdf skill"
    )


def test_eligibility_with_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentos.skills.eligibility.shutil.which",
        lambda name: "/usr/bin/python3" if name in {"python", "python3"} else None,
    )
    spec = _spec()
    assert spec is not None
    assert check_eligibility(spec, EligibilityContext.auto())


def _make_one_page_pdf(path: Path, label: str) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setFont("Helvetica", 14)
    c.drawString(72, 720, label)
    c.showPage()
    c.save()


def test_merge_split_extract_round_trip(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import extract  # type: ignore[import-not-found]
        import merge  # type: ignore[import-not-found]
        import split  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    _make_one_page_pdf(a, "ALPHA")
    _make_one_page_pdf(b, "BRAVO")

    combined = tmp_path / "combined.pdf"
    written = merge.merge([{"file": str(a)}, {"file": str(b)}], combined)
    assert written == 2
    assert combined.exists()

    out_dir = tmp_path / "split_out"
    parts = split.split(combined, "1,2", out_dir)
    assert len(parts.files) == 2
    assert parts.pages == [[1], [2]]
    assert parts.pages_out_of_range == []

    payload = extract.extract(combined, tables_strategy=None)
    assert payload["pages"] == 2
    page_texts = [item["content"] for item in payload["text"]]
    full_text = "\n".join(page_texts)
    assert "ALPHA" in full_text
    assert "BRAVO" in full_text


def test_split_range_parsing() -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import split  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    assert split.split_ranges("1-3") == [[1, 2, 3]]
    assert split.split_ranges("1,3,5") == [[1], [3], [5]]
    assert split.split_ranges("1-2,4-5") == [[1, 2], [4, 5]]
    # Reverse range gets normalized.
    assert split.split_ranges("5-3") == [[3, 4, 5]]


def test_merge_range_parsing() -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import merge  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    assert merge.parse_ranges(None, 4) == [1, 2, 3, 4]
    assert merge.parse_ranges("1,3", 4) == [1, 3]
    assert merge.parse_ranges("1-3", 5) == [1, 2, 3]
    # Out-of-range pages are filtered.
    assert merge.parse_ranges("1,99", 4) == [1]


def test_extract_creates_parent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import extract  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    pdf_file = tmp_path / "doc.pdf"
    _make_one_page_pdf(pdf_file, "TEST")

    out_file = tmp_path / "nested" / "dir" / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["extract.py", str(pdf_file), "--out", str(out_file)],
    )
    assert extract.main() == 0
    assert out_file.is_file()


def _extract_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import extract  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return extract


def _make_borderless_table_pdf(path: Path) -> None:
    """A platypus table with no ruling lines -- the layout `text` mode exists for."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.platypus import SimpleDocTemplate, Table

    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    # Four rows: pdfplumber's `text` mode needs ``min_words_vertical`` (default
    # 3) aligned words to accept a column edge, so stay clear of that threshold.
    doc.build([Table([["Name", "Qty"], ["Widget", "3"], ["Gadget", "7"], ["Gizmo", "9"]])])


def test_tables_strategy_text_detects_a_borderless_table(tmp_path: Path) -> None:
    """`--tables-strategy text` must switch both axes.

    Only `vertical_strategy` used to be set, so the horizontal axis kept
    looking for ruling lines a borderless table does not have and the flag's
    one documented use case returned no tables at all.
    """
    extract = _extract_module()
    pdf_file = tmp_path / "borderless.pdf"
    _make_borderless_table_pdf(pdf_file)

    payload = extract.extract(pdf_file, tables_strategy="text")

    cells = {cell for table in payload["tables"] for row in table["rows"] for cell in row if cell}
    assert {"Name", "Qty", "Widget", "3", "Gadget", "7", "Gizmo", "9"} <= cells


def test_tables_strategy_lines_still_the_default(tmp_path: Path) -> None:
    extract = _extract_module()
    pdf_file = tmp_path / "borderless.pdf"
    _make_borderless_table_pdf(pdf_file)

    assert extract._table_settings(None) == {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    }
    # A borderless table has no ruling lines, so the default finds nothing.
    assert extract.extract(pdf_file, tables_strategy=None)["tables"] == []


def test_tables_strategy_explicit_is_rejected_with_a_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """pdfplumber's `explicit` mode needs line lists this script cannot supply.

    It used to be advertised in `choices` and then crash inside pdfplumber
    with `TypeError: object of type 'NoneType' has no len()` on every call.
    """
    extract = _extract_module()
    pdf_file = tmp_path / "doc.pdf"
    _make_one_page_pdf(pdf_file, "TEST")

    with pytest.raises(ValueError, match="explicit"):
        extract.extract(pdf_file, tables_strategy="explicit")

    monkeypatch.setattr(sys, "argv", ["extract.py", str(pdf_file), "--tables-strategy", "explicit"])
    with pytest.raises(SystemExit) as exc_info:
        extract.main()
    assert exc_info.value.code == 2
    assert "invalid choice: 'explicit'" in capsys.readouterr().err


def _split_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import split  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return split


def _make_n_page_pdf(path: Path, pages: int) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=LETTER)
    for page in range(1, pages + 1):
        c.setFont("Helvetica", 14)
        c.drawString(72, 720, f"PAGE {page}")
        c.showPage()
    c.save()


def _page_count(path: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


def test_split_reports_the_pages_each_output_actually_holds(tmp_path: Path) -> None:
    """A range inside the document is unchanged, and now says what it wrote."""
    split = _split_module()
    src = tmp_path / "five.pdf"
    _make_n_page_pdf(src, 5)

    result = split.split(src, "1-3,5", tmp_path / "out")
    assert [p.name for p in result.files] == ["five_001.pdf", "five_002.pdf"]
    assert result.pages == [[1, 2, 3], [5]]
    assert result.total_pages == 5
    assert result.pages_out_of_range == []
    assert [_page_count(p) for p in result.files] == [3, 1]


def test_split_names_the_pages_an_overrunning_range_could_not_honour(
    tmp_path: Path,
) -> None:
    """`3-7` on a 5-page document wrote a 3-page file and said nothing."""
    split = _split_module()
    src = tmp_path / "five.pdf"
    _make_n_page_pdf(src, 5)

    result = split.split(src, "3-7", tmp_path / "out")
    assert len(result.files) == 1
    assert _page_count(result.files[0]) == 3
    assert result.pages == [[3, 4, 5]]
    assert result.pages_out_of_range == [6, 7]


def test_split_collects_out_of_range_pages_across_every_group(tmp_path: Path) -> None:
    """Each range contributes its own rejects, in the order they were asked for."""
    split = _split_module()
    src = tmp_path / "five.pdf"
    _make_n_page_pdf(src, 5)

    result = split.split(src, "4-6,1-2,9", tmp_path / "out")
    assert result.pages == [[4, 5], [1, 2]]
    assert result.pages_out_of_range == [6, 9]


def test_split_reports_a_page_below_the_first_as_out_of_range(tmp_path: Path) -> None:
    """The other end of the clamp: page 0 is not a page either."""
    split = _split_module()
    src = tmp_path / "five.pdf"
    _make_n_page_pdf(src, 5)

    result = split.split(src, "0-2", tmp_path / "out")
    assert result.pages == [[1, 2]]
    assert result.pages_out_of_range == [0]


def test_split_exits_non_zero_when_no_requested_page_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`7-9` on a 5-page document wrote nothing and exited 0."""
    split = _split_module()
    src = tmp_path / "five.pdf"
    _make_n_page_pdf(src, 5)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv", ["split.py", str(src), "--pages", "7-9", "--out", str(out_dir)]
    )
    assert split.main() == 2
    captured = capsys.readouterr()
    assert "nothing was written" in captured.err
    assert "1-5" in captured.err
    payload = json.loads(captured.out)
    assert payload["files"] == []
    assert payload["pages_out_of_range"] == [7, 8, 9]
    assert payload["total_pages"] == 5


def test_split_exits_zero_when_a_partial_range_still_produced_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A short file is a result, not a failure — the report is what carries it."""
    split = _split_module()
    src = tmp_path / "five.pdf"
    _make_n_page_pdf(src, 5)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv", ["split.py", str(src), "--pages", "3-7", "--out", str(out_dir)]
    )
    assert split.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["pages"] == [[3, 4, 5]]
    assert payload["pages_out_of_range"] == [6, 7]


def test_split_keeps_its_existing_summary_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`files` and `count` keep their shape; the new keys are additive."""
    split = _split_module()
    src = tmp_path / "five.pdf"
    _make_n_page_pdf(src, 5)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv", ["split.py", str(src), "--pages", "1-2,4", "--out", str(out_dir)]
    )
    assert split.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 2
    assert all(isinstance(name, str) for name in payload["files"])
    assert [Path(name).name for name in payload["files"]] == [
        "five_001.pdf",
        "five_002.pdf",
    ]
