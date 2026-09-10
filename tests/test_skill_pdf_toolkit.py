"""pdf-toolkit skill — load, eligibility, and merge→split→extract round-trip."""

from __future__ import annotations

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
    assert len(parts) == 2

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


def _make_borderless_table_pdf(path: Path) -> None:
    """A table with no ruling lines — the layout ``--tables-strategy text`` is for."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.platypus import SimpleDocTemplate, Table

    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    doc.build([Table([["Name", "Qty"], ["Widget", "3"], ["Gadget", "7"]])])


def _import_extract() -> object:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import extract  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return extract


def test_table_strategy_applies_to_both_axes() -> None:
    """A strategy names how to find edges, and pdfplumber asks per axis.

    Setting only ``vertical_strategy`` left the horizontal axis on its
    ``"lines"`` default, so ``text`` mode still hunted for ruling lines to
    find its rows.
    """
    extract = _import_extract()

    for strategy in ("lines", "text"):
        settings = extract.build_table_settings(strategy)
        assert settings["vertical_strategy"] == strategy
        assert settings["horizontal_strategy"] == strategy

    # The default is still lines, on both axes.
    assert extract.build_table_settings(None) == {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    }


def test_text_strategy_finds_a_borderless_table(tmp_path: Path) -> None:
    """The regression the flag exists for: a table with no ruling lines.

    Under the one-axis settings this came back as ``"tables": []`` — the
    horizontal axis was still looking for lines the table does not draw.
    """
    extract = _import_extract()

    pdf_file = tmp_path / "table.pdf"
    _make_borderless_table_pdf(pdf_file)

    payload = extract.extract(pdf_file, tables_strategy="text")
    rows = [cell for table in payload["tables"] for row in table["rows"] for cell in row]
    assert "Widget" in rows
    assert "Gadget" in rows

    # ``lines`` still reports nothing here, which is correct: there are none.
    assert extract.extract(pdf_file, tables_strategy="lines")["tables"] == []


def test_explicit_strategy_carries_the_edge_coordinates(tmp_path: Path) -> None:
    """``explicit`` reads the edges from the settings, so they must be sent."""
    extract = _import_extract()

    settings = extract.build_table_settings("explicit", [270.0, 315.0], [78.0, 96.0])
    assert settings["explicit_vertical_lines"] == [270.0, 315.0]
    assert settings["explicit_horizontal_lines"] == [78.0, 96.0]

    pdf_file = tmp_path / "table.pdf"
    _make_borderless_table_pdf(pdf_file)
    payload = extract.extract(
        pdf_file,
        tables_strategy="explicit",
        vertical_lines=[270.0, 315.0, 360.0],
        horizontal_lines=[78.0, 96.0, 114.0, 132.0],
    )
    assert payload["tables"], "explicit edges should yield a table"
    assert payload["tables"][0]["rows"][0] == ["Name", "Qty"]


@pytest.mark.parametrize(
    ("vertical", "horizontal"),
    [(None, None), ([270.0], None), (None, [78.0])],
)
def test_explicit_strategy_without_edges_is_a_clear_error(
    vertical: list[float] | None, horizontal: list[float] | None
) -> None:
    """Missing edges used to surface as ``TypeError`` from inside pdfplumber.

    ``get_edges()`` called ``len(None)`` on the absent
    ``explicit_vertical_lines``, so every ``--tables-strategy explicit`` run
    ended in a traceback naming a pdfplumber internal.
    """
    extract = _import_extract()

    with pytest.raises(ValueError, match="explicit-vertical-lines"):
        extract.build_table_settings("explicit", vertical, horizontal)


def test_explicit_strategy_reports_the_error_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    extract = _import_extract()

    pdf_file = tmp_path / "doc.pdf"
    _make_one_page_pdf(pdf_file, "TEST")
    monkeypatch.setattr(sys, "argv", ["extract.py", str(pdf_file), "--tables-strategy", "explicit"])
    assert extract.main() == 2
    assert "explicit-vertical-lines" in capsys.readouterr().err


def test_explicit_coordinates_reject_non_numbers() -> None:
    extract = _import_extract()

    assert extract.parse_coordinates("270, 315.5 ,360", "--explicit-vertical-lines") == [
        270.0,
        315.5,
        360.0,
    ]
    assert extract.parse_coordinates(None, "--explicit-vertical-lines") == []
    with pytest.raises(ValueError, match="expects numbers"):
        extract.parse_coordinates("270,left-edge", "--explicit-vertical-lines")
