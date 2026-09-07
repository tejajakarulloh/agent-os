"""xlsx skill — load, eligibility, and create→inspect→edit round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentos.skills.eligibility import EligibilityContext, check_eligibility
from agentos.skills.loader import SkillLoader

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "src" / "agentos" / "skills" / "bundled"
SCRIPTS = BUNDLED / "xlsx" / "scripts"


def _spec() -> object:
    return SkillLoader(bundled_dir=BUNDLED).get_by_name("xlsx")


def test_skill_loads() -> None:
    spec = _spec()
    assert spec is not None
    assert spec.name == "xlsx"
    assert spec.metadata is not None


def test_eligibility_with_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentos.skills.eligibility.shutil.which",
        lambda name: "/usr/bin/python3" if name in {"python", "python3"} else None,
    )
    spec = _spec()
    assert spec is not None
    assert check_eligibility(spec, EligibilityContext.auto())


def test_round_trip_with_formula_and_merge(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import create_xlsx  # type: ignore[import-not-found]
        import edit_xlsx  # type: ignore[import-not-found]
        import inspect_xlsx  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    spec = {
        "sheets": [
            {
                "name": "Sales",
                "rows": [
                    ["Region", "Revenue"],
                    ["NA", 1_200_000],
                    ["EU", 850_000],
                    ["Total", "=SUM(B2:B3)"],
                ],
                "merged": [{"range": "A1:B1"}],
                "freeze": "A2",
            }
        ]
    }
    src = tmp_path / "book.xlsx"
    create_xlsx.build(spec).save(str(src))
    assert src.exists()

    inspected = inspect_xlsx.inspect(src, data_only=False)
    sheet = next(s for s in inspected["sheets"] if s["name"] == "Sales")
    assert sheet["max_row"] == 4
    assert sheet["max_col"] == 2
    assert any("A1:B1" in r for r in sheet["merged"])
    assert sheet["freeze"] == "A2"

    last_row = sheet["rows"][3]
    assert last_row[1]["type"] == "f"
    assert last_row[1]["value"] == "=SUM(B2:B3)"

    from openpyxl import load_workbook

    wb = load_workbook(str(src))
    edit_xlsx.apply_ops(
        wb,
        [
            {"op": "set_cell", "sheet": "Sales", "row": 2, "col": 1, "value": "Americas"},
            {"op": "rename_sheet", "old": "Sales", "new": "Q3"},
        ],
    )
    out = tmp_path / "out.xlsx"
    wb.save(str(out))

    re_inspected = inspect_xlsx.inspect(out, data_only=False)
    sheet_q3 = next(s for s in re_inspected["sheets"] if s["name"] == "Q3")
    assert sheet_q3["rows"][1][0]["value"] == "Americas"


def test_text_escapes_formula(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import create_xlsx  # type: ignore[import-not-found]
        import edit_xlsx  # type: ignore[import-not-found]
        import inspect_xlsx  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    src = tmp_path / "book.xlsx"
    create_xlsx.build({"sheets": [{"name": "S", "rows": [["a"]]}]}).save(str(src))

    from openpyxl import load_workbook

    wb = load_workbook(str(src))
    edit_xlsx.apply_ops(
        wb,
        [
            {
                "op": "set_cell",
                "sheet": "S",
                "row": 2,
                "col": 1,
                "value": "=hello",
                "as_text": True,
            },
        ],
    )
    out = tmp_path / "out.xlsx"
    wb.save(str(out))

    inspected = inspect_xlsx.inspect(out, data_only=False)
    sheet = inspected["sheets"][0]
    cell_value = sheet["rows"][1][0]["value"]
    assert isinstance(cell_value, str)
    assert cell_value.lstrip("'") == "=hello"


def test_inspect_xlsx_creates_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import create_xlsx  # type: ignore[import-not-found]
        import inspect_xlsx  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    src = tmp_path / "book.xlsx"
    create_xlsx.build({"sheets": [{"name": "S", "rows": [["a"]]}]}).save(str(src))

    out = tmp_path / "nested" / "dir" / "out.json"
    monkeypatch.setattr(sys, "argv", ["inspect_xlsx.py", str(src), "--out", str(out)])
    assert inspect_xlsx.main() == 0
    assert out.is_file()


def _edit_and_reload(tmp_path: Path, ops: list[dict[str, object]]) -> object:
    """Run ``ops`` against a one-cell workbook and reload the saved result."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        import create_xlsx  # type: ignore[import-not-found]
        import edit_xlsx  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    from openpyxl import load_workbook

    src = tmp_path / "book.xlsx"
    create_xlsx.build({"sheets": [{"name": "S", "rows": [["a"]]}]}).save(str(src))
    wb = load_workbook(str(src))
    edit_xlsx.apply_ops(wb, ops)
    out = tmp_path / "out.xlsx"
    wb.save(str(out))
    return load_workbook(str(out))["S"]


def _set_cell(row: int, value: object, **extra: object) -> dict[str, object]:
    return {"op": "set_cell", "sheet": "S", "row": row, "col": 1, "value": value, **extra}


def test_as_text_keeps_the_formula_string_out_of_the_cell_value(tmp_path: Path) -> None:
    """``as_text: true`` stores ``=hello``, not Excel's input-mode apostrophe.

    The apostrophe is an escape typed into Excel's formula bar, not cell
    content. Prepending it left the cell holding ``'=hello`` — seven characters
    where the caller passed six — so a round-trip comparison against the
    original string failed and both Excel and LibreOffice rendered a stray
    apostrophe.
    """
    sheet = _edit_and_reload(tmp_path, [_set_cell(2, "=hello", as_text=True)])
    cell = sheet.cell(row=2, column=1)

    assert cell.value == "=hello"
    assert cell.data_type == "s"
    assert cell.quotePrefix is True


def test_inspect_reports_the_escaped_formula_without_an_apostrophe(
    tmp_path: Path,
) -> None:
    """The skill's own inspector is where a caller reads the value back.

    ``inspect_xlsx`` reported ``'=hello``, so the corruption was visible
    through the documented read path and not only through openpyxl directly.
    """
    sys.path.insert(0, str(SCRIPTS))
    try:
        import create_xlsx  # type: ignore[import-not-found]
        import edit_xlsx  # type: ignore[import-not-found]
        import inspect_xlsx  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    from openpyxl import load_workbook

    src = tmp_path / "book.xlsx"
    create_xlsx.build({"sheets": [{"name": "S", "rows": [["a"]]}]}).save(str(src))
    wb = load_workbook(str(src))
    edit_xlsx.apply_ops(wb, [_set_cell(2, "=hello", as_text=True)])
    out = tmp_path / "out.xlsx"
    wb.save(str(out))

    reported = inspect_xlsx.inspect(out, data_only=False)["sheets"][0]["rows"][1][0]

    assert reported["value"] == "=hello"
    assert reported["type"] == "s"


def test_without_as_text_a_formula_string_stays_a_formula(tmp_path: Path) -> None:
    """The default path is unchanged: openpyxl still records a formula cell."""
    sheet = _edit_and_reload(tmp_path, [_set_cell(2, "=SUM(A1:A1)")])
    cell = sheet.cell(row=2, column=1)

    assert cell.value == "=SUM(A1:A1)"
    assert cell.data_type == "f"
    assert cell.quotePrefix is False


def test_as_text_suppresses_the_iso_datetime_coercion(tmp_path: Path) -> None:
    """An explicit ``as_text`` request must win over the ISO-8601 heuristic.

    ``_coerce`` only consulted the flag on the formula branch, so there was no
    way to store a timestamp as text: every 19-character string with ``T`` at
    index 10 became a ``datetime`` cell regardless.
    """
    sheet = _edit_and_reload(tmp_path, [_set_cell(2, "2026-05-06T09:00:00", as_text=True)])
    cell = sheet.cell(row=2, column=1)

    assert cell.value == "2026-05-06T09:00:00"
    assert isinstance(cell.value, str)
    assert cell.data_type == "s"


def test_without_as_text_an_iso_string_still_becomes_a_datetime(tmp_path: Path) -> None:
    """The documented datetime convenience keeps working untouched."""
    from datetime import datetime

    sheet = _edit_and_reload(tmp_path, [_set_cell(2, "2026-05-06T09:00:00")])
    cell = sheet.cell(row=2, column=1)

    assert cell.value == datetime(2026, 5, 6, 9, 0)
    assert cell.data_type == "d"


def test_as_text_does_not_flag_an_ordinary_string(tmp_path: Path) -> None:
    """``quotePrefix`` only stands in for the escape a formula string needs.

    A plain string must not pick up the flag, or every text cell written with
    ``as_text`` would carry a style Excel attributes to manual escaping.
    """
    sheet = _edit_and_reload(tmp_path, [_set_cell(2, "plain", as_text=True)])
    cell = sheet.cell(row=2, column=1)

    assert cell.value == "plain"
    assert cell.data_type == "s"
    assert cell.quotePrefix is False


def test_as_text_leaves_non_string_values_alone(tmp_path: Path) -> None:
    """Numbers and booleans keep their own cell types; the flag is about text."""
    sheet = _edit_and_reload(
        tmp_path, [_set_cell(2, 42, as_text=True), _set_cell(3, True, as_text=True)]
    )

    assert sheet.cell(row=2, column=1).value == 42
    assert sheet.cell(row=2, column=1).data_type == "n"
    assert sheet.cell(row=3, column=1).value is True
    assert sheet.cell(row=3, column=1).data_type == "b"
