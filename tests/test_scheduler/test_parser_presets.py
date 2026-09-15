"""``@preset`` aliases resolve whatever case they are written in.

``_parse_field`` already folds case for month and day-of-week names, so
``0 9 * * MON-FRI`` and ``0 9 * * Mon-Fri`` both parse. The preset lookup was
the one cron spelling left case-sensitive: ``@DAILY`` raised
``CronParseError: Unknown preset '@DAILY'`` instead of resolving, and did so at
every layer that stores or fires a schedule, not only at parse time.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentos.scheduler.jobs import _next_run
from agentos.scheduler.parser import _PRESETS, CronParseError, parse_cron
from agentos.scheduler.schedule_normalizer import coerce_schedule
from agentos.scheduler.types import CronJob, ScheduleKind, SessionTarget


def _cron_job(expr: str) -> CronJob:
    return CronJob(
        id="job-1",
        cron_expr=expr,
        handler_key="agent_run",
        payload={"kind": "agent_turn", "task": "x", "agent_id": "main"},
        session_target=SessionTarget.ISOLATED,
        schedule_kind=ScheduleKind.CRON,
    )


@pytest.mark.parametrize("alias", sorted(_PRESETS))
@pytest.mark.parametrize("spell", [str.upper, str.title, lambda s: s])
def test_every_preset_resolves_in_every_case(alias: str, spell: object) -> None:
    """All seven aliases, not just the one the report happened to use."""
    expected = parse_cron(alias)
    written = spell(alias)  # type: ignore[operator]

    parsed = parse_cron(written)

    assert parsed.raw == expected.raw
    # Field-for-field, not just the expanded string: the preset has to schedule
    # the same minutes, not merely look the same.
    assert parsed.minute.values == expected.minute.values
    assert parsed.hour.values == expected.hour.values
    assert parsed.day_of_month.values == expected.day_of_month.values
    assert parsed.month.values == expected.month.values
    assert parsed.day_of_week.values == expected.day_of_week.values


def test_an_upper_case_preset_expands_to_the_standard_expression() -> None:
    assert parse_cron("@DAILY").raw == "0 0 * * *"
    assert parse_cron("@Weekly").raw == "0 0 * * 0"
    assert parse_cron("@HOURLY").raw == "0 * * * *"


def test_a_stored_upper_case_preset_still_fires() -> None:
    """The layer that matters to an operator: a saved job computing its next run.

    ``normalize_schedule`` stores the expression as the caller wrote it, so the
    uppercase spelling is what ``_next_run`` re-parses on every tick.
    """
    after = datetime(2026, 5, 15, 12, 30, tzinfo=UTC)

    assert _next_run(_cron_job("@DAILY"), after) == datetime(2026, 5, 16, 0, 0, tzinfo=UTC)
    assert _next_run(_cron_job("@HOURLY"), after) == datetime(2026, 5, 15, 13, 0, tzinfo=UTC)


def test_a_cron_schedule_accepts_an_upper_case_preset() -> None:
    """The validation gate every ``cron.add`` passes through."""
    kind, expr, _tz = coerce_schedule({"kind": "cron", "expr": "@DAILY"})

    assert kind == ScheduleKind.CRON
    assert expr == "@DAILY"


def test_an_unknown_preset_is_quoted_back_as_it_was_typed() -> None:
    """Case folding is for the lookup only; the message keeps the caller's text."""
    with pytest.raises(CronParseError, match=r"Unknown preset '@Fortnightly'"):
        parse_cron("@Fortnightly")


def test_an_unknown_preset_is_still_rejected() -> None:
    """Passes either way by design: folding case must not widen what is accepted."""
    for expr in ("@REBOOT", "@", "@daily daily"):
        with pytest.raises(CronParseError):
            parse_cron(expr)


def test_a_five_field_expression_is_untouched() -> None:
    """Passes either way by design: only an ``@`` expression takes the preset path."""
    assert parse_cron("0 9 * * MON-FRI").day_of_week.values == frozenset({1, 2, 3, 4, 5})
    assert parse_cron("*/15 * * * *").minute.values == frozenset(range(0, 60, 15))
