"""Regression tests for the FTS5 query sanitizer and the search_transcript path.

The sanitizer used to be an ASCII-only whitelist, which turned every
non-ASCII letter into a space and silently emptied queries in every
non-English locale (CJK, Cyrillic, Vietnamese, accented Latin, Arabic,
…). The FTS5 ``unicode61`` tokenizer handles the data side correctly, so
the bug is a pure query-side defect: a query that is well-formed Unicode
arrives at FTS5 as ``""`` (or a partial / wrong token), and
``search_transcript`` short-circuits to ``[]`` on the empty-guard.

These tests pin down the contract at two layers:

  * Pure unit tests on ``SessionStorage.sanitize_fts_query`` covering the
    individual behaviours (Unicode passes through, FTS5 syntax stripped,
    empty-guard preserved, 20-token cap preserved).
  * Integration tests against a real in-memory FTS5 index — the
    sanitizer output is an implementation detail, the hit count is the
    contract. Each integration case indexes a transcript in a different
    script and asserts the matching CJK / accented / Cyrillic / Vietnamese
    query actually returns the row.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage

# ── Unit tests on the sanitizer (string-level contract) ──────────────────────


class TestSanitizeFtsQueryUnit:
    """String-level assertions for ``SessionStorage.sanitize_fts_query``.

    These guard the implementation details: which characters survive,
    which are stripped, and the empty-query / token-cap invariants. The
    integration tests below then verify the contract (hit count) on a
    real FTS5 index.
    """

    def test_ascii_queries_pass_through_literally(self) -> None:
        result = SessionStorage.sanitize_fts_query("quantum widget")
        assert result == '"quantum" "widget"'

    def test_accented_latin_passes_through_intact(self) -> None:
        """``café déploiement`` must NOT be truncated to ``caf déploiement``
        — the previous ASCII whitelist produced ``"caf" "d" "ploiement"``
        which could match unrelated content."""
        result = SessionStorage.sanitize_fts_query("café déploiement")
        assert result == '"café" "déploiement"'

    def test_cjk_queries_pass_through_intact(self) -> None:
        """Chinese / Japanese / Korean characters must survive verbatim.
        The previous implementation collapsed these to an empty query."""
        result = SessionStorage.sanitize_fts_query("中文 报告")
        assert result == '"中文" "报告"'

    def test_cyrillic_queries_pass_through_intact(self) -> None:
        result = SessionStorage.sanitize_fts_query("отчёт готов")
        assert result == '"отчёт" "готов"'

    def test_vietnamese_with_diacritics_passes_through_intact(self) -> None:
        """The previous behaviour split ``triển khai`` into
        ``"tri" "n" "khai"`` — the ``ể`` broke the token and the two halves
        each matched unrelated content."""
        result = SessionStorage.sanitize_fts_query("triển khai")
        assert result == '"triển" "khai"'

    def test_arabic_queries_pass_through_intact(self) -> None:
        result = SessionStorage.sanitize_fts_query("مرحبا بالعالم")
        assert result == '"مرحبا" "بالعالم"'

    def test_mixed_script_queries_pass_through_intact(self) -> None:
        result = SessionStorage.sanitize_fts_query("hello café 中文 отчёт")
        assert result == '"hello" "café" "中文" "отчёт"'

    def test_fts5_syntax_characters_are_stripped(self) -> None:
        """``"``, ``(``, ``)``, ``*``, ``^`` and ``:`` are FTS5 operators
        and must NOT survive into the MATCH expression. The sanitizer
        replaces them with spaces so the surrounding words do not fuse."""
        for raw, expected in (
            ('"hello"', '"hello"'),
            ("hello*world", '"hello" "world"'),
            ("(foo OR bar)", '"foo" "OR" "bar"'),
            ("^anchor", '"anchor"'),
            ("field:value", '"field" "value"'),
        ):
            assert SessionStorage.sanitize_fts_query(raw) == expected

    def test_control_and_format_characters_are_stripped(self) -> None:
        """Control codepoints (Cc), format codepoints (Cf), and other
        ``\\p{C}`` categories must not survive — a query built from a
        string with stray formatting (zero-width joiners, RTL marks,
        NULL bytes from a corrupted log) must not inject MATCH syntax."""
        # Zero-width joiner (U+200D, Cf), right-to-left mark (U+200F, Cf),
        # NULL (U+0000, Cc), and a stray bell (U+0007, Cc).
        raw = "hello\u200dworld\u200ftest\u0000cafe\u0007baz"
        result = SessionStorage.sanitize_fts_query(raw)
        # All four control/format codepoints become spaces; words fuse
        # only where the joiner is removed.
        assert "\u200d" not in result
        assert "\u200f" not in result
        assert "\x00" not in result
        assert "\x07" not in result

    def test_punctuation_only_query_returns_empty_guard(self) -> None:
        """A query that is pure FTS5 syntax / control characters must
        short-circuit to the ``""`` empty-guard, not issue a MATCH for
        nothing. ``search_transcript`` relies on this to return ``[]``."""
        assert SessionStorage.sanitize_fts_query('"*^:()') == '""'
        assert SessionStorage.sanitize_fts_query("") == '""'
        assert SessionStorage.sanitize_fts_query("   ") == '""'

    def test_unicode_punctuation_passes_through_as_literal(self) -> None:
        """FTS5's ``unicode61`` tokenizer treats non-ASCII punctuation as
        a literal codepoint, not as MATCH syntax — so the sanitizer
        leaves it in place and the surviving token is quoted as a
        literal. This pins the contract that we are *not* over-stripping:
        a query like ``「量子」`` (Japanese full-width brackets) must
        still find a row whose transcript contains those brackets,
        rather than silently returning ``[]`` via the empty-guard."""
        # Full-width colon (U+FF1A), em dash (U+2014), bullet (U+2022),
        # full-width brackets (U+300C, U+300D) — all literal in FTS5.
        result = SessionStorage.sanitize_fts_query("：—•「量子」")
        assert result == '"：—•「量子」"'

    def test_token_cap_is_preserved(self) -> None:
        """The 20-token cap is part of the documented safety contract and
        must not regress when the sanitizer is changed to a blocklist."""
        raw = " ".join(f"word{i}" for i in range(30))
        result = SessionStorage.sanitize_fts_query(raw)
        assert len(result.split()) == 20

    def test_cap_counts_tokens_not_raw_words(self) -> None:
        """The cap is on the *post-sanitization* token count, so a long
        string of syntax characters that collapse to nothing must not
        waste the cap."""
        raw = "a" + "()" * 50 + "b"
        result = SessionStorage.sanitize_fts_query(raw)
        # The "()" runs become spaces; the only surviving tokens are
        # "a" and "b".
        assert result == '"a" "b"'


# ── Integration tests against a real in-memory FTS5 index ──────────────────


@pytest_asyncio.fixture
async def storage() -> AsyncGenerator[SessionStorage, None]:
    store = SessionStorage(":memory:")
    await store.connect()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def manager(storage: SessionStorage) -> SessionManager:
    return SessionManager(storage, inject_time_prefix=False)


async def _seed(manager: SessionManager, session_key: str, content: str) -> None:
    await manager.create(session_key, agent_id="main")
    await manager.append_message(session_key, role="user", content=content)


class TestSearchTranscriptUnicode:
    """End-to-end coverage on a real in-memory FTS5 index.

    The sanitizer string is an implementation detail; the *contract* is
    that a query in any script finds the row whose transcript contains
    that script. Each case below indexes one row in a different script
    and asserts the script-specific query returns exactly that row.
    """

    @pytest.mark.asyncio
    async def test_accented_latin_query_finds_accented_row(
        self, manager: SessionManager, storage: SessionStorage
    ) -> None:
        await _seed(manager, "agent:main:webchat:cafe0001", "Le café est ouvert.")
        await _seed(manager, "agent:main:webchat:cafe0002", "Nothing relevant here.")

        hits = await storage.search_transcript("café")
        assert [hit["session_key"] for hit in hits] == ["agent:main:webchat:cafe0001"]

    @pytest.mark.asyncio
    async def test_accented_query_does_not_match_ascii_fallback(
        self, manager: SessionManager, storage: SessionStorage
    ) -> None:
        """Regression for the wrong-match symptom called out in the
        issue: ``café`` previously sanitized to ``"caf"`` and would
        match a row containing only the ASCII prefix ``caf`` — i.e.
        it returned a *wrong* row instead of zero."""
        await _seed(manager, "agent:main:webchat:cafe0010", "Buy a cafeteria ticket.")
        await _seed(manager, "agent:main:webchat:cafe0011", "Le café est ouvert.")

        hits = await storage.search_transcript("café")
        assert [hit["session_key"] for hit in hits] == ["agent:main:webchat:cafe0011"]

    @pytest.mark.asyncio
    async def test_cjk_query_finds_cjk_row(
        self, manager: SessionManager, storage: SessionStorage
    ) -> None:
        await _seed(manager, "agent:main:webchat:cjk00001", "中文 报告 已发布")
        await _seed(manager, "agent:main:webchat:cjk00002", "Plain English row.")

        hits = await storage.search_transcript("中文 报告")
        assert [hit["session_key"] for hit in hits] == ["agent:main:webchat:cjk00001"]

    @pytest.mark.asyncio
    async def test_cyrillic_query_finds_cyrillic_row(
        self, manager: SessionManager, storage: SessionStorage
    ) -> None:
        await _seed(manager, "agent:main:webchat:cyr00001", "Отчёт готов к отправке.")
        await _seed(manager, "agent:main:webchat:cyr00002", "English placeholder.")

        hits = await storage.search_transcript("отчёт")
        assert [hit["session_key"] for hit in hits] == ["agent:main:webchat:cyr00001"]

    @pytest.mark.asyncio
    async def test_vietnamese_query_finds_vietnamese_row(
        self, manager: SessionManager, storage: SessionStorage
    ) -> None:
        await _seed(manager, "agent:main:webchat:vie00001", "Triển khai dự án mới.")
        await _seed(manager, "agent:main:webchat:vie00002", "Decoy transcript.")

        hits = await storage.search_transcript("triển khai")
        assert [hit["session_key"] for hit in hits] == ["agent:main:webchat:vie00001"]

    @pytest.mark.asyncio
    async def test_arabic_query_finds_arabic_row(
        self, manager: SessionManager, storage: SessionStorage
    ) -> None:
        await _seed(manager, "agent:main:webchat:ara00001", "مرحبا بالعالم الجديد")
        await _seed(manager, "agent:main:webchat:ara00002", "English decoy row.")

        hits = await storage.search_transcript("مرحبا")
        assert [hit["session_key"] for hit in hits] == ["agent:main:webchat:ara00001"]

    @pytest.mark.asyncio
    async def test_fts5_syntax_in_query_does_not_break_match(
        self, manager: SessionManager, storage: SessionStorage
    ) -> None:
        """A user who pastes a raw FTS5 query (e.g. ``"quantum"*`` from
        a snippet of a manual) must still get the matching row, not an
        empty result or a parse error. The sanitizer strips the syntax
        characters and the surviving token still matches."""
        await _seed(manager, "agent:main:webchat:fts00001", "quantum widget alpha")
        await _seed(manager, "agent:main:webchat:fts00002", "unrelated content")

        hits = await storage.search_transcript("quantum*")
        assert [hit["session_key"] for hit in hits] == ["agent:main:webchat:fts00001"]

    @pytest.mark.asyncio
    async def test_punctuation_only_query_returns_empty_list(
        self, manager: SessionManager, storage: SessionStorage
    ) -> None:
        """A query that is pure FTS5 syntax / control characters must
        short-circuit to ``[]`` via the ``""`` empty-guard, not raise."""
        hits = await storage.search_transcript('"*^:()')
        assert hits == []

    @pytest.mark.asyncio
    async def test_existing_ascii_behaviour_is_unchanged(
        self, manager: SessionManager, storage: SessionStorage
    ) -> None:
        """Regression guard: the blocklist must NOT change behaviour for
        pure-ASCII queries — the only safe way to change the sanitizer
        is to fix the Unicode bug without regressing the ASCII path."""
        await _seed(manager, "agent:main:webchat:asc00001", "quantum widget alpha")
        await _seed(manager, "agent:main:webchat:asc00002", "quantum widget beta")
        await _seed(manager, "agent:main:webchat:asc00003", "irrelevant")

        hits = await storage.search_transcript("quantum widget")
        assert {hit["session_key"] for hit in hits} == {
            "agent:main:webchat:asc00001",
            "agent:main:webchat:asc00002",
        }
