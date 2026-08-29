"""Tests for LexicalIndex and its lifecycle/concurrency optimizations."""

from __future__ import annotations

import concurrent.futures

from agentos.skills.retrieval import HybridRetriever
from agentos.skills.retrieval.lexical import Hit, LexicalIndex
from agentos.skills.types import SkillLayer, SkillSpec


def _make_sample_skills() -> list[SkillSpec]:
    return [
        SkillSpec(
            name="weather_reporter",
            description="Fetches weather forecast and temperature for cities",
            layer=SkillLayer.BUNDLED,
            always=False,
            triggers=["weather", "forecast", "rain", "temperature"],
            content="Weather skill content",
        ),
        SkillSpec(
            name="stock_analyzer",
            description="Analyzes market stock prices and equities",
            layer=SkillLayer.BUNDLED,
            always=False,
            triggers=["stocks", "finance", "shares"],
            content="Stock analyzer content",
        ),
        SkillSpec(
            name="crypto_tracker",
            description="Tracks Bitcoin and Solana cryptocurrency prices",
            layer=SkillLayer.BUNDLED,
            always=False,
            triggers=["crypto", "btc", "solana"],
            content="Crypto tracker content",
        ),
    ]


def test_lexical_index_basic_ranking() -> None:
    skills = _make_sample_skills()
    index = LexicalIndex(skills)

    hits = index.rank("forecast in tokyo")
    assert len(hits) > 0
    assert hits[0].skill_id == "weather_reporter"
    assert isinstance(hits[0], Hit)
    assert hits[0].rank == 1

    hits_crypto = index.rank("solana price")
    assert len(hits_crypto) > 0
    assert hits_crypto[0].skill_id == "crypto_tracker"
    index.close()


def test_lexical_index_lazy_build_and_connection_reuse() -> None:
    skills = _make_sample_skills()
    index = LexicalIndex(skills)

    # Before rank(), no connection should be created
    assert index._built is False
    assert index._conn is None

    # First rank() builds the connection
    hits1 = index.rank("weather")
    assert hits1[0].skill_id == "weather_reporter"
    assert index._built is True
    assert index._conn is not None
    conn_ref = index._conn

    # Subsequent rank() reuses the exact same SQLite connection
    hits2 = index.rank("stocks")
    assert index._conn is conn_ref
    assert hits2[0].skill_id == "stock_analyzer"

    # Close tears down the connection
    index.close()
    assert index._conn is None
    assert index._built is False

    # Close is idempotent
    index.close()


def test_lexical_index_concurrency_thread_safety() -> None:
    skills = _make_sample_skills()
    index = LexicalIndex(skills)

    queries = [
        "weather report",
        "stock forecast",
        "bitcoin crypto",
        "temperature rain",
        "finance shares",
    ]

    def _query_task(q: str) -> list[Hit]:
        return index.rank(q)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_query_task, q) for q in queries * 5]
        results = [f.result() for f in futures]

    assert len(results) == 25
    for r in results:
        assert len(r) > 0

    index.close()


def test_lexical_index_handles_special_fts_characters() -> None:
    skills = _make_sample_skills()
    index = LexicalIndex(skills)

    # Queries containing FTS5 reserved characters / syntax
    problematic_queries = [
        '* " " *',
        "weather AND OR NOT NEAR",
        "weather*",
        "'''weather'''",
        "---",
        "()()()",
    ]

    for q in problematic_queries:
        # Must not raise sqlite3.OperationalError or crash
        hits = index.rank(q)
        assert isinstance(hits, list)

    index.close()


def test_hybrid_retriever_invalidate_closes_lexical_index() -> None:
    skills = _make_sample_skills()
    retriever = HybridRetriever(strategy="lexical")

    retriever.retrieve(skills, "weather")
    assert retriever._lexical is not None
    assert retriever._lexical._conn is not None

    lex_ref = retriever._lexical
    retriever.invalidate()

    assert retriever._lexical is None
    assert lex_ref._conn is None


def test_hybrid_retriever_reuses_lexical_index_across_turns() -> None:
    skills = _make_sample_skills()
    retriever = HybridRetriever(strategy="lexical")

    retriever.retrieve(skills, "weather")
    lex_instance_1 = retriever._lexical
    assert lex_instance_1 is not None
    conn_1 = lex_instance_1._conn

    retriever.retrieve(skills, "stocks")  # same skill set, next turn
    lex_instance_2 = retriever._lexical
    assert lex_instance_2 is not None
    conn_2 = lex_instance_2._conn

    assert lex_instance_1 is lex_instance_2
    assert conn_1 is conn_2


def test_hybrid_retriever_closes_old_index_on_skill_set_change() -> None:
    skills_v1 = _make_sample_skills()
    skills_v2 = skills_v1 + [
        SkillSpec(
            name="new_skill",
            description="A newly installed skill",
            layer=SkillLayer.BUNDLED,
            always=False,
            triggers=["new"],
            content="New skill content",
        ),
    ]
    retriever = HybridRetriever(strategy="lexical")

    retriever.retrieve(skills_v1, "weather")
    old_lex = retriever._lexical
    assert old_lex is not None
    assert old_lex._conn is not None

    retriever.retrieve(skills_v2, "new")  # skill set changed

    assert retriever._lexical is not old_lex
    assert old_lex._conn is None
