from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentos.engine.pipeline import TurnContext
from agentos.engine.steps.skills_filter import filter_skills
from agentos.gateway.config import GatewayConfig
from agentos.skills.injector import (
    RENDER_MODE_COMPACT,
    RENDER_MODE_COMPACT_TRUNCATED,
    RENDER_MODE_FULL,
    RENDER_MODE_FULL_TRUNCATED,
    SkillInjector,
)
from agentos.skills.loader import SkillLoader
from agentos.skills.types import SkillLayer, SkillSpec

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "src" / "agentos" / "skills" / "bundled"


def _skill(
    name: str,
    *,
    layer: SkillLayer = SkillLayer.BUNDLED,
    description: str = "Use when exercising the injector.",
    disable_model_invocation: bool = False,
) -> SkillSpec:
    """A spec carrying the on-disk fields the injector used to leak."""
    base_dir = f"/opt/example/agentos/skills/{name}"
    return SkillSpec(
        name=name,
        description=description,
        layer=layer,
        always=False,
        triggers=[],
        content=f"# {name}",
        path=Path(base_dir),
        base_dir=base_dir,
        file_path=f"{base_dir}/SKILL.md",
        disable_model_invocation=disable_model_invocation,
    )


def test_neither_mode_emits_a_skill_location() -> None:
    skills = [_skill("alpha"), _skill("beta")]
    injector = SkillInjector()

    full = injector.inject_full("", skills)
    compact = injector.inject_compact("", skills)

    for prompt in (full, compact):
        assert "<location>" not in prompt
        assert "SKILL.md" not in prompt
        assert "/opt/example" not in prompt


def test_full_mode_is_used_when_it_fits_the_budget() -> None:
    skills = [_skill("alpha", description="Use when alpha things happen.")]

    block = SkillInjector().render("", skills, max_chars=10_000)
    prompt, dropped = block.text, block.dropped

    assert "<description>Use when alpha things happen.</description>" in prompt
    assert "<name>alpha</name>" in prompt
    assert dropped == []


def test_compact_mode_takes_over_when_descriptions_do_not_fit() -> None:
    skills = [_skill(f"skill-{i}", description="d" * 400) for i in range(10)]

    block = SkillInjector().render("", skills, max_chars=1_000)
    prompt, dropped = block.text, block.dropped

    assert "<description>" not in prompt
    for i in range(10):
        assert f"<name>skill-{i}</name>" in prompt
    assert dropped == []


def test_truncation_sacrifices_the_lowest_precedence_layers_first() -> None:
    keep = [
        _skill("workspace-0", layer=SkillLayer.WORKSPACE),
        *(_skill(f"managed-{i}", layer=SkillLayer.MANAGED) for i in range(3)),
    ]
    skills = [*(_skill(f"bundled-{i}") for i in range(20)), *keep]
    injector = SkillInjector()
    # Budget derived from a real render, not a magic number: the block's guidance
    # text is prose and does change, and a hardcoded budget silently turns this
    # into a different test (or a passing one that proves less) when it does.
    budget = len(injector.inject_compact("", keep))

    block = injector.render("", skills, max_chars=budget)
    prompt, dropped = block.text, block.dropped

    for spec in keep:
        assert f"<name>{spec.name}</name>" in prompt
    assert dropped
    assert all(name.startswith("bundled-") for name in dropped)


def test_dropped_names_match_what_is_missing_from_the_prompt() -> None:
    skills = [
        *(_skill(f"bundled-{i}") for i in range(30)),
        *(_skill(f"managed-{i}", layer=SkillLayer.MANAGED) for i in range(5)),
    ]

    block = SkillInjector().render("", skills, max_chars=700)
    prompt, dropped = block.text, block.dropped

    missing = {s.name for s in skills if f"<name>{s.name}</name>" not in prompt}
    # Without this the assertion below passes vacuously (empty == empty) the day
    # a wider budget stops truncation firing at all.
    assert dropped
    assert set(dropped) == missing
    assert len(dropped) == len(set(dropped))


def test_truncation_preserves_within_layer_order() -> None:
    skills = [_skill(f"bundled-{i:02d}") for i in range(30)]

    block = SkillInjector().render("", skills, max_chars=500)
    prompt, dropped = block.text, block.dropped

    kept = [s.name for s in skills if f"<name>{s.name}</name>" in prompt]
    assert kept == sorted(kept)
    assert dropped == [s.name for s in skills if s.name not in kept]


def test_model_invisible_skills_are_never_reported_as_dropped() -> None:
    skills = [
        _skill("hidden", disable_model_invocation=True),
        *(_skill(f"bundled-{i}") for i in range(20)),
    ]

    block = SkillInjector().render("", skills, max_chars=400)
    prompt, dropped = block.text, block.dropped

    assert "<name>hidden</name>" not in prompt
    assert "hidden" not in dropped


def test_a_tiny_budget_still_keeps_one_skill_and_the_guard() -> None:
    skills = [_skill(f"bundled-{i}") for i in range(5)]

    block = SkillInjector().render("", skills, max_chars=1)
    prompt, dropped = block.text, block.dropped

    assert "they are not callable tools" in prompt
    assert prompt.count("<name>") == 1
    assert len(dropped) == 4


def test_the_budget_costs_description_text_before_it_costs_descriptions() -> None:
    """One skill over budget used to drop every description in the block.

    Full mode overran by 3k of a 24k budget and the whole block fell to
    name-only — 3.5k rendered against a 24k allowance, so 20k of the operator's
    budget bought nothing and the model never saw a single description.
    """
    skills = [_skill(f"skill-{i}", description="word " * 100) for i in range(20)]
    injector = SkillInjector()
    untrimmed = injector.inject_full("", skills)
    # A budget just under what untrimmed descriptions need.
    budget = len(untrimmed) - 500

    block = injector.render("", skills, max_chars=budget)

    assert block.mode == RENDER_MODE_FULL_TRUNCATED
    assert block.description_max_chars is not None
    assert block.dropped == []
    # Every skill still listed, and still with description text to judge on.
    for i in range(20):
        assert f"<name>skill-{i}</name>" in block.text
    assert block.text.count("<description>") == 20
    assert len(block.text) <= budget
    # And the shortening is disclosed, so a clipped sentence is not read as the
    # whole description.
    assert "shortened to fit" in block.text


def test_the_widest_description_cap_that_fits_is_the_one_used() -> None:
    """The budget is what the operator agreed to spend, so spend it.

    A fixed ladder of caps settles for the first rung that fits — 320 chars
    where 451 also fitted, on a real install — and every skill between those two
    lengths is clipped for nothing.
    """
    skills = [
        _skill(f"skill-{i}", description="Use when " + "word " * (20 + i * 4)) for i in range(25)
    ]
    injector = SkillInjector()
    budget = len(injector.inject_full("", skills)) - 2_000

    block = injector.render("", skills, max_chars=budget)
    cap = block.description_max_chars

    assert block.mode == RENDER_MODE_FULL_TRUNCATED
    assert cap is not None
    assert len(block.text) <= budget
    # Nothing wider fits: the cap is the ceiling, not the first rung under it.
    assert len(injector.inject_full("", skills, description_max_chars=cap + 1)) > budget


def test_a_cap_that_would_clip_nothing_is_never_chosen() -> None:
    """Such a cap is the untrimmed block plus a claim that is not true of it.

    It also renders *larger* than untrimmed — the disclosure costs ~150 chars
    and saves none — so choosing one both misinforms the model and wastes budget.
    """
    skills = [_skill(f"skill-{i}", description="Use when " + "word " * 80) for i in range(20)]
    injector = SkillInjector()

    for budget in range(len(injector.inject_full("", skills)) - 1, 0, -900):
        block = injector.render("", skills, max_chars=budget)
        if block.mode != RENDER_MODE_FULL_TRUNCATED:
            continue
        # Shortening was claimed, so shortening must have happened.
        assert "shortened to fit" in block.text
        assert "…" in block.text
        assert len(block.text) <= budget


def test_a_render_that_fits_untrimmed_says_nothing_about_shortening() -> None:
    skills = [_skill("alpha", description="Use when alpha things happen.")]

    block = SkillInjector().render("", skills, max_chars=10_000)

    assert block.mode == RENDER_MODE_FULL
    assert block.description_max_chars is None
    assert block.degraded is False
    assert "shortened to fit" not in block.text


def test_compact_mode_points_at_skill_list_when_the_session_has_it() -> None:
    """Name-only mode has to name a route back to the descriptions.

    Asking for a skill_view per plausible name is not a route the model will
    take once the list is dozens of entries long, and skill_list answers the
    same question in one call — but the block never mentioned it existed.
    """
    skills = [_skill(f"skill-{i}") for i in range(40)]
    injector = SkillInjector()

    with_list = injector.inject_compact("", skills, skill_list_tool=True)
    without_list = injector.inject_compact("", skills, skill_list_tool=False)

    assert "`skill_list`" in with_list
    # Never advertise a tool the session does not expose.
    assert "skill_list" not in without_list
    assert "skill_view" in without_list


def test_render_modes_are_reported_for_every_degradation_step() -> None:
    skills = [_skill(f"skill-{i}", description="word " * 100) for i in range(30)]
    injector = SkillInjector()
    # Budgets derived from real renders rather than hardcoded: the block's
    # guidance is prose and changes, and a magic number quietly stops testing
    # the step it was chosen for when it does.
    untrimmed = len(injector.inject_full("", skills))
    narrowest = len(injector.inject_full("", skills, description_max_chars=120))
    names_only = len(injector.inject_compact("", skills))

    assert injector.render("", skills, max_chars=untrimmed).mode == RENDER_MODE_FULL
    assert injector.render("", skills, max_chars=untrimmed - 1).mode == RENDER_MODE_FULL_TRUNCATED
    assert injector.render("", skills, max_chars=narrowest - 1).mode == RENDER_MODE_COMPACT
    truncated = injector.render("", skills, max_chars=names_only - 1)
    assert truncated.mode == RENDER_MODE_COMPACT_TRUNCATED
    assert truncated.dropped


def _injected(ctx: TurnContext) -> str:
    prompt = ctx.system_prompt
    return prompt if isinstance(prompt, str) else "\n\n".join(prompt)


def _ctx(loader: SkillLoader, skills_config: object) -> TurnContext:
    tool_defs = [
        SimpleNamespace(name=name)
        for name in (
            "background_process",
            "cron",
            "exec_command",
            "memory_get",
            "memory_save",
            "memory_search",
            "process",
        )
    ]
    return TurnContext(
        message="please help with anything",
        session_key="agent:main:webchat:default",
        config=SimpleNamespace(
            tools=SimpleNamespace(profile="standard"),
            skills=skills_config,
        ),
        provider=None,
        model="test-model",
        tool_defs=tool_defs,
        system_prompt="base",
        metadata={"skill_loader": loader},
    )


@pytest.mark.asyncio
async def test_shipped_default_budget_keeps_every_managed_skill(tmp_path: Path) -> None:
    """The shipped default must fit the bundled set plus installed skills.

    The old 8000-char default could not hold the bundled set even in compact
    mode, and truncation kept a bundled-first prefix — so the skills an
    operator installed were the first thing dropped, silently.
    """
    managed = tmp_path / "managed"
    installed = [f"community-skill-{i}" for i in range(6)]
    for name in installed:
        skill_dir = managed / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\n"
            "description: Use when testing installed skills.\n"
            f"---\n\n# {name}\n",
            encoding="utf-8",
        )
    loader = SkillLoader(
        bundled_dir=BUNDLED,
        managed_dir=managed,
        snapshot_path=tmp_path / "snapshot.json",
    )

    ctx = await filter_skills(_ctx(loader, GatewayConfig().skills))

    prompt = _injected(ctx)
    for name in installed:
        assert f"<name>{name}</name>" in prompt
    assert ctx.metadata["skills_dropped_for_budget"] == []
    # Descriptions are the whole point of the budget: without them the model
    # cannot tell which skill matches the request.
    assert "<description>" in prompt
    assert ctx.metadata["skills_render_mode"] == RENDER_MODE_FULL


@pytest.mark.asyncio
async def test_budget_truncation_is_reported_in_metadata(tmp_path: Path) -> None:
    managed = tmp_path / "managed"
    skill_dir = managed / "community-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: community-skill\ndescription: Use when testing installed skills.\n---\n\n# c\n",
        encoding="utf-8",
    )
    loader = SkillLoader(
        bundled_dir=BUNDLED,
        managed_dir=managed,
        snapshot_path=tmp_path / "snapshot.json",
    )
    skills_config = SimpleNamespace(
        filter_enabled=False,
        max_skills_prompt_chars=300,
        injection_mode="system",
    )

    ctx = await filter_skills(_ctx(loader, skills_config))

    prompt = _injected(ctx)
    dropped = ctx.metadata["skills_dropped_for_budget"]
    assert dropped
    assert "community-skill" not in dropped
    assert "<name>community-skill</name>" in prompt
    assert ctx.metadata["skill_count"] == prompt.count("<name>")
    assert all(name not in ctx.metadata["filtered_skill_ids"] for name in dropped)


@pytest.mark.asyncio
async def test_a_stable_skill_list_lands_in_the_cached_system_prompt(tmp_path: Path) -> None:
    """The block is instructions, so it has to stay in the system message.

    It used to go into the dynamic suffix, which an enabled prompt cache splits
    off and delivers as a *user* message headed "not a user request ... use it
    only when it is relevant" — contradicting the block's own "read this before
    answering", from the weakest position in the request. With relevance
    filtering off the list is identical on every turn, so the cacheable half is
    also where it belongs on cost.
    """
    loader = SkillLoader(bundled_dir=BUNDLED, snapshot_path=tmp_path / "snapshot.json")
    skills_config = SimpleNamespace(
        filter_enabled=False,
        max_skills_prompt_chars=100_000,
        injection_mode="system",
    )

    ctx = await filter_skills(_ctx(loader, skills_config))

    # str, not (base, dynamic): nothing for the cache split to peel off.
    assert isinstance(ctx.system_prompt, str)
    assert ctx.system_prompt.startswith("base")
    assert "<available_skills>" in ctx.system_prompt


@pytest.mark.asyncio
async def test_a_stable_list_joins_the_base_and_leaves_the_recall_suffix_alone(
    tmp_path: Path,
) -> None:
    """Appending to the base must not eat an upstream dynamic block."""
    loader = SkillLoader(bundled_dir=BUNDLED, snapshot_path=tmp_path / "snapshot.json")
    skills_config = SimpleNamespace(
        filter_enabled=False,
        max_skills_prompt_chars=100_000,
        injection_mode="system",
    )
    ctx = _ctx(loader, skills_config)
    ctx.system_prompt = ("base", "RECALLED MEMORY BLOCK")

    ctx = await filter_skills(ctx)

    assert isinstance(ctx.system_prompt, tuple)
    base, dynamic = ctx.system_prompt
    assert base.startswith("base")
    assert "<available_skills>" in base
    assert dynamic == "RECALLED MEMORY BLOCK"


@pytest.mark.asyncio
async def test_a_per_message_skill_list_stays_out_of_the_cached_prefix(tmp_path: Path) -> None:
    """Relevance filtering re-picks the list per message, so caching it churns."""
    workspace = tmp_path / "workspace"
    for name in ("weather-local", "github-local"):
        skill_dir = workspace / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Use when testing {name}.\n---\n\n# {name}\n",
            encoding="utf-8",
        )
    loader = SkillLoader(workspace_dir=workspace, snapshot_path=tmp_path / "snapshot.json")
    skills_config = SimpleNamespace(
        filter_enabled=True,
        filter_top_k=1,
        filter_strategy="lexical",
        filter_lexical_top_n=20,
        filter_semantic_top_n=20,
        filter_rrf_k=60,
        filter_embedding_model="BAAI/bge-small-zh-v1.5",
        max_skills_prompt_chars=100_000,
        injection_mode="system",
    )

    ctx = _ctx(loader, skills_config)
    # Retrieval ranks against the message, so it needs one that actually matches.
    ctx.message = "please check the weather forecast"

    ctx = await filter_skills(ctx)

    assert ctx.metadata["skill_count"] == 1
    assert isinstance(ctx.system_prompt, tuple)
    assert ctx.system_prompt[0] == "base"
    assert "<available_skills>" in ctx.system_prompt[1]


def test_injector_escapes_xml_special_characters() -> None:
    """Quotes and XML tags in skill names/descriptions must be entity-escaped."""
    skills = [
        _skill(
            'bad<name>&"quoted"\'skill\'',
            description='Test "quotes", \'single\', & <tags> with </available_skills>',
        )
    ]
    injector = SkillInjector()

    full = injector.inject_full("", skills)
    assert "<name>bad&lt;name&gt;&amp;&quot;quoted&quot;&apos;skill&apos;</name>" in full
    assert (
        "<description>Test &quot;quotes&quot;, &apos;single&apos;, &amp; &lt;tags&gt; "
        "with &lt;/available_skills&gt;</description>"
    ) in full
    assert "</available_skills><available_skills>" not in full

    compact = injector.inject_compact("", skills)
    assert "<name>bad&lt;name&gt;&amp;&quot;quoted&quot;&apos;skill&apos;</name>" in compact

