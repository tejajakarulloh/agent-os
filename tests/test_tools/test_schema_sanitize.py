from __future__ import annotations

from agentos.tools.schema_sanitize import sanitize_input_schema


def test_clean_schema_passes_through_untouched() -> None:
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "A path."}},
        "required": ["path"],
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert fixes == []
    assert cleaned == schema


def test_object_without_properties_gains_an_empty_one() -> None:
    # llama.cpp's grammar converter fails the whole request on this shape.
    cleaned, fixes = sanitize_input_schema({"type": "object"})

    assert cleaned == {"type": "object", "properties": {}}
    assert "added_missing_properties" in fixes


def test_nullable_union_collapses_to_the_concrete_branch() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "description": "How many.",
            }
        },
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["properties"]["limit"] == {"type": "integer", "description": "How many."}
    assert "collapsed_nullable_union" in fixes


def test_nullable_union_collapses_when_null_branch_uses_a_type_list() -> None:
    schema = {
        "type": "object",
        "properties": {
            "filter": {"anyOf": [{"type": "string"}, {"type": ["null"]}]},
        },
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["properties"]["filter"] == {"type": "string"}
    assert "collapsed_nullable_union" in fixes


def test_nullable_union_collapses_when_null_branch_has_type_none() -> None:
    schema = {
        "type": "object",
        "properties": {
            "filter": {"anyOf": [{"type": "string"}, {"type": None}]},
        },
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["properties"]["filter"] == {"type": "string"}
    assert "collapsed_nullable_union" in fixes


def test_nullable_union_collapses_when_null_branch_is_a_const_null() -> None:
    schema = {
        "type": "object",
        "properties": {
            "filter": {"anyOf": [{"type": "string"}, {"const": None}]},
        },
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["properties"]["filter"] == {"type": "string"}
    assert "collapsed_nullable_union" in fixes


def test_nullable_union_collapses_when_null_branch_is_an_enum_of_null() -> None:
    schema = {
        "type": "object",
        "properties": {
            "filter": {"anyOf": [{"type": "string"}, {"enum": [None]}]},
        },
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["properties"]["filter"] == {"type": "string"}
    assert "collapsed_nullable_union" in fixes


def test_union_with_two_real_branches_is_left_alone() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "integer"}, {"type": "string"}]}},
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["properties"]["value"]["anyOf"] == [{"type": "integer"}, {"type": "string"}]
    assert "collapsed_nullable_union" not in fixes


def test_type_array_collapses_to_the_non_null_entry() -> None:
    schema = {"type": "object", "properties": {"name": {"type": ["string", "null"]}}}

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["properties"]["name"]["type"] == "string"
    assert "collapsed_type_array" in fixes


def test_local_ref_is_inlined_and_defs_are_dropped() -> None:
    schema = {
        "type": "object",
        "properties": {"point": {"$ref": "#/$defs/Point"}},
        "$defs": {
            "Point": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["properties"]["point"] == {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
    }
    assert "$defs" not in cleaned
    assert "inlined_ref" in fixes


def test_ref_siblings_survive_inlining() -> None:
    schema = {
        "type": "object",
        "properties": {
            "point": {"$ref": "#/$defs/Point", "description": "Where."},
        },
        "$defs": {"Point": {"type": "object", "properties": {}}},
    }

    cleaned, _ = sanitize_input_schema(schema)

    assert cleaned["properties"]["point"]["description"] == "Where."


def test_dangling_ref_is_dropped_without_failing_the_tool() -> None:
    schema = {"type": "object", "properties": {"thing": {"$ref": "#/$defs/Missing"}}}

    cleaned, fixes = sanitize_input_schema(schema)

    assert "$ref" not in cleaned["properties"]["thing"]
    assert "dropped_dangling_ref" in fixes


def test_self_referential_defs_terminate() -> None:
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert "cycle_truncated" in fixes
    assert cleaned["properties"]["node"]["type"] == "object"


def test_bare_string_property_value_is_dropped() -> None:
    schema = {"type": "object", "properties": {"broken": "object", "ok": {"type": "string"}}}

    cleaned, fixes = sanitize_input_schema(schema)

    assert "broken" not in cleaned["properties"]
    assert cleaned["properties"]["ok"] == {"type": "string"}
    assert "dropped_non_schema_property" in fixes


def test_required_entries_for_dropped_properties_are_pruned() -> None:
    schema = {
        "type": "object",
        "properties": {"broken": "object", "ok": {"type": "string"}},
        "required": ["broken", "ok"],
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["required"] == ["ok"]
    assert "pruned_required" in fixes


def test_non_object_root_is_forced_to_an_object() -> None:
    cleaned, fixes = sanitize_input_schema({"type": "string"})

    assert cleaned["type"] == "object"
    assert cleaned["properties"] == {}
    assert "forced_object_root" in fixes


def test_non_mapping_schema_is_replaced() -> None:
    cleaned, fixes = sanitize_input_schema("object")

    assert cleaned == {"type": "object", "properties": {}}
    assert fixes == ["replaced_non_object_schema"]


def test_additional_properties_false_is_preserved() -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": False}

    cleaned, _ = sanitize_input_schema(schema)

    assert cleaned["additionalProperties"] is False


def test_nested_properties_are_sanitized_too() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": ["integer", "null"]}},
            }
        },
    }

    cleaned, fixes = sanitize_input_schema(schema)

    assert cleaned["properties"]["outer"]["properties"]["inner"]["type"] == "integer"
    assert "collapsed_type_array" in fixes


def test_array_items_are_sanitized() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": ["string", "null"]}},
        },
    }

    cleaned, _ = sanitize_input_schema(schema)

    assert cleaned["properties"]["tags"]["items"]["type"] == "string"


def test_sanitizing_twice_changes_nothing_further() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "point": {"$ref": "#/$defs/Point"},
            "broken": "object",
        },
        "required": ["limit", "broken"],
        "$defs": {"Point": {"type": "object"}},
    }

    once, first_fixes = sanitize_input_schema(schema)
    twice, second_fixes = sanitize_input_schema(once)

    assert once == twice
    assert first_fixes != []
    assert second_fixes == []
