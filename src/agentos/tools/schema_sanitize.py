"""Normalize third-party JSON Schemas into the subset every backend accepts.

AgentOS's own tools declare their parameters through ``@tool(params=...)`` and
reach the provider as a typed :class:`~agentos.provider.types.ToolInputSchema`,
so they are well-formed by construction. MCP servers are not: their schemas
arrive as whatever the server chose to emit, and go out unchanged in the next
provider request. Shapes that OpenAI tolerates make other backends reject the
*entire* call, taking every other tool down with them:

* ``{"type": "object"}`` with no ``properties`` — llama.cpp's
  ``json-schema-to-grammar`` converter cannot build a parser for it and fails
  the request outright.
* ``anyOf``/``oneOf`` whose only purpose is to permit ``null`` — the shape
  Pydantic emits for every ``Optional[...]`` field. Anthropic rejects it at the
  top of ``input_schema``.
* ``"type": ["string", "null"]`` — many grammar converters accept only a
  single-string ``type``.
* ``$ref`` into ``$defs`` — routine Pydantic output, and unsupported by
  several providers, so it has to be inlined before the request goes out.
* A schema value that is a bare string instead of an object, from a server
  that serialized its own types wrong.

Sanitizing happens once at registration, not per request. Everything here is
pure, idempotent, and never raises: a schema this module cannot repair is
returned as it arrived, because a tool the model can still call beats a
discovery that aborted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Deep enough for any hand-written tool schema; short enough that a
# self-referential $defs chain cannot spin.
_MAX_DEPTH = 12

# Bookkeeping keys that mean nothing to a provider once refs are inlined.
_DROPPED_KEYS = frozenset({"$schema", "$id", "$defs", "definitions"})

# Keys whose value is itself a schema.
_SCHEMA_VALUED_KEYS = ("items", "additionalProperties", "contains", "not")

# Keys whose value is a list of schemas.
_SCHEMA_LIST_KEYS = ("allOf", "anyOf", "oneOf", "prefixItems")


def _is_null_schema(node: Any) -> bool:
    if not isinstance(node, Mapping):
        return False
    if "type" in node:
        node_type = node["type"]
        if node_type is None or node_type == "null":
            return True
        if isinstance(node_type, list) and node_type and all(t == "null" for t in node_type):
            return True
    if "const" in node and node["const"] is None:
        return True
    if node.get("enum") == [None]:
        return True
    return False


def _collect_defs(schema: Mapping[str, Any]) -> dict[str, Any]:
    defs: dict[str, Any] = {}
    for container in ("$defs", "definitions"):
        section = schema.get(container)
        if isinstance(section, Mapping):
            for name, value in section.items():
                defs[f"#/{container}/{name}"] = value
    return defs


def _resolve_ref(ref: str, defs: Mapping[str, Any]) -> Any:
    target = defs.get(ref)
    return target if isinstance(target, Mapping) else None


def _sanitize_node(
    node: Any,
    *,
    defs: Mapping[str, Any],
    depth: int,
    seen: frozenset[str],
    fixes: list[str],
) -> Any:
    """Return a cleaned copy of *node*, or ``None`` when it must be dropped."""

    if depth > _MAX_DEPTH:
        fixes.append("depth_truncated")
        return {}
    if isinstance(node, bool):
        # `additionalProperties: false` is legal and meaningful — keep it.
        return node
    if not isinstance(node, Mapping):
        # A bare string where a schema belongs. There is nothing to salvage.
        return None

    ref = node.get("$ref")
    if isinstance(ref, str):
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        target = _resolve_ref(ref, defs)
        if target is None:
            fixes.append("dropped_dangling_ref")
            node = siblings
        elif ref in seen:
            fixes.append("cycle_truncated")
            node = siblings
        else:
            fixes.append("inlined_ref")
            merged = dict(target)
            merged.update(siblings)
            return _sanitize_node(
                merged,
                defs=defs,
                depth=depth + 1,
                seen=seen | {ref},
                fixes=fixes,
            )

    for union_key in ("anyOf", "oneOf"):
        branches = node.get(union_key)
        if not isinstance(branches, list):
            continue
        non_null = [branch for branch in branches if not _is_null_schema(branch)]
        if len(non_null) == 1 and len(non_null) < len(branches):
            fixes.append("collapsed_nullable_union")
            branch = non_null[0]
            collapsed: dict[str, Any] = dict(branch) if isinstance(branch, Mapping) else {}
            # Siblings win: a description written next to the union describes
            # the field, not the branch it was lifted from.
            collapsed.update({k: v for k, v in node.items() if k != union_key})
            return _sanitize_node(
                collapsed,
                defs=defs,
                depth=depth + 1,
                seen=seen,
                fixes=fixes,
            )

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED_KEYS:
            continue

        if key == "type" and isinstance(value, list):
            concrete = [entry for entry in value if entry != "null"]
            out["type"] = concrete[0] if concrete else "string"
            fixes.append("collapsed_type_array")
            continue

        if key == "properties":
            if not isinstance(value, Mapping):
                fixes.append("dropped_invalid_properties")
                continue
            cleaned_properties: dict[str, Any] = {}
            for name, child in value.items():
                sanitized = _sanitize_node(
                    child, defs=defs, depth=depth + 1, seen=seen, fixes=fixes
                )
                if sanitized is None:
                    fixes.append("dropped_non_schema_property")
                    continue
                cleaned_properties[str(name)] = sanitized
            out["properties"] = cleaned_properties
            continue

        if key in _SCHEMA_VALUED_KEYS:
            sanitized = _sanitize_node(value, defs=defs, depth=depth + 1, seen=seen, fixes=fixes)
            if sanitized is None:
                fixes.append("dropped_non_schema_value")
                continue
            out[key] = sanitized
            continue

        if key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            cleaned_list: list[Any] = []
            for child in value:
                sanitized = _sanitize_node(
                    child, defs=defs, depth=depth + 1, seen=seen, fixes=fixes
                )
                if sanitized is None:
                    fixes.append("dropped_non_schema_value")
                    continue
                cleaned_list.append(sanitized)
            if cleaned_list:
                out[key] = cleaned_list
            continue

        out[key] = value

    if out.get("type") == "object" and not isinstance(out.get("properties"), Mapping):
        # A grammar converter has nothing to constrain without this.
        out["properties"] = {}
        fixes.append("added_missing_properties")

    required = out.get("required")
    if isinstance(required, list):
        properties = out.get("properties")
        if isinstance(properties, Mapping):
            kept = [name for name in required if name in properties]
            if len(kept) != len(required):
                fixes.append("pruned_required")
            out["required"] = kept

    return out


def sanitize_input_schema(schema: Any) -> tuple[dict[str, Any], list[str]]:
    """Clean one tool input schema.

    Returns the sanitized schema and the sorted names of the repairs applied,
    which is empty when the schema was already fine. Callers log the names to
    identify which server is emitting broken schemas.
    """

    if not isinstance(schema, Mapping):
        return {"type": "object", "properties": {}}, ["replaced_non_object_schema"]

    fixes: list[str] = []
    try:
        cleaned = _sanitize_node(
            schema,
            defs=_collect_defs(schema),
            depth=0,
            seen=frozenset(),
            fixes=fixes,
        )
    except Exception:  # noqa: BLE001 — a usable tool beats a failed discovery
        return dict(schema), ["sanitize_failed"]

    if not isinstance(cleaned, dict):
        return {"type": "object", "properties": {}}, ["replaced_non_object_schema"]

    if cleaned.get("type") != "object":
        cleaned["type"] = "object"
        fixes.append("forced_object_root")
    if not isinstance(cleaned.get("properties"), Mapping):
        cleaned["properties"] = {}
        fixes.append("added_missing_properties")

    return cleaned, sorted(set(fixes))
