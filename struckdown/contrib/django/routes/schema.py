"""Input and output schemas for a route.

A route's inputs are the variables its prompt reads (``{{ essay }}``) and its
outputs are the slots it fills (``[[summary]]``). With no schema set, every
input is a required string capped at the route's token limit and every slot is
returned. A schema narrows or reshapes that.

Schemas are JSON Schema, restricted to what maps onto a pydantic field without
surprises: a flat object whose properties are ``string``, ``integer``,
``number``, ``boolean``, ``array`` (of those) or ``object``, with ``enum``,
``minLength``/``maxLength``, ``minimum``/``maximum`` and ``maxItems``.
A string property with ``"format": "image"`` is an uploaded image, not text.
:func:`check_schema` refuses anything else, so a schema that saves is one the
API can enforce.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

CHARS_PER_TOKEN = 4  # rough, for a length cap people can reason about

_PRIMITIVE = {"string": str, "integer": int, "number": float, "boolean": bool}
_KEYWORDS = {
    "type",
    "description",
    "title",
    "enum",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "maxItems",
    "items",
    "format",
    "default",
}


class SchemaError(ValueError):
    pass


def analyse(prompt: str) -> dict:
    """``{"inputs": [...], "slots": [{"name", "type"}...], "error": str|None}``."""
    from struckdown.parsing import extract_slots
    from struckdown.playground.core import extract_required_inputs, validate_syntax

    checked = validate_syntax(prompt)
    if not checked["valid"]:
        e = checked["error"]
        return {"inputs": [], "slots": [], "error": f"line {e['line']}: {e['message']}"}
    found = extract_required_inputs(prompt)
    slots = [
        {"name": s.name, "type": s.type}
        for s in extract_slots(prompt)
        if not s.is_action and s.name in found["slots_defined"]
    ]
    return {"inputs": found["inputs_required"], "slots": slots, "error": None}


def derived_input_schema(prompt: str, max_tokens: int) -> dict:
    """The schema a route uses when none is set: each input a required string."""
    inputs = analyse(prompt)["inputs"]
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "maxLength": max_tokens * CHARS_PER_TOKEN}
            for name in inputs
        },
        "required": inputs,
    }


_SLOT_TYPES = {"bool": "boolean", "boolean": "boolean", "int": "integer", "number": "number"}


def derived_output_schema(prompt: str) -> dict:
    """Every slot, typed where the slot's type says how."""
    slots = analyse(prompt)["slots"]
    return {
        "type": "object",
        "properties": {
            s["name"]: ({"type": _SLOT_TYPES[s["type"]]} if s["type"] in _SLOT_TYPES else {})
            for s in slots
        },
    }


def image_fields(schema: dict | None) -> set[str]:
    return {
        name
        for name, prop in ((schema or {}).get("properties") or {}).items()
        if prop.get("format") == "image"
    }


def _field_type(name: str, prop: dict, max_chars: int):
    unknown = set(prop) - _KEYWORDS
    if unknown:
        raise SchemaError(f"{name}: unsupported keyword(s) {sorted(unknown)}")
    kind = prop.get("type")
    if kind is None:
        return Any, {}
    if kind in ("array", "object") and "enum" in prop:
        raise SchemaError(f"{name}: enum is only for strings and numbers")
    if "enum" in prop:
        return Literal[tuple(prop["enum"])], {}
    if kind == "string":
        return str, {
            "max_length": min(prop.get("maxLength", max_chars), max_chars),
            **({"min_length": prop["minLength"]} if "minLength" in prop else {}),
        }
    if kind in ("integer", "number"):
        return _PRIMITIVE[kind], {
            k: prop[j] for j, k in (("minimum", "ge"), ("maximum", "le")) if j in prop
        }
    if kind == "boolean":
        return bool, {}
    if kind == "array":
        item_type, _ = _field_type(f"{name}[]", prop.get("items") or {}, max_chars)
        return list[item_type], {"max_length": prop.get("maxItems", 100)}
    if kind == "object":
        return dict, {}
    raise SchemaError(f"{name}: unsupported type {kind!r}")


def to_model(schema: dict, *, max_chars: int, name: str = "Inputs") -> type[BaseModel]:
    """A pydantic model for ``schema``. Unknown fields are refused; image
    fields are left out (they arrive as files and are checked separately)."""
    if schema.get("type", "object") != "object":
        raise SchemaError("the schema must describe an object")
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        raise SchemaError("properties must be an object")
    required = set(schema.get("required") or [])
    fields = {}
    for field_name, prop in properties.items():
        if not isinstance(prop, dict):
            raise SchemaError(f"{field_name}: each property must be an object")
        if prop.get("format") == "image":
            continue
        annotation, constraints = _field_type(field_name, prop, max_chars)
        if field_name in required:
            fields[field_name] = (annotation, Field(..., **constraints))
        else:
            fields[field_name] = (Optional[annotation], Field(prop.get("default"), **constraints))
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def check_schema(schema: dict | None, *, max_chars: int) -> None:
    """Raise :class:`SchemaError` unless ``schema`` is one :func:`to_model` takes."""
    if schema:
        to_model(schema, max_chars=max_chars)
