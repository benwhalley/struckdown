"""Input and output schemas for a route.

A route's inputs are the variables its prompt reads (``{{ essay }}``) and its
outputs are the slots it fills (``[[summary]]``). With no schema set, every
input is a required string capped at the route's token limit and every slot is
returned. A schema narrows or reshapes that.

Schemas are JSON Schema, restricted to what maps onto a pydantic field without
surprises: a flat object whose properties are ``string``, ``integer``,
``number``, ``boolean`` or ``array`` (of those, at most 100), with ``enum``,
``minLength``/``maxLength``, ``minimum``/``maximum`` and ``maxItems``. Every
input needs a ``type``; an output schema may also use ``object`` or leave the
type out. Property names are identifiers.
A string property with ``"format": "image"`` is an uploaded image, not text.
:func:`check_schema` refuses anything else, so a schema that saves is one the
API can enforce.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Optional

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


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_INT_KEYWORDS = ("minLength", "maxLength", "maxItems")
_NUMBER_KEYWORDS = ("minimum", "maximum")
MAX_ITEMS = 100


def _check_keywords(name: str, prop: dict) -> None:
    """Each keyword's value is of the kind JSON Schema gives it, so a typo is a
    form error on save rather than a crash on the first call."""
    unknown = set(prop) - _KEYWORDS
    if unknown:
        raise SchemaError(f"{name}: unsupported keyword(s) {sorted(unknown)}")
    for key in _INT_KEYWORDS:
        if key in prop and (type(prop[key]) is not int or prop[key] < 0):
            raise SchemaError(f"{name}: {key} must be a whole number of 0 or more")
    for key in _NUMBER_KEYWORDS:
        if key in prop and type(prop[key]) not in (int, float):
            raise SchemaError(f"{name}: {key} must be a number")
    if "enum" in prop:
        values = prop["enum"]
        if not isinstance(values, list) or not values:
            raise SchemaError(f"{name}: enum must be a non-empty list")
        if any(type(v) not in (str, int, float, bool) for v in values):
            raise SchemaError(f"{name}: enum values must be strings, numbers or true/false")
    if "format" in prop and prop["format"] != "image":
        raise SchemaError(f"{name}: the only format supported is \"image\"")
    for key in ("description", "title"):
        if key in prop and not isinstance(prop[key], str):
            raise SchemaError(f"{name}: {key} must be text")


def _field_type(name: str, prop: dict, max_chars: int, *, inputs: bool):
    """The annotation and pydantic constraints for one property. Inputs are held
    to more than outputs: every input needs a type, and nothing unbounded
    (a free-form object, an array of objects) is accepted from a caller."""
    _check_keywords(name, prop)
    kind = prop.get("type")
    if kind is None:
        if inputs:
            raise SchemaError(f"{name}: give each input a type")
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
        items = prop.get("items") or {}
        if not isinstance(items, dict):
            raise SchemaError(f"{name}: items must be an object")
        if inputs and items.get("type") not in _PRIMITIVE:
            raise SchemaError(f"{name}: an input array must hold strings, numbers or true/false")
        item_type, item_constraints = _field_type(f"{name}[]", items, max_chars, inputs=inputs)
        if item_constraints:
            item_type = Annotated[item_type, Field(**item_constraints)]
        return list[item_type], {"max_length": min(prop.get("maxItems", MAX_ITEMS), MAX_ITEMS)}
    if kind == "object" and not inputs:
        return dict, {}
    raise SchemaError(f"{name}: unsupported type {kind!r}")


def to_model(
    schema: dict, *, max_chars: int, name: str = "Inputs", inputs: bool = True
) -> type[BaseModel]:
    """A pydantic model for ``schema``. Unknown fields are refused; image
    fields are left out (they arrive as files and are checked separately)."""
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise SchemaError("the schema must describe an object")
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        raise SchemaError("properties must be an object")
    required = schema.get("required") or []
    if not isinstance(required, list) or any(not isinstance(r, str) for r in required):
        raise SchemaError("required must be a list of property names")
    fields = {}
    for field_name, prop in properties.items():
        # names reach the example HTML page and Python script, and the template
        if not _NAME.match(field_name):
            raise SchemaError(
                f"{field_name!r}: a property name must be letters, digits and _ "
                "(not starting with a digit)"
            )
        if not isinstance(prop, dict):
            raise SchemaError(f"{field_name}: each property must be an object")
        if prop.get("format") == "image":
            _check_keywords(field_name, prop)
            continue
        annotation, constraints = _field_type(field_name, prop, max_chars, inputs=inputs)
        if field_name in set(required):
            fields[field_name] = (annotation, Field(..., **constraints))
        else:
            fields[field_name] = (Optional[annotation], Field(prop.get("default"), **constraints))
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def check_schema(schema: dict | None, *, max_chars: int, inputs: bool = True) -> None:
    """Raise :class:`SchemaError` unless ``schema`` is one :func:`to_model` takes."""
    if schema:
        try:
            to_model(schema, max_chars=max_chars, inputs=inputs)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SchemaError):
                raise
            raise SchemaError(f"the schema could not be read: {exc}") from exc
