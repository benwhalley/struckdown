"""Jinja inside a slot's options, and slot syntax errors that used to pass silently."""

import uuid
from unittest.mock import patch

import pytest
from box import Box
from typer.testing import CliRunner

import struckdown as sd
from struckdown.errors import TemplateError
from struckdown.jinja_analysis import analyze_template
from struckdown.sd_cli import app

CREDS = sd.LLMCredentials(api_key="test")


def last_choice(return_type):
    """A valid answer for any slot: the last pick option, or some text."""
    annotation = return_type.model_fields["response"].annotation
    choices = [a for a in getattr(annotation, "__args__", []) if a is not type(None)]
    literals = getattr(choices[0], "__args__", None) if choices else None
    return literals[-1] if literals else "blue"


def run(template, context=None):
    """complete() with the model call replaced; returns (result, llm kwargs per call)."""
    calls = []

    def fake_chat(*args, **kwargs):
        calls.append(kwargs.get("extra_kwargs"))
        return_type = kwargs["return_type"]
        return return_type(response=last_choice(return_type)), Box({"usage": {}})

    with patch("struckdown.llm.structured_chat", side_effect=fake_chat):
        result = sd.complete(f"{uuid.uuid4()} {template}", context or {}, credentials=CREDS)
    return result, calls


def option_values(result, key):
    return [o.value for o in result.results[key].options]


def test_pick_options_from_context():
    result, _ = run("Which? [[pick:choice|{{ opts }}]]", {"opts": "red,green,blue"})
    assert option_values(result, "choice") == ["red", "green", "blue"]
    assert result["choice"] == "blue"


def test_pick_options_from_a_list():
    result, _ = run("Which? [[pick:choice|{{ opts|join(',') }}]]", {"opts": ["cat", "dog"]})
    assert option_values(result, "choice") == ["cat", "dog"]


def test_options_can_use_an_earlier_slot():
    result, _ = run('Colour? [[colour]] Shade? [[pick:shade|"light {{ colour }}","dark {{ colour }}"]]')
    assert option_values(result, "shade") == ["light blue", "dark blue"]


def test_llm_parameters_from_context():
    _, calls = run("Say hi [[greeting|temperature={{ t }}]]", {"t": 0.3})
    assert calls[-1] == {"temperature": 0.3}


def test_earlier_slot_value_triggers_rerender():
    analysis = analyze_template("[[colour]] [[pick:shade|{{ colour }},grey]]")
    assert [s.key for s in analysis.slots] == ["colour", "shade"]
    assert analysis.triggers == {"colour": ["shade"]}


def test_invalid_rendered_options_raise_naming_the_slot():
    with pytest.raises(TemplateError, match=r"slot \[\[x\]\] rendered as \[\[pick:x\|a,,b\]\]"):
        run("[[pick:x|{{ opts }}]]", {"opts": "a,,b"})


def test_jinja_in_slot_name_or_type_raises():
    with pytest.raises(TemplateError, match="only go in a slot's options"):
        run("[[pick:{{ name }}|a,b]]", {"name": "x"})


def test_invalid_slot_syntax_raises_instead_of_running_nothing():
    with pytest.raises(TemplateError, match="Could not parse the template's slots"):
        run("Choose [[pick:x|red,]]")


def test_help_keeps_slot_brackets():
    result = CliRunner().invoke(app, ["chat", "--help"], terminal_width=120)
    assert 'sd chat "tell a joke [[joke]]"' in result.output
