"""Jinja inside a slot's options, and slot syntax errors that used to pass silently."""

import uuid
from unittest.mock import patch

import pytest
from box import Box
from typer.testing import CliRunner

import struckdown as sd
from struckdown.errors import TemplateError
from struckdown.jinja_analysis import analyze_template
from struckdown.parsing import DYNAMIC_TOKEN, get_slot_names, parse_syntax
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


# --- parse_syntax: static validation of templates that use dynamic options -----

DYNAMIC = "Mood? [[pick:mood|{{ moods|join(',') }}]]\n\nSay hi [[greeting|temperature={{ temp }}]]"


def test_parse_syntax_accepts_jinja_in_slot_options():
    """Editors and validators call parse_syntax; it rejected what complete() runs."""
    [section] = parse_syntax(DYNAMIC)
    assert list(section) == ["mood", "greeting"]
    assert section["mood"].action_type == "pick"


def test_parse_syntax_marks_options_that_come_from_a_render():
    [section] = parse_syntax(DYNAMIC)
    assert [o.value for o in section["mood"].options] == [DYNAMIC_TOKEN]


def test_get_slot_names_sees_slots_with_dynamic_options():
    assert get_slot_names(DYNAMIC) == {"mood", "greeting"}


def test_parse_syntax_still_rejects_jinja_in_slot_name_or_type():
    with pytest.raises(TemplateError, match="only go in a slot's options"):
        parse_syntax("[[pick:{{ name }}|a,b]]")


def test_parse_syntax_still_rejects_broken_slots():
    with pytest.raises(Exception, match="Unexpected"):
        parse_syntax("Choose [[pick:x|red,]]")


def test_parse_syntax_agrees_with_complete():
    """What complete() can run, parse_syntax accepts."""
    parse_syntax(DYNAMIC)
    result, _ = run(DYNAMIC, {"moods": ["happy", "sad"], "temp": 0.2})
    assert result.results["mood"].output in {"happy", "sad"}


def test_explain_handles_dynamic_options(tmp_path):
    prompt = tmp_path / "dynamic.sd"
    prompt.write_text(DYNAMIC)
    result = CliRunner().invoke(app, ["explain", str(prompt)], terminal_width=120)
    assert result.exit_code == 0, result.output
    assert "mood" in result.output
    assert "Unexpected token" not in result.output
