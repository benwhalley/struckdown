"""Test optional pick selections with required flag and ! prefix."""

import pytest

from struckdown.return_type_models import selection_response_model


def test_pick_optional_by_default():
    """Test that pick selections are optional by default (can return None)."""
    model = selection_response_model(
        ["option1", "option2", "option3"], required_prefix=False
    )

    # Check that the response field accepts None
    instance = model(response=None)
    assert instance.response is None

    # Check that valid options still work
    instance = model(response="option1")
    assert instance.response == "option1"


def test_pick_required_with_prefix():
    """Test that pick selections with ! prefix (required_prefix=True) cannot be None."""
    model = selection_response_model(
        ["option1", "option2", "option3"], required_prefix=True
    )

    # Should raise validation error when None provided
    with pytest.raises(Exception):  # Pydantic ValidationError
        model(response=None)

    # Valid options should still work
    instance = model(response="option1")
    assert instance.response == "option1"


def test_pick_required_with_keyvalue():
    """Test that pick selections with required=true key-value cannot be None."""
    model = selection_response_model(
        ["option1", "option2", "required=true"], required_prefix=False
    )

    # Should raise validation error when None provided
    with pytest.raises(Exception):  # Pydantic ValidationError
        model(response=None)

    # Valid options should still work
    instance = model(response="option1")
    assert instance.response == "option1"

    # 'required=true' should be filtered out from options
    with pytest.raises(Exception):
        model(response="required=true")


def test_pick_with_required_as_option():
    """Test that 'required' (without =) can be a valid option value."""
    model = selection_response_model(
        ["required", "optional", "maybe"], required_prefix=False
    )

    # 'required' should be a valid selection (it's just a string option)
    instance = model(response="required")
    assert instance.response == "required"

    # Should allow None (optional by default)
    instance = model(response=None)
    assert instance.response is None


def test_pick_must_have_at_least_one_option():
    """Test that we get an error if only meta-options are provided."""
    with pytest.raises(ValueError, match="at least one selectable option"):
        selection_response_model(["required=true"], required_prefix=False)


def test_pick_multi_selection_with_quantifier():
    """Test that multi-selection with quantifier still works."""
    model = selection_response_model(
        ["option1", "option2", "option3"], quantifier=(1, 2)
    )

    # Should accept list of selections
    instance = model(response=["option1", "option2"])
    assert instance.response == ["option1", "option2"]

    # Should validate min/max constraints
    with pytest.raises(Exception):  # Too few items
        model(response=[])

    with pytest.raises(Exception):  # Too many items
        model(response=["option1", "option2", "option3"])


def test_pick_required_false():
    """Test that required=false makes field optional."""
    model = selection_response_model(
        ["option1", "option2", "required=false"], required_prefix=False
    )

    # Should allow None
    instance = model(response=None)
    assert instance.response is None

    # Should work with valid option
    instance = model(response="option1")
    assert instance.response == "option1"


def test_optional_pick_schema_is_openai_strict_compatible():
    """Optional picks must emit a top-level ``type`` key on the response
    property. Without this, OpenAI strict tool-calling rejects the schema
    with: ``In context=('properties', 'response'), schema must have a 'type'
    key.`` See selection_response_model() and _flatten_nullable_anyof().
    """
    from pydantic import TypeAdapter

    model = selection_response_model(
        ["respond", "reaction", "skip"], required_prefix=False
    )

    # both schema-generation paths (classmethod override and TypeAdapter
    # via __get_pydantic_json_schema__) must produce a valid schema
    for schema in (model.model_json_schema(), TypeAdapter(model).json_schema()):
        prop = schema["properties"]["response"]
        assert "anyOf" not in prop, (
            "Optional pick schema still uses anyOf; OpenAI strict mode "
            "rejects properties without a top-level 'type' key."
        )
        assert prop.get("type") == ["string", "null"]
        assert prop.get("enum") == ["respond", "reaction", "skip"]
