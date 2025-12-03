"""Tests for struckdown.playground.core module."""

import pytest
from pathlib import Path
import tempfile

from struckdown.playground.core import (
    extract_required_inputs,
    validate_syntax,
    encode_state,
    decode_state,
    load_xlsx_data,
)


class TestExtractRequiredInputs:
    """Tests for extract_required_inputs function."""

    def test_basic_variable_and_slot(self):
        """Variables not filled by slots are required inputs."""
        syntax = "Tell me about {{topic}}\n[[response]]"
        result = extract_required_inputs(syntax)
        assert result["inputs_required"] == ["topic"]
        assert result["slots_defined"] == ["response"]

    def test_slot_fills_variable(self):
        """Variables filled by earlier slots are not required."""
        syntax = "[[topic]]\nExpand on {{topic}}\n[[expansion]]"
        result = extract_required_inputs(syntax)
        assert result["inputs_required"] == []
        assert "topic" in result["slots_defined"]
        assert "expansion" in result["slots_defined"]

    def test_typed_slot(self):
        """Typed slots correctly extracted."""
        syntax = "Is valid? [[!bool:valid]]\n{% if valid %}Yes{% endif %}"
        result = extract_required_inputs(syntax)
        assert "valid" in result["slots_defined"]

    def test_multiple_variables(self):
        """Multiple variables correctly identified."""
        syntax = "{{name}} from {{city}} says [[greeting]]"
        result = extract_required_inputs(syntax)
        assert sorted(result["inputs_required"]) == ["city", "name"]
        assert result["slots_defined"] == ["greeting"]

    def test_no_variables_or_slots(self):
        """Plain text returns empty lists."""
        syntax = "Just plain text"
        result = extract_required_inputs(syntax)
        assert result["inputs_required"] == []
        assert result["slots_defined"] == []

    def test_only_slots(self):
        """Template with only slots needs no inputs."""
        syntax = "[[first]] and [[second]]"
        result = extract_required_inputs(syntax)
        assert result["inputs_required"] == []
        assert sorted(result["slots_defined"]) == ["first", "second"]

    def test_slot_with_options(self):
        """Slot with options correctly extracts key."""
        syntax = "Pick: [[!pick:color|red,green,blue]]"
        result = extract_required_inputs(syntax)
        assert result["slots_defined"] == ["color"]

    def test_action_slot(self):
        """Action slots correctly extracted."""
        syntax = "[[@search:results|query={{topic}}]]"
        result = extract_required_inputs(syntax)
        assert "results" in result["slots_defined"]
        assert "topic" in result["inputs_required"]

    def test_jinja_conditionals(self):
        """Variables in jinja conditionals detected."""
        syntax = "{% if show_details %}Details: {{details}}{% endif %}\n[[output]]"
        result = extract_required_inputs(syntax)
        assert "show_details" in result["inputs_required"]
        assert "details" in result["inputs_required"]

    def test_empty_syntax(self):
        """Empty syntax returns empty lists."""
        result = extract_required_inputs("")
        assert result["inputs_required"] == []
        assert result["slots_defined"] == []


class TestValidateSyntax:
    """Tests for validate_syntax function."""

    def test_valid_simple(self):
        """Valid simple syntax returns no error."""
        result = validate_syntax("Hello [[greeting]]")
        assert result["valid"] is True
        assert result["error"] is None

    def test_valid_complex(self):
        """Valid complex syntax returns no error."""
        syntax = """
        <system>You are helpful</system>
        Tell me about {{topic}}
        [[!bool:is_good]]
        {% if is_good %}Great!{% endif %}
        [[summary]]
        """
        result = validate_syntax(syntax)
        assert result["valid"] is True

    def test_empty_syntax(self):
        """Empty syntax is valid."""
        result = validate_syntax("")
        assert result["valid"] is True

    def test_whitespace_only(self):
        """Whitespace-only syntax is valid."""
        result = validate_syntax("   \n\t  ")
        assert result["valid"] is True

    def test_plain_text(self):
        """Plain text without slots is valid."""
        result = validate_syntax("Just plain text without any special syntax")
        assert result["valid"] is True


class TestEncodeDecodeState:
    """Tests for encode_state and decode_state functions."""

    def test_roundtrip_basic(self):
        """State encoding/decoding preserves data."""
        syntax = "[[test]]"
        model = "gpt-4"
        inputs = {"x": "y"}
        encoded = encode_state(syntax=syntax, model=model, inputs=inputs)
        decoded = decode_state(encoded)
        assert decoded["syntax"] == syntax
        assert decoded["model"] == model
        assert decoded["inputs"] == inputs

    def test_roundtrip_complex(self):
        """Complex state roundtrips correctly."""
        syntax = """
        Tell me about {{topic}}
        [[response]]
        Rate it: [[!number:rating|1-10]]
        """
        model = "anthropic/claude-3-sonnet"
        inputs = {"topic": "artificial intelligence", "detail_level": "high"}
        encoded = encode_state(syntax=syntax, model=model, inputs=inputs)
        decoded = decode_state(encoded)
        assert decoded["syntax"] == syntax
        assert decoded["model"] == model
        assert decoded["inputs"] == inputs

    def test_roundtrip_empty(self):
        """Empty state roundtrips correctly."""
        encoded = encode_state(syntax="", model="", inputs={})
        decoded = decode_state(encoded)
        assert decoded["syntax"] == ""
        assert decoded["model"] == ""
        assert decoded["inputs"] == {}

    def test_roundtrip_unicode(self):
        """Unicode content roundtrips correctly."""
        syntax = "Say hello in French: [[greeting]]"
        inputs = {"name": "Jean-Pierre"}
        encoded = encode_state(syntax=syntax, model="", inputs=inputs)
        decoded = decode_state(encoded)
        assert decoded["syntax"] == syntax
        assert decoded["inputs"] == inputs

    def test_decode_invalid(self):
        """Invalid encoded string returns empty state."""
        decoded = decode_state("not-valid-base64!!!")
        assert decoded["syntax"] == ""
        assert decoded["model"] == ""
        assert decoded["inputs"] == {}

    def test_encoded_is_url_safe(self):
        """Encoded state is URL-safe."""
        syntax = "Test [[slot]] with special chars: <>&\""
        encoded = encode_state(syntax=syntax, model="test", inputs={"a": "b"})
        # URL-safe base64 uses only alphanumeric, -, _, =
        import re
        assert re.match(r'^[A-Za-z0-9_=-]+$', encoded)

    def test_compression_works(self):
        """Long repeated content compresses well."""
        syntax = "[[slot]]" * 1000
        encoded = encode_state(syntax=syntax, model="", inputs={})
        # Compressed should be much smaller than raw
        raw_length = len(syntax)
        assert len(encoded) < raw_length / 2


class TestLoadXlsxData:
    """Tests for load_xlsx_data function."""

    def test_load_csv(self):
        """CSV file loads correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age\n")
            f.write("Alice,30\n")
            f.write("Bob,25\n")
            f.flush()
            path = Path(f.name)

        try:
            result = load_xlsx_data(path)
            assert result["row_count"] == 2
            assert result["columns"] == ["name", "age"]
            assert len(result["rows"]) == 2
            assert result["rows"][0]["name"] == "Alice"
            assert result["rows"][1]["name"] == "Bob"
        finally:
            path.unlink()

    def test_load_xlsx(self):
        """XLSX file loads correctly."""
        import pandas as pd

        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            path = Path(f.name)

        try:
            df = pd.DataFrame({
                "topic": ["cats", "dogs"],
                "style": ["funny", "serious"],
            })
            df.to_excel(path, index=False)

            result = load_xlsx_data(path)
            assert result["row_count"] == 2
            assert "topic" in result["columns"]
            assert "style" in result["columns"]
            assert result["rows"][0]["topic"] == "cats"
        finally:
            path.unlink()

    def test_handles_nan_values(self):
        """NaN values converted to empty strings."""
        import pandas as pd
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            path = Path(f.name)

        try:
            df = pd.DataFrame({
                "name": ["Alice", np.nan, "Charlie"],
                "value": [1.0, 2.0, np.nan],
            })
            df.to_excel(path, index=False)

            result = load_xlsx_data(path)
            # NaN should be converted to empty string
            assert result["rows"][1]["name"] == ""
        finally:
            path.unlink()
