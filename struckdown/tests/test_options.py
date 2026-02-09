"""Tests for the options parser."""

import pytest

from struckdown.validation import ParsedOptions, parse_options


class TestParseOptions:
    """Test the parse_options function."""

    def test_empty_options(self):
        """Empty options returns default ParsedOptions."""
        opts = parse_options(None)
        assert opts.required is False
        assert opts.ge is None
        assert opts.le is None
        assert opts.positional == []
        assert opts.kwargs == {}

    def test_empty_list(self):
        """Empty list returns default ParsedOptions."""
        opts = parse_options([])
        assert opts.required is False
        assert opts.positional == []

    def test_required_keyword(self):
        """Test required=true parsing."""
        opts = parse_options(["required=true"])
        assert opts.required is True

        opts = parse_options(["required=yes"])
        assert opts.required is True

        opts = parse_options(["required=1"])
        assert opts.required is True

        opts = parse_options(["required=false"])
        assert opts.required is False

    def test_bare_required(self):
        """Bare 'required' is a positional value, not a keyword.

        This is important for 'pick' type where 'required' could be a valid option.
        Use 'required=true' to set the required flag.
        Use is_required_flag_set() to check for both.
        """
        opts = parse_options(["required"])
        assert opts.required is False  # bare "required" doesn't set the flag
        assert "required" in opts.positional
        # but is_required_flag_set() checks both
        assert opts.is_required_flag_set() is True

    def test_min_max_aliases(self):
        """Test min/max are aliases for ge/le."""
        opts = parse_options(["min=0", "max=100"])
        assert opts.ge == 0.0
        assert opts.le == 100.0

    def test_ge_le_direct(self):
        """Test ge/le direct usage."""
        opts = parse_options(["ge=5", "le=10"])
        assert opts.ge == 5.0
        assert opts.le == 10.0

    def test_gt_lt(self):
        """Test gt/lt parsing."""
        opts = parse_options(["gt=0", "lt=100"])
        assert opts.gt == 0.0
        assert opts.lt == 100.0

    def test_min_length_max_length(self):
        """Test string length constraints."""
        opts = parse_options(["min_length=5", "max_length=100"])
        assert opts.min_length == 5
        assert opts.max_length == 100

    def test_positional_args(self):
        """Test positional (non-keyword) arguments."""
        opts = parse_options(["apple", "banana", "cherry"])
        assert opts.positional == ["apple", "banana", "cherry"]
        assert opts.kwargs == {}

    def test_mixed_args(self):
        """Test mixed positional and keyword arguments."""
        opts = parse_options(["apple", "min=0", "banana", "max=10"])
        assert opts.positional == ["apple", "banana"]
        assert opts.ge == 0.0
        assert opts.le == 10.0

    def test_kwargs_dict(self):
        """Test that all key=value pairs are in kwargs."""
        opts = parse_options(["min=0", "max=100", "custom=value"])
        assert opts.kwargs["min"] == 0
        assert opts.kwargs["max"] == 100
        assert opts.kwargs["custom"] == "value"

    def test_get_method(self):
        """Test the get() method."""
        opts = parse_options(["foo=bar"])
        assert opts.get("foo") == "bar"
        assert opts.get("missing") is None
        assert opts.get("missing", "default") == "default"

    def test_contains(self):
        """Test the __contains__ method."""
        opts = parse_options(["foo=bar"])
        assert "foo" in opts
        assert "missing" not in opts

    def test_value_type_coercion(self):
        """Test that values are coerced to appropriate types."""
        opts = parse_options(["int_val=42", "float_val=3.14", "str_val=hello"])
        assert opts.kwargs["int_val"] == 42
        assert opts.kwargs["float_val"] == 3.14
        assert opts.kwargs["str_val"] == "hello"

    def test_whitespace_handling(self):
        """Test that whitespace is stripped."""
        opts = parse_options(["  min = 0  ", "  max=100  "])
        assert opts.ge == 0.0
        assert opts.le == 100.0

    def test_empty_string_option(self):
        """Test handling of empty strings."""
        opts = parse_options(["", "  ", "valid"])
        assert opts.positional == ["valid"]


    def test_pattern_option(self):
        """Test pattern/regex option parsing."""
        opts = parse_options(["pattern=\\d{3}-\\d{4}"])
        assert opts.pattern == "\\d{3}-\\d{4}"

    def test_regex_alias(self):
        """Test regex is an alias for pattern."""
        opts = parse_options(["regex=[A-Z]+"])
        assert opts.pattern == "[A-Z]+"

    def test_pattern_with_quotes(self):
        """Test pattern option with quoted value (as would come from grammar)."""
        # When parsed from grammar, quotes are stripped by the parser
        opts = parse_options(["pattern=\\d{3}"])
        assert opts.pattern == "\\d{3}"


class TestParsedOptions:
    """Test the ParsedOptions dataclass."""

    def test_defaults(self):
        """Test default values."""
        opts = ParsedOptions()
        assert opts.required is False
        assert opts.ge is None
        assert opts.le is None
        assert opts.gt is None
        assert opts.lt is None
        assert opts.min_length is None
        assert opts.max_length is None
        assert opts.pattern is None
        assert opts.positional == []
        assert opts.kwargs == {}
