"""Tests for pattern validation in response models."""

import pytest
from pydantic import ValidationError

from struckdown.return_type_models import default_response_model


class TestPatternValidation:
    """Test pattern/regex constraint validation."""

    def test_pattern_creates_constrained_model(self):
        """Test that pattern option creates a model with regex constraint."""
        model = default_response_model(["pattern=\\d{3}"])

        # Valid: matches pattern
        instance = model(response="123")
        assert instance.response == "123"

        # Invalid: doesn't match pattern
        with pytest.raises(ValidationError):
            model(response="abc")

    def test_pattern_with_complex_regex(self):
        """Test pattern with more complex regex."""
        model = default_response_model(["pattern=\\d{3}-\\d{4}"])

        # Valid phone-like pattern
        instance = model(response="123-4567")
        assert instance.response == "123-4567"

        # Invalid
        with pytest.raises(ValidationError):
            model(response="1234567")

    def test_pattern_with_anchors(self):
        """Test pattern with start/end anchors."""
        model = default_response_model(["pattern=^[A-Z]{2}\\d{4}$"])

        # Valid: exactly 2 uppercase letters followed by 4 digits
        instance = model(response="AB1234")
        assert instance.response == "AB1234"

        # Invalid: lowercase
        with pytest.raises(ValidationError):
            model(response="ab1234")

    def test_pattern_combined_with_required(self):
        """Test pattern with required constraint."""
        model = default_response_model(["pattern=[A-Z]+", "required=true"])

        # Valid
        instance = model(response="HELLO")
        assert instance.response == "HELLO"

        # Invalid: doesn't match
        with pytest.raises(ValidationError):
            model(response="hello")

    def test_pattern_combined_with_length(self):
        """Test pattern combined with length constraints."""
        model = default_response_model(["pattern=\\w+", "min_length=3", "max_length=10"])

        # Valid: matches pattern and length
        instance = model(response="hello")
        assert instance.response == "hello"

        # Invalid: too short
        with pytest.raises(ValidationError):
            model(response="ab")

        # Invalid: too long
        with pytest.raises(ValidationError):
            model(response="verylongword")

    def test_regex_alias_works(self):
        """Test that regex= is an alias for pattern=."""
        model = default_response_model(["regex=\\d+"])

        instance = model(response="12345")
        assert instance.response == "12345"

        with pytest.raises(ValidationError):
            model(response="abc")
