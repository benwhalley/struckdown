"""
Test cases for quantifier functionality in pick operations.
Tests parsing, Pydantic model generation, and validation.
"""

import unittest
from typing import get_args, get_origin

from pydantic import ValidationError

from struckdown.parsing import parse_syntax
from struckdown.return_type_models import selection_response_model


class QuantifierParsingTestCase(unittest.TestCase):
    """Test that quantifier syntax is parsed correctly"""

    def test_parse_single_pick_no_quantifier(self):
        """Test backward compatibility: [[pick:letter|a,b,c]]"""
        template = "Choose a letter [[pick:letter|a,b,c]]"
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 1)
        section = sections[0]
        self.assertIn("letter", section)

        part = section["letter"]
        self.assertEqual(part.key, "letter")
        # options are now OptionValue namedtuples
        option_values = [opt.value for opt in part.options]
        self.assertEqual(option_values, ["a", "b", "c"])
        self.assertIsNone(part.quantifier)

    def test_parse_exact_quantifier(self):
        """Test [[pick{3}:letters|a,b,c]]"""
        template = "Choose exactly 3 [[pick{3}:letters|a,b,c]]"
        sections = parse_syntax(template)

        part = sections[0]["letters"]
        self.assertEqual(part.quantifier, (3, 3))

    def test_parse_range_quantifier(self):
        """Test [[pick{1,3}:letters|a,b,c]]"""
        template = "Choose 1-3 [[pick{1,3}:letters|a,b,c]]"
        sections = parse_syntax(template)

        part = sections[0]["letters"]
        self.assertEqual(part.quantifier, (1, 3))

    def test_parse_at_least_quantifier(self):
        """Test [[pick{2,}:letters|a,b,c]]"""
        template = "Choose at least 2 [[pick{2,}:letters|a,b,c]]"
        sections = parse_syntax(template)

        part = sections[0]["letters"]
        self.assertEqual(part.quantifier, (2, None))

    def test_parse_zero_or_more_quantifier(self):
        """Test [[pick*:letters|a,b,c]]"""
        template = "Choose any [[pick*:letters|a,b,c]]"
        sections = parse_syntax(template)

        part = sections[0]["letters"]
        self.assertEqual(part.quantifier, (0, None))

    def test_parse_one_or_more_quantifier(self):
        """Test [[pick+:letters|a,b,c]]"""
        template = "Choose one or more [[pick+:letters|a,b,c]]"
        sections = parse_syntax(template)

        part = sections[0]["letters"]
        self.assertEqual(part.quantifier, (1, None))

    def test_parse_zero_or_one_quantifier(self):
        """Test [[pick?:letters|a,b,c]]"""
        template = "Choose optionally [[pick?:letters|a,b,c]]"
        sections = parse_syntax(template)

        part = sections[0]["letters"]
        self.assertEqual(part.quantifier, (0, 1))


class QuantifierModelGenerationTestCase(unittest.TestCase):
    """Test that quantifiers produce correct Pydantic models"""

    def test_single_pick_model(self):
        """Test model without quantifier produces single selection"""
        model = selection_response_model(["a", "b", "c"], quantifier=None)

        # Should be a single Literal, not a List
        field_info = model.model_fields["response"]
        annotation = field_info.annotation

        # Check it's a Literal type, not a List
        self.assertNotEqual(get_origin(annotation), list)

    def test_multi_pick_model_exact(self):
        """Test model with exact quantifier {3}"""
        model = selection_response_model(["a", "b", "c"], quantifier=(3, 3))

        field_info = model.model_fields["response"]
        annotation = field_info.annotation

        # Should be a List type
        self.assertEqual(get_origin(annotation), list)

        # Check constraints - Pydantic stores min/max in separate metadata objects
        metadata_objs = field_info.metadata
        min_len = next(
            (m.min_length for m in metadata_objs if hasattr(m, "min_length")), None
        )
        max_len = next(
            (m.max_length for m in metadata_objs if hasattr(m, "max_length")), None
        )

        self.assertEqual(min_len, 3)
        self.assertEqual(max_len, 3)

    def test_multi_pick_model_range(self):
        """Test model with range quantifier {1,3}"""
        model = selection_response_model(["a", "b", "c"], quantifier=(1, 3))

        field_info = model.model_fields["response"]

        # Check constraints
        metadata_objs = field_info.metadata
        min_len = next(
            (m.min_length for m in metadata_objs if hasattr(m, "min_length")), None
        )
        max_len = next(
            (m.max_length for m in metadata_objs if hasattr(m, "max_length")), None
        )

        self.assertEqual(min_len, 1)
        self.assertEqual(max_len, 3)

    def test_multi_pick_model_at_least(self):
        """Test model with at_least quantifier {2,}"""
        model = selection_response_model(["a", "b", "c"], quantifier=(2, None))

        field_info = model.model_fields["response"]

        # Check constraints
        metadata_objs = field_info.metadata
        min_len = next(
            (m.min_length for m in metadata_objs if hasattr(m, "min_length")), None
        )
        max_len = next(
            (m.max_length for m in metadata_objs if hasattr(m, "max_length")), None
        )

        self.assertEqual(min_len, 2)
        self.assertIsNone(max_len)  # No max constraint

    def test_multi_pick_model_zero_or_more(self):
        """Test model with * quantifier"""
        model = selection_response_model(["a", "b", "c"], quantifier=(0, None))

        field_info = model.model_fields["response"]

        # Check constraints - should allow empty list
        metadata_objs = field_info.metadata
        min_len = next(
            (m.min_length for m in metadata_objs if hasattr(m, "min_length")), None
        )
        max_len = next(
            (m.max_length for m in metadata_objs if hasattr(m, "max_length")), None
        )

        self.assertEqual(min_len, 0)
        self.assertIsNone(max_len)  # No max constraint


class QuantifierValidationTestCase(unittest.TestCase):
    """Test that Pydantic models validate correctly with quantifiers"""

    def test_validate_single_pick(self):
        """Test single pick validates correctly"""
        model = selection_response_model(["a", "b", "c"], quantifier=None)

        # Valid single selection
        instance = model(response="a")
        self.assertEqual(instance.response, "a")

        # Invalid - not in options
        with self.assertRaises(ValidationError):
            model(response="d")

    def test_validate_exact_count(self):
        """Test exact count {3} validates correctly"""
        model = selection_response_model(["a", "b", "c", "d"], quantifier=(3, 3))

        # Valid - exactly 3
        instance = model(response=["a", "b", "c"])
        self.assertEqual(len(instance.response), 3)

        # Invalid - too few
        with self.assertRaises(ValidationError):
            model(response=["a", "b"])

        # Invalid - too many
        with self.assertRaises(ValidationError):
            model(response=["a", "b", "c", "d"])

    def test_validate_range(self):
        """Test range {1,3} validates correctly"""
        model = selection_response_model(["a", "b", "c", "d"], quantifier=(1, 3))

        # Valid - 1 item
        instance = model(response=["a"])
        self.assertEqual(len(instance.response), 1)

        # Valid - 2 items
        instance = model(response=["a", "b"])
        self.assertEqual(len(instance.response), 2)

        # Valid - 3 items
        instance = model(response=["a", "b", "c"])
        self.assertEqual(len(instance.response), 3)

        # Invalid - 0 items
        with self.assertRaises(ValidationError):
            model(response=[])

        # Invalid - 4 items
        with self.assertRaises(ValidationError):
            model(response=["a", "b", "c", "d"])

    def test_validate_at_least(self):
        """Test at_least {2,} validates correctly"""
        model = selection_response_model(["a", "b", "c", "d"], quantifier=(2, None))

        # Valid - 2 items
        instance = model(response=["a", "b"])
        self.assertEqual(len(instance.response), 2)

        # Valid - 4 items
        instance = model(response=["a", "b", "c", "d"])
        self.assertEqual(len(instance.response), 4)

        # Invalid - 1 item
        with self.assertRaises(ValidationError):
            model(response=["a"])

    def test_validate_zero_or_one(self):
        """Test ? quantifier validates correctly"""
        model = selection_response_model(["a", "b", "c"], quantifier=(0, 1))

        # Valid - 0 items
        instance = model(response=[])
        self.assertEqual(len(instance.response), 0)

        # Valid - 1 item
        instance = model(response=["a"])
        self.assertEqual(len(instance.response), 1)

        # Invalid - 2 items
        with self.assertRaises(ValidationError):
            model(response=["a", "b"])

    def test_validate_invalid_options(self):
        """Test that invalid options are rejected regardless of quantifier"""
        model = selection_response_model(["a", "b", "c"], quantifier=(1, 3))

        # Invalid - contains option not in list
        with self.assertRaises(ValidationError):
            model(response=["a", "z"])

        # Invalid - all options invalid
        with self.assertRaises(ValidationError):
            model(response=["x", "y", "z"])


class QuantifierDescriptionTestCase(unittest.TestCase):
    """Test that field descriptions are generated correctly"""

    def test_description_exact(self):
        """Test description for exact count"""
        model = selection_response_model(["a", "b", "c"], quantifier=(3, 3))
        field_info = model.model_fields["response"]
        description = field_info.description

        self.assertIn("exactly 3", description)

    def test_description_range(self):
        """Test description for range"""
        model = selection_response_model(["a", "b", "c"], quantifier=(1, 3))
        field_info = model.model_fields["response"]
        description = field_info.description

        self.assertIn("between 1 and 3", description)

    def test_description_at_least(self):
        """Test description for at_least"""
        model = selection_response_model(["a", "b", "c"], quantifier=(2, None))
        field_info = model.model_fields["response"]
        description = field_info.description

        self.assertIn("at least 2", description)

    def test_description_up_to(self):
        """Test description for up_to (min=0)"""
        model = selection_response_model(["a", "b", "c"], quantifier=(0, 3))
        field_info = model.model_fields["response"]
        description = field_info.description

        self.assertIn("up to 3", description)

    def test_description_any_number(self):
        """Test description for any number (0, None)"""
        model = selection_response_model(["a", "b", "c"], quantifier=(0, None))
        field_info = model.model_fields["response"]
        description = field_info.description

        self.assertIn("any number of", description)


if __name__ == "__main__":
    unittest.main()
