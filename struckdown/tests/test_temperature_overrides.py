"""Tests for temperature and model parameter overrides in completion slots."""

import unittest

from struckdown.parsing import parse_syntax
from struckdown.return_type_models import (DefaultResponse, ExtractedResponse,
                                           IntegerResponse, PoeticalResponse,
                                           date_response_model,
                                           number_response_model,
                                           selection_response_model)


class TemperatureDefaultsTestCase(unittest.TestCase):
    """Test that response models have appropriate default temperatures."""

    def test_default_response_temperature(self):
        """DefaultResponse should have temperature 0.7"""
        self.assertEqual(DefaultResponse.llm_config.temperature, 0.7)

    def test_extract_response_temperature(self):
        """ExtractedResponse should have temperature 0.0 for deterministic extraction"""
        self.assertEqual(ExtractedResponse.llm_config.temperature, 0.0)

    def test_poem_response_temperature(self):
        """PoeticalResponse should have temperature 1.5 for high creativity"""
        self.assertEqual(PoeticalResponse.llm_config.temperature, 1.5)

    def test_integer_response_temperature(self):
        """IntegerResponse should have temperature 0.0 for deterministic numbers"""
        self.assertEqual(IntegerResponse.llm_config.temperature, 0.0)

    def test_selection_model_temperature(self):
        """Selection model should have temperature 0.0 for deterministic picking"""
        model = selection_response_model(["option1", "option2"])
        self.assertEqual(model.llm_config.temperature, 0.0)

    def test_number_model_temperature(self):
        """Number model should have temperature 0.1 for flexible extraction"""
        model = number_response_model()
        self.assertEqual(model.llm_config.temperature, 0.1)

    def test_date_model_temperature(self):
        """Date model should have temperature 0.1 for flexible temporal extraction"""
        model = date_response_model()
        self.assertEqual(model.llm_config.temperature, 0.1)


class OptionParsingTestCase(unittest.TestCase):
    """Test parsing of key=value pairs in completion slot options."""

    def test_parse_temperature_override(self):
        """Test parsing [[extract:quote|temperature=0.5]]"""
        template = "Extract this: [[extract:quote|temperature=0.5]]"
        sections = parse_syntax(template)
        prompt_part = list(sections[0].values())[0]

        self.assertEqual(prompt_part.llm_kwargs, {"temperature": 0.5})
        self.assertEqual(prompt_part.options, [])  # No plain options

    def test_parse_model_override(self):
        """Test parsing [[extract:quote|model=gpt-4o-mini]]"""
        template = "Extract this: [[extract:quote|model=gpt-4o-mini]]"
        sections = parse_syntax(template)
        prompt_part = list(sections[0].values())[0]

        self.assertEqual(prompt_part.llm_kwargs, {"model": "gpt-4o-mini"})
        self.assertEqual(prompt_part.options, [])

    def test_parse_multiple_overrides(self):
        """Test parsing [[extract:quote|temperature=0.5,model=gpt-4o-mini]]"""
        template = "Extract this: [[extract:quote|temperature=0.5,model=gpt-4o-mini]]"
        sections = parse_syntax(template)
        prompt_part = list(sections[0].values())[0]

        self.assertEqual(
            prompt_part.llm_kwargs, {"temperature": 0.5, "model": "gpt-4o-mini"}
        )
        self.assertEqual(prompt_part.options, [])

    def test_parse_mixed_options(self):
        """Test parsing [[date:when|required,temperature=0.2]]"""
        template = "When: [[date:when|required,temperature=0.2]]"
        sections = parse_syntax(template)
        prompt_part = list(sections[0].values())[0]

        self.assertEqual(prompt_part.llm_kwargs, {"temperature": 0.2})
        self.assertEqual(prompt_part.options, ["required"])  # Plain option preserved

    def test_parse_number_with_constraints_and_temp(self):
        """Test parsing [[number:score|min=0,max=100,temperature=0.0]]"""
        template = "Score: [[number:score|min=0,max=100,temperature=0.0]]"
        sections = parse_syntax(template)
        prompt_part = list(sections[0].values())[0]

        # min/max should be in options (used by number_response_model)
        # temperature should be in llm_kwargs
        self.assertEqual(prompt_part.llm_kwargs, {"temperature": 0.0})
        self.assertIn("min=0", prompt_part.options)
        self.assertIn("max=100", prompt_part.options)

    def test_temperature_validation(self):
        """Test that invalid temperatures raise errors"""
        with self.assertRaises(ValueError) as cm:
            template = "Extract: [[extract:quote|temperature=3.0]]"
            parse_syntax(template)
        self.assertIn("Invalid value for 'temperature'", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            template = "Extract: [[extract:quote|temperature=-1.0]]"
            parse_syntax(template)
        self.assertIn("Invalid value for 'temperature'", str(cm.exception))

    def test_parse_no_overrides(self):
        """Test that slots without overrides have empty llm_kwargs"""
        template = "Extract this: [[extract:quote]]"
        sections = parse_syntax(template)
        prompt_part = list(sections[0].values())[0]

        self.assertEqual(prompt_part.llm_kwargs, {})
        self.assertEqual(prompt_part.options, [])


class BackwardCompatibilityTestCase(unittest.TestCase):
    """Test that existing templates still work without changes."""

    def test_simple_completion_still_works(self):
        """Test [[var]] still parses correctly"""
        template = "Hello [[response]]"
        sections = parse_syntax(template)
        self.assertEqual(len(sections), 1)
        prompt_part = list(sections[0].values())[0]
        self.assertEqual(prompt_part.key, "response")
        self.assertEqual(prompt_part.llm_kwargs, {})

    def test_typed_completion_still_works(self):
        """Test [[extract:quote]] still parses correctly"""
        template = "Extract: [[extract:quote]]"
        sections = parse_syntax(template)
        self.assertEqual(len(sections), 1)
        prompt_part = list(sections[0].values())[0]
        self.assertEqual(prompt_part.key, "quote")
        self.assertEqual(prompt_part.action_type, "extract")
        self.assertEqual(prompt_part.llm_kwargs, {})

    def test_pick_with_options_still_works(self):
        """Test [[pick:choice|option1,option2]] still parses correctly"""
        template = "Choose: [[pick:choice|red,green,blue]]"
        sections = parse_syntax(template)
        prompt_part = list(sections[0].values())[0]
        self.assertEqual(prompt_part.options, ["red", "green", "blue"])
        self.assertEqual(prompt_part.llm_kwargs, {})

    def test_quantifier_still_works(self):
        """Test [[pick*:choices|a,b,c]] still parses correctly"""
        template = "Pick many: [[pick*:choices|a,b,c]]"
        sections = parse_syntax(template)
        prompt_part = list(sections[0].values())[0]
        self.assertEqual(prompt_part.quantifier, (0, None))
        self.assertEqual(prompt_part.options, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
