"""
Tests for optional completion placeholder in final segments.

The final segment of a template can omit the completion placeholder,
which will automatically be replaced with [[response]]. Non-final segments
still require explicit completion placeholders.
"""

import unittest

from struckdown.parsing import parse_syntax


class OptionalCompletionTestCase(unittest.TestCase):
    """Test optional completion placeholder functionality"""

    def test_parse_single_segment_no_completion(self):
        """Test parsing a single segment without completion placeholder"""
        template = "What is 2+2?"
        sections = parse_syntax(template)

        # Should have 1 section with 1 completion
        self.assertEqual(len(sections), 1)
        # The completion should be 'response' (auto-added)
        self.assertIn("response", sections[0])

        part = sections[0]["response"]
        self.assertEqual(part.text, "What is 2+2?")

    def test_parse_single_segment_with_explicit_completion(self):
        """Test parsing with explicit completion (backward compatibility)"""
        template = "What is 2+2? [[answer]]"
        sections = parse_syntax(template)

        # Should have 1 section with 1 completion
        self.assertEqual(len(sections), 1)
        # The completion should be 'answer' (as specified)
        self.assertIn("answer", sections[0])

        part = sections[0]["answer"]
        self.assertEqual(part.text, "What is 2+2?")

    def test_parse_multiple_segments_final_no_completion(self):
        """Test parsing multiple segments where final has no completion"""
        template = """Generate a topic [[topic]]

<checkpoint>

Write an essay about {{topic}}"""

        sections = parse_syntax(template)

        # Should have 2 sections
        self.assertEqual(len(sections), 2)

        # First section should have 'topic'
        self.assertIn("topic", sections[0])
        self.assertEqual(sections[0]["topic"].text, "Generate a topic")

        # Second section should have 'response' (auto-added)
        self.assertIn("response", sections[1])
        self.assertIn("{{topic}}", sections[1]["response"].text)

    def test_parse_multiple_segments_all_explicit(self):
        """Test backward compatibility with explicit completions in all segments"""
        template = """Generate a topic [[topic]]

<checkpoint>

Write an essay about {{topic}} [[essay]]"""

        sections = parse_syntax(template)

        # Should have 2 sections
        self.assertEqual(len(sections), 2)

        # First section should have 'topic'
        self.assertIn("topic", sections[0])

        # Second section should have 'essay' (as specified)
        self.assertIn("essay", sections[1])

    def test_parse_with_typed_completion(self):
        """Test that explicit typed completions work as before"""
        template = "Rate this from 1-10 [[int:rating]]"
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 1)
        self.assertIn("rating", sections[0])

        part = sections[0]["rating"]
        self.assertEqual(part.action_type, "int")

    def test_parse_final_segment_with_variables(self):
        """Test final segment with template variables and no completion"""
        template = """What is your name? [[name]]

<checkpoint>

Hello {{name}}, how are you?"""

        sections = parse_syntax(template)

        self.assertEqual(len(sections), 2)
        self.assertIn("name", sections[0])
        self.assertIn("response", sections[1])

        # Check that the template variable is preserved
        self.assertIn("{{name}}", sections[1]["response"].text)

    def test_parse_with_shared_header_no_completion(self):
        """Test system message with no completion in final segment"""
        template = """<system>You are a helpful assistant.</system>

Tell me a joke"""

        sections = parse_syntax(template)

        self.assertEqual(len(sections), 1)
        self.assertIn("response", sections[0])

        part = sections[0]["response"]
        self.assertEqual(part.system_message, "You are a helpful assistant.")
        self.assertEqual(part.text, "Tell me a joke")

    def test_parse_complex_multi_segment(self):
        """Test complex scenario with multiple segments and mixed completions"""
        # Use default+ instead of pick+ since pick requires options to select from
        template = """Generate 3 random topics [[default+:topics]]

<checkpoint>

Pick one topic from {{topics}} [[chosen]]

<checkpoint>

Write a paragraph about {{chosen}}"""

        sections = parse_syntax(template)

        # Should have 3 sections
        self.assertEqual(len(sections), 3)

        # First has explicit completion
        self.assertIn("topics", sections[0])

        # Second has explicit completion
        self.assertIn("chosen", sections[1])

        # Third gets auto-added 'response'
        self.assertIn("response", sections[2])
        self.assertIn("{{chosen}}", sections[2]["response"].text)

    def test_whitespace_handling(self):
        """Test that various whitespace scenarios work correctly"""
        # Test with trailing whitespace
        template1 = "Tell me something   "
        sections1 = parse_syntax(template1)
        self.assertIn("response", sections1[0])

        # Test with leading whitespace
        template2 = "   Tell me something"
        sections2 = parse_syntax(template2)
        self.assertIn("response", sections2[0])

        # Test with internal newlines
        template3 = """Tell me something

With multiple lines"""
        sections3 = parse_syntax(template3)
        self.assertIn("response", sections3[0])
        self.assertIn("With multiple lines", sections3[0]["response"].text)

    def test_list_completion_with_quantifier(self):
        """Test that list completions work with quantifier syntax"""
        template = "Generate 5 ideas [[default{5}:idea]]"
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 1)
        self.assertIn("idea", sections[0])

        part = sections[0]["idea"]
        # Should have quantifier (5, 5) for exactly 5 items
        self.assertEqual(part.quantifier, (5, 5))

    def test_options_backward_compatibility(self):
        """Test that completion options still work"""
        template = "Pick a color [[pick:color|red,blue,green]]"
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 1)
        self.assertIn("color", sections[0])

        part = sections[0]["color"]
        self.assertEqual(part.action_type, "pick")
        self.assertIn("red", part.options)
        self.assertIn("blue", part.options)
        self.assertIn("green", part.options)


class NonFinalSegmentValidationTestCase(unittest.TestCase):
    """Test that non-final segments without completions are allowed (for context setup)"""

    def test_non_final_segment_without_completion_allowed(self):
        """Test that non-final segments without completions are allowed.

        This is valid for segments that just set up context (e.g., <system> tags).
        """
        # This template has a non-final segment without completion
        template = """First task without completion

<checkpoint>

Second task [[second]]"""

        # This should now succeed - non-final segments without completions are allowed
        sections = parse_syntax(template)
        # Should have parsed successfully
        self.assertIsNotNone(sections)

    def test_middle_segment_without_completion_allowed(self):
        """Test that middle segments without completions are allowed."""
        template = """First task [[first]]

<checkpoint>

Second task without completion

<checkpoint>

Third task [[third]]"""

        # This should now succeed - middle segments can be context-only
        sections = parse_syntax(template)
        self.assertIsNotNone(sections)


if __name__ == "__main__":
    unittest.main()
