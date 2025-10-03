"""
Tests for parallel LLM calling functionality.
This file tests the parallel processing capabilities of the chatter function.
"""

import asyncio
import unittest
from collections import OrderedDict
from unittest.mock import Mock, patch

from struckdown import ChatterResult, SegmentDependencyGraph, chatter, chatter_async


class ParallelLLMCallsTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.mock_credentials = Mock()

    def test_single_segment_with_multiple_completions(self):
        """Test a single segment with multiple completion variables"""

        # Create a single segment with multiple completions
        segment = OrderedDict(
            [
                ("A", Mock(text="Think of a number", return_type=Mock(), options=[])),
                (
                    "B",
                    Mock(
                        text="Think of another number", return_type=Mock(), options=[]
                    ),
                ),
            ]
        )

        # Mock parser to return single segment
        mock_parser = Mock()
        mock_parser.return_value.parse.return_value = [segment]

        # Mock Django Template
        mock_template = Mock()
        mock_template.render.return_value = "rendered prompt"
        mock_template_class = Mock(return_value=mock_template)

        # Mock structured_chat to return predictable results
        call_count = 0

        def mock_structured_chat(prompt, llm, credentials, return_type, extra_kwargs=None):
            nonlocal call_count
            call_count += 1
            result = Mock()
            if call_count == 1:
                result.response = "7"  # First number
            elif call_count == 2:
                result.response = "15"  # Second number
            else:
                result.response = f"result_{call_count}"
            return result, Mock()

        with (
            patch("struckdown.parser", mock_parser),
            patch(
                "struckdown.structured_chat", side_effect=mock_structured_chat
            ),
            patch("struckdown.Template", mock_template_class),
        ):
            result = chatter(
                "think of a number [[A]] think of another number [[B]]",
                self.mock_llm,
                self.mock_credentials,
            )

            # Verify results
            self.assertIsInstance(result, ChatterResult)
            self.assertEqual(result["A"], "7")
            self.assertEqual(result["B"], "15")
            self.assertEqual(len(result), 2)

            # Verify keys are in correct order
            keys = list(result.keys())
            self.assertEqual(keys, ["A", "B"])

    def test_parallel_independent_segments(self):
        """Test parallel processing of independent segments preserves key order"""

        # Create independent segments
        segment1 = OrderedDict(
            [("season", Mock(text="Favorite season", return_type=Mock(), options=[]))]
        )
        segment2 = OrderedDict(
            [("city", Mock(text="Favorite city", return_type=Mock(), options=[]))]
        )

        segments = [segment1, segment2]

        # Mock parser
        mock_parser = Mock()
        mock_parser.return_value.parse.return_value = segments

        # Mock Django Template
        mock_template = Mock()
        mock_template.render.return_value = "rendered prompt"
        mock_template_class = Mock(return_value=mock_template)

        # Mock structured_chat - return unique responses so we can track which went where
        call_count = 0

        def mock_structured_chat(prompt, llm, credentials, return_type, extra_kwargs=None):
            nonlocal call_count
            call_count += 1
            result = Mock()
            result.response = f"response_{call_count}"
            return result, Mock()

        async def run_test():
            with (
                patch("struckdown.parser", mock_parser),
                patch(
                    "struckdown.structured_chat",
                    side_effect=mock_structured_chat,
                ),
                patch("struckdown.Template", mock_template_class),
            ):
                result = await chatter_async(
                    "Fav season [[season]] ¡OBLIVIATE Fav city [[city]]",
                    self.mock_llm,
                    self.mock_credentials,
                )

                # Most important: verify ordering is preserved regardless of which response went where
                self.assertIsInstance(result, ChatterResult)
                keys = list(result.keys())
                self.assertEqual(
                    keys, ["season", "city"], "Keys should be in source order"
                )

                # Verify both segments completed (don't care about specific values)
                self.assertEqual(len(result), 2)
                self.assertIn(
                    "response_", str(result["season"])
                )  # Contains some response
                self.assertIn(
                    "response_", str(result["city"])
                )  # Contains some response

        asyncio.run(run_test())

    def test_dependent_segments_sequential_execution(self):
        """Test that dependent segments execute sequentially and preserve key order"""

        # Create dependent segments
        segment1 = OrderedDict(
            [("season", Mock(text="Favorite season", return_type=Mock(), options=[]))]
        )
        segment2 = OrderedDict(
            [("city", Mock(text="Favorite city", return_type=Mock(), options=[]))]
        )
        segment3 = OrderedDict(
            [
                (
                    "poem",
                    Mock(
                        text="Describe {{city}} in {{season}}",
                        return_type=Mock(),
                        options=[],
                    ),
                )
            ]
        )

        segments = [segment1, segment2, segment3]

        # Mock parser
        mock_parser = Mock()
        mock_parser.return_value.parse.return_value = segments

        # Mock Django Template
        mock_template = Mock()
        mock_template.render.return_value = "rendered prompt"
        mock_template_class = Mock(return_value=mock_template)

        # Mock structured_chat - return unique responses to track execution
        call_count = 0

        def mock_structured_chat(prompt, llm, credentials, return_type, extra_kwargs=None):
            nonlocal call_count
            call_count += 1
            result = Mock()
            result.response = f"response_{call_count}"
            return result, Mock()

        async def run_test():
            with (
                patch("struckdown.parser", mock_parser),
                patch(
                    "struckdown.structured_chat",
                    side_effect=mock_structured_chat,
                ),
                patch("struckdown.Template", mock_template_class),
            ):
                result = await chatter_async(
                    "Fav season [[season]] ¡OBLIVIATE Fav city [[city]] ¡OBLIVIATE Describe {{city}} in {{season}} [[poem]]",
                    self.mock_llm,
                    self.mock_credentials,
                )

                # Most important: verify ordering is preserved
                self.assertIsInstance(result, ChatterResult)
                keys = list(result.keys())
                self.assertEqual(
                    keys, ["season", "city", "poem"], "Keys should be in source order"
                )

                # Verify all segments completed
                self.assertEqual(len(result), 3)
                self.assertIn("response_", str(result["season"]))
                self.assertIn("response_", str(result["city"]))
                self.assertIn("response_", str(result["poem"]))

        asyncio.run(run_test())

    def test_mixed_completion_types(self):
        """Test different completion types (speak, pick, think, etc.)"""

        # Create segment with different completion types
        segment = OrderedDict(
            [
                ("A", Mock(text="Say something funny", return_type=Mock(), options=[])),
                ("B", Mock(text="Another thing", return_type=Mock(), options=[])),
                (
                    "C",
                    Mock(
                        text="How funny are you?",
                        return_type=Mock(),
                        options=["unfunny", "abitfunny", "veryfunny"],
                    ),
                ),
                (
                    "D",
                    Mock(
                        text="What is happening here?", return_type=Mock(), options=[]
                    ),
                ),
            ]
        )

        # Mock parser
        mock_parser = Mock()
        mock_parser.return_value.parse.return_value = [segment]

        # Mock Django Template
        mock_template = Mock()
        mock_template.render.return_value = "rendered prompt"
        mock_template_class = Mock(return_value=mock_template)

        # Mock structured_chat
        call_count = 0

        def mock_structured_chat(prompt, llm, credentials, return_type, extra_kwargs=None):
            nonlocal call_count
            call_count += 1
            result = Mock()
            responses = [
                "Why did the chicken cross the road?",
                "And another funny thing...",
                "veryfunny",
                "This is a test conversation",
            ]
            if call_count <= len(responses):
                result.response = responses[call_count - 1]
            else:
                result.response = f"result_{call_count}"
            return result, Mock()

        with (
            patch("struckdown.parser", mock_parser),
            patch(
                "struckdown.structured_chat", side_effect=mock_structured_chat
            ),
            patch("struckdown.Template", mock_template_class),
        ):
            result = chatter(
                "Say something funny [[speak:A]] And another thing: [[B]] How funny are you? [[pick:C|unfunny,abitfunny,veryfunny]] Think about this [[think:D]]",
                self.mock_llm,
                self.mock_credentials,
            )

            # Verify all completion types worked
            self.assertIsInstance(result, ChatterResult)
            self.assertEqual(result["A"], "Why did the chicken cross the road?")
            self.assertEqual(result["B"], "And another funny thing...")
            self.assertEqual(result["C"], "veryfunny")
            self.assertEqual(result["D"], "This is a test conversation")
            self.assertEqual(len(result), 4)

            # Verify ordering
            keys = list(result.keys())
            self.assertEqual(keys, ["A", "B", "C", "D"])

    def test_context_variables_with_segments(self):
        """Test that context variables work correctly with segments"""

        # Create segments using context variables
        segment1 = OrderedDict(
            [
                (
                    "joke",
                    Mock(
                        text="Tell me a joke about {{topic}}",
                        return_type=Mock(),
                        options=[],
                    ),
                )
            ]
        )
        segment2 = OrderedDict(
            [
                (
                    "evaluation",
                    Mock(
                        text="Is this joke funny: {{joke}}",
                        return_type=Mock(),
                        options=[],
                    ),
                )
            ]
        )

        segments = [segment1, segment2]

        # Mock parser
        mock_parser = Mock()
        mock_parser.return_value.parse.return_value = segments

        # Mock Django Template to simulate context rendering
        mock_template = Mock()
        mock_template.render.return_value = "rendered prompt with context"
        mock_template_class = Mock(return_value=mock_template)

        # Mock structured_chat
        call_count = 0

        def mock_structured_chat(prompt, llm, credentials, return_type, extra_kwargs=None):
            nonlocal call_count
            call_count += 1
            result = Mock()
            if call_count == 1:
                result.response = (
                    "Why don't apples ever get lost? Because they have a core GPS!"
                )
            elif call_count == 2:
                result.response = "Yes, that's quite funny!"
            else:
                result.response = f"result_{call_count}"
            return result, Mock()

        with (
            patch("struckdown.parser", mock_parser),
            patch(
                "struckdown.structured_chat", side_effect=mock_structured_chat
            ),
            patch("struckdown.Template", mock_template_class),
        ):
            result = chatter(
                "Tell me a joke about {{topic}} [[joke]] ¡OBLIVIATE Is this joke funny: {{joke}} [[evaluation]]",
                self.mock_llm,
                self.mock_credentials,
                context={"topic": "apples"},
            )

            # Verify context was used
            self.assertIsInstance(result, ChatterResult)
            self.assertEqual(
                result["joke"],
                "Why don't apples ever get lost? Because they have a core GPS!",
            )
            self.assertEqual(result["evaluation"], "Yes, that's quite funny!")
            self.assertEqual(len(result), 2)

            # Verify Template.render was called with context
            self.assertTrue(mock_template.render.called)

    def test_dependency_graph_creation(self):
        """Test that the dependency graph is created correctly"""

        # Create segments with known dependencies
        segment1 = OrderedDict([("var1", Mock(text="First variable"))])
        segment2 = OrderedDict([("var2", Mock(text="Second using {{var1}}"))])
        segment3 = OrderedDict(
            [("var3", Mock(text="Third using {{var1}} and {{var2}}"))]
        )

        segments = [segment1, segment2, segment3]

        # Create dependency graph
        graph = SegmentDependencyGraph(segments)

        # Verify dependencies are detected correctly
        self.assertEqual(graph.dependency_graph["segment_0"], set())  # No dependencies
        self.assertEqual(
            graph.dependency_graph["segment_1"], {"segment_0"}
        )  # Depends on segment_0
        self.assertEqual(
            graph.dependency_graph["segment_2"], {"segment_0", "segment_1"}
        )  # Depends on both

        # Verify execution plan
        execution_plan = graph.get_execution_plan()
        expected_plan = [["segment_0"], ["segment_1"], ["segment_2"]]
        self.assertEqual(execution_plan, expected_plan)


if __name__ == "__main__":
    unittest.main()
