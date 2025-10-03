import asyncio
from collections import OrderedDict
from unittest.mock import Mock, patch

from chatter import ChatterResult
from django.test import TestCase

from llmtools.llm_calling import chatter, chatter_async
from llmtools.models import LLM, LLMCredentials
from llmtools.return_type_models import ExtractedResponse


class ChatterOrderingTestCase(TestCase):
    def setUp(self):
        self.llm = LLM.objects.create(model_name="test-model")
        self.credentials = LLMCredentials.objects.create(
            llm_api_key="test-key", llm_base_url="http://test.com"
        )

    def test_chatter_preserves_key_order_single_segment(self):
        """Test that chatter preserves the order of keys as they appear in the prompt"""

        # Mock the parser to return a single segment with multiple completions
        mock_segment = OrderedDict(
            [
                (
                    "first",
                    Mock(
                        text="First completion",
                        return_type=ExtractedResponse,
                        options=[],
                    ),
                ),
                (
                    "second",
                    Mock(
                        text="Second completion",
                        return_type=ExtractedResponse,
                        options=[],
                    ),
                ),
                (
                    "third",
                    Mock(
                        text="Third completion",
                        return_type=ExtractedResponse,
                        options=[],
                    ),
                ),
            ]
        )

        # Mock structured_chat to return predictable responses
        def mock_structured_chat(prompt, llm, credentials, return_type, extra_body={}):
            mock_response = Mock()
            mock_response.response = f"Response for {return_type}"
            return mock_response, Mock()

        with (
            patch("llmtools.llm_calling.parser") as mock_parser,
            patch(
                "llmtools.llm_calling.structured_chat", side_effect=mock_structured_chat
            ),
            patch("llmtools.llm_calling.Template") as mock_template_class,
        ):
            # Setup mocks
            mock_parser.return_value.parse.return_value = [mock_segment]
            mock_template = Mock()
            mock_template.render.return_value = "rendered prompt"
            mock_template_class.return_value = mock_template

            # Call chatter
            result = chatter(
                "First [[first]] Second [[second]] Third [[third]]",
                self.llm,
                self.credentials,
            )

            # Verify results
            self.assertIsInstance(result, ChatterResult)
            keys = list(result.keys())
            self.assertEqual(keys, ["first", "second", "third"])
            self.assertEqual(len(result), 3)

    def test_chatter_preserves_key_order_parallel_segments(self):
        """Test that chatter preserves key order even when segments are processed in parallel"""

        # Create mock segments that would complete in different orders
        segment1 = OrderedDict(
            [
                (
                    "apple",
                    Mock(text="Apple text", return_type=ExtractedResponse, options=[]),
                )
            ]
        )
        segment2 = OrderedDict(
            [
                (
                    "banana",
                    Mock(text="Banana text", return_type=ExtractedResponse, options=[]),
                )
            ]
        )
        segment3 = OrderedDict(
            [
                (
                    "cherry",
                    Mock(
                        text="Cherry {{apple}} {{banana}}",
                        return_type=ExtractedResponse,
                        options=[],
                    ),
                )
            ]
        )

        mock_segments = [segment1, segment2, segment3]

        # Mock structured_chat to simulate different completion times
        call_count = 0

        def mock_structured_chat(prompt, llm, credentials, return_type, extra_body={}):
            nonlocal call_count
            call_count += 1
            mock_response = Mock()

            # Simulate responses that might complete out of order
            if "Apple" in prompt:
                mock_response.response = "apple_result"
            elif "Banana" in prompt:
                mock_response.response = "banana_result"
            elif "Cherry" in prompt:
                mock_response.response = "cherry_result"
            else:
                mock_response.response = f"result_{call_count}"

            return mock_response, Mock()

        async def run_test():
            with (
                patch("llmtools.llm_calling.parser") as mock_parser,
                patch(
                    "llmtools.llm_calling.structured_chat",
                    side_effect=mock_structured_chat,
                ),
                patch("llmtools.llm_calling.Template") as mock_template_class,
            ):
                # Setup mocks
                mock_parser.return_value.parse.return_value = mock_segments
                mock_template = Mock()
                mock_template.render.return_value = "rendered prompt"
                mock_template_class.return_value = mock_template

                # Call chatter_async to test parallel processing
                result = await chatter_async(
                    "Apple [[apple]] ¡OBLIVIATE Banana [[banana]] ¡OBLIVIATE Cherry [[cherry]]",
                    self.llm,
                    self.credentials,
                )

                # Verify results are in the correct order
                self.assertIsInstance(result, ChatterResult)
                keys = list(result.keys())
                # Keys should be in the order they appear in the original prompt
                self.assertEqual(keys, ["apple", "banana", "cherry"])
                self.assertEqual(len(result), 3)

                # Verify the values
                self.assertEqual(result["apple"], "apple_result")
                self.assertEqual(result["banana"], "banana_result")
                self.assertEqual(result["cherry"], "cherry_result")

        # Run the async test
        asyncio.run(run_test())

    def test_chatter_result_properties(self):
        """Test ChatterResult helper properties"""
        result = ChatterResult()
        result["first"] = "first_value"
        result["second"] = "second_value"
        result["last"] = "last_value"

        # Test .response property returns the last item
        self.assertEqual(result.response, "last_value")

        # Test .outputs property returns a Box object
        outputs = result.outputs
        self.assertEqual(outputs.first, "first_value")
        self.assertEqual(outputs.second, "second_value")
        self.assertEqual(outputs.last, "last_value")

    def test_chatter_with_context(self):
        """Test that context variables are properly passed through"""

        mock_segment = OrderedDict(
            [
                (
                    "greeting",
                    Mock(
                        text="Hello {{name}}", return_type=ExtractedResponse, options=[]
                    ),
                )
            ]
        )

        def mock_structured_chat(prompt, llm, credentials, return_type, extra_body={}):
            mock_response = Mock()
            # Check that the context was properly rendered
            if "Hello Alice" in prompt:
                mock_response.response = "Hello Alice, nice to meet you!"
            else:
                mock_response.response = "Generic greeting"
            return mock_response, Mock()

        with (
            patch("llmtools.llm_calling.parser") as mock_parser,
            patch(
                "llmtools.llm_calling.structured_chat", side_effect=mock_structured_chat
            ),
            patch("llmtools.llm_calling.Template") as mock_template_class,
        ):
            # Setup mocks
            mock_parser.return_value.parse.return_value = [mock_segment]
            mock_template = Mock()
            # Simulate template rendering with context
            mock_template.render.return_value = "Hello Alice"
            mock_template_class.return_value = mock_template

            # Call chatter with context
            result = chatter(
                "Hello {{name}} [[greeting]]",
                self.llm,
                self.credentials,
                context={"name": "Alice"},
            )

            # Verify the context was used
            self.assertEqual(result["greeting"], "Hello Alice, nice to meet you!")
