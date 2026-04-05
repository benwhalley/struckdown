"""Tests for thinking token exposure on SlotResult and StruckdownResult."""

import unittest

from box import Box

from struckdown.results import StruckdownResult, SlotResult


class TestSlotResultThinking(unittest.TestCase):
    def test_thinking_from_dict_completion(self):
        seg = SlotResult(
            name="answer",
            output="42",
            prompt="question",
            completion={"_thinking": "Let me reason step by step..."},
        )
        self.assertEqual(seg.thinking, "Let me reason step by step...")

    def test_thinking_from_box_completion(self):
        seg = SlotResult(
            name="answer",
            output="42",
            prompt="question",
            completion=Box({"_thinking": "Reasoning here"}),
        )
        self.assertEqual(seg.thinking, "Reasoning here")

    def test_thinking_none_when_no_completion(self):
        seg = SlotResult(name="answer", output="42", prompt="question")
        self.assertIsNone(seg.thinking)

    def test_thinking_none_when_key_missing(self):
        """Old cached results without _thinking key return None."""
        seg = SlotResult(
            name="answer",
            output="42",
            prompt="question",
            completion={"usage": {}, "_hidden_params": {}},
        )
        self.assertIsNone(seg.thinking)

    def test_thinking_none_when_stripped(self):
        seg = SlotResult(
            name="answer",
            output="42",
            prompt="question",
            completion=None,
        )
        self.assertIsNone(seg.thinking)

    def test_thinking_survives_roundtrip(self):
        seg = SlotResult(
            name="answer",
            output="42",
            prompt="question",
            completion={"_thinking": "deep thought"},
        )
        dumped = seg.model_dump()
        restored = SlotResult.model_validate(dumped)
        self.assertEqual(restored.thinking, "deep thought")


class TestStruckdownResultThinking(unittest.TestCase):
    def test_thinking_aggregates_slots(self):
        result = StruckdownResult()
        result["a"] = SlotResult(
            name="a", output="1", prompt="",
            completion={"_thinking": "thought A"},
        )
        result["b"] = SlotResult(
            name="b", output="2", prompt="",
            completion={"_thinking": "thought B"},
        )
        self.assertEqual(result.thinking, {"a": "thought A", "b": "thought B"})

    def test_thinking_excludes_none(self):
        result = StruckdownResult()
        result["a"] = SlotResult(
            name="a", output="1", prompt="",
            completion={"_thinking": "thought A"},
        )
        result["b"] = SlotResult(
            name="b", output="2", prompt="",
            completion={"usage": {}},
        )
        self.assertEqual(result.thinking, {"a": "thought A"})

    def test_thinking_empty_when_no_thinking(self):
        result = StruckdownResult()
        result["a"] = SlotResult(name="a", output="1", prompt="")
        self.assertEqual(result.thinking, {})

    def test_strip_debug_data_removes_thinking(self):
        result = StruckdownResult()
        result["a"] = SlotResult(
            name="a", output="1", prompt="q",
            completion={"_thinking": "thought A"},
        )
        stripped = result.strip_debug_data()
        self.assertIsNone(stripped.results["a"].thinking)
        self.assertEqual(stripped.thinking, {})


class TestBuildCompletionDictThinking(unittest.TestCase):
    def test_thinking_extracted_from_model_response(self):
        from struckdown.llm import _build_completion_dict

        class FakeResponse:
            usage = None
            thinking = "I need to consider..."
            def cost(self):
                return None

        result = _build_completion_dict(FakeResponse(), [])
        self.assertEqual(result["_thinking"], "I need to consider...")

    def test_empty_thinking_normalised_to_none(self):
        from struckdown.llm import _build_completion_dict

        class FakeResponse:
            usage = None
            thinking = ""
            def cost(self):
                return None

        result = _build_completion_dict(FakeResponse(), [])
        self.assertIsNone(result["_thinking"])

    def test_no_thinking_attribute(self):
        from struckdown.llm import _build_completion_dict

        class FakeResponse:
            usage = None
            def cost(self):
                return None

        result = _build_completion_dict(FakeResponse(), [])
        self.assertIsNone(result["_thinking"])

    def test_none_model_response(self):
        from struckdown.llm import _build_completion_dict

        result = _build_completion_dict(None, [])
        self.assertIsNone(result["_thinking"])


if __name__ == "__main__":
    unittest.main()
