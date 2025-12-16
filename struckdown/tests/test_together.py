"""Tests for <together> parallel slot processing."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from struckdown.execution import SegmentDependencyGraph
from struckdown.parsing import parse_syntax


class TestTogetherParsing:
    """Test grammar and transformer for <together> tags."""

    def test_together_basic_parsing(self):
        """<together> block parses correctly."""
        template = """
        <together>
        Question 1? [[answer1]]
        Question 2? [[answer2]]
        </together>
        """
        sections = parse_syntax(template)
        assert len(sections) == 1

        # Both slots should have the same together_group
        slot1 = sections[0]["answer1"]
        slot2 = sections[0]["answer2"]
        assert slot1.together_group is not None
        assert slot1.together_group == slot2.together_group

    def test_together_group_ids_unique(self):
        """Multiple <together> blocks get different group IDs."""
        template = """
        <together>
        [[a]]
        [[b]]
        </together>

        <together>
        [[c]]
        [[d]]
        </together>
        """
        sections = parse_syntax(template)

        group_ab = sections[0]["a"].together_group
        group_cd = sections[0]["c"].together_group

        assert group_ab is not None
        assert group_cd is not None
        assert group_ab != group_cd
        assert sections[0]["a"].together_group == sections[0]["b"].together_group
        assert sections[0]["c"].together_group == sections[0]["d"].together_group

    def test_slots_outside_together_have_no_group(self):
        """Slots outside <together> have together_group=None."""
        template = """
        [[before]]
        <together>
        [[inside]]
        </together>
        [[after]]
        """
        sections = parse_syntax(template)

        assert sections[0]["before"].together_group is None
        assert sections[0]["inside"].together_group is not None
        assert sections[0]["after"].together_group is None

    def test_together_validation_catches_invalid_reference(self):
        """Referencing a slot from the same <together> block raises error."""
        template = """
        <together>
        What is X? [[x]]
        {{x}} is it valid? [[validity]]
        </together>
        """
        with pytest.raises(ValueError, match="Invalid reference in <together> block"):
            parse_syntax(template)

    def test_together_allows_external_references(self):
        """Variables defined outside <together> can be referenced inside."""
        template = """
        What is the topic? [[topic]]

        <together>
        {{topic}} - is it clear? [[clarity]]
        {{topic}} - is it relevant? [[relevance]]
        </together>
        """
        # Should not raise
        sections = parse_syntax(template)
        assert sections[0]["clarity"].together_group is not None
        assert sections[0]["relevance"].together_group is not None

    def test_unmatched_together_close_raises(self):
        """</together> without <together> raises error."""
        template = """
        [[something]]
        </together>
        """
        with pytest.raises(ValueError, match="without matching"):
            parse_syntax(template)

    def test_together_with_typed_slots(self):
        """<together> works with typed completion slots."""
        template = """
        <together>
        Is it valid? [[bool:is_valid]]
        Rate it 1-10: [[number:rating|min=1,max=10]]
        </together>
        """
        sections = parse_syntax(template)

        assert sections[0]["is_valid"].together_group is not None
        assert sections[0]["rating"].together_group is not None
        assert (
            sections[0]["is_valid"].together_group
            == sections[0]["rating"].together_group
        )

    def test_together_preserves_slot_order(self):
        """Slots in <together> maintain their original order."""
        template = """
        <together>
        First [[first]]
        Second [[second]]
        Third [[third]]
        </together>
        """
        sections = parse_syntax(template)

        keys = list(sections[0].keys())
        # Find indices of our slots (response is auto-added at the end)
        first_idx = keys.index("first")
        second_idx = keys.index("second")
        third_idx = keys.index("third")

        assert first_idx < second_idx < third_idx

    def test_together_with_system_message(self):
        """<together> works alongside system messages."""
        template = """
        <system>You are a helpful assistant.</system>

        <together>
        [[q1]]
        [[q2]]
        </together>
        """
        sections = parse_syntax(template)

        assert sections[0]["q1"].together_group is not None
        assert sections[0]["q1"].system_message == "You are a helpful assistant."

    def test_nested_together_not_supported(self):
        """Nested <together> blocks work (each is independent)."""
        # Note: Nested together blocks aren't explicitly forbidden,
        # they just create separate groups
        template = """
        <together>
        [[outer1]]
        <together>
        [[inner1]]
        </together>
        [[outer2]]
        </together>
        """
        sections = parse_syntax(template)

        # outer1 and outer2 should be in the same group
        # inner1 should be in a different group
        outer_group = sections[0]["outer1"].together_group
        inner_group = sections[0]["inner1"].together_group
        outer2_group = sections[0]["outer2"].together_group

        # All should be in the outermost group since together is stack-based
        assert outer_group == outer2_group
        assert inner_group != outer_group

    def test_together_empty_block(self):
        """Empty <together> block parses without error."""
        template = """
        <together>
        </together>
        [[response]]
        """
        # Should not raise
        sections = parse_syntax(template)
        assert len(sections) == 1


class TestTogetherWithCheckpoints:
    """Test <together> interaction with checkpoints."""

    def test_together_before_checkpoint(self):
        """<together> works before a checkpoint."""
        template = """
        <together>
        [[a]]
        [[b]]
        </together>

        <checkpoint>

        [[after]]
        """
        sections = parse_syntax(template)

        # First segment has a and b in together group
        assert sections[0]["a"].together_group is not None
        assert sections[0]["a"].together_group == sections[0]["b"].together_group

    def test_together_after_checkpoint(self):
        """<together> works after a checkpoint."""
        template = """
        [[before]]

        <checkpoint>

        <together>
        [[a]]
        [[b]]
        </together>
        """
        sections = parse_syntax(template)

        # Second segment has a and b in together group
        assert sections[1]["a"].together_group is not None
        assert sections[1]["a"].together_group == sections[1]["b"].together_group


class TestSegmentDependencyGraph:
    """Test dependency analysis between checkpoint segments."""

    def test_independent_segments_detected(self):
        """Segments with no shared variables can run in parallel."""
        template = """
        [[summary]]

        <checkpoint>

        <together>
        [[q1]]
        [[q2]]
        </together>
        """
        segments = parse_syntax(template)
        graph = SegmentDependencyGraph(segments)
        plan = graph.get_execution_plan()

        # Both segments should be in the same batch (parallel)
        assert len(plan) == 1
        assert set(plan[0]) == {"segment_0", "segment_1"}

    def test_dependent_segments_sequential(self):
        """Segment referencing prior segment's variable must wait."""
        template = """
        [[x]]

        <checkpoint>

        {{x}} - related question [[y]]
        """
        segments = parse_syntax(template)
        graph = SegmentDependencyGraph(segments)
        plan = graph.get_execution_plan()

        # segment_1 depends on segment_0, so sequential
        assert len(plan) == 2
        assert plan[0] == ["segment_0"]
        assert plan[1] == ["segment_1"]

    def test_mixed_dependencies_three_segments(self):
        """Three segments: 0 and 1 independent, 2 depends on both."""
        template = """
        [[a]]

        <checkpoint>

        [[b]]

        <checkpoint>

        {{a}} and {{b}} combined [[c]]
        """
        segments = parse_syntax(template)
        graph = SegmentDependencyGraph(segments)
        plan = graph.get_execution_plan()

        # segment_0 and segment_1 can run in parallel, segment_2 waits
        assert len(plan) == 2
        assert set(plan[0]) == {"segment_0", "segment_1"}
        assert plan[1] == ["segment_2"]

    def test_chain_dependencies(self):
        """Each segment depends on the previous one."""
        template = """
        [[a]]

        <checkpoint>

        {{a}} leads to [[b]]

        <checkpoint>

        {{b}} leads to [[c]]
        """
        segments = parse_syntax(template)
        graph = SegmentDependencyGraph(segments)
        plan = graph.get_execution_plan()

        # All sequential: 0 -> 1 -> 2
        assert len(plan) == 3
        assert plan[0] == ["segment_0"]
        assert plan[1] == ["segment_1"]
        assert plan[2] == ["segment_2"]

    def test_together_sd_pattern(self):
        """Test pattern from examples/together.sd - independent segments."""
        # Simulates: summary in segment 0, together block in segment 1
        template = """
        <system>You are a helpful assistant.</system>

        # Student work
        > Some essay content here...

        3 line summary: [[summary]]

        <checkpoint>

        # Feedback task

        <together>
        Does it have a clear argument? [[clear_argument]]
        Does it discuss medical ethics? [[med_ethics]]
        Does it have a clear conclusion? [[clear_conclusion]]
        </together>
        """
        segments = parse_syntax(template)
        graph = SegmentDependencyGraph(segments)

        # Verify no dependencies between segments
        assert graph.dependency_graph["segment_0"] == set()
        assert graph.dependency_graph["segment_1"] == set()

        plan = graph.get_execution_plan()

        # Both can run in parallel
        assert len(plan) == 1
        assert set(plan[0]) == {"segment_0", "segment_1"}


class TestTogetherConversationHistory:
    """Test that slots after <together> blocks see the Q&A history."""

    def test_together_tags_not_exposed_to_llm(self):
        """<together> and </together> tags should never appear in LLM messages."""
        from struckdown import LLM, chatter_async

        messages_sent = []

        def mock_structured_chat(*args, **kwargs):
            messages_sent.append(kwargs.get("messages", []))
            mock_res = MagicMock()
            mock_res.response = "mock response"
            mock_res.model_dump.return_value = {"response": "mock response"}
            mock_com = {"_run_id": "test", "usage": {}}
            return mock_res, mock_com

        template = """
        <together>
        Question A? [[a]]
        Question B? [[b]]
        </together>

        Final question: [[final]]
        """

        async def run_test():
            with patch(
                "struckdown.llm.structured_chat", side_effect=mock_structured_chat
            ):
                return await chatter_async(template, model=LLM())

        asyncio.run(run_test())

        # Check no messages contain together tags
        for call_msgs in messages_sent:
            for msg in call_msgs:
                content = msg["content"]
                assert "<together>" not in content, f"<together> found in: {content}"
                assert "</together>" not in content, f"</together> found in: {content}"

    def test_slot_after_together_sees_qa_history(self):
        """A slot after </together> should see Q&A pairs in its messages."""
        from struckdown import LLM, chatter_async

        # Track messages passed to each LLM call
        call_messages = []

        def mock_structured_chat(*args, **kwargs):
            call_messages.append(kwargs.get("messages", []))
            mock_res = MagicMock()
            mock_res.response = "mock response"
            mock_res.model_dump.return_value = {"response": "mock response"}
            mock_com = {"_run_id": "test", "usage": {}}
            return mock_res, mock_com

        template = """
        <together>
        Question A? [[a]]
        Question B? [[b]]
        </together>

        Based on above, final answer: [[final]]
        """

        async def run_test():
            with patch(
                "struckdown.llm.structured_chat", side_effect=mock_structured_chat
            ):
                return await chatter_async(template, model=LLM())

        asyncio.run(run_test())

        # Should have 3 calls: a, b (parallel), then final (sequential)
        assert len(call_messages) == 3, f"Expected 3 calls, got {len(call_messages)}"

        # The final slot should see Q&A from together block in its messages
        final_messages = call_messages[2]

        # Extract user messages (questions) and assistant messages (answers)
        user_msgs = [m["content"] for m in final_messages if m["role"] == "user"]
        assistant_msgs = [
            m["content"] for m in final_messages if m["role"] == "assistant"
        ]

        # Should see both questions from together block
        assert any(
            "Question A?" in msg for msg in user_msgs
        ), f"Missing Q A in messages: {user_msgs}"
        assert any(
            "Question B?" in msg for msg in user_msgs
        ), f"Missing Q B in messages: {user_msgs}"

        # Should see both answers (mock responses)
        assert (
            len(assistant_msgs) >= 2
        ), f"Expected at least 2 assistant messages, got {assistant_msgs}"


class TestParallelSegmentGlobals:
    """Test that system/header messages propagate to parallel segments."""

    def test_parallel_segments_see_system_message(self):
        """Segments in same parallel batch should all see segment 0's system message."""
        from struckdown import LLM, chatter_async

        messages_per_call = []

        def mock_structured_chat(*args, **kwargs):
            messages_per_call.append(kwargs.get("messages", []))
            mock_res = MagicMock()
            mock_res.response = "mock response"
            mock_res.model_dump.return_value = {"response": "mock response"}
            mock_com = {"_run_id": "test", "usage": {}}
            return mock_res, mock_com

        # Segment 0 has system + header, segment 1 has only body
        # They're independent so should run in parallel
        template = """
        <system>You are a test assistant.</system>

        <header>
        # Test content
        Some important context here.
        </header>

        First question: [[q1]]

        <checkpoint>

        Second question: [[q2]]
        """

        async def run_test():
            with patch(
                "struckdown.llm.structured_chat", side_effect=mock_structured_chat
            ):
                return await chatter_async(template, model=LLM())

        asyncio.run(run_test())

        # Should have 2 calls (one per segment)
        assert (
            len(messages_per_call) == 2
        ), f"Expected 2 calls, got {len(messages_per_call)}"

        # BOTH calls should have the system message
        for i, msgs in enumerate(messages_per_call):
            system_msgs = [m for m in msgs if m.get("role") == "system"]
            assert len(system_msgs) >= 1, f"Call {i} missing system message"
            assert (
                "test assistant" in system_msgs[0]["content"]
            ), f"Call {i} has wrong system message"

        # BOTH calls should have the header content
        for i, msgs in enumerate(messages_per_call):
            all_content = " ".join(m.get("content", "") for m in msgs)
            assert (
                "important context" in all_content
            ), f"Call {i} missing header content"


class TestParallelExecutionTiming:
    """Test that parallel slots actually start simultaneously."""

    def test_together_slots_start_within_20ms(self):
        """All slots in a together block should start within 20ms of each other."""
        from struckdown import LLM, chatter_async

        # Track when each LLM call starts
        call_times = []

        def mock_structured_chat(*args, **kwargs):
            call_times.append(time.monotonic())
            # Return a mock response
            mock_res = MagicMock()
            mock_res.response = "mock response"
            mock_res.model_dump.return_value = {"response": "mock response"}
            mock_com = {"_run_id": "test", "usage": {}}
            return mock_res, mock_com

        template = """
        <system>You are helpful.</system>

        # Content
        Some content here.

        <checkpoint>

        # Summary
        [[summary]]

        <checkpoint>

        # Questions
        <together>
        Question 1? [[q1]]
        Question 2? [[q2]]
        Question 3? [[q3]]
        </together>
        """

        async def run_test():
            with patch(
                "struckdown.llm.structured_chat", side_effect=mock_structured_chat
            ):
                return await chatter_async(template, model=LLM())

        asyncio.run(run_test())

        # Should have 4 calls: summary + 3 together slots
        assert len(call_times) == 4, f"Expected 4 calls, got {len(call_times)}"

        # All calls should start within 20ms of each other
        min_time = min(call_times)
        max_time = max(call_times)
        spread_ms = (max_time - min_time) * 1000

        assert spread_ms < 20, f"Calls spread over {spread_ms:.1f}ms, expected < 20ms"

    def test_independent_segments_start_within_20ms(self):
        """Independent checkpoint segments should start within 20ms of each other."""
        from struckdown import LLM, chatter_async

        call_times = []

        def mock_structured_chat(*args, **kwargs):
            call_times.append(time.monotonic())
            mock_res = MagicMock()
            mock_res.response = "mock response"
            mock_res.model_dump.return_value = {"response": "mock response"}
            mock_com = {"_run_id": "test", "usage": {}}
            return mock_res, mock_com

        # Three independent segments (no variable references between them)
        template = """
        First segment [[a]]

        <checkpoint>

        Second segment [[b]]

        <checkpoint>

        Third segment [[c]]
        """

        async def run_test():
            with patch(
                "struckdown.llm.structured_chat", side_effect=mock_structured_chat
            ):
                return await chatter_async(template, model=LLM())

        asyncio.run(run_test())

        # Should have 3 calls
        assert len(call_times) == 3, f"Expected 3 calls, got {len(call_times)}"

        # All calls should start within 20ms of each other
        min_time = min(call_times)
        max_time = max(call_times)
        spread_ms = (max_time - min_time) * 1000

        assert spread_ms < 20, f"Calls spread over {spread_ms:.1f}ms, expected < 20ms"

    def test_dependent_segments_run_sequentially(self):
        """Segments with dependencies should NOT start simultaneously."""
        from struckdown import LLM, chatter_async

        call_times = []

        def mock_structured_chat(*args, **kwargs):
            call_times.append(time.monotonic())
            # Add small delay to make timing measurable
            time.sleep(0.05)
            mock_res = MagicMock()
            mock_res.response = "mock response"
            mock_res.model_dump.return_value = {"response": "mock response"}
            mock_com = {"_run_id": "test", "usage": {}}
            return mock_res, mock_com

        # Second segment depends on first (references {{a}})
        template = """
        First: [[a]]

        <checkpoint>

        Second uses {{a}}: [[b]]
        """

        async def run_test():
            with patch(
                "struckdown.llm.structured_chat", side_effect=mock_structured_chat
            ):
                return await chatter_async(template, model=LLM())

        asyncio.run(run_test())

        assert len(call_times) == 2, f"Expected 2 calls, got {len(call_times)}"

        # Second call should start AFTER first completes (at least 50ms apart)
        time_diff_ms = (call_times[1] - call_times[0]) * 1000
        assert (
            time_diff_ms >= 40
        ), f"Dependent segment started too soon: {time_diff_ms:.1f}ms apart"
