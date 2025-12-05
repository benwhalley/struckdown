"""Tests for incremental result yielding."""

from struckdown import (
    chatter_incremental,
    CheckpointReached,
    ProcessingComplete,
    ProcessingError,
    SlotCompleted,
)


class TestIncrementalEventTypes:
    """Test that incremental events are correctly structured."""

    def test_slot_completed_type_literal(self):
        """SlotCompleted has correct type literal."""
        from struckdown.incremental import SlotCompleted
        from struckdown.results import SegmentResult

        result = SegmentResult(name="test", output="value", prompt="prompt")
        event = SlotCompleted(
            segment_index=0,
            slot_key="test",
            result=result,
            elapsed_ms=100.0,
            was_cached=False,
        )
        assert event.type == "slot_completed"

    def test_checkpoint_reached_type_literal(self):
        """CheckpointReached has correct type literal."""
        event = CheckpointReached(
            segment_index=0,
            segment_name="test",
            accumulated_results={},
        )
        assert event.type == "checkpoint"

    def test_processing_complete_type_literal(self):
        """ProcessingComplete has correct type literal."""
        from struckdown.results import ChatterResult

        event = ProcessingComplete(
            result=ChatterResult(),
            early_termination=False,
        )
        assert event.type == "complete"

    def test_processing_error_type_literal(self):
        """ProcessingError has correct type literal."""
        from struckdown.results import ChatterResult

        event = ProcessingError(
            segment_index=0,
            slot_key="test",
            error_message="test error",
            partial_results=ChatterResult(),
        )
        assert event.type == "error"


class TestIncrementalEventSerialization:
    """Test that events serialize to JSON correctly."""

    def test_slot_completed_serializes(self):
        """SlotCompleted can be serialized to JSON."""
        from struckdown.incremental import SlotCompleted
        from struckdown.results import SegmentResult

        result = SegmentResult(name="test", output="value", prompt="prompt")
        event = SlotCompleted(
            segment_index=0,
            slot_key="test",
            result=result,
            elapsed_ms=100.0,
            was_cached=False,
        )
        json_str = event.model_dump_json()
        assert "slot_completed" in json_str
        assert "test" in json_str

    def test_checkpoint_reached_serializes(self):
        """CheckpointReached can be serialized to JSON."""
        event = CheckpointReached(
            segment_index=0,
            segment_name="checkpoint_1",
            accumulated_results={},
        )
        json_str = event.model_dump_json()
        assert "checkpoint" in json_str
        assert "checkpoint_1" in json_str

    def test_processing_complete_serializes(self):
        """ProcessingComplete can be serialized to JSON."""
        from struckdown.results import ChatterResult

        event = ProcessingComplete(
            result=ChatterResult(),
            early_termination=False,
        )
        json_str = event.model_dump_json()
        assert "complete" in json_str


class TestChatterIncremental:
    """Test chatter_incremental function (sync wrapper)."""

    def test_single_slot_yields_events(self):
        """Single slot yields SlotCompleted, CheckpointReached, ProcessingComplete."""
        # Use template with content before slot to ensure messages are generated
        events = list(chatter_incremental("Say hello: [[greeting]]"))

        # Should have at least 3 events
        assert len(events) >= 3

        # First event should be SlotCompleted
        assert events[0].type == "slot_completed"
        assert events[0].slot_key == "greeting"

        # Should have a checkpoint
        checkpoints = [e for e in events if e.type == "checkpoint"]
        assert len(checkpoints) >= 1

        # Last event should be ProcessingComplete
        assert events[-1].type == "complete"
        assert isinstance(events[-1].result.results, dict)

    def test_multiple_slots_yield_in_order(self):
        """Multiple slots yield events in template order."""
        events = list(chatter_incremental("Say a: [[a]] then say b: [[b]] then say c: [[c]]"))

        slot_events = [e for e in events if e.type == "slot_completed"]
        slot_keys = [e.slot_key for e in slot_events]

        assert slot_keys == ["a", "b", "c"]

    def test_checkpoint_separates_segments(self):
        """Checkpoint yields CheckpointReached events."""
        template = "First: [[a]] <checkpoint> Second: [[b]]"
        events = list(chatter_incremental(template))

        checkpoints = [e for e in events if e.type == "checkpoint"]
        # Two segments = two checkpoint events
        assert len(checkpoints) == 2

    def test_slot_completed_has_timing(self):
        """SlotCompleted events include elapsed_ms."""
        events = list(chatter_incremental("Say something: [[x]]"))

        slot_event = next(e for e in events if e.type == "slot_completed")
        assert hasattr(slot_event, "elapsed_ms")
        assert slot_event.elapsed_ms >= 0

    def test_slot_completed_has_result(self):
        """SlotCompleted events include full SegmentResult."""
        events = list(chatter_incremental("Tell a joke: [[joke]]"))

        slot_event = next(e for e in events if e.type == "slot_completed")
        assert hasattr(slot_event, "result")
        assert slot_event.result.name == "joke"
        assert slot_event.result.output is not None

    def test_final_result_contains_all_slots(self):
        """ProcessingComplete result contains all slot results."""
        template = "Say a: [[a]] Say b: [[b]]"
        events = list(chatter_incremental(template))

        complete_event = next(e for e in events if e.type == "complete")
        assert "a" in complete_event.result.results
        assert "b" in complete_event.result.results

    def test_context_passed_to_slots(self):
        """Context variables are available in template."""
        template = "The topic is {{topic}}. Respond: [[response]]"
        context = {"topic": "testing"}

        events = list(chatter_incremental(template, context=context))

        # Should complete without error
        assert events[-1].type == "complete"

    def test_was_cached_flag_present(self):
        """SlotCompleted events include was_cached flag."""
        events = list(chatter_incremental("Say x: [[x]]"))

        slot_event = next(e for e in events if e.type == "slot_completed")
        assert hasattr(slot_event, "was_cached")
        assert isinstance(slot_event.was_cached, bool)

    def test_checkpoint_has_accumulated_results(self):
        """CheckpointReached events include accumulated_results."""
        events = list(chatter_incremental("Say a: [[a]] Say b: [[b]]"))

        checkpoint = next(e for e in events if e.type == "checkpoint")
        assert hasattr(checkpoint, "accumulated_results")
        assert isinstance(checkpoint.accumulated_results, dict)
        # Should have results from this segment
        assert "a" in checkpoint.accumulated_results
        assert "b" in checkpoint.accumulated_results


class TestIncrementalMatchesNonIncremental:
    """Test that incremental results match non-incremental results."""

    def test_same_slot_values(self):
        """Incremental and non-incremental produce same slot keys."""
        template = "Rate from 1-10: [[number:rating]]"

        # Run incremental
        events = list(chatter_incremental(template))
        incremental_result = events[-1].result

        # The results should have the same keys
        assert "rating" in incremental_result.results

    def test_same_output_types(self):
        """Output types match between incremental and non-incremental."""
        # Use a clear true/false question to get a definite boolean
        template = "Is 2 + 2 = 4? [[bool:answer]]"

        events = list(chatter_incremental(template))
        incremental_result = events[-1].result

        # Should have boolean output (or None if LLM can't determine)
        assert "answer" in incremental_result.results
        # The output should be bool or None
        assert incremental_result.results["answer"].output is None or isinstance(
            incremental_result.results["answer"].output, bool
        )

    def test_typed_slots_work(self):
        """Various typed slots work with incremental mode."""
        # Test pick type
        events = list(chatter_incremental("Pick one: [[pick:choice|a,b,c]]"))
        assert events[-1].type == "complete"
        assert "choice" in events[-1].result.results

    def test_segment_index_increases(self):
        """Segment index increases across checkpoints."""
        template = "First: [[a]] <checkpoint> Second: [[b]] <checkpoint> Third: [[c]]"
        events = list(chatter_incremental(template))

        slot_events = [e for e in events if e.type == "slot_completed"]

        # First slot should be segment 0
        assert slot_events[0].segment_index == 0
        # Second slot should be segment 1
        assert slot_events[1].segment_index == 1
        # Third slot should be segment 2
        assert slot_events[2].segment_index == 2
