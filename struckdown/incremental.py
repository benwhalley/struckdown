"""Incremental result events for struckdown.

This module provides event types for incremental (slot-by-slot) result yielding.
Events are yielded as each slot completes, allowing consumers to display
progress in real-time.

Terminology:
- Incremental: Slot results yielded as they complete (this module)
- Streaming: Token-by-token LLM output in real-time (future Phase 2)
"""

from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from .results import ChatterResult, SegmentResult


class SlotCompleted(BaseModel):
    """Emitted when a slot is filled (LLM completion or action execution)."""

    type: Literal["slot_completed"] = "slot_completed"
    segment_index: int
    slot_key: str
    result: SegmentResult
    elapsed_ms: float
    was_cached: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CheckpointReached(BaseModel):
    """Emitted when a <checkpoint> boundary is crossed."""

    type: Literal["checkpoint"] = "checkpoint"
    segment_index: int
    segment_name: Optional[str] = None
    accumulated_results: Dict[str, SegmentResult]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProcessingComplete(BaseModel):
    """Final event with aggregated results."""

    type: Literal["complete"] = "complete"
    result: ChatterResult
    early_termination: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProcessingError(BaseModel):
    """Emitted when an error occurs during processing."""

    type: Literal["error"] = "error"
    segment_index: int
    slot_key: Optional[str] = None
    error_message: str
    partial_results: ChatterResult

    model_config = ConfigDict(arbitrary_types_allowed=True)


IncrementalEvent = Union[SlotCompleted, CheckpointReached, ProcessingComplete, ProcessingError]
