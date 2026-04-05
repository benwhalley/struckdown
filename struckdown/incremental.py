"""Incremental result events for struckdown.

This module provides event types for incremental (slot-by-slot) result yielding
and token-by-token streaming for free-form text slots.

Event flow for a streaming free-text slot:
  SlotStreamStart → TokenDelta → TokenDelta → ... → SlotCompleted

Event flow for a constrained or non-streaming slot:
  SlotCompleted
"""

from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from .results import StruckdownResult, SlotResult


class SlotCompleted(BaseModel):
    """Emitted when a slot is filled (LLM completion or action execution)."""

    type: Literal["slot_completed"] = "slot_completed"
    segment_index: int
    slot_key: str
    result: SlotResult
    elapsed_ms: float
    was_cached: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CheckpointReached(BaseModel):
    """Emitted when a <checkpoint> boundary is crossed."""

    type: Literal["checkpoint"] = "checkpoint"
    segment_index: int
    segment_name: Optional[str] = None
    accumulated_results: Dict[str, SlotResult]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProcessingComplete(BaseModel):
    """Final event with aggregated results."""

    type: Literal["complete"] = "complete"
    result: StruckdownResult
    early_termination: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SlotStreamStart(BaseModel):
    """Emitted when a streamable (free-text) slot begins generating."""

    type: Literal["stream_start"] = "stream_start"
    segment_index: int
    slot_key: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TokenDelta(BaseModel):
    """Emitted for each token chunk during streaming of a free-text slot."""

    type: Literal["token_delta"] = "token_delta"
    segment_index: int
    slot_key: str
    delta: str
    accumulated: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProcessingError(BaseModel):
    """Emitted when an error occurs during processing."""

    type: Literal["error"] = "error"
    segment_index: int
    slot_key: Optional[str] = None
    error_message: str
    partial_results: StruckdownResult

    model_config = ConfigDict(arbitrary_types_allowed=True)


IncrementalEvent = Union[
    SlotCompleted, SlotStreamStart, TokenDelta,
    CheckpointReached, ProcessingComplete, ProcessingError,
]
