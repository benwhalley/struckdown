# Phase 1: Chunked Segment Responses

**Goal:** Yield `SegmentResult` objects as each slot completes, rather than waiting for all segments to finish before returning.

## Current Architecture

### Data Flow (current)

```
chatter_async(template, context)
    |
    v
resolve_includes() --> handle <include 'file.sd'/> tags
    |
    v
split_by_checkpoint() --> list of (segment_text, segment_name) tuples
    |
    v
FOR each segment:
    |
    +-- extract_system_message() --> separate system from body
    +-- analyze_template() --> find conditional triggers
    |
    v
    process_segment_with_delta(body, context, ...)
        |
        v
        LOOP:
            find_slots_with_positions() --> all [[slot]] in rendered output
            find first unfilled slot
            IF none: break

            extract delta (text between last filled slot and current)

            IF action with _executor:
                call executor function
            ELSE:
                structured_chat() --> blocking LLM call

            store in results[slot_key]
            update accumulated_context

            IF triggers_rerender(slot_key):
                re-render template (positions shift!)
        |
        v
        RETURN ChatterResult
    |
    v
    final.update(segment_result)
    update accumulated_context for next segment
|
v
RETURN final ChatterResult
```

### Key Files

| File | Role |
|------|------|
| `__init__.py` | `chatter_async()` - main entry point, orchestrates segments |
| `segment_processor.py` | `process_segment_with_delta()` - handles slot-by-slot processing |
| `results.py` | `SegmentResult`, `ChatterResult`, progress callbacks |
| `execution.py` | `process_single_segment_()` - older path, not currently used by chatter |

---

## Proposed Design

### New Public API

```python
# __init__.py

from typing import AsyncGenerator, Generator

async def chatter_stream_async(
    multipart_prompt: str,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    context: dict = {},
    extra_kwargs: dict = None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Process a struckdown template, yielding events as slots complete.

    Same parameters as chatter_async(), but yields StreamEvent objects
    instead of returning a single ChatterResult.

    Events are yielded in processing order:
    - SlotCompleted: after each slot is filled (LLM call or action)
    - CheckpointReached: after all slots in a segment complete
    - ProcessingComplete: final event with aggregated ChatterResult

    Example:
        async for event in chatter_stream_async(template):
            if event.type == "slot_completed":
                print(f"{event.slot_key}: {event.result.output}")
    """
    ...

def chatter_stream(
    multipart_prompt: str,
    ...
) -> Generator[StreamEvent, None, None]:
    """Synchronous wrapper for chatter_stream_async."""
    ...
```

### Event Types

```python
# streaming.py (new file)

from dataclasses import dataclass
from typing import Literal, Union, Dict, Optional
from .results import SegmentResult, ChatterResult

@dataclass
class SlotCompleted:
    """Emitted when a slot is filled (LLM completion or action execution)."""
    type: Literal["slot_completed"] = "slot_completed"
    segment_index: int          # which checkpoint segment (0-based)
    slot_key: str               # the variable name
    result: SegmentResult       # full result with output, prompt, messages, etc.
    elapsed_ms: float           # time for this slot
    was_cached: bool            # true if result came from cache

@dataclass
class CheckpointReached:
    """Emitted when a <checkpoint> boundary is crossed."""
    type: Literal["checkpoint"] = "checkpoint"
    segment_index: int
    segment_name: Optional[str]                  # name if <checkpoint name="foo">
    accumulated_results: Dict[str, SegmentResult]  # all results so far

@dataclass
class ProcessingComplete:
    """Final event with aggregated results."""
    type: Literal["complete"] = "complete"
    result: ChatterResult
    early_termination: bool = False  # True if [[@break]] or [[end]] triggered

@dataclass
class ProcessingError:
    """Emitted when an error occurs (slot still fails, but partial results available)."""
    type: Literal["error"] = "error"
    segment_index: int
    slot_key: Optional[str]     # None if error before slot processing
    error: Exception
    partial_results: ChatterResult

StreamEvent = Union[SlotCompleted, CheckpointReached, ProcessingComplete, ProcessingError]
```

### Implementation Approach

**Option A: Create parallel generator function**

Create `process_segment_with_delta_streaming()` as a new async generator that mirrors `process_segment_with_delta()` but yields instead of accumulating.

Pros:
- Clean separation, no risk of breaking existing code
- Can optimise streaming path independently

Cons:
- Code duplication with `process_segment_with_delta()`
- Two code paths to maintain

**Option B: Refactor existing function to support both modes**

Add a `streaming: bool = False` parameter to `process_segment_with_delta()`. When True, yield events; when False, accumulate and return.

Pros:
- Single source of truth
- Changes benefit both paths

Cons:
- More complex function signature
- Risk of regressions

**Option C: Make non-streaming consume streaming internally**

Refactor `process_segment_with_delta()` to always be a generator, then have the non-streaming path consume it:

```python
async def process_segment_with_delta(...) -> ChatterResult:
    """Non-streaming: consume generator and return final result."""
    result = ChatterResult()
    async for event in process_segment_with_delta_streaming(...):
        if isinstance(event, SlotCompleted):
            result[event.slot_key] = event.result
    return result

async def process_segment_with_delta_streaming(...) -> AsyncGenerator[StreamEvent, None]:
    """Streaming: yield events as slots complete."""
    # ... actual implementation ...
```

Pros:
- Single implementation
- Guarantees parity between streaming and non-streaming

Cons:
- Small overhead for non-streaming case (generator machinery)
- Need to ensure non-streaming tests still pass

**Decision: Option C** -- single implementation, non-streaming consumes streaming.

---

## Identified Difficulties

### 1. Re-rendering and Position Shifts

**Problem:** When a slot triggers conditional re-rendering (`analysis.triggers_rerender(slot_key)`), the template is re-rendered and slot positions change. The consumer might have already received events based on old positions.

**Current code (segment_processor.py:290-296):**
```python
if analysis.triggers_rerender(slot_key):
    logger.debug(f"Slot {slot_key} triggers re-render")
    rendered = render_template(body_template, accumulated_context)
    # Reset last_slot_end since positions changed
    last_slot_end = 0
```

**Impact on streaming:**
- Consumer has no way to know slots might appear/disappear after a re-render
- If conditional adds new slots, they'll appear in subsequent events
- If conditional removes slots, they simply won't be yielded

**Proposed solution:**
- This is actually fine for chunked streaming -- we yield completed slots, not pending ones
- Consumer only sees slots that actually get filled
- Could optionally add a `ReRenderOccurred` event if useful for debugging

### 2. Early Termination (`StruckdownEarlyTermination`)

**Problem:** `[[@break]]` and `[[end]]` raise `StruckdownEarlyTermination` exception with `partial_results`. How should this work with streaming?

**Current code (execution.py:336-342, segment_processor doesn't have this):**
```python
if accumulated_context.get('_break_requested'):
    raise StruckdownEarlyTermination(
        f"Break requested: {break_msg}",
        partial_results=results
    )
```

**Note:** The `segment_processor.py` version doesn't handle `[[@break]]` -- this is only in `execution.py:process_single_segment_()`. Need to verify which code path handles this.

**Proposed solution:**
- Yield all completed slots before the break
- Yield a `ProcessingComplete` event with `early_termination=True` flag
- Don't raise exception -- the generator just stops

### 3. Error Handling Mid-Stream

**Problem:** If slot 3 of 5 fails, what happens?

**Current behaviour:** Exception raised, no partial results (except via `StruckdownEarlyTermination`).

**Proposed solution:**
- Yield `ProcessingError` event with partial results
- Let consumer decide whether to continue or stop
- Generator stops after error event

### 4. Two Processing Paths

**Problem:** There are two different processing functions:
- `segment_processor.py:process_segment_with_delta()` -- used by `chatter_async()`
- `execution.py:process_single_segment_()` -- older implementation

**Impact:** Only need to modify `process_segment_with_delta()` since that's what `chatter_async()` uses.

### 5. Progress Callback Integration

**Current:** `progress_tracking()` context manager fires callback after each slot.

**Question:** Should streaming replace this, or coexist?

**Proposed solution:** Coexist -- streaming is the new way, but progress callbacks still work for non-streaming callers. Internally, both can share the same yield/callback point.

### 6. Synchronous Wrapper Complexity

**Problem:** `chatter_stream()` needs to be a synchronous generator wrapping an async generator.

**Implementation:**
```python
def chatter_stream(...) -> Generator[StreamEvent, None, None]:
    """Synchronous wrapper."""
    async def collect():
        events = []
        async for event in chatter_stream_async(...):
            events.append(event)
        return events

    # Run async generator to completion, yield collected events
    events = anyio.run(collect)
    yield from events
```

**Note:** This doesn't provide true streaming in sync context -- events are collected then yielded. True streaming requires async. This is a limitation we should document.

**Alternative:** Use `anyio.from_thread.run_sync()` pattern if caller is in a thread with running event loop.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Event granularity | Slot-level only | Simpler; SegmentStarted slot counts misleading due to re-rendering |
| Error handling | Stop immediately | Yield `ProcessingError` with partial results, generator ends |
| Non-streaming refactor | Single implementation | `chatter_async()` consumes streaming generator internally |
| Cache indication | Include `was_cached` | Useful for debugging and UI |

---

## Implementation Plan

### Step 1: Create streaming.py with event types
- Define `StreamEvent` union type
- Define `SlotCompleted`, `CheckpointReached`, `ProcessingComplete`, `ProcessingError`
- Export from `__init__.py`

### Step 2: Create process_segment_with_delta_streaming()
- Async generator version of `process_segment_with_delta()`
- Yields `SlotCompleted` after each slot
- Returns nothing (generator)

### Step 3: Refactor process_segment_with_delta() to use streaming
- Make it consume `process_segment_with_delta_streaming()` internally
- Verify all existing tests pass

### Step 4: Create chatter_stream_async()
- Orchestrates segments like `chatter_async()`
- Yields events from each segment
- Yields `CheckpointReached` between segments
- Yields `ProcessingComplete` at end

### Step 5: Create chatter_stream() sync wrapper
- Document limitation (not truly streaming in sync context)

### Step 6: Add tests
- Test event ordering
- Test checkpoint boundaries
- Test error handling
- Test early termination

### Step 7: Update __init__.py exports
- Add `chatter_stream`, `chatter_stream_async`
- Add event types to `__all__`

---

## Test Cases

```python
# tests/test_chunking.py

import pytest
from struckdown import chatter_stream_async, SlotCompleted, CheckpointReached, ProcessingComplete

@pytest.mark.asyncio
async def test_single_slot_yields_three_events():
    """Single slot: SlotCompleted, CheckpointReached, ProcessingComplete"""
    events = [e async for e in chatter_stream_async("[[greeting]]")]

    assert len(events) == 3
    assert events[0].type == "slot_completed"
    assert events[0].slot_key == "greeting"
    assert events[1].type == "checkpoint"
    assert events[2].type == "complete"

@pytest.mark.asyncio
async def test_multiple_slots_yield_in_order():
    """Multiple slots yield in template order"""
    events = [e async for e in chatter_stream_async("[[a]] then [[b]] then [[c]]")]

    slot_events = [e for e in events if e.type == "slot_completed"]
    assert [e.slot_key for e in slot_events] == ["a", "b", "c"]

@pytest.mark.asyncio
async def test_checkpoint_separates_segments():
    """Checkpoint yields CheckpointReached between segments"""
    template = "[[a]] <checkpoint> [[b]]"
    events = [e async for e in chatter_stream_async(template)]

    checkpoints = [e for e in events if e.type == "checkpoint"]
    assert len(checkpoints) == 2  # one after each segment

@pytest.mark.asyncio
async def test_final_result_matches_non_streaming():
    """Final ChatterResult matches what chatter_async returns"""
    template = "[[joke]] then [[rating:number]]"

    # Streaming
    events = [e async for e in chatter_stream_async(template)]
    streaming_result = events[-1].result

    # Non-streaming
    from struckdown import chatter_async
    direct_result = await chatter_async(template)

    assert streaming_result.keys() == direct_result.keys()

@pytest.mark.asyncio
async def test_elapsed_time_is_reasonable():
    """Elapsed time is measured for each slot"""
    events = [e async for e in chatter_stream_async("[[x]]")]

    slot_event = events[0]
    assert slot_event.elapsed_ms > 0
    assert slot_event.elapsed_ms < 60000  # less than 60 seconds

@pytest.mark.asyncio
async def test_conditional_slots_only_yield_if_rendered():
    """Slots inside false conditionals don't yield events"""
    template = """
    [[bool:show_extra]]
    {% if show_extra %}
    [[extra_content]]
    {% endif %}
    """
    # If show_extra is False, extra_content slot won't exist
    # This depends on LLM response -- may need mocking
```

---

## References

- `segment_processor.py:122-298` -- current `process_segment_with_delta()` implementation
- `__init__.py:89-189` -- current `chatter_async()` implementation
- `results.py:56-85` -- progress callback pattern
- `playground/core.py:295-377` -- existing batch streaming with anyio memory channels
