# Phase 2: Token Streaming

> **Status:** Future work. Complete Phase 1 (chunking) first.
>
> See `chunking.md` for Phase 1 implementation.

## Overview

Phase 2 adds real-time token streaming from LLM completions, allowing consumers to display partial responses as they arrive.

## Key Challenge

The current stack uses instructor + litellm with structured output:

```python
instructor.from_litellm(litellm.completion).chat.completions.create_with_completion(
    model=model_name,
    response_model=return_type,  # Pydantic model
    messages=messages,
)
```

**Problem:** Instructor's structured output mode doesn't support streaming. The response must be complete JSON to validate against the Pydantic schema.

## Potential Approaches

### Approach A: Hybrid by slot type

Different slot types have different streaming characteristics:

| Slot Type | Can Stream? | Notes |
|-----------|-------------|-------|
| `[[text]]` (default) | Yes | Free-form text |
| `[[bool:x]]` | No | Short response |
| `[[number:x]]` | No | Short response |
| `[[pick:x\|a,b,c]]` | No | Constrained output |
| `[[extract:x\|...]]` | Maybe | Stream text, validate JSON at end |
| `[[@action:x]]` | No | Not LLM calls |

For streamable types, bypass instructor and use raw litellm streaming:

```python
async for chunk in litellm.acompletion(..., stream=True):
    yield chunk.choices[0].delta.content
```

Then validate the complete response at the end.

### Approach B: Use instructor's partial streaming

Instructor has experimental partial streaming support:
https://python.useinstructor.com/concepts/partial/

This yields partially populated Pydantic models as tokens arrive. Worth investigating.

### Approach C: Stream-then-parse

1. Stream tokens from LLM without structured output
2. Buffer complete response
3. Extract JSON from response (may be in markdown code block)
4. Validate against Pydantic schema

## Caching Considerations

Token streaming is incompatible with current caching:
- Cache stores complete results
- Can't cache token stream

Options:
1. Check cache first, stream only on miss
2. Cache final result after streaming completes
3. Disable cache for streaming mode

## New Event Type

```python
@dataclass
class TokenReceived:
    type: Literal["token"] = "token"
    segment_index: int
    slot_key: str
    token: str
    buffer: str  # accumulated tokens so far
```

## Dependencies

- Requires Phase 1 (chunking) to be complete
- May require instructor upgrade or bypass
- Need to test with various LLM providers

## Open Questions

1. How to handle structured output validation with streaming?
2. Should streaming be opt-in per slot, or global?
3. How to handle errors mid-stream?
4. Cache strategy for streaming mode?

---

*This document will be expanded when Phase 1 is complete.*
