# Prompt Caching Support for Struckdown

## Summary of Prompt Caching by Provider

### OpenAI / Azure OpenAI
- **Automatic** - no explicit markers needed
- Minimum 1,024 tokens, first 1,024 must be identical
- Enabled by default, no opt-out
- Response shows `cached_tokens` in `usage.prompt_tokens_details`
- **No struckdown changes needed** - already works

### Anthropic / Claude
- **Requires explicit marking**: `cache_control: {"type": "ephemeral"}` on content blocks
- Minimum tokens: 1,024-4,096 depending on model
- Up to 4 explicit breakpoints per request
- TTL: 5 minutes default (1.25x write cost, 0.1x read), optional 1-hour (2x write)
- Two modes:
  - **Automatic**: Single `cache_control` at request level - system moves breakpoint to last cacheable block
  - **Explicit**: Place `cache_control` on specific content blocks

### LiteLLM Support
- Has `cache_control_injection_points` parameter that auto-injects `cache_control` for Anthropic
- Struckdown already uses litellm via `instructor.from_litellm()`

## Current Struckdown Architecture

- **LLM calls**: `llm.py:_call_llm_cached()` → `instructor` → `litellm.completion()`
- **Messages built in**: `segment_processor.py:process_segment_with_delta_incremental()`
- **Message order**: system → header → accumulated Q&A history → current content
- **extra_kwargs**: Passed through to litellm call at line 453

This message structure is already cache-friendly (stable content at beginning).

## Can Struckdown Support This Automatically?

**Yes.** Here's the analysis:

| Provider | Status | Changes Needed |
|----------|--------|----------------|
| OpenAI/Azure | Works now | None - automatic |
| Anthropic | Needs changes | Inject cache_control |
| Other providers | Unknown | Likely automatic like OpenAI |

## Recommended Implementation

### Option A: Use LiteLLM's cache_control_injection_points (Simplest)

For Anthropic models, pass this in `extra_kwargs`:
```python
{
    "cache_control_injection_points": [
        {"location": "message", "role": "system", "index": -1},  # cache system
    ]
}
```

**Pros**: Minimal code, uses existing litellm feature
**Cons**: Less control, depends on litellm implementation

### Option B: Directly inject cache_control on messages (More control)

Transform messages before sending to litellm:
```python
# For Anthropic models, add cache_control to system message
if is_anthropic_model(model_name):
    for msg in messages:
        if msg["role"] == "system":
            msg["cache_control"] = {"type": "ephemeral"}
```

**Pros**: Full control, works even if litellm changes
**Cons**: More code to maintain

### Option C: Request-level automatic caching (Anthropic's recommended approach)

Pass `cache_control={"type": "ephemeral"}` at request top level - Anthropic automatically moves breakpoint to last cacheable block.

```python
if is_anthropic_model(model_name):
    extra_kwargs["cache_control"] = {"type": "ephemeral"}
```

**Pros**: Simplest, Anthropic-recommended for conversations
**Cons**: May not work via litellm (needs testing)

## Implementation Plan

### 1. Add model detection (`llm.py`)
```python
def is_anthropic_model(model_name: str) -> bool:
    return model_name.lower().startswith(("claude", "anthropic/"))
```

### 2. Add config (`llm.py`)
```python
PROMPT_CACHING_ENABLED = env_config("STRUCKDOWN_PROMPT_CACHING", default=True, cast=bool)
```

### 3. Inject cache_control for Anthropic (`llm.py:_call_llm_cached`)
Before the API call, transform system messages to use content blocks with cache_control:
```python
if PROMPT_CACHING_ENABLED and is_anthropic_model(model_name):
    messages = _inject_cache_control(messages)
```

### 4. Cache stats already implemented
The completion object already has `cached_prompt_tokens` and `cache_creation_tokens` properties (from commit 36578df in `results.py:327-349`). These read from `usage.prompt_tokens_details.cached_tokens` and `usage.prompt_tokens_details.cache_creation_tokens`.

Once we enable cache_control for Anthropic, the stats will automatically populate.

### 5. TODO: Enable by default
Add TODO in code noting this is enabled by default for supported models.

## Files to Modify
- `struckdown/llm.py` - main changes (detection, injection, config)

## Decisions
- **Enabled by default** - auto-enable for Anthropic models, users can disable with `STRUCKDOWN_PROMPT_CACHING=0`
- **Track cache stats** - expose `cached_prompt_tokens` and `cache_creation_tokens` properties on completion object

## Verification
1. Run with Anthropic model, check response contains `cache_read_input_tokens > 0` on subsequent calls
2. Run with OpenAI model, check response contains `cached_tokens > 0`
3. Test multi-turn conversations to verify cache hits accumulate
4. Compare costs before/after with prompt caching enabled

## Reference Documentation
- [LiteLLM Prompt Caching](https://docs.litellm.ai/docs/tutorials/prompt_caching)
- [Anthropic Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Azure OpenAI Prompt Caching](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/prompt-caching)
