---
layout: default
title: API
parent: Reference
nav_order: 2
---

# API Reference

## Main Functions

### complete

```python
def complete(
    multipart_prompt: str,
    context: Union[dict, List[dict]] = {},
    *,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    spec: Optional[ModelSpec] = None,
    registry: Optional[ModelRegistry] = None,
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
    strict_params: bool = False,
    max_concurrent: Optional[int] = None,
    on_complete: Optional[callable] = None,
    stop_at: Optional[str] = None,
    on_halt: str = "raise",
    tools=None,
    deps=None,
    deps_type=None,
    limits=None,
) -> Union[StruckdownResult, List[StruckdownResult]]
```

Process a struckdown template with one or more contexts.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `multipart_prompt` | `str` | Struckdown template string |
| `context` | `dict` or `List[dict]` | Variables for template rendering. A list runs the template once per context |
| `model` | `LLM` | Model configuration |
| `credentials` | `LLMCredentials` | API credentials -- pass explicitly when using struckdown as a library |
| `spec` | `ModelSpec` | Model and credentials together, instead of `model` + `credentials` |
| `registry` | `ModelRegistry` | Resolve the model by name or alias from a registry |
| `extra_kwargs` | `dict` | Additional LLM parameters |
| `template_path` | `Path` | Path for resolving includes |
| `include_paths` | `List[Path]` | Additional include search paths |
| `strict_undefined` | `bool` | Raise on undefined template variables |
| `strict_params` | `bool` | Raise on unsupported LLM parameters instead of warning |
| `max_concurrent` | `int` | Max concurrent requests (list mode) |
| `on_complete` | `callable` | Callback after each completion |
| `stop_at` | `str` | Slot name to stop after; later segments are skipped |
| `on_halt` | `str` | `"raise"` (default) or `"return"` -- see [Halting](#halting) |
| `tools` | `list` | Python functions a `use_tools=true` slot may call |
| `deps` | `Any` | Dependency object passed to tools |
| `deps_type` | `type` | Type of `deps`, for tool signatures |
| `limits` | `UsageLimits` | Ceiling on requests, tool calls and output tokens for tool slots |

**Returns:** `StruckdownResult` for a single context, `List[StruckdownResult]` for a list.

**Raises:** `Halted` when a `[[halt:...]]` slot trips and `on_halt="raise"`.

**Example:**

```python
from struckdown import complete

result = complete("Tell me a joke [[joke]]")
print(result["joke"])
```

---

### complete_async

```python
async def complete_async(
    multipart_prompt: str,
    context: Union[dict, List[dict]] = {},
    **kwargs
) -> Union[StruckdownResult, List[StruckdownResult]]
```

Async version of `complete()`. Same parameters.

**Example:**

```python
import asyncio
from struckdown import complete_async

async def main():
    result = await complete_async("Tell me a joke [[joke]]")
    print(result["joke"])

asyncio.run(main())
```

---

### complete_incremental_async

```python
async def complete_incremental_async(
    multipart_prompt: str,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    context={},
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
    stream: bool = True,
    strict_params: bool = False,
    stop_at: Optional[str] = None,
    on_halt: str = "raise",
    tools=None,
    deps=None,
    deps_type=None,
    limits=None,
    *,
    spec: Optional[ModelSpec] = None,
    registry: Optional[ModelRegistry] = None,
) -> AsyncGenerator[IncrementalEvent, None]
```

Process a template one slot at a time, yielding events as work completes.
Takes a single context, not a list. With `stream=True` (the default),
free-text slots also yield their tokens as they arrive.

**Events:**

| Event | Fields | When |
|-------|--------|------|
| `SlotStreamStart` | `segment_index`, `slot_key` | A slot begins streaming |
| `TokenDelta` | `segment_index`, `slot_key`, `delta`, `accumulated` | Each chunk of streamed text |
| `ThinkingDelta` | `segment_index`, `slot_key`, `delta`, `accumulated` | Each chunk of streamed reasoning |
| `SlotCompleted` | `segment_index`, `slot_key`, `result`, `elapsed_ms`, `was_cached` | A slot has its final value |
| `SlotRetracted` | `segment_index`, `slot_key`, `reason` | Streamed text must be withdrawn |
| `ToolStarted` | `segment_index`, `slot_key`, `tool_name`, `arguments` | A tool call begins |
| `ToolCompleted` | ...plus `output`, `ok`, `error`, `elapsed_ms`, `was_cached` | A tool call returns |
| `CheckpointReached` | `segment_index`, `segment_name`, `accumulated_results` | A `<checkpoint>` boundary |
| `ProcessingComplete` | `result`, `early_termination` | Final event, with the aggregated `StruckdownResult` |
| `ProcessingError` | `segment_index`, `slot_key`, `error_message`, `partial_results` | The run failed |

Every event type is importable from `struckdown.incremental`, which is the
import path to prefer. Most are also re-exported from `struckdown` itself;
`ToolStarted` and `ToolCompleted` are not.

**Example:**

```python
from struckdown import complete_incremental_async, SlotCompleted
from struckdown.incremental import TokenDelta

async for event in complete_incremental_async(prompt, context=ctx,
                                              model=model, credentials=creds):
    if isinstance(event, TokenDelta):
        print(event.delta, end="", flush=True)
    elif isinstance(event, SlotCompleted):
        print(f"\n{event.slot_key} done in {event.elapsed_ms:.0f}ms")
```

Closing the generator cancels the call in flight, so a stop button also ends
the spending.

---

### complete_incremental

```python
def complete_incremental(
    multipart_prompt: str,
    ...,
    stream: bool = False,
) -> Generator[IncrementalEvent, None, None]
```

Synchronous version of `complete_incremental_async()`. Same events and
parameters, except that `stream` defaults to `False`.

---

## Halting

A `[[halt:...]]` slot stops a run when its verdict holds. See
[Halting a Run](../explanation/template-syntax.md#halting-a-run) for the
template side.

### Halted

```python
from struckdown import Halted
```

Raised when a guard trips and `on_halt="raise"` (the default).

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `slot` | `str` | Name of the halt slot that tripped |
| `reason` | `str` | The model's one-sentence justification, written for a log |
| `results` | `StruckdownResult` | The slots that finished before the run stopped |
| `when` | `bool` | The slot's `when=` setting |

```python
from struckdown import complete, Halted

try:
    result = complete(prompt, context=ctx, model=model, credentials=creds)
except Halted as halted:
    log.warning("halted at [[halt:%s]]: %s", halted.slot, halted.reason)
    return "I can't help with that."
```

With `on_halt="return"`, `complete()` returns the partial results instead of
raising, and the incremental generators yield
`ProcessingComplete(early_termination=True)` rather than raising.

`results` holds every slot that finished, which can include one that ran
beside the guard and completed before the verdict came back. They are there
to log and to bill, not to show. A slot still streaming when the guard trips
is withdrawn with a `SlotRetracted` event.

`reason` is not for the person being judged: showing it to them describes how
the guard works. A guard is an LLM call reading the same untrusted text, so
it is a signal to count, not a security boundary.

---

### readonly

```python
from struckdown import readonly

@readonly
def search_handbook(query: str) -> list[dict]:
    ...
```

Marks a tool as having no side effects, so a guard running speculatively
beside a tool slot need not be settled before the tool runs. Leave it off
anything that writes: undeclared means "might write", which forces the join.

---

## Other Functions

### get_embedding

```python
def get_embedding(
    texts: List[str],
    model: Optional[str] = None,
    credentials: Optional[LLMCredentials] = None,
    dimensions: Optional[int] = None,
    batch_size: int = 100,
    max_tokens_per_batch: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    cost_callback: Optional[Callable[[float, int, int], None]] = None,
) -> EmbeddingResultList
```

Get embeddings for texts using API or local models.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `texts` | `List[str]` | Texts to embed |
| `model` | `str` | Model name (e.g., `"text-embedding-3-small"`) |
| `credentials` | `LLMCredentials` | API credentials |
| `dimensions` | `int` | Output dimensions (model-specific) |
| `batch_size` | `int` | Texts per API batch |
| `max_tokens_per_batch` | `int` | Token ceiling per batch |
| `progress_callback` | `Callable` | Called with count completed |
| `cost_callback` | `Callable` | Called with cost and token counts |

**Returns:** `EmbeddingResultList` containing `EmbeddingResult` arrays.

**Example:**

```python
from struckdown import get_embedding
import numpy as np

results = get_embedding(["hello", "world"])
similarity = np.dot(results[0], results[1])
print(f"Cost: ${results.total_cost}")
```

---

### get_embedding_async

```python
async def get_embedding_async(
    texts: List[str],
    **kwargs
) -> EmbeddingResultList
```

Async version of `get_embedding()`. Same parameters.

---

### structured_chat

```python
def structured_chat(
    prompt: str = None,
    messages: List[Dict] = None,
    return_type: BaseModel = None,
    llm: LLM = None,
    credentials: LLMCredentials = None,
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
    extra_kwargs: Optional[dict] = None,
    strict_params: bool = False,
) -> Tuple[BaseModel, Box]
```

Low-level function for structured LLM calls with Pydantic models. No template
parsing: this is the call `complete()` makes for each slot.

**Returns:** Tuple of (parsed response, completion object).

---

## Result Classes

### StruckdownResult

Container for template processing results.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `results` | `Dict[str, SlotResult]` | Results by slot name |
| `response` | `Any` | Last slot's output |
| `outputs` | `Box` | All outputs as a Box dict |
| `thinking` | `Dict[str, str]` | Reasoning text by slot name, where the model returned any |
| `total_cost` | `float` | Total USD cost. Counts an unknown cost as 0.0 (check `has_unknown_costs`) and cached slots at their original cost |
| `prompt_tokens` | `int` | Total input tokens |
| `completion_tokens` | `int` | Total output tokens |
| `total_tokens` | `int` | Total tokens |
| `cached_prompt_tokens` | `int` | Input tokens served from the provider's cache |
| `cache_creation_tokens` | `int` | Input tokens written to the provider's cache |
| `has_unknown_costs` | `bool` | Any unknown costs |
| `all_costs_unknown` | `bool` | Every cost unknown |
| `fresh_call_count` | `int` | Fresh API calls |
| `cached_call_count` | `int` | Cache hits |
| `fresh_cost` | `float` | Cost of fresh calls only: what this run spent |
| `cached_cost` | `float` | Original cost of the slots served from the response cache: what the cache saved |

**Methods:**

```python
result["slot_name"]       # Get slot output
result.keys()             # List slot names
len(result)               # Number of slots
result.strip_debug_data() # Drop prompts and raw completions
```

---

### SlotResult

One slot's outcome, as found in `StruckdownResult.results`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Slot name |
| `prompt` | `str` | The prompt sent for this slot |
| `output` | `Any` | The parsed value |
| `completion` | `Box` | Raw provider response, with usage and cost |
| `action` | `Optional[str]` | Action or type name, where the slot had one |
| `options` | `list` | Parsed slot options |
| `params` | `dict` | Per-slot LLM parameters |
| `response_schema` | `dict` | Schema sent to the provider |

---

### EmbeddingResult

Numpy array subclass with cost metadata.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `cost` | `Optional[float]` | USD cost (None if unknown) |
| `tokens` | `int` | Token count |
| `model` | `str` | Model name |
| `cached` | `bool` | From cache |

Works as a normal numpy array:

```python
import numpy as np
emb = results[0]
similarity = np.dot(emb, other_emb)
```

---

### EmbeddingResultList

List of `EmbeddingResult` with aggregate properties.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `total_cost` | `Optional[float]` | Total cost (None if any unknown) |
| `total_tokens` | `int` | Total tokens |
| `cached_count` | `int` | Cached embeddings |
| `fresh_count` | `int` | Fresh embeddings |
| `fresh_cost` | `Optional[float]` | Cost from fresh only |
| `has_unknown_costs` | `bool` | Any unknown costs |
| `model` | `str` | Model name |

---

### CostSummary

Aggregate costs across multiple results.

```python
from struckdown import CostSummary

summary = CostSummary.from_results([result1, result2])
print(summary.format_summary())
# Total cost: $0.0012 (1,520 in / 310 out)
#   This run: $0.0004 (3 fresh, 2 cached)
```

`format_summary(include_breakdown=True)` returns the line the CLI prints; the second line appears only when some calls were cached. When some costs are unknown the total is shown as a lower bound (`>=$...`), and as `unknown` when none is known.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_cost` | `float` | Combined `StruckdownResult.total_cost` |
| `fresh_cost` | `float` | Combined `fresh_cost` |
| `total_prompt_tokens` | `int` | Combined input tokens |
| `total_completion_tokens` | `int` | Combined output tokens |
| `cached_prompt_tokens` | `int` | Input tokens read from the provider's cache |
| `cache_creation_tokens` | `int` | Input tokens written to the provider's cache |
| `fresh_count` | `int` | Fresh API calls |
| `cached_count` | `int` | Response-cache hits |
| `has_unknown_costs` | `bool` | Any result has an unknown cost |
| `all_costs_unknown` | `bool` | No result has a known cost |

---

## Configuration Classes

### LLM

Model configuration. Defaults are read from environment variables for CLI convenience but should be passed explicitly when using struckdown as a library.

```python
from struckdown import LLM, complete

llm = LLM(model_name="openai:gpt-4o")
result = complete("...", model=llm)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `DEFAULT_LLM` env var, or `gpt-4.1-mini` | Model identifier in `provider:model` format |

---

### LLMCredentials

API credentials for LLM calls. Defaults are read from environment variables for CLI convenience. When embedding struckdown in a web application, always pass credentials explicitly (e.g. from a database-stored `Credential` via `ModelSpec.as_credentials()`).

```python
from struckdown import LLMCredentials, complete

creds = LLMCredentials(
    api_key="sk-...",
    base_url="https://api.openai.com/v1"
)
result = complete("...", credentials=creds)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | `str` | `LLM_API_KEY` env var | API key for the provider |
| `base_url` | `str` | `LLM_API_BASE` env var | Base URL (set for proxies, leave empty for direct provider access) |

---

### ModelSpec

Portable, self-contained specification for a model endpoint. Combines identity, credentials, pricing, and metadata. Preferred over passing `LLM` + `LLMCredentials` separately -- `complete()` takes it directly as `spec=`.

```python
from struckdown.model_spec import ModelSpec

spec = ModelSpec(
    model_name="openai:gpt-4o",
    api_key="sk-...",
    input_cost_per_mtok=2.50,
    output_cost_per_mtok=10.0,
)

result = complete("...", spec=spec)

# or convert to LLM + credentials for structured_chat
llm = spec.as_llm()
credentials = spec.as_credentials()
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | (required) | Model identifier in `provider:model` format |
| `model_type` | `"llm"` or `"embedding"` | `"llm"` | Model type |
| `api_key` | `SecretStr` | `None` | API key (masked in repr) |
| `base_url` | `str` | `None` | Base URL for proxies |
| `data_residency` | `str` | `None` | Data residency region |
| `display_name` | `str` | `None` | Human-readable name |
| `input_cost_per_mtok` | `float` | `None` | Input cost per million tokens (USD) |
| `output_cost_per_mtok` | `float` | `None` | Output cost per million tokens (USD) |
| `cache_read_cost_per_mtok` | `float` | `None` | Price of a prompt token read from the provider's cache (USD per million); the input rate if unset |
| `cache_write_cost_per_mtok` | `float` | `None` | Price of a prompt token written to the provider's cache (USD per million); the input rate if unset |
| `model_ref` | `str` | `None` | Your own identifier for this model, copied onto every usage record |

**Computed fields:** `provider` (extracted from model_name), `bare_name` (model name without provider prefix), `provider_display` (human-readable provider name).

When both the input and output rates are set, struckdown uses them for cost calculation instead of looking up prices via pydantic-ai or genai-prices. `complete(spec=...)` sets them, with `model_ref`, for the calls it makes; `as_llm()` and `as_credentials()` do not, so a `structured_chat` call made with those needs `set_model_pricing(...)`. See [Cost Tracking](../explanation/cost-tracking.md).

---

### ModelRegistry

Collection of `ModelSpec` instances with alias resolution. Used by pipeline systems (e.g. soaking) to manage multiple models. `complete()` takes one as `registry=`.

```python
from struckdown.model_spec import ModelSpec, ModelRegistry

registry = ModelRegistry(
    models={
        "openai:gpt-4o": ModelSpec(model_name="openai:gpt-4o", api_key="sk-..."),
        "openai:gpt-4o-mini": ModelSpec(model_name="openai:gpt-4o-mini", api_key="sk-..."),
    },
    aliases={"default": "openai:gpt-4o-mini", "best": "openai:gpt-4o"},
    default_llm="openai:gpt-4o-mini",
)

spec = registry.resolve("best")  # returns the gpt-4o spec
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `models` | `Dict[str, ModelSpec]` | `{}` | Registered models keyed by model_name |
| `aliases` | `Dict[str, str]` | `{}` | Alias to model_name mappings |
| `default_llm` | `str` | `None` | Default LLM model name |
| `default_embedding` | `str` | `None` | Default embedding model name |

**Methods:**

| Method | Description |
|--------|-------------|
| `resolve(name_or_alias)` | Resolve a name or alias to a `ModelSpec` |
| `resolve_embedding(name)` | Resolve an embedding model |
| `register(spec)` | Add a `ModelSpec` to the registry |
| `llms()` | List all LLM specs |
| `embeddings()` | List all embedding specs |
| `from_env()` | Build a minimal registry from `DEFAULT_LLM` / `LLM_API_KEY` / `LLM_API_BASE` env vars (CLI convenience) |

---

## Utility Functions

### clear_cache

```python
from struckdown import clear_cache
clear_cache()  # Clear LLM response cache
```

### clear_embedding_cache

```python
from struckdown.embedding_cache import clear_embedding_cache
clear_embedding_cache()  # Clear embedding cache
```

### progress_tracking

```python
from struckdown import progress_tracking, complete

def on_call():
    print("API call completed!")

with progress_tracking(on_api_call=on_call):
    result = complete(prompt)
```

Context manager that fires a callback after each LLM completion, for progress
reporting without changing the `complete()` signature.

### set_model_pricing

```python
from struckdown import set_model_pricing

set_model_pricing(
    input_cost_per_mtok=0.40,
    output_cost_per_mtok=1.60,
    cache_read_cost_per_mtok=0.10,    # optional
    cache_write_cost_per_mtok=None,   # optional
)
set_model_pricing(None, None)         # clear
```

Stored pricing for the calls that follow in this context, whatever their model. `set_audio_pricing(cost_per_minute)` does the same for transcription, and `set_model_ref(ref)` names the model on usage records. All three are context variables: set them immediately before the call they belong to. See the [Usage Ledger reference](usage-ledger.md).

### Usage records

`register_usage_handler`, `usage_tracking`, `UsageRecord`, `CostBreakdown` and the Django ledger are documented in the [Usage Ledger reference](usage-ledger.md).

### mark_struckdown_safe

```python
from struckdown import mark_struckdown_safe

safe_content = mark_struckdown_safe("<system>...</system>")
```

Mark content as safe to bypass auto-escaping.
