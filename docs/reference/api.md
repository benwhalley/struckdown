---
layout: default
title: API
parent: Reference
nav_order: 2
---

# API Reference

## Main Functions

### chatter

```python
def chatter(
    multipart_prompt: str,
    context: Union[dict, List[dict]] = {},
    *,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
    max_concurrent: Optional[int] = None,
    on_complete: Optional[callable] = None,
) -> Union[ChatterResult, List[ChatterResult]]
```

Process a struckdown template with one or more contexts.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `multipart_prompt` | `str` | Struckdown template string |
| `context` | `dict` or `List[dict]` | Variables for template rendering |
| `model` | `LLM` | Model configuration (default from env) |
| `credentials` | `LLMCredentials` | API credentials (default from env) |
| `extra_kwargs` | `dict` | Additional LLM parameters |
| `template_path` | `Path` | Path for resolving includes |
| `include_paths` | `List[Path]` | Additional include search paths |
| `strict_undefined` | `bool` | Raise on undefined variables |
| `max_concurrent` | `int` | Max concurrent requests (list mode) |
| `on_complete` | `callable` | Callback after each completion |

**Returns:** `ChatterResult` for single context, `List[ChatterResult]` for list.

**Example:**

```python
from struckdown import chatter

result = chatter("Tell me a joke [[joke]]")
print(result["joke"])
```

---

### chatter_async

```python
async def chatter_async(
    multipart_prompt: str,
    context: Union[dict, List[dict]] = {},
    **kwargs
) -> Union[ChatterResult, List[ChatterResult]]
```

Async version of `chatter()`. Same parameters.

**Example:**

```python
import asyncio
from struckdown import chatter_async

async def main():
    result = await chatter_async("Tell me a joke [[joke]]")
    print(result["joke"])

asyncio.run(main())
```

---

### get_embedding

```python
def get_embedding(
    texts: List[str],
    model: Optional[str] = None,
    credentials: Optional[LLMCredentials] = None,
    dimensions: Optional[int] = None,
    batch_size: int = 100,
    progress_callback: Optional[Callable[[int], None]] = None,
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
| `progress_callback` | `Callable` | Called with count completed |

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
) -> Tuple[BaseModel, Box]
```

Low-level function for structured LLM calls with Pydantic models.

**Returns:** Tuple of (parsed response, completion object).

---

## Result Classes

### ChatterResult

Container for template processing results.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `results` | `Dict[str, SegmentResult]` | Results by slot name |
| `response` | `Any` | Last slot's output |
| `outputs` | `Box` | All outputs as Box dict |
| `total_cost` | `float` | Total USD cost |
| `prompt_tokens` | `int` | Total input tokens |
| `completion_tokens` | `int` | Total output tokens |
| `total_tokens` | `int` | Total tokens |
| `has_unknown_costs` | `bool` | Any unknown costs |
| `fresh_call_count` | `int` | Fresh API calls |
| `cached_call_count` | `int` | Cache hits |

**Methods:**

```python
result["slot_name"]       # Get slot output
result.keys()             # List slot names
len(result)               # Number of slots
```

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
print(summary)  # "Total cost: $0.0012 (5 calls, 2 cached)"
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `total_cost` | `float` | Combined cost |
| `total_tokens` | `int` | Combined tokens |
| `prompt_tokens` | `int` | Combined input |
| `completion_tokens` | `int` | Combined output |
| `fresh_call_count` | `int` | Total fresh calls |
| `cached_call_count` | `int` | Total cache hits |
| `has_unknown_costs` | `bool` | Any unknown |

---

## Configuration Classes

### LLM

Model configuration.

```python
from struckdown import LLM

llm = LLM(model_name="gpt-4")
result = chatter("...", model=llm)
```

**Fields:**

| Field | Type | Default |
|-------|------|---------|
| `model_name` | `str` | `DEFAULT_LLM` env var |

---

### LLMCredentials

API credentials.

```python
from struckdown import LLMCredentials

creds = LLMCredentials(
    api_key="sk-...",
    base_url="https://api.openai.com/v1"
)
result = chatter("...", credentials=creds)
```

**Fields:**

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `str` | `LLM_API_KEY` env var |
| `base_url` | `str` | `LLM_API_BASE` env var |

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

### get_run_id / new_run

```python
from struckdown import get_run_id, new_run

run_id = get_run_id()  # Current run ID
new_run()              # Start new run (for cache detection)
```

### mark_struckdown_safe

```python
from struckdown import mark_struckdown_safe

safe_content = mark_struckdown_safe("<system>...</system>")
```

Mark content as safe to bypass auto-escaping.
