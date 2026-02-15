---
layout: default
title: Cost Tracking
parent: Explanation
nav_order: 1
---

# Cost Tracking

Struckdown tracks API costs for both LLM completions and embeddings, allowing you to monitor spend and enforce budgets.

## Overview

Cost information flows from the underlying API responses through litellm's pricing database. Costs are tracked per-call and aggregated across operations.

**Key points:**

- Costs are in USD
- Cached responses have zero cost (no API call made)
- Unknown costs return `None`, not `0.0`
- Token counts are always available, even when cost is unknown


## ChatterResult Cost Properties

When you call `chatter()` or `chatter_async()`, the returned `ChatterResult` provides cost information:

```python
from struckdown import chatter

result = chatter("Tell me a joke [[joke]]")

# Token counts
result.prompt_tokens      # input tokens across all segments
result.completion_tokens  # output tokens across all segments
result.total_tokens       # prompt_tokens + completion_tokens

# Cost (USD)
result.total_cost         # total cost across all segments
result.fresh_cost         # cost from fresh API calls only
result.cached_cost        # cost from cached calls (always 0.0)

# Cache statistics
result.fresh_call_count   # number of fresh API calls
result.cached_call_count  # number of cache hits

# Cost reliability
result.has_unknown_costs  # True if ANY segment has unknown cost
result.all_costs_unknown  # True if ALL segments have unknown cost
```

### Handling Unknown Costs

Cost may be unknown when:
- Using a custom API endpoint with non-standard pricing
- The model isn't in litellm's pricing database
- The API response doesn't include usage information

```python
result = chatter("...")

if result.has_unknown_costs:
    print(f"Cost is at least ${result.total_cost:.4f} (some unknown)")
else:
    print(f"Total cost: ${result.total_cost:.4f}")
```


## EmbeddingResult Cost Properties

The `get_embedding()` and `get_embedding_async()` functions return an `EmbeddingResultList` containing `EmbeddingResult` objects:

```python
from struckdown import get_embedding

results = get_embedding(["hello", "world"], model="text-embedding-3-small")

# Aggregate properties on the list
results.total_cost      # total USD cost (None if any unknown)
results.total_tokens    # total tokens across all embeddings
results.cached_count    # number retrieved from cache
results.fresh_count     # number from fresh API calls
results.fresh_cost      # cost from fresh calls only (None if any unknown)
results.has_unknown_costs  # True if any fresh embedding has unknown cost
results.model           # model name used

# Per-embedding properties
results[0].cost         # cost for this embedding (None if unknown, 0.0 if cached)
results[0].tokens       # tokens for this embedding
results[0].model        # model name
results[0].cached       # True if retrieved from cache
```

### Backwards Compatibility

`EmbeddingResult` is a numpy array subclass, so existing code works unchanged:

```python
import numpy as np

results = get_embedding(["hello", "world"])

# Still works as before
for emb in results:
    similarity = np.dot(emb, other_embedding)

# Array operations work
matrix = np.stack(list(results))
```


## CostSummary

For aggregating costs across multiple operations, use `CostSummary`:

```python
from struckdown import CostSummary

# Aggregate multiple ChatterResults
summary = CostSummary.from_results([result1, result2, result3])

summary.total_cost        # combined cost
summary.total_tokens      # combined tokens
summary.prompt_tokens     # combined input tokens
summary.completion_tokens # combined output tokens
summary.fresh_call_count  # total fresh API calls
summary.cached_call_count # total cache hits
summary.has_unknown_costs # True if any result has unknown costs
```


## Cost Sources

Costs are calculated using litellm's pricing database, which covers major providers:

- **OpenAI**: GPT-4, GPT-3.5, embeddings (text-embedding-3-small/large, ada-002)
- **Anthropic**: Claude models
- **Azure OpenAI**: Same pricing as OpenAI
- **Other providers**: Cohere, Google, etc.

For custom endpoints or unlisted models, costs will be `None`. Token counts are still available from the API response.


## Caching Behaviour

Struckdown caches both LLM completions and embeddings:

- **LLM completions**: Cached based on messages, model, and parameters
- **Embeddings**: Cached per-text in `~/.struckdown/cache/embeddings/`

Cached responses:
- Have `cached=True` on the result
- Have `cost=0.0` (no API call made)
- Don't contribute to "unknown costs" status


## Environment Variables

- `STRUCKDOWN_CACHE`: Cache directory (default: `~/.struckdown/cache`). Set to `0` or `false` to disable.
- `SD_MAX_CONCURRENCY`: Maximum concurrent API calls (default: 20)
- `SD_EMBEDDING_BATCH_SIZE`: Texts per embedding batch (default: 100)
