---
layout: default
title: Caching
parent: Explanation
nav_order: 2
---

# Caching

Struckdown automatically caches LLM responses and embeddings to reduce API costs and improve performance.

## LLM Response Caching

LLM completions are cached based on:

- Messages (system, user, assistant)
- Model name
- Return type schema
- Extra parameters (temperature, etc.)
- Struckdown version

### Cache Location

By default, responses are cached in `~/.struckdown/cache/`. Control this with:

```bash
# Custom location
export STRUCKDOWN_CACHE=/path/to/cache

# Disable caching
export STRUCKDOWN_CACHE=0
# or
export STRUCKDOWN_CACHE=false
```

### Cache Behaviour

- **Cache hits**: Return instantly with `cost=0.0`
- **Cache misses**: Make API call, cache response
- **Deterministic errors**: Content policy violations are cached to avoid repeated failures

### Detecting Cache Hits

```python
from struckdown import chatter, get_run_id

result = chatter("Tell me a joke [[joke]]")

# Check if response was cached
for key, segment in result.results.items():
    if segment.completion:
        cached = segment.completion.get("_run_id") != get_run_id()
        print(f"{key}: {'cached' if cached else 'fresh'}")

# Aggregate counts
print(f"Fresh calls: {result.fresh_call_count}")
print(f"Cached calls: {result.cached_call_count}")
```


## Embedding Caching

Embeddings are cached per-text using a separate disk cache:

- **Location**: `~/.struckdown/cache/embeddings/`
- **Key format**: `{model}:{dimensions}:{sha256(text)[:32]}`
- **Storage**: LRU eviction with 5GB limit

### Cache Behaviour

```python
from struckdown import get_embedding

# First call - makes API request
results = get_embedding(["hello", "world"])
print(results.cached_count)  # 0
print(results.fresh_count)   # 2

# Second call - returns from cache
results = get_embedding(["hello", "world"])
print(results.cached_count)  # 2
print(results.fresh_count)   # 0
print(results.total_cost)    # 0.0 (all cached)
```

### Partial Cache Hits

When embedding multiple texts, cached and fresh results are merged:

```python
# "hello" is cached from before, "new text" is not
results = get_embedding(["hello", "new text"])
print(results.cached_count)  # 1
print(results.fresh_count)   # 1
print(results[0].cached)     # True
print(results[1].cached)     # False
```


## Clearing the Cache

### Programmatically

```python
from struckdown import clear_cache
from struckdown.embedding_cache import clear_embedding_cache

# Clear LLM response cache
clear_cache()

# Clear embedding cache
clear_embedding_cache()
```

### Manually

```bash
# Remove entire cache directory
rm -rf ~/.struckdown/cache/

# Remove only embeddings
rm -rf ~/.struckdown/cache/embeddings/
```


## Cache Invalidation

The cache is automatically invalidated when:

- **Struckdown version changes** -- responses cached with older versions are not reused
- **Return type schema changes** -- different Pydantic models produce different cache keys
- **Model name changes** -- each model has separate cache entries

To force a fresh API call without clearing the entire cache, change a parameter that affects the cache key (e.g., add whitespace to the prompt).


## Concurrency Control

API calls are limited by a global semaphore to prevent overwhelming the provider:

```bash
# Default: 20 concurrent calls
export SD_MAX_CONCURRENCY=20
```

This applies to both LLM completions and embedding batches.


## Environment Variables Summary

| Variable | Description | Default |
|----------|-------------|---------|
| `STRUCKDOWN_CACHE` | Cache directory, or `0`/`false` to disable | `~/.struckdown/cache` |
| `SD_MAX_CONCURRENCY` | Max concurrent API calls | `20` |
| `SD_EMBEDDING_BATCH_SIZE` | Texts per embedding batch | `100` |
