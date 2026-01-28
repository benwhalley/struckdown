# Phase 2: Migrate LLM Response Caching to diskcache

## Goal

Migrate all caching from joblib.Memory to diskcache for consistency, replacing the existing joblib-based LLM response cache with diskcache.

## Current State (After Phase 1)

- Embedding cache uses diskcache (per-string caching, thread-safe)
- LLM response cache still uses joblib.Memory (`@memory.cache` on `_call_llm_cached`)

## Changes Required

### 1. Replace `struckdown/cache.py` core

```python
# Before (joblib)
from joblib import Memory
memory = Memory(location=str(cache_dir), verbose=0)

# After (diskcache)
from diskcache import FanoutCache
cache = FanoutCache(
    directory=str(cache_dir),
    shards=8,
    size_limit=cache_size_limit,
    eviction_policy='least-recently-used',
)
```

### 2. Replace `@memory.cache` on `_call_llm_cached`

```python
# Before
@memory.cache(ignore=["return_type", "llm", "credentials"])
def _call_llm_cached(...):

# After
@cache.memoize(ignore={"return_type", "llm", "credentials"})
def _call_llm_cached(...):
```

### 3. Update `hash_return_type()`

Keep as-is -- still needed for cache key generation.

### 4. Update tests

- `test_cache.py` - update for new API
- `test_cache_integration.py` - update decorator tests

## Benefits

- Single caching system across all LLM operations
- Better LRU eviction (built-in vs manual reduce_size)
- Less disk fragmentation (SQLite vs many pickle files)
- Consistent API across all cached functions
- Thread-safe by default

## Risks

- Existing cache invalidated (users need to clear cache)
- Different serialisation (joblib pickle vs diskcache pickle)
- Potential subtle behaviour differences

## Migration Path

1. Release Phase 1 (embeddings only) -- **DONE**
2. Add deprecation warning when joblib cache is used
3. Release Phase 2 with full migration
4. Document cache clearing in release notes

## Notes

- The `reduce_size` functionality from joblib Memory can be replaced with diskcache's built-in size limiting
- The `cache_version` parameter in `_call_llm_cached` ensures cache invalidation on version upgrades
- Consider adding a migration script to detect and clear old joblib cache on first run
