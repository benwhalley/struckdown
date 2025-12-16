#!/usr/bin/env python3
"""
Manual verification script for LLM call caching.

This script makes the same LLM call twice and verifies that:
1. The first call hits the API
2. The second call uses the cache
3. Cache files are created on disk

Run this to verify caching works after API changes.
"""

import time
from pathlib import Path

from struckdown import chatter
from struckdown.cache import get_cache_dir


def get_cache_size(cache_path):
    """Calculate total size of cache directory"""
    if not cache_path or not cache_path.exists():
        return 0
    total = 0
    for f in cache_path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def count_cache_files(cache_path):
    """Count number of cache files"""
    if not cache_path or not cache_path.exists():
        return 0
    return len([f for f in cache_path.rglob("*") if f.is_file()])


print("=" * 70)
print("STRUCKDOWN CACHE VERIFICATION")
print("=" * 70)

# Check cache configuration
cache_dir = get_cache_dir()
print(f"\n📁 Cache directory: {cache_dir}")

if cache_dir is None:
    print("⚠️  CACHING IS DISABLED!")
    print("   Set STRUCKDOWN_CACHE environment variable to enable caching")
    exit(1)

# Get initial cache state
initial_size = get_cache_size(cache_dir)
initial_files = count_cache_files(cache_dir)

print(f"   Initial size: {initial_size:,} bytes")
print(f"   Initial files: {initial_files}")

# Make first call
print("\n" + "=" * 70)
print("FIRST CALL (should hit API)")
print("=" * 70)

template = """What is 2+2? Give a one word answer [[response]]"""

start = time.time()
result1 = chatter(template, extra_kwargs={"max_tokens": 10})
duration1 = time.time() - start

print(f"✓ Response: {result1.response}")
print(f"✓ Duration: {duration1:.3f}s")

# Check cache after first call
after_first_size = get_cache_size(cache_dir)
after_first_files = count_cache_files(cache_dir)

print(
    f"✓ Cache size: {after_first_size:,} bytes (+{after_first_size - initial_size:,})"
)
print(f"✓ Cache files: {after_first_files} (+{after_first_files - initial_files})")

if after_first_size <= initial_size:
    print("⚠️  WARNING: Cache size didn't increase!")
    print("   This might indicate caching is not working properly")

# Make second call (identical)
print("\n" + "=" * 70)
print("SECOND CALL (should use cache)")
print("=" * 70)

start = time.time()
result2 = chatter(template, extra_kwargs={"max_tokens": 10})
duration2 = time.time() - start

print(f"✓ Response: {result2.response}")
print(f"✓ Duration: {duration2:.3f}s")

# Check cache after second call
after_second_size = get_cache_size(cache_dir)
after_second_files = count_cache_files(cache_dir)

print(
    f"✓ Cache size: {after_second_size:,} bytes (+{after_second_size - after_first_size:,})"
)
print(
    f"✓ Cache files: {after_second_files} (+{after_second_files - after_first_files})"
)

# Verify results
print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

success = True

# Check responses match
if result1.response == result2.response:
    print("✅ Responses match")
else:
    print("❌ Responses don't match (they should be identical)")
    print(f"   First:  {result1.response}")
    print(f"   Second: {result2.response}")
    success = False

# Check second call was faster (cache hit)
speedup = duration1 / duration2 if duration2 > 0 else float("inf")
print(f"\n⏱️  Speed comparison:")
print(f"   First call:  {duration1:.3f}s")
print(f"   Second call: {duration2:.3f}s")
print(f"   Speedup:     {speedup:.1f}x")

if duration2 < duration1 * 0.5:  # Second call should be at least 2x faster
    print("✅ Second call was significantly faster (cache hit!)")
else:
    print("⚠️  Second call wasn't much faster (possible cache miss)")
    print("   Note: If using a very fast mock or local model, this is expected")

# Check cache files were created
if after_first_files > initial_files:
    print(f"\n✅ Cache files created: {after_first_files - initial_files} new files")
else:
    print("\n❌ No new cache files created")
    success = False

# Check no duplicate caching
if after_second_files == after_first_files:
    print("✅ No duplicate cache files (second call reused cache)")
else:
    print("⚠️  Additional cache files created on second call")
    print(f"   This might indicate cache keys are not matching properly")

# Final verdict
print("\n" + "=" * 70)
if success:
    print("✅ CACHE VERIFICATION PASSED")
    print("   Caching is working correctly after API changes")
else:
    print("❌ CACHE VERIFICATION FAILED")
    print("   There may be issues with the caching implementation")
print("=" * 70)

# Show cache location
print(f"\n💡 Cache location: {cache_dir}")
print(f"   You can inspect the cache files manually")
print(f"   Clear cache with: rm -rf {cache_dir}")
