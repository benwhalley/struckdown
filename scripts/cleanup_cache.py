#!/usr/bin/env python
"""CLI script for cache cleanup. Run via cron or manually."""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

from struckdown.playground.prompt_cache import cleanup_cache, get_cache_stats

if __name__ == "__main__":
    stats_before = get_cache_stats()
    print(
        f"Cache before: {stats_before['file_count']} files, {stats_before['total_size_mb']}MB"
    )

    result = cleanup_cache()

    if result["files_deleted"] > 0:
        print(
            f"Deleted {result['files_deleted']} files, freed {result['bytes_freed']} bytes"
        )
        stats_after = get_cache_stats()
        print(
            f"Cache after: {stats_after['file_count']} files, {stats_after['total_size_mb']}MB"
        )
    else:
        print("No cleanup needed")
