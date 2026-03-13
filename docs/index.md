---
layout: default
title: Home
nav_order: 1
permalink: /
---

# Struckdown

Markdown-based syntax for **structured** conversations with language models.

Struckdown makes it easy to extract structured, typed data from text using LLMs. Instead of parsing free-form responses, you define exactly what you want with a simple template syntax.

## Quick Example

Turn unstructured text into structured data:

```bash
sd batch *.txt "Purpose, <5 words: [[purpose]]"
```

Output:
```json
[
  {"filename": "butter_robot.txt", "purpose": "Pass butter, question existence."},
  {"filename": "portal_gun.txt", "purpose": "Interdimensional travel device."}
]
```

Or with type constraints:

```bash
sd batch *.txt "Price: [[number:price]] Currency: [[pick:currency|USD,GBP,EUR]]"
```

## Key Features

- **Slot syntax** -- `[[variable]]` for completions, `[[type:variable]]` for typed extraction
- **Type safety** -- Extract booleans, numbers, dates, or pick from options
- **Batch processing** -- Process hundreds of files with progress bars
- **Token management** -- Use `<checkpoint>` to save tokens between steps
- **Caching** -- Automatic disk caching saves money and time
- **Custom actions** -- Extend with Python functions (RAG, APIs, databases)
- **Web integration** -- Fetch URLs and search the web directly in prompts
- **Multiple outputs** -- JSON, CSV, Excel, or stdout

## Installation

```bash
# Requires UV (https://docs.astral.sh/uv/)
uv tool install git+https://github.com/benwhalley/struckdown
```

Configure your LLM:

```bash
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.openai.com/v1"
export DEFAULT_LLM="gpt-4o-mini"
```

## Basic Usage

### Single Prompts

```bash
# Simple extraction
sd chat "Is Python compiled? [[bool:answer]]"

# Pick from options
sd chat "Sentiment of 'I love it': [[pick:sentiment|positive,negative,neutral]]"

# Number with constraints
sd chat "Rate 1-10: 'Great!' [[int:rating|min=1,max=10]]"
```

### Batch Processing

```bash
# Process files to JSON
sd batch *.txt "Summarise: [[summary]]" -o results.json

# Extract to CSV
sd batch documents/*.txt "Name: [[extract:name]] Email: [[extract:email]]" -o contacts.csv

# Chain operations
sd batch *.txt "Company: [[extract:company]]" | \
  sd batch "Find {{company}} stock ticker: [[ticker]]" -k
```

### Multi-Step Reasoning

```bash
sd chat "
Document: {{input}}
Extract key points: [[extract+:points]]

<checkpoint>

Based on: {{points}}
Recommendation: [[recommendation]]
" -s document.txt
```

## Python API

```python
from struckdown import chatter

result = chatter("""
Analyse this review: {{review}}

Sentiment: [[pick:sentiment|positive,negative,neutral]]
Rating: [[int:rating|min=1,max=5]]
""", context={"review": "Great product but slow shipping"})

print(result["sentiment"])  # "positive"
print(result["rating"])     # 4
print(result.total_cost)    # 0.0001 (USD)
```

## Documentation

| Section | Description |
|---------|-------------|
| [Getting Started](tutorials/getting-started.md) | Installation and first steps |
| [Template Syntax](explanation/template-syntax.md) | Complete syntax reference |
| [CLI Reference](reference/cli.md) | All CLI commands |
| [Custom Actions](how-to/custom-actions.md) | Extend with Python plugins |
| [Caching](explanation/caching.md) | How caching works |
| [API Reference](reference/api.md) | Python API documentation |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for LLM provider | Required |
| `LLM_API_BASE` | API base URL (OpenAI-compatible) | OpenAI default |
| `DEFAULT_LLM` | Default model name | `gpt-4o-mini` |
| `STRUCKDOWN_CACHE` | Cache directory (0 to disable) | `~/.struckdown/cache` |
| `STRUCKDOWN_CACHE_SIZE` | Cache size limit in MB | `10240` (10 GB) |
| `SD_MAX_CONCURRENCY` | Max concurrent API calls | `20` |

## Links

- [GitHub Repository](https://github.com/benwhalley/struckdown)
- [PyPI Package](https://pypi.org/project/struckdown/)
