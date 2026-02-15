---
layout: default
title: Home
nav_order: 1
permalink: /
---

# struckdown

Structured LLM prompting with template syntax.

Struckdown is a Python library for building structured conversations with language models. It combines Jinja2 templating with a simple slot syntax to extract typed, validated responses from LLMs.

## Quick Example

```python
from struckdown import chatter

result = chatter("""
You are a helpful assistant.

Analyse this text and extract the main topic and sentiment.

Text: {{text}}

[[topic:str]]
[[sentiment:str]]
""", context={"text": "I love sunny days at the beach!"})

print(result["topic"])      # "beach/outdoor activities"
print(result["sentiment"])  # "positive"
print(result.total_cost)    # 0.0001 (USD)
```

## Key Features

- **Slot syntax** -- `[[name:type]]` extracts typed values from LLM responses
- **Jinja2 templating** -- Full template power with variables, loops, conditionals
- **Structured outputs** -- Automatic validation via Pydantic models
- **Cost tracking** -- Monitor API spend per call and in aggregate
- **Caching** -- Automatic caching of LLM responses and embeddings
- **Async support** -- `chatter_async()` for concurrent processing
- **Embeddings** -- `get_embedding()` with cost tracking and caching

## Installation

```bash
pip install struckdown
```

For local embeddings (sentence-transformers):
```bash
pip install struckdown[local]
```

## Documentation

| Section | Description |
|---------|-------------|
| [Tutorials](tutorials/) | Learn struckdown step by step |
| [How-To Guides](how-to/) | Solve specific problems |
| [Explanation](explanation/) | Understand how things work |
| [Reference](reference/) | Technical specifications |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for LLM provider | Required |
| `LLM_API_BASE` | API base URL (OpenAI-compatible) | OpenAI default |
| `DEFAULT_LLM` | Default model name | `gpt-4.1-mini` |
| `DEFAULT_EMBEDDING_MODEL` | Default embedding model | `text-embedding-3-large` |
| `STRUCKDOWN_CACHE` | Cache directory | `~/.struckdown/cache` |
| `SD_MAX_CONCURRENCY` | Max concurrent API calls | `20` |

## Links

- [GitHub Repository](https://github.com/benwhalley/struckdown)
- [PyPI Package](https://pypi.org/project/struckdown/)
