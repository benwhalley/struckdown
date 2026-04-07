---
layout: default
title: Using Different Providers
parent: How-To Guides
nav_order: 3
---

# Using Different LLM Providers

Struckdown supports multiple LLM providers via [pydantic-ai](https://ai.pydantic.dev/). Model names use the `provider:model` format.

## Supported Providers

| Provider | Prefix | Env Var for API Key | Example Model |
|----------|--------|---------------------|---------------|
| OpenAI | `openai:` | `OPENAI_API_KEY` | `openai:gpt-4o` |
| Anthropic | `anthropic:` | `ANTHROPIC_API_KEY` | `anthropic:claude-sonnet-4-20250514` |
| Google | `google-gla:` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `google-gla:gemini-2.0-flash` |
| Mistral | `mistral:` | `MISTRAL_API_KEY` | `mistral:mistral-large-latest` |
| Azure OpenAI | `azure:` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | `azure:gpt-4o` |
| Ollama | `ollama:` | (none needed) | `ollama:llama3` |
| OpenAI-compatible proxy | (bare name) | `LLM_API_KEY` + `LLM_API_BASE` | `gpt-4o` |

## CLI Examples

### OpenAI (direct)

```bash
export OPENAI_API_KEY=sk-...
sd chat "Tell a joke [[joke]]" --model-name openai:gpt-4o
sd chat "Tell a joke [[joke]]" --model-name openai:gpt-4.1-mini
```

### Anthropic (direct)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
sd chat "Tell a joke [[joke]]" --model-name anthropic:claude-sonnet-4-20250514
sd chat "Tell a joke [[joke]]" --model-name anthropic:claude-3-5-haiku-20241022
```

### Google Gemini (direct)

```bash
export GOOGLE_API_KEY=AI...
sd chat "Tell a joke [[joke]]" --model-name google-gla:gemini-2.0-flash
sd chat "Tell a joke [[joke]]" --model-name google-gla:gemini-2.5-pro
```

### Mistral (direct)

```bash
export MISTRAL_API_KEY=...
sd chat "Tell a joke [[joke]]" --model-name mistral:mistral-large-latest
sd chat "Tell a joke [[joke]]" --model-name mistral:mistral-small-latest
```

### Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com
sd chat "Tell a joke [[joke]]" --model-name azure:gpt-4o
```

### Ollama (local)

```bash
# No API key needed -- Ollama runs locally on port 11434
sd chat "Tell a joke [[joke]]" --model-name ollama:llama3
sd chat "Tell a joke [[joke]]" --model-name ollama:qwen3
```

### LiteLLM proxy (backward compatible)

When `LLM_API_BASE` is set, all requests route through that proxy. The provider prefix is stripped before sending to the proxy, so bare model names work too.

```bash
export LLM_API_KEY=sk-...
export LLM_API_BASE=http://litellm.example.com/v1

# Both of these send "claude-sonnet-4-20250514" to the proxy:
sd chat "Tell a joke [[joke]]" --model-name claude-sonnet-4-20250514
sd chat "Tell a joke [[joke]]" --model-name anthropic:claude-sonnet-4-20250514
```

## How Provider Routing Works

Struckdown determines how to route a request based on two things:

1. **Is `base_url` set?** (via `LLM_API_BASE` env var or from credentials)
   - **Yes** -- proxy mode. All requests go through the proxy as OpenAI-compatible. Provider prefix is stripped.
   - **No** -- native provider mode. The `provider:` prefix determines which pydantic-ai provider to use.

2. **Is `api_key` set explicitly?** (via credentials from database or env var)
   - **Yes** -- the key is injected into the provider.
   - **No** -- pydantic-ai reads the standard env var for that provider (e.g. `ANTHROPIC_API_KEY`).

## Setting a Default Model

Use the `DEFAULT_LLM` environment variable:

```bash
export DEFAULT_LLM=anthropic:claude-sonnet-4-20250514
sd chat "Tell a joke [[joke]]"  # uses Anthropic
```

Bare names (without a provider prefix) default to OpenAI:

```bash
export DEFAULT_LLM=gpt-4.1-mini
# Equivalent to openai:gpt-4.1-mini
```

## Per-Slot Model Overrides

You can use different models for different slots within the same template:

```
Quickly extract the key quote:
[[extract:quote|model=openai:gpt-4o-mini]]

Now reason carefully about it:
[[think:analysis|model=anthropic:claude-sonnet-4-20250514,thinking=high]]
```

Note: per-slot model overrides use the same `provider:model` format. The provider prefix in the slot override determines the provider, and the credentials must be available (via env vars or database).

## Python API

```python
from struckdown import LLM, LLMCredentials, complete

# Direct to Anthropic
llm = LLM(model_name="anthropic:claude-sonnet-4-20250514")
credentials = LLMCredentials(api_key="sk-ant-...")

result = complete("Tell a joke [[joke]]", model=llm, credentials=credentials)

# Via proxy
llm = LLM(model_name="claude-sonnet-4-20250514")
credentials = LLMCredentials(api_key="proxy-key", base_url="http://proxy:8000/v1")

result = complete("Tell a joke [[joke]]", model=llm, credentials=credentials)
```
