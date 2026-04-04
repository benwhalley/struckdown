---
layout: default
title: Model Overrides
parent: How-To Guides
nav_order: 2
---

# Per-Slot LLM Parameters

## Overview

Struckdown supports per-slot control over LLM parameters -- temperature, thinking level, model selection, and more. Parameters are set inline using pipe syntax and validated at parse time.

## Supported Parameters

| Parameter | Type | Range / Values | Description |
|-----------|------|----------------|-------------|
| `temperature` | float | 0.0 -- 2.0 | Randomness. Lower = deterministic, higher = creative |
| `thinking` | string | off, minimal, low, medium, high, xhigh | Extended reasoning level (provider-dependent) |
| `model` | string | any model name | Override the LLM model for this slot |
| `max_tokens` | int | > 0 | Maximum tokens in response |
| `seed` | int | >= 0 | For reproducible outputs |

Additional parameters supported via `extra_kwargs` (not per-slot):

| Parameter | Type | Description |
|-----------|------|-------------|
| `top_p` | float | Nucleus sampling |
| `timeout` | float | Request timeout in seconds |
| `presence_penalty` | float | Penalise tokens already in output |
| `frequency_penalty` | float | Penalise frequent tokens |

## Default Temperatures

Each response type has a sensible default temperature:

| Response Type | Default Temp | Rationale |
|--------------|--------------|-----------|
| `extract` | 0.0 | Deterministic verbatim extraction |
| `pick`/`decide`/`bool` | 0.0 | Consistent selection/decision making |
| `int`/`date_rule` | 0.0 | Structured data needs precision |
| `number`/`date`/`time`/`duration` | 0.1 | Slight flexibility for interpretation |
| `think` | 0.5 | Balanced reasoning |
| `respond`/`default` | 0.7 | Natural but controlled responses |
| `speak` | 0.8 | More conversational variety |
| `poem` | 1.5 | Maximum creativity |

## Per-Slot Syntax

Override parameters on any completion slot using pipe syntax:

```
[[extract:quote|temperature=0.5]]
[[think:reasoning|temperature=0.3]]
[[poem:verse|temperature=1.8]]
[[extract:data|model=gpt-4o-mini]]
[[think:analysis|temperature=0.4,model=gpt-5]]
```

Model-specific options (like `min`, `max`, `required`) are preserved alongside LLM parameters:

```
[[number:score|min=0,max=100,temperature=0.0]]
[[date:when|required,temperature=0.2]]
```

The parser separates:
- **LLM parameters**: `temperature`, `thinking`, `model`, `max_tokens`, `seed` -- passed to the LLM
- **Slot options**: `min`, `max`, `required` -- used by response model factories

## Thinking / Extended Reasoning

Use `thinking` to enable extended reasoning (chain-of-thought) on models that support it. The thinking parameter controls how much reasoning the model performs before producing its answer.

```
[[think:analysis|thinking=high]]
[[think:deep_reasoning|thinking=xhigh,temperature=0.3]]
[[pick:choice|yes,no|thinking=low]]
```

**Levels:**
- `off` -- explicitly disable thinking (for models where it's on by default)
- `minimal`, `low`, `medium`, `high`, `xhigh` -- increasing reasoning depth

**Omitting `thinking`** means struckdown does not interfere -- the provider's default behaviour applies. This is distinct from `thinking=off`, which explicitly disables it.

**Provider support:** Thinking is supported by Claude (Opus, Sonnet with extended thinking), OpenAI o-series models, and other providers via pydantic-ai's unified `ModelSettings.thinking` field. If a provider does not support the requested thinking level, the error propagates -- it is not silently dropped.

### Example: mixing thinking levels

```
Analyse this document carefully:
{{source}}

First, reason through the key themes:
[[think:reasoning|thinking=high]]

Then pick the dominant theme:
[[pick:theme|politics,economics,culture,science|thinking=off]]

Finally, write a summary:
[[respond:summary|temperature=0.7]]
```

## Streaming

Free-text slots (`respond`, `speak`, `think`, `extract`, `poem`) are streamed token-by-token by default when using the CLI or the async incremental API. Constrained slots (`pick`, `bool`, `int`, etc.) complete atomically.

Streaming is transparent to template authors -- no syntax changes required. It is controlled by the `stream` parameter on `chatter_incremental_async()` (default: `True` for async, `False` for sync wrapper).

## Unsupported Parameters

When a parameter is not recognised by struckdown, the default behaviour is to log a warning and drop it:

```
WARNING: Dropped unsupported LLM parameters: top_k, custom_param
```

For stricter handling, enable `strict_params` to raise an error instead:

```python
# Python API
result = chatter(template, strict_params=True)
```

```bash
# CLI
sd chat --strict-params -p template.sd
```

This is useful for catching typos or ensuring all parameters are supported by the current provider.

## Priority Order

LLM parameters are applied in this priority order (highest to lowest):

1. **Slot-specific overrides**: `[[type:var|temperature=X]]`
2. **Return type defaults**: `ResponseModel.llm_config`
3. **Global extra_kwargs**: Passed to `chatter()` function

## Examples

### Basic usage

```python
from struckdown import chatter

# uses default temperature for each type
result = chatter("""
Extract the quote: "Hello world"
[[extract:quote]]

Think about it:
[[think:analysis]]

Be creative:
[[poem:verse]]
""")

# quote uses temp=0.0 (deterministic)
# analysis uses temp=0.5 (balanced)
# verse uses temp=1.5 (creative)
```

### With overrides

```python
result = chatter("""
Extract carefully with slight flexibility:
[[extract:quote|temperature=0.1]]

Think very precisely:
[[think:analysis|temperature=0.2]]

Use a specific model:
[[think:reasoning|model=gpt-4o-mini]]
""")
```

### With thinking

```python
result = chatter("""
Reason deeply about this problem:
[[think:reasoning|thinking=high]]

Quick classification (no extended reasoning needed):
[[pick:category|A,B,C|thinking=off]]
""")
```

### Cost optimisation

```python
result = chatter("""
Simple extraction (cheap, deterministic):
[[extract:data|model=gpt-4o-mini,temperature=0.0]]

Complex reasoning (expensive, careful):
[[think:analysis|model=gpt-5,temperature=0.3,thinking=high]]
""")
```
