---
layout: default
title: Actions
parent: Reference
nav_order: 3
---

# Actions Reference

Actions are Python functions callable from templates using `[[@action:var|params]]` syntax.

## Syntax

```
[[@action:variable]]                    # No parameters
[[@action:variable|param=value]]        # Named parameter
[[@action:variable|param1=v1,param2=v2]] # Multiple parameters
[[@action:variable|positional_arg]]     # Positional parameter
```

### Parameter Types

| Syntax | Type | Description |
|--------|------|-------------|
| `param=varname` | Variable | Reference context variable |
| `param="literal"` | String | Literal string value |
| `param=123` | Number | Literal number |
| `param=true` | Boolean | Literal boolean |

### Examples

```python
# Variable reference - looks up 'topic' in context
[[@search:results|query=topic]]

# Literal string
[[@search:results|query="climate change"]]

# Mixed
[[@search:results|query=topic,limit=10]]
```


## Built-in Actions

### @fetch

Fetch content from a URL.

```
[[@fetch:content|url="https://example.com"]]
[[@fetch:content|url=user_url]]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | `str` | Yes | URL to fetch |

**Returns:** Page content as text (HTML converted to markdown).

---

### @search

Web search using DuckDuckGo.

```
[[@search:results|query="python tutorials",n=5]]
[[@search:results|query=user_query]]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` | Yes | - | Search query |
| `n` | `int` | No | `10` | Number of results |

**Returns:** Search results as formatted text.

---

### @evidence

Retrieve evidence from a vector store (RAG).

```
[[@evidence|topic]]
[[@evidence:docs|query=extracted_topic,n=3]]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` | Yes | - | Search query |
| `n` | `int` | No | `5` | Number of documents |

**Returns:** Retrieved documents as formatted text.

Requires evidence store configuration. See [RAG Tutorial](../tutorials/rag-retrieval.md).


## Registering Custom Actions

```python
from struckdown import Actions

@Actions.register('myaction')
def my_action(context, param1: str, param2: int = 10):
    """Action description"""
    return f"Result: {param1} x {param2}"
```

### Registration Options

```python
@Actions.register(
    'myaction',
    on_error='propagate',      # Error handling: 'propagate', 'return_empty', 'return_default'
    default='fallback value',  # Default for 'return_default'
    return_type=MyModel,       # Pydantic model for JSON parsing
)
def my_action(context, ...):
    ...
```

### Error Handling

| Mode | Behaviour |
|------|-----------|
| `propagate` | Raise exception (default) |
| `return_empty` | Return empty string on error |
| `return_default` | Return `default` value on error |


## Context Object

The first parameter `context` is a dict-like object containing:

- All template variables
- Previously extracted slot values
- Results from prior checkpoints

```python
@Actions.register('summarize')
def summarize(context):
    # Access extracted values
    name = context.get('name', 'unknown')
    items = context.get('items', [])
    return f"{name} has {len(items)} items"
```


## Type Coercion

Parameters are automatically coerced based on function signature:

```python
@Actions.register('multiply')
def multiply(context, value: int, factor: float = 2.0):
    return str(value * factor)

# String "10" converted to int, "1.5" to float
[[@multiply:result|value="10",factor="1.5"]]
```

Supported types:

| Type | Coercion |
|------|----------|
| `str` | Direct (default) |
| `int` | `int(value)` |
| `float` | `float(value)` |
| `bool` | `"true"` → `True`, `"false"` → `False` |
| `List[T]` | JSON parse |
| `Dict[str, T]` | JSON parse |


## Return Values

Actions must return a string or a JSON-serializable object:

```python
# String return
@Actions.register('greet')
def greet(context, name: str):
    return f"Hello, {name}!"

# Object return (serialized to JSON)
@Actions.register('get_user')
def get_user(context, id: int):
    return {"name": "Alice", "id": id}
```

### Pydantic Models

Specify `return_type` for automatic deserialization:

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str

@Actions.register('get_user', return_type=User)
def get_user(context, id: int):
    return '{"name": "Alice", "email": "alice@example.com"}'

# In template, result is a User object
[[@get_user:user|id=123]]
```


## Actions vs Slots

| Feature | Action `[[@...]]` | Slot `[[...]]` |
|---------|-------------------|----------------|
| Execution | Python function | LLM call |
| Cost | Free | API tokens |
| Determinism | Deterministic | Non-deterministic |
| Use case | Data retrieval, transforms | Generation, reasoning |
