---
layout: default
title: Actions
parent: Reference
nav_order: 3
---

# Actions Reference

Actions are Python functions called from templates with `[[@action:var|params]]`
syntax. They run instead of an LLM call.

## Syntax

```
[[@action:variable]]                    # No parameters
[[@action:variable|param=value]]        # Named parameter
[[@action:variable|param1=v1,param2=v2]] # Multiple parameters
[[@action:variable|positional_arg]]     # Positional parameter
[[@action]]                             # Result stored under the action's name
```

Positional parameters fill the function's parameters in order, skipping
`context`.

### Parameter Types

| Syntax | Type | Description |
|--------|------|-------------|
| `param=varname` | Variable | Looks up `varname` in the context |
| `param="literal"` | String | Literal string; double quotes only |
| `param=123`, `param=1.5` | Number | Literal number |
| `varname` (positional) | Variable | An unquoted or quoted identifier is looked up in the context |
| `"two words"` (positional) | String | A quoted value that is not an identifier stays literal |

Any unquoted identifier is treated as a variable name. If the name is not in
the context, struckdown logs a warning and passes the name itself as a
string. So `raw=true` passes the string `"true"` (with a warning), which a
`bool` parameter then coerces to `True`; see [Type Coercion](#type-coercion).

Single-quoted values do not parse.

### Examples

```
# Variable reference - looks up 'topic' in context
[[@search:results|query=topic]]

# Literal string
[[@search:results|query="climate change"]]

# Mixed
[[@search:results|query=topic,max_results=10]]
```


## Built-in Actions

| Action | Purpose |
|--------|---------|
| [`@fetch`](#fetch) | Fetch a URL as markdown or HTML |
| [`@search`](#search) | Web search via DuckDuckGo |
| [`@evidence`](#evidence) | Keyword search over local text files |
| [`@timestamp`](#timestamp) | Current date and time |
| [`@history`](#history) | Load conversation turns from a file |
| [`@markdownify`](#markdownify) | Convert HTML to markdown |
| [`@set`](#set) | Placeholder; currently always returns an empty string |
| [`@noop`](#noop) | Does nothing; stands in for unknown actions |

### @fetch

Fetch content from a URL.

```
[[@fetch:content|url="https://example.com"]]
[[@fetch:content|url=user_url]]
[[@fetch:page|url=user_url,max_chars=0]]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | `str` | Yes | - | URL to fetch |
| `raw` | `bool` | No | `false` | Return raw HTML instead of markdown |
| `timeout` | `int` | No | `8` | Request timeout in seconds; the default comes from `STRUCKDOWN_WEB_FETCH_TIMEOUT` |
| `max_chars` | `int` | No | `64000` | Maximum characters returned; `0` means no limit |
| `playwright` | `bool` | No | `false` | Fetch with a headless browser. Without it, requests are made directly and fall back to Playwright on a 401 or 403 |

Playwright needs the `playwright` package (`pip install playwright`, or
struckdown's `dev` extra) and `playwright install chromium`.

**Returns:** Page content as text. By default the main content is extracted
and converted to markdown.

Blocked in the hosted playground (`sd serve`). An empty or invalid URL raises
an error.

---

### @search

Web search using DuckDuckGo. By default each result page is fetched and its
content included.

```
[[@search:results|query="python tutorials",max_results=3]]
[[@search:results|query=user_query]]
[[@search:results|query=user_query,embed=false]]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` | Yes | - | Search query |
| `max_results` | `int` | No | `5` | Maximum number of results |
| `embed` | `bool` | No | `true` | Include each page's fetched content; `false` gives titles and snippets only |
| `raw` | `bool` | No | `false` | Keep fetched pages as HTML instead of markdown |
| `timeout` | `int` | No | `8` | Per-request timeout in seconds (from `STRUCKDOWN_WEB_FETCH_TIMEOUT`) |
| `max_tokens` | `int` | No | `10000` | Maximum characters per fetched page; `0` means no limit |
| `playwright` | `bool` | No | `false` | Fetch pages with a headless browser |

**Returns:** Search results as formatted markdown text, or
`No results found for: <query>`.

Blocked in the hosted playground (`sd serve`).

---

### @evidence

Keyword search (BM25) over `.txt` and `.md` files in an `evidence/`
directory. There is no vector store or embedding step.

```
[[@evidence|topic]]
[[@evidence:docs|query=extracted_topic,n=3]]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` or list of `str` | Yes | - | Search query, or several; results are merged |
| `n` | `int` | No | `3` | Number of passages returned |

**Where it looks:**

1. If the context has `evidence_folder` (a path or list of paths), only those.
2. Otherwise `evidence/` next to the template file, and `evidence/` in the
   current directory.

Files are split into passages of about 500 characters. In the hosted
playground, uploaded evidence is searched instead.

**Returns:** The top `n` passages, each headed by its file name, separated by
`---`. Returns an empty string if no folder or no match is found, or if the
search fails.

```bash
sd chat -p prompt.sd -c evidence_folder=./my_docs
```

See [RAG Tutorial](../tutorials/rag-retrieval.md).

---

### @timestamp

The current local date and time.

```
[[@timestamp]]
[[@timestamp:today|format="%Y-%m-%d"]]
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `format` | `str` | No | ISO 8601 | `strftime` format string |

---

### @history

Loads conversation turns from a text file and adds them to the conversation
as separate messages. Each non-empty line is one turn; roles alternate.

```
[[@history]]
[[@history|n=5]]
[[@history|role="assistant"]]
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `filename` | `str` | No | `context["_history_file"]` | Path to the file, relative to the current directory |
| `first` | `str` | No | `"assistant_first"` | Who speaks first: `"assistant_first"` or `"user_first"` |
| `n` | `int` | No | all | Keep only the last `n` turns (after role filtering) |
| `role` | `str` | No | all | Keep only `"user"`, `"assistant"` or `"system"` turns |

`sd chat --history FILE` sets `_history_file`. In `sd chat -i`, the live
conversation is used instead of the file. Unknown parameters are logged and
ignored. With no file given, the action raises an error. Registered with
`default_save=False`, so a bare `[[@history]]` is not stored in the context.

---

### @markdownify

Converts HTML to markdown.

```
[[@markdownify:md|html=raw_html]]
[[@markdownify:md|html=raw_html,extract_content=false]]
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `html` | `str` | Yes | - | HTML to convert |
| `extract_content` | `bool` | No | `true` | Extract the main readable content first; `false` converts the whole page |

---

### @set

Intended for setting a variable without an LLM call. The current
implementation ignores its parameters and always returns an empty string, so
`[[@set:greeting|value="Hello"]]` sets `greeting` to `""`, not `"Hello"`.

---

### @noop

Returns an empty string and saves nothing. When a template uses an action
name that is not registered, the parser warns (`Unknown action '@name'`) and
substitutes `@noop`.


## Registering Custom Actions

```python
from struckdown import Actions

@Actions.register('myaction')
def my_action(context, param1: str, param2: int = 10):
    """Action description"""
    return f"Result: {param1} x {param2}"
```

Sync functions run in a thread; `async def` functions are awaited.

### Registration Options

```python
@Actions.register(
    'myaction',
    on_error='propagate',      # 'propagate', 'return_empty' or 'return_default'
    default='fallback value',  # returned on error when on_error='return_default'
    default_save=True,         # store the result of a bare [[@myaction]]
    return_type=MyModel,       # Pydantic model used when reloading stored results
    allow_remote_use=True,     # allow in the hosted playground
    role='user',               # message role of the output in the conversation
)
def my_action(context, ...):
    ...
```

| Option | Default | Meaning |
|--------|---------|---------|
| `on_error` | `"propagate"` | Error handling; see below |
| `default` | `""` | Value returned on error when `on_error="return_default"` |
| `default_save` | `True` | Whether `[[@action]]` (no variable name) stores its result under the action's name. `[[@action:var]]` always stores it |
| `return_type` | `None` | Pydantic model for this action's output; see [Pydantic Models](#pydantic-models) |
| `allow_remote_use` | `True` | `False` blocks the action in the hosted playground (`sd serve`); `@fetch` and `@search` use this |
| `role` | `"user"` | Role of the message carrying the output: `"user"`, `"assistant"` or `"system"`. Ignored when the action returns a `MessageList` |

Actions can also be loaded from files: an `actions/` directory next to the
template or in the current directory is discovered automatically, and
`sd chat --tools` / `sd batch --tools` load a file or directory explicitly.

### Error Handling

| Mode | Behaviour |
|------|-----------|
| `propagate` | Raise exception (default) |
| `return_empty` | Return empty string on error |
| `return_default` | Return `default` value on error |


## Context Object

The first parameter, `context`, is a dict containing:

- All template variables
- Values of slots and actions that ran earlier in the template
- Internal keys set by struckdown or the CLI, such as `_template_path` and
  `_history_file`

```python
@Actions.register('summarise')
def summarise(context):
    name = context.get('name', 'unknown')
    items = context.get('items', [])
    return f"{name} has {len(items)} items"
```


## Type Coercion

Before the call, parameters are validated against the function's type hints
with a Pydantic model in its default (lax) mode:

```python
@Actions.register('multiply')
def multiply(context, value: int, factor: float = 2.0):
    return str(value * factor)

# String "10" converted to int, "1.5" to float
[[@multiply:result|value="10",factor="1.5"]]
```

Template values arrive as strings, except unquoted numbers, which arrive as
`int` or `float`, and variable references, which arrive as whatever the
context holds. What lax mode does with them:

| Type | Coercion |
|------|----------|
| `str` | Strings unchanged. Unannotated parameters are treated as `str`. An unquoted number (`n=3`) arrives as a number, and lax mode does not turn a number into a string, so it fails |
| `int` | `"10"` becomes `10`; `"1.5"` fails |
| `float` | `"1.5"` becomes `1.5` |
| `bool` | `"true"`, `"yes"`, `"1"`, `"on"` become `True`; `"false"`, `"no"`, `"0"`, `"off"` become `False` |
| `List[T]`, `Dict[str, T]` | A JSON string is **not** parsed and fails validation; the value must already be a list or dict, for example a context variable holding one |

If any parameter fails validation, coercion is abandoned for the whole call:
every parameter is passed to the function exactly as rendered, uncoerced,
and only a debug-level log records the failure. A function that relies on
coercion should check its inputs.


## Return Values

The returned value is stored unchanged as the slot's output and in the
context. What goes into the conversation is `str(value)`, sent as one
message with the action's `role`:

```python
# String return
@Actions.register('greet')
def greet(context, name: str):
    return f"Hello, {name}!"

# Dict return: stored as a dict; the prompt sees its Python repr
@Actions.register('get_user')
def get_user(context, id: int):
    return {"name": "Alice", "id": id}
```

Return `json.dumps(...)` if the model should see JSON rather than a Python
repr.

An action that returns a `MessageList` (as `@history` does) adds each message
separately, with its own role.

### Pydantic Models

An action can return a Pydantic model instance directly; it is stored as the
object and the conversation gets `str()` of it.

`return_type` on `register` does not convert what the function returns. A
returned JSON string stays a string. `return_type` is used only when saved
results are loaded back from JSON: if a stored output for that action is a
dict, it is rebuilt as an instance of `return_type` (and left as a dict,
with a warning, if that fails).

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str

@Actions.register('get_user', return_type=User)
def get_user(context, id: int):
    return User(name="Alice", email="alice@example.com")
```

```
[[@get_user:user|id=123]]
```


## Actions vs Slots

| Feature | Action `[[@...]]` | Slot `[[...]]` |
|---------|-------------------|----------------|
| Execution | Python function | LLM call |
| Cost | No API tokens (though `@search`/`@fetch` make web requests) | API tokens |
| Determinism | Depends on the function | Non-deterministic |
| Use case | Data retrieval, transforms | Generation, reasoning |
