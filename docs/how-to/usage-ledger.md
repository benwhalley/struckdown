---
layout: default
title: Record LLM Usage
parent: How-To Guides
nav_order: 5
---

# Record LLM Usage

Keep a record of every request struckdown makes: which model, how many tokens, what it cost, and which slot it was for. This page covers the plain Python hooks. For a Django project, [Record Usage in Django](django-usage-ledger.md) writes the records to tables and gives you a costs page.

## Listen for calls

Register a handler. struckdown calls it once per provider request -- a completion, one round of a tool loop, an embedding batch, a transcription, a response served from struckdown's cache, a request that failed -- with a `UsageRecord`:

```python
import struckdown as sd

def keep(record: sd.UsageRecord):
    print(record.kind, record.model_name, record.input_tokens, record.output_tokens)
    if record.cost is not None:
        print(f"${record.cost.total_cost:.5f} ({record.cost.source})")

sd.register_usage_handler(keep)          # every call in this process

with sd.usage_tracking(keep):            # or only the calls inside a block
    sd.complete("Tell a joke.\n[[joke]]", model=model, credentials=creds)
```

`unregister_usage_handler(keep)` removes a process-wide handler. `usage_tracking` holds one handler per context: a nested block replaces the outer one until it ends. A handler that raises is logged and never fails the call.

The fields a handler usually reads:

| Field | Meaning |
|---|---|
| `kind` | `chat`, `embedding` or `transcription` |
| `model_name`, `provider`, `base_url_host` | what was called, and where the request went (the host only, never the key) |
| `input_tokens`, `output_tokens` | token counts; `input_tokens` includes cached tokens |
| `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens` | cache reads and writes are part of input, reasoning part of output |
| `cost` | a `CostBreakdown`, or `None` when nothing could price the call |
| `cache_hit` | served from struckdown's own response cache; cost 0 |
| `ok`, `error_class` | a failed request is still a record |
| `slot` | the template slot the call was made for, when there is one |

The full list is in the [Usage Ledger reference](../reference/usage-ledger.md#usagerecord).

A tool loop is several requests, and each is its own record. Only the last carries the run's duration, because pydantic-ai does not time the rounds separately. An embedding call is one record per batch sent. Embeddings served from the embedding cache and local embedding models make no request and produce no record.

## Price the calls

A record's `cost` comes from the first of: rates you supplied, pydantic-ai's price data, the genai-prices snapshot. To supply rates, set them on the `ModelSpec` you pass to `complete(spec=...)`:

```python
from struckdown import ModelSpec, complete

spec = ModelSpec(
    model_name="openai:gpt-4.1-mini",
    api_key="sk-...",
    input_cost_per_mtok=0.40,
    output_cost_per_mtok=1.60,
    cache_read_cost_per_mtok=0.10,
    model_ref="gpt-4.1-mini@openai",
)
result = complete("Tell a joke.\n[[joke]]", spec=spec)
```

or call `set_model_pricing` before calls made with `model=` and `credentials=`:

```python
from struckdown import set_model_pricing

set_model_pricing(0.40, 1.60, cache_read_cost_per_mtok=0.10)
```

Both an input and an output rate are needed. Cached prompt tokens are charged at the cache rate when one is given and at the input rate if not. [Cost Tracking](../explanation/cost-tracking.md) explains the arithmetic.

`model_ref` (or `set_model_ref(...)`) is your own identifier for the model -- a database row id, say. It is copied onto every record, so a ledger can join back to your model table without matching on the name.

The rates and `model_ref` are held in context variables and apply to every call that follows in the same context, whichever model it uses. Set them immediately before the call they belong to. If you resolve another model in between -- an embedding model to embed a query, for instance -- set them again before the call.

## Keep the request and response bodies

Building the messages sent and the output received costs memory on every call, so it is only done when a handler asks. Set `wants_payload = True` on the handler, as an attribute or as a property that decides at call time:

```python
def keep(record):
    if record.payload:
        store(record.payload.request, record.payload.response)

keep.wants_payload = True
```

What `payload` holds depends on the call. For a completion, `request` is `{"messages": [...]}` and `response` is `{"output": ...}`. For a tool loop, only the first round's record carries the request, and each round's `response` is `{"parts": [...]}`. For an embedding batch, `request` is `{"texts": <count>}` and `response` is `None`. Transcriptions carry no payload.

## Threads and event loops

struckdown moves between threads and event loops on your behalf: a sync `complete()` runs an event loop, and `structured_chat_async` runs the sync call on a worker thread. Records are not dispatched from those threads. Each hop holds its records back and dispatches them once it returns:

- from the sync API (`complete`, `complete_incremental`, `get_embedding`), your handler is called in your own thread, after the call has finished;
- from the async API, it is called on your event loop, and if it returns an awaitable, that is awaited.

A handler that writes to a database can therefore use your thread's connection, or, from async code, hand the write to a thread with `sync_to_async` and return the coroutine.

`structured_chat` and `transcribe` called directly make no hop, so they call the handler in the thread that made the call.

The held records are dispatched whether or not the call raises. A sync `complete()` whose second slot fails reports the first slot's call and an `ok=False` record for the second.

If your own code runs struckdown on a worker thread and you want the records back in the calling thread, do what struckdown does:

```python
from struckdown import held_usage

with held_usage():
    result = run_in_worker(lambda: sd.complete(prompt, spec=spec))
```

The records are dispatched when the block exits, including when it raises. The worker must run in a copy of the caller's context (`contextvars.copy_context().run`, or a helper that copies it, such as `asyncio.to_thread`) for the records to reach the block. From async code, use `async with struckdown.ledger.aheld_usage():`.
