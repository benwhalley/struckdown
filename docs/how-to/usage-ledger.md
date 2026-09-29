---
layout: default
title: Usage Ledger
parent: How-To Guides
nav_order: 5
---

# Usage Ledger

Keep a record of every call struckdown makes: what model, how many tokens, what it cost, and what it belonged to.

## Listening for calls

struckdown prices every call it makes. The ledger is how it tells you. Each provider request -- a completion, one round of a tool loop, an embedding batch, a transcription, a cache hit, a request that failed -- becomes a `UsageRecord`, handed to the handlers you registered:

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

The record's fields:

| Field | Meaning |
|---|---|
| `kind` | `chat`, `embedding` or `transcription` |
| `model_name`, `provider`, `base_url_host` | what was called, and where the request went (host only, never the key) |
| `model_ref` | your own identifier for the model, if you set one (below) |
| `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens` | token counts; cache reads and writes are a subset of input, reasoning a subset of output |
| `audio_seconds` | for transcription |
| `cost` | a `CostBreakdown`: `input_cost`, `output_cost`, `total_cost`, the per-Mtok prices used and their `source` (`stored`, `pydantic_ai`, `genai_prices`, `audio_rate`). `None` when nothing could price the call |
| `duration_ms`, `started_at` | timing |
| `cache_hit` | served from struckdown's own response cache; costs 0 |
| `ok`, `error_class` | a failed request is still a record |
| `slot` | the template slot the call was made for, when there is one |
| `payload` | the request and response bodies, only if a handler asked (below) |

A tool loop is several requests; each is its own record, and only the last carries the run's duration, because the provider does not time them separately.

A handler that raises is logged and never fails the call.

### Bodies

Assembling the messages sent and the parts received costs memory on every call, so it is only done when a handler asks. Set `wants_payload = True` on the handler (an attribute, or a property that decides at call time):

```python
def keep(record):
    if record.payload:
        store(record.payload.request, record.payload.response)

keep.wants_payload = True
```

### Threads and event loops

struckdown moves between threads and event loops on your behalf: a sync `complete()` runs an event loop, an async `structured_chat_async` runs the sync call on a worker thread. Records are never dispatched from those worker threads. For the sync API (`complete`, `complete_incremental`, `structured_chat`, `get_embedding`, `transcribe`) your handler is called in your own thread, after the work; for the async API it is called on your event loop, and if it returns an awaitable, that is awaited. A handler that writes to a database can therefore do so on your connection, or hand the write to a thread with `sync_to_async` and return the coroutine.

## Pricing

Cost comes from the first of: prices you supplied, pydantic-ai's own price data, the `genai-prices` snapshot. To supply prices, put them on the `ModelSpec` or call `set_model_pricing`:

```python
from struckdown.llm import set_model_pricing

set_model_pricing(2.0, 8.0, cache_read_cost_per_mtok=0.2, cache_write_cost_per_mtok=2.5)
```

Cached prompt tokens are charged at the cache rate when one is given and at the input rate if not, which overstates rather than hides. `ModelSpec.model_ref` (or `set_model_ref(...)`) names your own record for the model -- a database row id, say -- and rides on every record, so a ledger can join back to it without matching on the name.

## The Django tables

`struckdown.contrib.django` (app label `sd_models`) writes every record to a table, so a project on struckdown gets a costs page without touching its call sites. It needs `django.contrib.contenttypes` and `django.contrib.auth`.

```python
INSTALLED_APPS += ["struckdown.contrib.django"]
MIDDLEWARE += ["struckdown.contrib.django.spans.SpanMiddleware"]   # after AuthenticationMiddleware
```

```python
# where the Celery app is built
from struckdown.contrib.django.spans import install_celery_hooks
install_celery_hooks()
```

Then `manage.py migrate`. Three tables:

- **`LLMCall`**, one row per provider request. The model's name, prices and residency are copied onto the row as they were when the call was made, so repricing or deleting an `AvailableModel` leaves history alone; the foreign key to it is for filtering only. `input_cost` and `output_cost` are USD, null when the call could not be priced (never 0); `total_cost` is a generated column. Cache hits and failures are rows too.
- **`LLMSpan`**, what a call belonged to. The middleware opens one per request and the Celery hooks one per task, named `http:<view name>` and `celery:<task name>`; a row is written only if a call was made inside it. You can name a block yourself, with the user and the record that holds the text:

  ```python
  from struckdown.contrib.django.spans import llm_span

  with llm_span("assistant.turn", user=request.user, obj=conversation):
      sd.complete(...)
  ```

  Spans nest. `open_span()` / `close_span()` serve a generator that cannot use `with`; `aopen_span()` / `aclose_span()` serve async code. Pass `capture=True` to keep the bodies of the calls inside.
- **`LLMCallPayload`**, the bodies, one-to-one with a call. Written only inside a `capture=True` span or when `STRUCKDOWN_LEDGER_CAPTURE_PAYLOADS` is on.

Calls that do not go through struckdown -- a direct OpenAI SDK call -- are recorded with `ledger.record_openai_response(response, model_name=..., available_model=row)` or the lower-level `ledger.record_call(...)`, priced from the `AvailableModel` row's stored rates.

`AvailableModel.get_llm_and_credentials()` and `to_spec()` set the stored prices (cache rates included) and the `model_ref` for the calls that follow, so a project that resolves its models through those rows gets exact pricing and exact joins.

### The costs page

`/admin/sd_models/llmcosts/` totals calls by span root ("feature"), by model and by month, with counts and averages per user and never a name; calls that could not be priced are counted apart and left out of every total. Read-only changelists for calls, spans and payloads sit beside it. The admins are plain `ModelAdmin`s registered on the default site; a project with a themed admin subclasses them (`admin_ledger.LLMCostsAdmin` and friends) and re-registers, and `LLMCostsAdmin.extra_sections()` adds a project's own tables to the page.

### Settings and retention

| Setting | Default | |
|---|---|---|
| `STRUCKDOWN_LEDGER_ENABLED` | `True` | write rows at all |
| `STRUCKDOWN_LEDGER_CAPTURE_PAYLOADS` | `False` | keep every call's bodies |
| `STRUCKDOWN_LEDGER_PAYLOAD_DAYS` | `30` | payload retention; 0 keeps them |
| `STRUCKDOWN_LEDGER_CALL_DAYS` | `400` | call and span retention; 0 keeps them |

`manage.py sd_prune_ledger [--dry-run]` applies both windows. Schedule it nightly.

### What is not recorded

Local embedding and cross-encoder models make no API call and cost nothing, so they leave no row. A call made inside a database transaction that then rolls back loses its row with the transaction; the ledger writes on the caller's connection on purpose, so it never holds a second one.
