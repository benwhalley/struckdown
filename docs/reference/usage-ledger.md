---
layout: default
title: Usage Ledger
parent: Reference
nav_order: 5
---

# Usage Ledger Reference

The records struckdown emits for every provider request, the functions that control them, and the Django tables, settings and commands that store them. For step-by-step use see [Record LLM Usage](../how-to/usage-ledger.md) and [Record Usage in Django](../how-to/django-usage-ledger.md); for how costs are worked out see [Cost Tracking](../explanation/cost-tracking.md).

## Core API

Importable from `struckdown` unless another module is given.

| Function | Description |
|---|---|
| `register_usage_handler(handler)` | Call `handler` for every record in this process. Registering the same handler twice has no effect. |
| `unregister_usage_handler(handler)` | Remove a process-wide handler. |
| `usage_tracking(handler)` | Context manager: call `handler` for records made inside the block, in this context only. One per context; a nested block replaces the outer handler until it ends. |
| `set_model_pricing(input_cost_per_mtok, output_cost_per_mtok, cache_read_cost_per_mtok=None, cache_write_cost_per_mtok=None)` | Stored pricing (USD per million tokens) for calls that follow in this context. Needs both input and output rates; `None` for either clears it. |
| `struckdown.llm.get_model_pricing()` | The current stored pricing as a `StoredPricing`, or `None`. |
| `set_model_ref(ref)` | Your own identifier for the model, copied onto records that follow in this context. `None` clears it. |
| `struckdown.ledger.get_model_ref()` | The current `model_ref`. |
| `set_audio_pricing(cost_per_minute)` | Per-minute rate for transcriptions that follow in this context. `None` clears it. |
| `deferred_usage()` | Context manager yielding a list. Records made inside the block, including on threads running in a copy of this context, are appended to it instead of dispatched. |
| `flush_usage(pending)` | Dispatch held records from sync code, in this thread. Inside an enclosing `deferred_usage` block, passes them up to it instead. |
| `struckdown.ledger.flush_usage_async(pending)` | The same from async code; awaits handlers that return an awaitable. |

`set_model_pricing`, `set_model_ref` and `set_audio_pricing` set [context variables](https://docs.python.org/3/library/contextvars.html). They stay set until replaced, and apply to every call in the context whatever its model. `complete(spec=...)` and `complete(registry=...)` set pricing and `model_ref` from the resolved `ModelSpec`.

### Handlers

```python
Handler = Callable[[UsageRecord], None | Awaitable[None]]
```

From async code, an awaitable a handler returns is awaited. From sync code it is run to completion, or scheduled on the running loop if the thread has one. A handler that raises is logged and does not fail the call.

A handler with a truthy `wants_payload` attribute (or property) receives `record.payload`; otherwise `payload` is `None` and the bodies are never built.

### UsageRecord

`struckdown.UsageRecord`, a dataclass. One per provider request.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | | `chat`, `embedding` or `transcription` |
| `model_name` | `str` | | Model as called, e.g. `openai:gpt-4.1-mini` |
| `provider` | `str` | `""` | From the response, or the `provider:` prefix of the name |
| `base_url_host` | `str` | `""` | Host of the credentials' `base_url`; never the key |
| `model_ref` | `str` or `None` | `None` | The `model_ref` set when the record was made |
| `input_tokens` | `int` | `0` | All prompt tokens, cached included |
| `output_tokens` | `int` | `0` | All completion tokens, reasoning included |
| `cache_read_tokens` | `int` | `0` | Prompt tokens read from the provider's cache |
| `cache_write_tokens` | `int` | `0` | Prompt tokens written to the provider's cache |
| `reasoning_tokens` | `int` | `0` | Part of `output_tokens` |
| `audio_seconds` | `float` or `None` | `None` | Transcription only |
| `cost` | `CostBreakdown` or `None` | `None` | `None` when the call could not be priced, and for failed requests |
| `duration_ms` | `float` or `None` | `None` | For a tool loop, set on the last round's record only |
| `started_at` | `datetime` or `None` | `None` | UTC |
| `cache_hit` | `bool` | `False` | Served from struckdown's response cache; cost 0 |
| `ok` | `bool` | `True` | `False` for a failed request |
| `error_class` | `str` | `""` | Exception class name of a failed request |
| `slot` | `str` or `None` | `None` | Template slot the call was made for |
| `provider_request_id` | `str` | `""` | The provider's response id, where given |
| `finish_reason` | `str` | `""` | Where given |
| `payload` | `UsagePayload` or `None` | `None` | Bodies, when a handler asked |
| `extra` | `dict` | `{}` | Unused by struckdown; free for handlers |

`record.total_cost` is `cost.total_cost`, or `None`.

### CostBreakdown

`struckdown.CostBreakdown`, a dataclass.

| Field | Type | Description |
|---|---|---|
| `input_cost` | `float` | USD: uncached input, cache reads and cache writes at their own rates; the whole cost of an embedding or transcription |
| `output_cost` | `float` | USD: completion tokens, reasoning included |
| `input_price`, `output_price` | `float` or `None` | USD per million tokens, as used; set for `stored` pricing only |
| `cache_read_price`, `cache_write_price` | `float` or `None` | As used; the input rate when no cache rate was set |
| `source` | `str` | `stored`, `pydantic_ai`, `genai_prices`, `audio_rate`, or `cache` for a response-cache hit |

`total_cost` is `input_cost + output_cost`.

### StoredPricing and cost_from_stored

`struckdown.ledger.StoredPricing(input_per_mtok, output_per_mtok, cache_read_per_mtok=None, cache_write_per_mtok=None)` holds caller-supplied rates.

`struckdown.ledger.cost_from_stored(pricing, *, input_tokens, output_tokens, cache_read_tokens=0, cache_write_tokens=0)` returns the `CostBreakdown` for those counts. `input_tokens` is the whole prompt; cache reads and writes are subtracted from it and charged at their own rates, or at the input rate where none is set.

### UsagePayload

`struckdown.UsagePayload(request=None, response=None)`.

| Call | `request` | `response` |
|---|---|---|
| Completion | `{"messages": [...]}` | `{"output": ...}` |
| Tool loop round | `{"messages": [...]}` on the first round, else `None` | `{"parts": [...]}` |
| Embedding batch | `{"texts": <count>}` | `None` |
| Transcription | no payload | |

### OpenTelemetry names

`struckdown.ledger.SEMCONV` maps record fields to the OpenTelemetry [`gen_ai` semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

| Field | Attribute |
|---|---|
| `kind` | `gen_ai.operation.name` |
| `model_name` | `gen_ai.request.model` |
| `provider` | `gen_ai.provider.name` |
| `input_tokens` | `gen_ai.usage.input_tokens` |
| `output_tokens` | `gen_ai.usage.output_tokens` |
| `cache_read_tokens` | `gen_ai.usage.cache_read.input_tokens` |
| `cache_write_tokens` | `gen_ai.usage.cache_creation.input_tokens` |
| `provider_request_id` | `gen_ai.response.id` |
| `finish_reason` | `gen_ai.response.finish_reasons` |
| `error_class` | `error.type` |

---

## Django

`struckdown.contrib.django`, app label `sd_models`. Requires Django 5.0+, `django.contrib.contenttypes` and `django.contrib.auth`.

### Settings

| Setting | Default | Description |
|---|---|---|
| `STRUCKDOWN_LEDGER_ENABLED` | `True` | Write rows at all |
| `STRUCKDOWN_LEDGER_CAPTURE_PAYLOADS` | `False` | Keep every call's bodies, not only those in a `capture=True` span |
| `STRUCKDOWN_LEDGER_PAYLOAD_DAYS` | `30` | Payload retention for `sd_prune_ledger`; `0` keeps them |
| `STRUCKDOWN_LEDGER_CALL_DAYS` | `400` | Call and span retention for `sd_prune_ledger`; `0` keeps them |

### Django tables

All three tables are written by the ledger; the admin shows them read-only.

#### LLMSpan (`llm_span`)

What a call belonged to: a request, a task, or a block a caller named.

| Field | Type | Description |
|---|---|---|
| `id` | UUID | |
| `name` | char(160) | `http:<view name>`, `celery:<task name>`, or the name given to `llm_span` |
| `parent` | FK `LLMSpan`, null | Enclosing span |
| `user` | FK user, null | |
| `content_type`, `object_id`, `obj` | generic FK | The record that holds the text of the exchange |
| `started_at`, `ended_at` | datetime | For a streaming response's root span, `ended_at` is when the view returned, before the body streamed |
| `attributes` | JSON | Extra keyword arguments to the span; `method` and `path` for a request |
| `capture_payloads` | bool | Opened with `capture=True` |

`root_name` (property) is the outermost span's name.

#### LLMCall (`llm_call`)

One row per provider request. The model's name, prices and residency are copied onto the row as they were at call time; `available_model` is a link for filtering, not a source of facts.

| Field | Type | Description |
|---|---|---|
| `created_at` | datetime | When the row was written; the costs page and pruning use this |
| `started_at` | datetime, null | When the request started |
| `duration_ms` | int, null | |
| `span` | FK `LLMSpan`, null | Innermost span; null when unattributed |
| `root_name` | char(160) | Outermost span's name, copied for grouping; `""` when unattributed |
| `slot` | char(120) | |
| `kind` | char(20) | `chat`, `embedding`, `transcription` |
| `model_name`, `provider`, `base_url_host` | char | |
| `data_residency` | char(10) | From the linked `AvailableModel` |
| `available_model` | FK `AvailableModel`, null | See [linking](#how-a-call-is-linked-to-an-availablemodel) |
| `input_price`, `output_price`, `cache_read_price`, `cache_write_price` | decimal(12,6), null | USD per million tokens, as used; null unless priced from stored rates |
| `price_source` | char(20) | As `CostBreakdown.source`; `""` when unpriced |
| `input_tokens` | int | All input, cached included |
| `cache_read_tokens`, `cache_write_tokens` | int | |
| `output_tokens` | int | |
| `reasoning_tokens` | int | Part of output |
| `audio_seconds` | float, null | |
| `input_cost`, `output_cost` | decimal(14,8), null | USD. Null means unpriced, never 0; set together or not at all |
| `total_cost` | generated decimal(14,8), null | `input_cost + output_cost`, stored |
| `cache_hit` | bool | Served from struckdown's response cache |
| `ok` | bool | |
| `error_class` | char(120) | |
| `provider_request_id` | char(120) | |
| `finish_reason` | char(40) | |

`unpriced` (property) is true when `input_cost` is null. The costs page counts a row as unpriced when its cost is null, `ok` is true and it is not a cache hit.

#### LLMCallPayload (`llm_call_payload`)

| Field | Type | Description |
|---|---|---|
| `call` | one-to-one `LLMCall`, cascade | |
| `request`, `response` | JSON, null | As `UsagePayload` |
| `created_at` | datetime | Used by pruning |

Written only when `STRUCKDOWN_LEDGER_CAPTURE_PAYLOADS` is on or the call's innermost span has `capture_payloads`.

#### LLMCosts

A proxy of `LLMCall` that stores nothing. Its admin changelist is the costs page, at `/admin/sd_models/llmcosts/`, and its `view_llmcosts` permission gates it.

### AvailableModel additions

| Member | Description |
|---|---|
| `cache_read_cost_per_mtok` | decimal, null. Blank means cache reads are charged at the input rate |
| `cache_write_cost_per_mtok` | decimal, null. Blank means cache writes are charged at the input rate |
| `stored_pricing()` | The row's rates as a `StoredPricing`, or `None` if the input or output rate is blank |
| `to_spec(default_credential=None)` | Now includes the cache rates and `model_ref=<row id>` |
| `get_llm_and_credentials()` | Also sets the cache rates, `model_ref` and the audio rate for the calls that follow |

`sd_update_prices` and `update_prices()` fill the cache rates where the price source publishes them.

### How a call is linked to an AvailableModel

1. The row whose id is the record's `model_ref`, if it exists.
2. Otherwise the only row with that `model_name`.
3. Otherwise, of the rows with that name, the only one whose credential's `base_url` contains the record's `base_url_host`.
4. Otherwise no link.

When struckdown could not price a successful, non-cached chat or embedding call and the linked row has stored rates, the row is written priced from those rates with `price_source` `stored`.

### Spans

`struckdown.contrib.django.spans`.

| Function | Description |
|---|---|
| `llm_span(name, *, user=None, obj=None, capture=False, **attributes)` | Context manager; opens a span, writes its row, yields the `SpanHandle`, closes it on exit |
| `open_span(name, *, user=None, obj=None, capture=False, eager=True, root=False, name_fn=None, **attributes)` | Opens a span in this context and returns its handle. `eager` writes the row now; `root` ignores any enclosing span; `name_fn` is a callable that supplies the final name later |
| `close_span(handle)` | Sets `ended_at` and restores the enclosing span |
| `aopen_span(name, *, user=None, obj=None, capture=False, **attributes)` | `open_span` from async code; the row is written on a thread |
| `aclose_span(handle)` | `close_span` from async code |
| `current_span()` | The current `SpanHandle`, or `None` |
| `install_celery_hooks()` | Connects `task_prerun` / `task_postrun` so each task runs in a root span named `celery:<task name>` |

`user` is a user instance, a primary key, or a callable returning either; an anonymous user is stored as null. `obj` is any model instance.

`SpanMiddleware` opens a lazy root span per request -- the row is written only if a call is made -- named `http:<view name>`, or `http:<path>` when the URL did not resolve. It is sync- and async-capable, and must come after `AuthenticationMiddleware`. For a streaming response it runs the body under the request's span: a sync body around each `next()`, an async body around each `yield` (so calls an async body makes before yielding are not attributed to it). It resets the span context when the view returns, discarding any span the view opened and left open; open spans for streamed work inside the body.

### Ledger writer

`struckdown.contrib.django.ledger`.

| Function | Description |
|---|---|
| `record_call(*, model_name, input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0, kind="chat", available_model=None, duration_ms=None, started_at=None, ok=True, error_class="", provider_request_id="", finish_reason="", request=None, response=None)` | Write an `LLMCall` for a call that did not go through struckdown, priced from `available_model`'s stored rates (or a row matched by name). Returns the row, or `None` if the ledger is disabled or the write failed |
| `record_openai_response(response, *, model_name, available_model=None, **kwargs)` | `record_call` from an OpenAI SDK chat completion or final stream chunk: reads `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`, `id` and the first choice's `finish_reason`. There is no cache-write count in this shape |
| `write_record(record)` | Write a `UsageRecord`. Logs and returns `None` on failure |
| `resolve_model(record)` | The linked `AvailableModel`, as above, or `None` |
| `handler` | The registered handler instance. Writes inline from sync code; from async code returns a `sync_to_async(thread_sensitive=True)` coroutine for struckdown to await |

A failed write is logged and never fails the LLM call.

### Costs aggregations

`struckdown.contrib.django.costs` returns plain dicts, for a dashboard tile or a report. `days` is one of the windows in `WINDOWS` (1, 7, 30, 365, or 0 for all time).

| Function | Returns |
|---|---|
| `dashboard(days)` | All of the below in one dict |
| `headline(days)` | Totals: `calls`, `users`, `input_tokens`, `cache_read_tokens`, `output_tokens`, `cost_in`, `cost_out`, `cost_total`, `unpriced`, `cache_hits`, `failed`, `per_call`, `per_user`, `cache_share` |
| `by_feature(days)` | The same per `root_name` (`"unattributed"` for none) |
| `by_model(days)` | The same per model name, provider and kind |
| `by_month(months=12)` | The same per calendar month |
| `per_user(days)` | `users`, `mean`, `median`, `max`; no names |

`per_call` is `cost_total` divided by calls minus unpriced calls and cache hits. Failed calls are not subtracted.

### Admin

`struckdown.contrib.django.admin_ledger` defines `LLMCallAdmin`, `LLMSpanAdmin`, `LLMCallPayloadAdmin` (read-only; deleting is allowed) and `LLMCostsAdmin` (the costs page). `LLMCostsAdmin.extra_sections(request, days)` returns extra tables for the page as a list of `{"title", "columns", "rows", "note"}` dicts, `rows` being a list of lists.

`register(site=admin.site)` registers the four admins on `site`, skipping any model already registered. The app calls it for the default site when its admin module loads.

### Management commands

| Command | Description |
|---|---|
| `sd_prune_ledger [--dry-run]` | Delete payloads older than `STRUCKDOWN_LEDGER_PAYLOAD_DAYS`, calls older than `STRUCKDOWN_LEDGER_CALL_DAYS`, and spans older than that with no calls left. `--dry-run` counts only; its span count leaves out spans that would become empty when their calls go |
| `sd_update_prices [--force] [--dry-run]` | Refresh stored prices, cache rates included, for active `AvailableModel`s from their credential's pricing source. Skips rows with `prices_updated_manually` unless `--force` |
