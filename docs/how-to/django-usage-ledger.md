---
layout: default
title: Record Usage in Django
parent: How-To Guides
nav_order: 6
---

# Record Usage in Django

Write every LLM call your Django project makes to a table, attribute each one to the request, task or feature it ran in, and read the totals on a costs page in the admin. No call site has to change. The table and field definitions are in the [Usage Ledger reference](../reference/usage-ledger.md#django-tables).

## Install the app

The contrib app needs Django 5.0 or later (the ledger uses a generated column) and the `django` extra:

```bash
pip install "struckdown[django]"
```

Add it after the Django apps it depends on. `contenttypes` and `auth` are required; `admin` is needed for the costs page.

```python
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.admin",
    # ...
    "struckdown.contrib.django",   # app label: sd_models
]
```

Create the tables:

```bash
python manage.py migrate sd_models
```

Migration `0008_usage_ledger` adds `llm_span`, `llm_call` and `llm_call_payload`, and the cache price columns on `AvailableModel`; `0009_llmcosts` adds the proxy model behind the costs page.

From now on every call struckdown makes in this process writes a row: the app registers its handler in `AppConfig.ready()`. Run the migration before you deploy code that makes calls. Until the tables exist each write fails and is logged; the call itself still succeeds. To stop writing rows, set `STRUCKDOWN_LEDGER_ENABLED = False`.

## Attribute calls to requests and tasks

Add the middleware after `AuthenticationMiddleware`, so the span knows the user:

```python
MIDDLEWARE = [
    # ...
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    # ...
    "struckdown.contrib.django.spans.SpanMiddleware",
]
```

If you use Celery, install the task hooks once, where the Celery app is built:

```python
# myproject/celery.py
from celery import Celery
from struckdown.contrib.django.spans import install_celery_hooks

app = Celery("myproject")
app.config_from_object("django.conf:settings", namespace="CELERY")
install_celery_hooks()
```

Each request now opens a span named `http:<view name>` (or `http:<path>` if the URL did not resolve), and each task a span named `celery:<task name>`. The span's row is written only when a call is made inside it, so a request that makes no LLM call writes nothing.

A call made outside both -- in a management command or a shell -- is recorded with no span, and the costs page lists it as "unattributed". Wrap such code in `llm_span` (next section).

Spans are held in a [context variable](https://docs.python.org/3/library/contextvars.html). They follow `sync_to_async`, asyncio tasks and struckdown's own worker threads, but not a plain `threading.Thread` or `ThreadPoolExecutor.submit`. Start such a thread with `contextvars.copy_context().run`:

```python
import contextvars

pool.submit(contextvars.copy_context().run, transcribe_piece, piece)
```

## Name a block of calls

Open a span around the calls that make up one piece of work, with the user and the record that holds the text of the exchange:

```python
from struckdown.contrib.django.spans import llm_span

with llm_span("assistant.turn", user=request.user, obj=conversation):
    result = sd.complete(prompt, spec=spec)
```

Extra keyword arguments are stored on the span as JSON attributes: `llm_span("triage", obj=question, queue="urgent")`. Spans nest; each call is linked to the innermost span, and the costs page groups by the outermost, which it calls the feature. Inside a request, that is the request's `http:` span.

A span you name is written when it opens, so it survives a crash inside it.

Where `with` does not fit:

- in a generator, `span = open_span(...)` and `close_span(span)` in a `finally`;
- in async code, `span = await aopen_span(...)` and `await aclose_span(span)`, which do their database work on a thread.

### Streaming responses

Open the span inside the body that produces the stream, not in the view before it returns the response:

```python
from django.http import StreamingHttpResponse
from struckdown.contrib.django.spans import aclose_span, aopen_span

async def answer(request, pk):
    conversation = await Conversation.objects.aget(pk=pk)

    async def body():
        span = await aopen_span("assistant.turn", user=request.user, obj=conversation)
        try:
            async for event in sd.complete_incremental_async(prompt, spec=spec):
                yield render(event)
        finally:
            await aclose_span(span)

    return StreamingHttpResponse(body())
```

When the view returns, the middleware restores the span context to what it was before the request, which discards a span the view opened. A body that opens no span of its own still has its calls recorded under the request's `http:` span, sync or async.

## Price calls from your model table

Resolve each model through its `AvailableModel` row:

```python
row = AvailableModel.objects.get(pk=model_id)

result = sd.complete(prompt, spec=row.to_spec())

# or, where you need the LLM and credentials separately
llm, credentials = row.get_llm_and_credentials()
result = sd.complete(prompt, model=llm, credentials=credentials)
```

Either way struckdown prices the calls at the row's stored rates (cache rates included) and names the row as their `model_ref`, so each `LLMCall` links to the row exactly. `complete(spec=...)` sets these as the call starts; `get_llm_and_credentials()` sets them when you call it. It also sets the per-minute audio rate from `cost_per_audio_minute`, which is the only way a transcription gets priced.

The rates and `model_ref` are context variables and stay set for everything that follows in the same context. Resolving a second model before the call -- an embedding model to embed a query, say -- re-points them, and the call is then priced and linked as the second model. Call `get_llm_and_credentials()` immediately before the call it is for, or pass `spec=` so that the call sets them itself. After `complete_async(spec=...)` they remain set in your task for later calls.

A call made without a `model_ref` is linked by name: the row whose `model_name` matches, or, if several do, the one whose credential's `base_url` contains the host the call went to. If that does not settle it, the call has no link. When struckdown could not price a call and the linked row has rates, the ledger prices it from the row. It does not do this for transcriptions.

To fill and refresh the rates, including cache rates where the price source publishes them:

```bash
python manage.py sd_update_prices            # skips rows whose prices were set by hand
python manage.py sd_update_prices --force    # updates those too
```

## Record calls made outside struckdown

A direct OpenAI SDK call does not pass through struckdown, so record it yourself:

```python
from struckdown.contrib.django.ledger import record_openai_response

response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
record_openai_response(response, model_name="gpt-4.1-mini", available_model=row)
```

This reads the token counts, cached tokens, response id and finish reason from the response (or from the final chunk of a stream that included usage) and prices the call from the row's stored rates. For anything else, `record_call(model_name=..., input_tokens=..., output_tokens=..., available_model=row, ...)` takes the counts directly. A call through a row with no rates is written with a null cost. Both functions attribute the call to the current span.

## Keep request and response bodies

Bodies are not kept by default. To keep them for the calls in one block:

```python
with llm_span("assistant.turn", user=request.user, obj=conversation, capture=True):
    ...
```

`capture` applies to calls whose innermost span has it set: a nested span without `capture=True` turns it off for the calls inside it. To keep every call's bodies, set `STRUCKDOWN_LEDGER_CAPTURE_PAYLOADS = True`. Bodies are pruned after `STRUCKDOWN_LEDGER_PAYLOAD_DAYS` (30 by default).

## Read the costs page

The page is at `/admin/sd_models/llmcosts/`, for staff with the `sd_models.view_llmcosts` permission. It shows, for the last 24 hours, 7 days, 30 days, year or all time:

- the headline: calls, users, tokens, the share of input read from the provider's cache, cost, unpriced calls, cache hits and failures;
- the same figures by feature (outermost span) and by model;
- the last twelve months, month by month;
- per-user counts and averages -- mean, median, maximum -- with nobody named.

Calls that could not be priced are counted apart and left out of every total. The "per call" average currently includes failed calls in its denominator, which pulls it down when calls fail.

Read-only changelists for calls, spans and payloads sit beside it.

The admins are plain `ModelAdmin`s registered on the default `admin.site` when the app's admin module loads. For a themed admin, subclass them from `struckdown.contrib.django.admin_ledger` and register your subclasses in their place:

```python
from django.contrib import admin
from struckdown.contrib.django import admin_ledger
from struckdown.contrib.django.models import LLMCosts


class CostsAdmin(admin_ledger.LLMCostsAdmin, ThemedModelAdmin):
    def extra_sections(self, request, days):
        return [{
            "title": "Spend before the ledger",
            "columns": ["Feature", "Cost (USD)"],
            "rows": [["Triage", "12.40"]],
            "note": "From the old per-feature cost columns.",
        }]


if admin.site.is_registered(LLMCosts):
    admin.site.unregister(LLMCosts)
admin.site.register(LLMCosts, CostsAdmin)
```

Put the ledger class first, so its `changelist_view` (which renders the page) wins. The `is_registered` check matters because the app registers its own admins only if a model is not registered yet, and either admin module may load first. `extra_sections()` adds your own tables below the ledger's. The other classes are `LLMCallAdmin`, `LLMSpanAdmin` and `LLMCallPayloadAdmin`, for `LLMCall`, `LLMSpan` and `LLMCallPayload`.

## Prune old rows

Run the prune command nightly:

```bash
python manage.py sd_prune_ledger --dry-run   # count only
python manage.py sd_prune_ledger
```

It deletes payloads older than `STRUCKDOWN_LEDGER_PAYLOAD_DAYS` (default 30), calls older than `STRUCKDOWN_LEDGER_CALL_DAYS` (default 400, enough for a year-on-year comparison), and spans older than that which no longer have any calls. A window of 0 keeps rows for ever. With Celery beat:

```python
# myproject/tasks.py
from celery import shared_task
from django.core.management import call_command

@shared_task
def prune_llm_ledger():
    call_command("sd_prune_ledger")
```

```python
# settings.py
from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    "prune-llm-ledger": {
        "task": "myproject.tasks.prune_llm_ledger",
        "schedule": crontab(hour=3, minute=55),
    },
}
```

## What is not recorded

- Local embedding and cross-encoder models make no API call and cost nothing, so they leave no row.
- A call made inside a database transaction that then rolls back loses its row with the transaction. The ledger writes on the caller's connection so that it never holds a second one.
- A tool loop that stops part-way, at its usage limits or on an exception, writes one failed row without token counts, not a row per round it completed. See [Limitations](../explanation/cost-tracking.md#limitations).
