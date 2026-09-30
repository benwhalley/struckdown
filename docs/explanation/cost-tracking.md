---
layout: default
title: Cost Tracking
parent: Explanation
nav_order: 1
---

# Cost Tracking

When struckdown makes a call it works out what the call cost, where it can. You see that figure in two places:

- on what the call returns: `StruckdownResult.total_cost` from `complete()`, and `EmbeddingResultList.total_cost` from `get_embedding()`;
- on the usage records struckdown hands to your handlers, one per provider request, which is what a ledger such as the Django tables keeps. See [Record LLM Usage](../how-to/usage-ledger.md).

This page explains where the figures come from, how cached prompt tokens are charged, why an unknown cost is kept apart from a zero one, and where the two views give different answers. Every figure is an estimate: token counts multiplied by list prices. The provider's invoice is the authority.


## Where a price comes from

For each provider response struckdown uses the first of these that can price it:

1. **Stored pricing** -- rates you supplied for the current context. They come from the `ModelSpec` fields when `complete(spec=...)` or `complete(registry=...)` resolves a spec, from `set_model_pricing(...)`, or, in Django, from `AvailableModel.get_llm_and_credentials()`. Both an input and an output rate are needed; with only one, stored pricing is off. The breakdown's `source` is `stored`.
2. **pydantic-ai's price data**, via the response's `cost()` method. `source` is `pydantic_ai`.
3. **The [genai-prices](https://github.com/pydantic/genai-prices) snapshot**, looked up by model name and provider. `source` is `genai_prices`.

If none of them knows the model, the cost is unknown (below).

Two kinds of call are priced differently:

- **Embeddings** use the stored input rate if one is set, otherwise pydantic-ai's price for the batch. The cost is all on the input side.
- **Transcription** is priced per minute of audio, at the rate set with `set_audio_pricing(cost_per_minute)` (in Django, `AvailableModel.cost_per_audio_minute`, which `get_llm_and_credentials()` sets). The cost sits on the input side with an output cost of 0, and `source` is `audio_rate`. Without a rate, a transcription is unpriced.

Stored pricing lives in a [context variable](https://docs.python.org/3/library/contextvars.html), not on the model object, and stays set until something replaces it. The consequences are under [Limitations](#limitations).


## Input and output sides

A priced call carries a `CostBreakdown`:

- `input_cost` -- uncached prompt tokens, cache reads and cache writes, each at its own rate;
- `output_cost` -- completion tokens, reasoning tokens included;
- `total_cost` -- the two added together.

With stored pricing, the breakdown also records the per-million-token rates it used (`input_price`, `output_price`, `cache_read_price`, `cache_write_price`). pydantic-ai and genai-prices return a cost for each side but not the rates, so for those sources the rate fields are `None`.


## Cached prompt tokens

Providers bill prompt tokens read from their own prompt cache at a discount -- Anthropic at a tenth of the input rate, OpenAI at between a tenth and a half depending on the model -- and Anthropic charges a premium, 1.25 times the input rate, to write tokens into the cache. This is the provider's cache, not struckdown's response cache (see [Caching](caching.md)).

pydantic-ai reports `input_tokens` as the whole prompt, cache reads and writes included. With stored pricing struckdown therefore charges:

```
input_cost  = (input_tokens - cache_read_tokens - cache_write_tokens) x input rate
            + cache_read_tokens  x cache read rate
            + cache_write_tokens x cache write rate
output_cost = output_tokens x output rate
```

For example, a call with 100,000 prompt tokens, 80,000 of them read from the cache, and 1,000 output tokens, at $3, $0.30 and $15 per million for input, cache read and output:

| Part | Tokens | Rate per million | Cost |
|---|---:|---:|---:|
| Uncached input | 20,000 | $3.00 | $0.0600 |
| Cache read | 80,000 | $0.30 | $0.0240 |
| Output | 1,000 | $15.00 | $0.0150 |
| **Total** | | | **$0.0990** |

Charged at the input rate throughout, the same call would come to $0.3150.

A cache rate that is not set falls back to the input rate. That overstates the cost of cache-heavy use rather than hiding it. Set the rates with `cache_read_cost_per_mtok` and `cache_write_cost_per_mtok` on `ModelSpec` or `set_model_pricing`; in Django, `sd_update_prices` fills them on `AvailableModel` where the price source publishes them. genai-prices often publishes only the read rate; OpenRouter publishes both.

pydantic-ai and genai-prices price cache tokens from their own data.

Before 0.16, stored pricing charged every prompt token at the full input rate.


## Unknown is not zero

A call nobody could price is not free, and adding it to a total as 0 would make the total look complete when it is not. So:

- a usage record's `cost` is `None`, and the Django ledger writes `input_cost` and `output_cost` as null. The costs page counts these calls apart and leaves them out of every total;
- a failed request also has no cost;
- `EmbeddingResultList.total_cost` and `fresh_cost` are `None` if any fresh embedding's cost is unknown.

`StruckdownResult.total_cost` is the exception. It is always a float and adds an unknown cost as 0.0, so check `has_unknown_costs` before treating it as a complete figure:

```python
result = complete(prompt, spec=spec)

if result.has_unknown_costs:
    print(f"At least ${result.total_cost:.4f}; some calls could not be priced")
else:
    print(f"${result.total_cost:.4f}")
```

`all_costs_unknown` is true when no slot could be priced.

An API embedding whose batch could not be priced currently comes back with `cost` 0.0 rather than `None`, so `EmbeddingResultList.has_unknown_costs` stays false for it. The usage record for that batch has `cost=None`, so a ledger counts it correctly.


## Response cache hits

A slot served from struckdown's response cache makes no request. The two views record this differently:

- The **usage record** has `cache_hit=True` and a cost of 0, with `source` `cache`. A ledger counts what you spent.
- The **result** keeps the cost of the original call in the cached completion, so `StruckdownResult.total_cost` includes it. `fresh_cost` is what this run spent; `cached_cost` is what the cached slots cost when they were first made, which is what the cache saved.

Cached embeddings produce no usage record. A cached `EmbeddingResult` carries the cost it had when first computed; `EmbeddingResultList.fresh_cost` leaves it out.


## Result properties

`complete()` and `complete_async()` return a `StruckdownResult`:

```python
result.prompt_tokens          # input tokens across all slots
result.completion_tokens      # output tokens across all slots
result.total_tokens           # the two added together
result.cached_prompt_tokens   # input tokens read from the provider's cache
result.cache_creation_tokens  # input tokens written to the provider's cache

result.total_cost             # USD; cached slots at their original cost, unknown as 0.0
result.fresh_cost             # USD spent by this run
result.cached_cost            # original cost of the slots served from struckdown's cache
result.fresh_call_count
result.cached_call_count
result.has_unknown_costs      # any slot unpriced
result.all_costs_unknown      # every slot unpriced
```

A tool slot is several provider requests. Its cost on the result is the sum over them, or unknown if any of them could not be priced.

`get_embedding()` returns an `EmbeddingResultList`, a list of numpy arrays that carry `cost`, `tokens`, `model` and `cached`; the list adds `total_cost`, `fresh_cost`, `total_tokens`, `cached_count`, `fresh_count` and `has_unknown_costs`.

`CostSummary.from_results([...])` adds several results together, and `format_summary()` gives the line the CLI prints. Its fields are listed in the [API reference](../reference/api.md#costsummary).


## Limitations

This is how the current version behaves, with the workaround for each.

- **Pricing and `model_ref` travel in context variables.** `set_model_pricing`, `set_model_ref`, `get_llm_and_credentials()` and `complete(spec=...)` set them for everything that follows in the same context, and resolving a second model re-points them. If you resolve a chat model, then resolve an embedding model (to embed a query, say), then make the chat call, the chat call is priced at the embedding model's rates and joined to its row. Resolve a model immediately before the call that uses it, with nothing in between that resolves another. Changes made inside a `sync_to_async` call are copied back into the caller's context, so a resolution there counts too.
- **Anthropic cache writes through OpenRouter are charged at the input rate.** OpenRouter's OpenAI-shaped usage reports `cached_tokens` (reads) but no cache-write count, so written tokens are counted as ordinary input. For Anthropic models this leaves out the write premium. Reads are priced correctly.
- **Records are lost when a call raises inside a deferral.** Where struckdown hops threads or event loops (see [Threads and event loops](../how-to/usage-ledger.md#threads-and-event-loops)) it holds records back and dispatches them after the hop returns. If the hop raises, the held records are not dispatched. So a sync `complete()` whose later slot raises loses the records for the whole run, including slots that succeeded and were paid for, and the `ok=False` record for a failed request is usually lost. A tool loop's records are emitted when the run finishes, so a run stopped by its usage limits or by an exception produces none. Treat failure counts as a lower bound.
- **The costs page's "per call" average counts failed calls.** Its denominator leaves out unpriced calls and cache hits but not failed requests, which have no cost, so failures pull the average down.
