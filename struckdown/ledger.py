"""One record per provider request, handed to whoever is listening.

struckdown prices every call it makes but, until now, told nobody: the figure
rode on the completion dict and was the caller's to keep or drop. This module
is the other half. Each call path in :mod:`struckdown.llm` and
:mod:`struckdown.audio` builds a :class:`UsageRecord` and passes it to the
registered handlers, so a host application can keep a ledger without touching
its call sites.

Two ways to listen::

    from struckdown import register_usage_handler, usage_tracking

    register_usage_handler(write_row)          # process-wide, e.g. in AppConfig.ready()

    with usage_tracking(collect):              # this context only
        complete(...)

A handler is ``Callable[[UsageRecord], None | Awaitable[None]]``. From an async
call path struckdown awaits whatever it returns, so a handler that must do its
work on a thread (a Django ORM write, say) returns the coroutine and nothing is
fired and forgotten. A handler that raises is logged and never fails the call.

A handler that wants the request and response bodies sets ``wants_payload =
True`` on itself; the payload is only assembled when someone asked, because
rendering it on every call costs memory the ledger does not need.

Field names follow the OpenTelemetry ``gen_ai.*`` semantic conventions where one
exists; :data:`SEMCONV` is the mapping, so an exporter is a rename, not a remodel.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterator, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# -- what a call cost ---------------------------------------------------------


@dataclass
class StoredPricing:
    """Per-Mtok prices a caller supplied for the current context (USD)."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: Optional[float] = None
    cache_write_per_mtok: Optional[float] = None


@dataclass
class CostBreakdown:
    """The two sides of a call's cost, and the prices that produced them.

    ``input_cost`` covers uncached input, cache reads and cache writes at their
    own rates; ``output_cost`` covers completion tokens, reasoning included.
    Transcription is priced per minute and lands in ``input_cost`` with an
    ``output_cost`` of 0, so ``total_cost`` still adds up.
    """

    input_cost: float
    output_cost: float
    input_price: Optional[float] = None  # per Mtok, as used
    output_price: Optional[float] = None
    cache_read_price: Optional[float] = None
    cache_write_price: Optional[float] = None
    source: str = "unknown"  # stored | pydantic_ai | genai_prices | audio_rate

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


def cost_from_stored(
    pricing: StoredPricing,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> CostBreakdown:
    """Price token counts with caller-supplied rates.

    ``input_tokens`` is the total prompt, cache reads and writes included, which
    is how pydantic-ai reports it for every provider. A cache rate that is not
    known falls back to the full input rate, which overstates rather than
    hides.
    """
    read_rate = pricing.cache_read_per_mtok
    write_rate = pricing.cache_write_per_mtok
    read_rate = pricing.input_per_mtok if read_rate is None else read_rate
    write_rate = pricing.input_per_mtok if write_rate is None else write_rate
    uncached = max(input_tokens - cache_read_tokens - cache_write_tokens, 0)
    input_cost = (
        uncached * pricing.input_per_mtok
        + cache_read_tokens * read_rate
        + cache_write_tokens * write_rate
    ) / 1_000_000
    output_cost = output_tokens * pricing.output_per_mtok / 1_000_000
    return CostBreakdown(
        input_cost=input_cost,
        output_cost=output_cost,
        input_price=pricing.input_per_mtok,
        output_price=pricing.output_per_mtok,
        cache_read_price=read_rate,
        cache_write_price=write_rate,
        source="stored",
    )


def cost_from_price_calc(price_calc, source: str) -> Optional[CostBreakdown]:
    """A pydantic-ai or genai-prices price calculation as a breakdown.

    Both libraries return an object (or mapping) with ``input_price``,
    ``output_price`` and ``total_price``. Older pydantic-ai returned only a
    ``total``; that becomes an input-only breakdown, the least wrong reading.
    """
    if price_calc is None:
        return None

    def _get(name):
        if isinstance(price_calc, dict):
            return price_calc.get(name)
        return getattr(price_calc, name, None)

    total = _get("total_price")
    if total is None:
        total = _get("total")
    if total is None:
        return None
    input_price = _get("input_price")
    output_price = _get("output_price")
    if input_price is None and output_price is None:
        return CostBreakdown(input_cost=float(total), output_cost=0.0, source=source)
    return CostBreakdown(
        input_cost=float(input_price or 0),
        output_cost=float(output_price or 0),
        source=source,
    )


# -- the record ---------------------------------------------------------------


@dataclass
class UsagePayload:
    """The bodies, for a handler that asked: what was sent and what came back."""

    request: Any = None
    response: Any = None


@dataclass
class UsageRecord:
    kind: str  # chat | embedding | transcription
    model_name: str
    provider: str = ""
    base_url_host: str = ""
    model_ref: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    audio_seconds: Optional[float] = None
    cost: Optional[CostBreakdown] = None
    duration_ms: Optional[float] = None
    started_at: Optional[datetime] = None
    cache_hit: bool = False
    ok: bool = True
    error_class: str = ""
    slot: Optional[str] = None
    provider_request_id: str = ""
    finish_reason: str = ""
    payload: Optional[UsagePayload] = None
    extra: dict = field(default_factory=dict)

    @property
    def total_cost(self) -> Optional[float]:
        return self.cost.total_cost if self.cost is not None else None


#: OpenTelemetry ``gen_ai`` semantic-convention names for the record's fields.
SEMCONV = {
    "kind": "gen_ai.operation.name",
    "model_name": "gen_ai.request.model",
    "provider": "gen_ai.provider.name",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "cache_read_tokens": "gen_ai.usage.cache_read.input_tokens",
    "cache_write_tokens": "gen_ai.usage.cache_creation.input_tokens",
    "provider_request_id": "gen_ai.response.id",
    "finish_reason": "gen_ai.response.finish_reasons",
    "error_class": "error.type",
}


# -- context: what the caller told us about this call ---------------------------

_model_ref: ContextVar[Optional[str]] = ContextVar("struckdown_model_ref", default=None)
_current_slot: ContextVar[Optional[str]] = ContextVar("struckdown_current_slot", default=None)


def set_model_ref(ref: Optional[str]) -> None:
    """Name the caller's own model record for the calls that follow in this context.

    struckdown only knows a model name; a host that keeps its own table of
    models (the Django contrib's ``AvailableModel``) sets its row id here so a
    record can be joined back without guessing from the name.
    """
    _model_ref.set(ref)


def get_model_ref() -> Optional[str]:
    return _model_ref.get()


@contextmanager
def slot_context(name: Optional[str]) -> Iterator[None]:
    """The slot whose call is about to be made, for the records it produces."""
    token = _current_slot.set(name)
    try:
        yield
    finally:
        _current_slot.reset(token)


def current_slot() -> Optional[str]:
    return _current_slot.get()


# -- handlers -----------------------------------------------------------------

Handler = Callable[[UsageRecord], Union[None, Awaitable[None]]]

_handlers: list[Handler] = []
_context_handler: ContextVar[Optional[Handler]] = ContextVar(
    "struckdown_usage_handler", default=None
)
# tasks scheduled from a sync emit inside a running loop; held so they are not
# collected before they run
_pending: set = set()
# records held back until the caller's own thread can dispatch them
_deferred: ContextVar[Optional[list]] = ContextVar("struckdown_usage_deferred", default=None)


def register_usage_handler(handler: Handler) -> None:
    if handler not in _handlers:
        _handlers.append(handler)


def unregister_usage_handler(handler: Handler) -> None:
    if handler in _handlers:
        _handlers.remove(handler)


@contextmanager
def usage_tracking(handler: Handler) -> Iterator[None]:
    """Receive every record produced inside the block, in this context only."""
    token = _context_handler.set(handler)
    try:
        yield
    finally:
        _context_handler.reset(token)


def active_handlers() -> list[Handler]:
    ctx = _context_handler.get()
    return _handlers + ([ctx] if ctx is not None else [])


def wants_payload() -> bool:
    """True if any listening handler asked for request and response bodies."""
    return any(getattr(h, "wants_payload", False) for h in active_handlers())


def _call(handler: Handler, record: UsageRecord):
    try:
        return handler(record)
    except Exception:
        logger.exception("usage handler %r raised", handler)
        return None


@contextmanager
def deferred_usage() -> Iterator[list]:
    """Hold records back instead of dispatching them, until flushed.

    struckdown hops between threads and event loops on the caller's behalf: a
    sync ``complete()`` runs an event loop, an async ``structured_chat_async``
    runs the sync call on a worker thread. A handler that writes to a
    database wants to be called in the caller's own thread, on the caller's
    own connection -- not on a worker thread that will hold a connection open
    afterwards, and not on the loop. So each hop collects the records made on
    the far side and dispatches them on the near side once it returns. The
    list is shared with any context copied from this one, so a record made on
    the other side of the hop lands in it.
    """
    pending: list = []
    token = _deferred.set(pending)
    try:
        yield pending
    finally:
        _deferred.reset(token)


def _hand_up(pending: list) -> bool:
    """Pass held-back records to an enclosing deferral, if there is one.

    Hops nest: ``complete()`` defers around its event loop, and inside it
    ``structured_chat_async`` defers around a worker thread. The inner flush
    must not dispatch on the loop when the outer caller is waiting to dispatch
    in its own thread.
    """
    outer = _deferred.get()
    if outer is None:
        return False
    outer.extend(pending)
    pending.clear()
    return True


def flush_usage(pending: list) -> None:
    """Dispatch held-back records from sync code, in this thread."""
    if _hand_up(pending):
        return
    token = _deferred.set(None)
    try:
        while pending:
            emit(pending.pop(0))
    finally:
        _deferred.reset(token)


async def flush_usage_async(pending: list) -> None:
    """Dispatch held-back records from async code, on this loop."""
    if _hand_up(pending):
        return
    token = _deferred.set(None)
    try:
        while pending:
            await emit_async(pending.pop(0))
    finally:
        _deferred.reset(token)


def emit(record: UsageRecord) -> None:
    """Hand ``record`` to every handler, from sync code.

    A handler that returns an awaitable is scheduled on the running loop if
    this thread has one, else run to completion here.
    """
    pending = _deferred.get()
    if pending is not None:
        pending.append(record)
        return
    for handler in active_handlers():
        result = _call(handler, record)
        if not inspect.isawaitable(result):
            continue
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            try:
                asyncio.run(result)
            except Exception:
                logger.exception("usage handler %r raised", handler)
            continue
        task = loop.create_task(result)
        _pending.add(task)
        task.add_done_callback(_pending.discard)


async def emit_async(record: UsageRecord) -> None:
    """Hand ``record`` to every handler and await any that need awaiting."""
    pending = _deferred.get()
    if pending is not None:
        pending.append(record)
        return
    for handler in active_handlers():
        result = _call(handler, record)
        if inspect.isawaitable(result):
            try:
                await result
            except Exception:
                logger.exception("usage handler %r raised", handler)


# -- building records from provider responses ---------------------------------


def provider_from_model_name(model_name: str) -> str:
    if model_name and ":" in model_name:
        return model_name.split(":", 1)[0]
    return ""


def host_of(base_url: Optional[str]) -> str:
    if not base_url:
        return ""
    return urlparse(base_url).hostname or ""


def now() -> datetime:
    return datetime.now(timezone.utc)


def record_from_response(
    *,
    kind: str,
    model_name: str,
    response,
    cost: Optional[CostBreakdown],
    credentials=None,
    started_at: Optional[datetime] = None,
    duration_ms: Optional[float] = None,
    cache_hit: bool = False,
    ok: bool = True,
    error_class: str = "",
    payload: Optional[UsagePayload] = None,
) -> UsageRecord:
    """A record from a pydantic-ai ``ModelResponse`` (or anything shaped like one).

    Duck-typed on purpose: the same builder serves a real response, a cached
    completion dict rebuilt into a stand-in, and the fakes in tests.
    """
    usage = getattr(response, "usage", None)
    details = getattr(usage, "details", None) or {}
    provider = getattr(response, "provider_name", None) or provider_from_model_name(model_name)
    return UsageRecord(
        kind=kind,
        model_name=model_name,
        provider=provider or "",
        base_url_host=host_of(getattr(credentials, "base_url", None)),
        model_ref=get_model_ref(),
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_write_tokens", 0) or 0),
        reasoning_tokens=int(details.get("reasoning_tokens", 0) or 0),
        cost=cost,
        duration_ms=duration_ms,
        started_at=started_at,
        cache_hit=cache_hit,
        ok=ok,
        error_class=error_class,
        slot=current_slot(),
        provider_request_id=str(getattr(response, "provider_response_id", "") or ""),
        finish_reason=str(getattr(response, "finish_reason", "") or ""),
        payload=payload,
    )


class _UsageStandIn:
    """A completion dict's usage, shaped like ``ModelResponse.usage``.

    A cache hit returns the stored dict, not the response it came from; this
    lets the same builder read it.
    """

    def __init__(self, usage: dict):
        self.input_tokens = usage.get("prompt_tokens", 0) or 0
        self.output_tokens = usage.get("completion_tokens", 0) or 0
        details = usage.get("prompt_tokens_details") or {}
        self.cache_read_tokens = details.get("cached_tokens", 0) or 0
        self.cache_write_tokens = details.get("cache_creation_tokens", 0) or 0
        self.details = {}


class ResponseStandIn:
    """A completion dict as a response, for the cache-hit path."""

    def __init__(self, com_dict: dict):
        self.usage = _UsageStandIn(com_dict.get("usage") or {})
        self.provider_name = None
        self.provider_response_id = ""
        self.finish_reason = ""
