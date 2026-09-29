"""Write struckdown's usage records to the ledger tables.

Registered as a process-wide usage handler in ``SdModelsConfig.ready()``. From
sync code the row is written inline, on the caller's connection; from async
code the handler returns a ``sync_to_async`` coroutine that struckdown awaits,
so the ORM is never touched on the event loop and nothing is fired and
forgotten. A failed write is logged and never fails the LLM call: this is the
one place that rule is worth bending, because a ledger outage must not turn
into an answer outage.

:func:`record_call` is the public entry for calls struckdown never saw (a
direct OpenAI SDK call, say).
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Optional

from asgiref.sync import sync_to_async
from django.conf import settings

from struckdown.ledger import CostBreakdown, UsageRecord, cost_from_stored

logger = logging.getLogger(__name__)


def setting(name: str, default):
    return getattr(settings, f"STRUCKDOWN_LEDGER_{name}", default)


def enabled() -> bool:
    return bool(setting("ENABLED", True))


def capture_payloads() -> bool:
    """Payloads are kept when the setting says so or the current span asked."""
    if setting("CAPTURE_PAYLOADS", False):
        return True
    from .spans import current_span

    span = current_span()
    return bool(span is not None and span.capture_payloads)


def _in_async_context() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class LedgerHandler:
    """The registered handler. An instance, so ``wants_payload`` can be live."""

    @property
    def wants_payload(self) -> bool:
        return enabled() and capture_payloads()

    def __call__(self, record: UsageRecord):
        if not enabled():
            return None
        if _in_async_context():
            return sync_to_async(write_record, thread_sensitive=True)(record)
        write_record(record)
        return None


handler = LedgerHandler()


def _money(value: Optional[float]) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(str(value)).quantize(Decimal("0.00000001"))


def _price(value: Optional[float]) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(str(value)).quantize(Decimal("0.000001"))


def resolve_model(record: UsageRecord):
    """The ``AvailableModel`` this call went through, or None.

    ``model_ref`` (set by ``get_llm_and_credentials`` / ``to_spec``) is exact.
    Without it the name is matched, preferring a row whose credential points at
    the host the call went to; an ambiguous name resolves to nothing rather
    than to a guess.
    """
    from .models import AvailableModel

    if record.model_ref:
        row = AvailableModel.objects.filter(pk=record.model_ref).first()
        if row is not None:
            return row
    rows = list(
        AvailableModel.objects.filter(model_name=record.model_name).select_related("credential")
    )
    if len(rows) == 1:
        return rows[0]
    if record.base_url_host:
        by_host = [
            r for r in rows if r.credential and record.base_url_host in (r.credential.base_url or "")
        ]
        if len(by_host) == 1:
            return by_host[0]
    return None


def write_record(record: UsageRecord):
    """Write one ``LLMCall`` (and its payload, if asked). Never raises."""
    try:
        return _write(record)
    except Exception:
        logger.exception("ledger write failed for %s %s", record.kind, record.model_name)
        return None


def _write(record: UsageRecord):
    from .models import LLMCall, LLMCallPayload
    from .spans import current_span, materialise

    model = resolve_model(record)
    cost = record.cost
    # a call struckdown could not price, through a model whose row has a price
    if cost is None and model is not None and record.ok and not record.cache_hit:
        pricing = model.stored_pricing()
        if pricing is not None and record.kind != "transcription":
            cost = cost_from_stored(
                pricing,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cache_read_tokens=record.cache_read_tokens,
                cache_write_tokens=record.cache_write_tokens,
            )

    handle = current_span()
    span = materialise(handle) if handle is not None else None

    call = LLMCall.objects.create(
        started_at=record.started_at,
        duration_ms=int(record.duration_ms) if record.duration_ms is not None else None,
        span=span,
        root_name=handle.root_name if handle is not None else "",
        slot=record.slot or "",
        kind=record.kind,
        model_name=record.model_name,
        provider=record.provider or (model.provider if model else ""),
        base_url_host=record.base_url_host,
        data_residency=model.data_residency if model else "",
        available_model=model,
        input_price=_price(cost.input_price) if cost else None,
        output_price=_price(cost.output_price) if cost else None,
        cache_read_price=_price(cost.cache_read_price) if cost else None,
        cache_write_price=_price(cost.cache_write_price) if cost else None,
        price_source=cost.source if cost else "",
        input_tokens=record.input_tokens,
        cache_read_tokens=record.cache_read_tokens,
        cache_write_tokens=record.cache_write_tokens,
        output_tokens=record.output_tokens,
        reasoning_tokens=record.reasoning_tokens,
        audio_seconds=record.audio_seconds,
        input_cost=_money(cost.input_cost) if cost else None,
        output_cost=_money(cost.output_cost) if cost else None,
        cache_hit=record.cache_hit,
        ok=record.ok,
        error_class=record.error_class[:120],
        provider_request_id=record.provider_request_id[:120],
        finish_reason=record.finish_reason[:40],
    )
    if record.payload is not None and capture_payloads():
        LLMCallPayload.objects.create(
            call=call, request=record.payload.request, response=record.payload.response
        )
    return call


def record_call(
    *,
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    kind: str = "chat",
    available_model=None,
    duration_ms: Optional[float] = None,
    started_at=None,
    ok: bool = True,
    error_class: str = "",
    provider_request_id: str = "",
    finish_reason: str = "",
    request=None,
    response=None,
):
    """Record a call that did not go through struckdown.

    The cost comes from ``available_model``'s stored prices (or the row that
    matches ``model_name``); a call through an unpriced model is written with
    null cost, which the costs page counts as unknown rather than free.
    """
    from struckdown.ledger import UsagePayload

    record = UsageRecord(
        kind=kind,
        model_name=model_name,
        model_ref=available_model.id if available_model is not None else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        duration_ms=duration_ms,
        started_at=started_at,
        ok=ok,
        error_class=error_class,
        provider_request_id=provider_request_id or "",
        finish_reason=finish_reason or "",
        payload=(
            UsagePayload(request=request, response=response)
            if request is not None or response is not None
            else None
        ),
    )
    if available_model is not None and record.ok:
        pricing = available_model.stored_pricing()
        if pricing is not None:
            record.cost = cost_from_stored(
                pricing,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
    if not enabled():
        return None
    return write_record(record)


def record_openai_response(response, *, model_name: str, available_model=None, **kwargs):
    """``record_call`` from an OpenAI SDK chat completion (or its final stream chunk)."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    choices = getattr(response, "choices", None) or []
    return record_call(
        model_name=model_name,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cache_read_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        available_model=available_model,
        provider_request_id=getattr(response, "id", "") or "",
        finish_reason=(getattr(choices[0], "finish_reason", "") or "") if choices else "",
        **kwargs,
    )


def cost_summary() -> Optional[CostBreakdown]:
    """Convenience for tests and shells: the last call's cost, or None."""
    from .models import LLMCall

    call = LLMCall.objects.order_by("-created_at").first()
    if call is None or call.input_cost is None:
        return None
    return CostBreakdown(float(call.input_cost), float(call.output_cost))
