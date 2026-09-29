"""The ledger writer: a usage record becomes an LLMCall row."""

import asyncio
from decimal import Decimal
from unittest.mock import patch

import pytest
from pydantic_ai.models.test import TestModel

import struckdown as sd
from struckdown.contrib.django import ledger
from struckdown.contrib.django.models import (AvailableModel, Credential,
                                              LLMCall, LLMCallPayload)
from struckdown.contrib.django.spans import llm_span
from struckdown.ledger import (CostBreakdown, UsagePayload, UsageRecord,
                               cost_from_stored)

pytestmark = pytest.mark.django_db


def _record(**kwargs):
    base = dict(
        kind="chat",
        model_name="stub",
        input_tokens=1000,
        output_tokens=100,
        cache_read_tokens=500,
        duration_ms=120.5,
    )
    base.update(kwargs)
    return UsageRecord(**base)


def test_a_priced_record_copies_prices_and_computes_the_total():
    cost = cost_from_stored(
        sd.llm.StoredPricing(2.0, 8.0, cache_read_per_mtok=0.2),
        input_tokens=1000,
        output_tokens=100,
        cache_read_tokens=500,
    )
    call = ledger.write_record(_record(cost=cost))
    call.refresh_from_db()
    assert call.input_price == Decimal("2.000000")
    assert call.cache_read_price == Decimal("0.200000")
    assert call.price_source == "stored"
    # 500 uncached at $2 + 500 cached at $0.20 per Mtok
    assert call.input_cost == Decimal("0.00110000")
    assert call.output_cost == Decimal("0.00080000")
    assert call.total_cost == Decimal("0.00190000")
    assert call.duration_ms == 120


def test_an_unpriced_record_has_null_costs_not_zero():
    call = ledger.write_record(_record(cost=None))
    call.refresh_from_db()
    assert call.input_cost is None and call.total_cost is None
    assert call.unpriced


def test_a_cache_hit_is_a_row_with_zero_cost():
    call = ledger.write_record(
        _record(cache_hit=True, cost=CostBreakdown(0.0, 0.0, source="cache"))
    )
    call.refresh_from_db()
    assert call.cache_hit and call.total_cost == 0


def test_a_failed_call_is_recorded():
    call = ledger.write_record(_record(ok=False, error_class="RateLimitError", cost=None))
    assert not call.ok and call.error_class == "RateLimitError"


def test_model_ref_resolves_the_row_and_copies_residency(priced_model):
    call = ledger.write_record(_record(model_ref=priced_model.id))
    assert call.available_model == priced_model
    assert call.data_residency == "eu"


def test_a_bare_name_resolves_when_unambiguous(priced_model):
    call = ledger.write_record(_record(model_name="stub"))
    assert call.available_model == priced_model


def test_an_ambiguous_name_prefers_the_host_else_nothing(priced_model, credential):
    other_cred = Credential.objects.create(
        name="Other", api_key="k", base_url="http://other.invalid/v1"
    )
    other = AvailableModel.objects.create(
        model_name="stub", model_type="llm", name="Stub 2", credential=other_cred
    )
    assert ledger.write_record(_record(model_name="stub")).available_model is None
    assert (
        ledger.write_record(_record(model_name="stub", base_url_host="other.invalid")).available_model
        == other
    )


def test_an_unpriced_record_through_a_priced_row_is_priced_by_the_ledger(priced_model):
    call = ledger.write_record(_record(model_ref=priced_model.id, cost=None))
    call.refresh_from_db()
    assert call.price_source == "stored"
    assert call.total_cost == Decimal("0.00190000")


def test_payload_is_written_only_when_capture_is_on(settings):
    payload = UsagePayload(request={"messages": []}, response={"output": "hi"})
    ledger.write_record(_record(payload=payload))
    assert LLMCallPayload.objects.count() == 0

    settings.STRUCKDOWN_LEDGER_CAPTURE_PAYLOADS = True
    call = ledger.write_record(_record(payload=payload))
    assert call.payload.response == {"output": "hi"}


def test_a_capturing_span_turns_payloads_on():
    payload = UsagePayload(request={"messages": []}, response={"output": "hi"})
    with llm_span("debug.session", capture=True):
        assert ledger.handler.wants_payload
        call = ledger.write_record(_record(payload=payload))
    assert call.payload is not None
    assert call.span.capture_payloads


def test_disabled_ledger_writes_nothing(settings):
    settings.STRUCKDOWN_LEDGER_ENABLED = False
    assert ledger.handler(_record()) is None
    assert LLMCall.objects.count() == 0


def test_a_broken_write_does_not_raise():
    with patch.object(ledger, "_write", side_effect=RuntimeError("db gone")):
        assert ledger.write_record(_record()) is None


def test_record_call_prices_from_the_row(priced_model):
    call = ledger.record_call(
        model_name="stub",
        input_tokens=1_000_000,
        output_tokens=0,
        available_model=priced_model,
        request={"messages": [{"role": "user", "content": "hi"}]},
    )
    call.refresh_from_db()
    assert call.total_cost == Decimal("2.00000000")
    assert call.available_model == priced_model
    assert not LLMCallPayload.objects.exists()


def test_record_openai_response_reads_the_sdk_shape(priced_model):
    class Details:
        cached_tokens = 400

    class Usage:
        prompt_tokens = 1000
        completion_tokens = 50
        prompt_tokens_details = Details()

    class Choice:
        finish_reason = "stop"

    class Response:
        id = "chatcmpl-1"
        usage = Usage()
        choices = [Choice()]

    call = ledger.record_openai_response(Response(), model_name="stub", available_model=priced_model)
    assert (call.input_tokens, call.cache_read_tokens, call.output_tokens) == (1000, 400, 50)
    assert call.provider_request_id == "chatcmpl-1"
    assert call.finish_reason == "stop"


@pytest.mark.django_db(transaction=True)
def test_from_async_code_the_handler_hands_back_a_coroutine_and_the_row_lands():
    async def go():
        result = ledger.handler(_record())
        assert asyncio.iscoroutine(result)
        await result

    asyncio.run(go())
    assert LLMCall.objects.count() == 1


def test_a_struckdown_call_through_a_priced_row_lands_with_its_slot(priced_model):
    sd.clear_cache()
    model, creds = priced_model.get_llm_and_credentials()
    with patch.object(sd.LLM, "get_pydantic_model", lambda self, c=None: TestModel()):
        with llm_span("test.feature"):
            sd.complete("Say hi.\n[[greeting]]", model=model, credentials=creds)
    call = LLMCall.objects.get()
    assert call.slot == "greeting"
    assert call.available_model == priced_model
    assert call.price_source == "stored"
    assert call.root_name == "test.feature"
    assert call.span.name == "test.feature"
    assert call.total_cost is not None
