"""Private spans: no payload row and no cache, whatever the settings say."""

import asyncio
import threading
from unittest.mock import patch

import pytest
from asgiref.sync import async_to_sync, sync_to_async
from pydantic_ai.models.test import TestModel

import struckdown as sd
from struckdown.cache import cache_off, maybe_cached
from struckdown.contrib.django import ledger
from struckdown.contrib.django.models import LLMCall, LLMCallPayload, LLMSpan
from struckdown.contrib.django.spans import (aclose_span, aopen_span, close_span, llm_span,
                                             open_span)
from struckdown.ledger import UsagePayload, UsageRecord, emit, wants_payload

pytestmark = pytest.mark.django_db

PAYLOAD = UsagePayload(request={"messages": ["secret"]}, response={"output": "secret"})


def _record():
    return UsageRecord(kind="chat", model_name="stub", input_tokens=10, payload=PAYLOAD)


def test_a_private_span_keeps_no_payload_even_when_capture_is_on(settings):
    settings.STRUCKDOWN_LEDGER_CAPTURE_PAYLOADS = True
    with llm_span("contacts.note", private=True) as handle:
        assert not ledger.handler.wants_payload
        assert not wants_payload()
        call = ledger.write_record(_record())
    assert LLMCallPayload.objects.count() == 0
    assert call.span == handle.row
    row = LLMSpan.objects.get(pk=handle.row.pk)
    assert row.private and not row.capture_payloads


def test_capture_cannot_be_asked_for_inside_a_private_span(settings):
    with llm_span("contacts.note", private=True):
        with llm_span("helper", capture=True) as inner:
            assert inner.private
            call = ledger.write_record(_record())
    assert LLMCallPayload.objects.count() == 0
    assert LLMSpan.objects.get(pk=inner.row.pk).private
    assert call.span == inner.row


def test_emit_strips_the_payload_before_any_handler_sees_it():
    seen = []
    with sd.usage_tracking(seen.append), sd.private():
        emit(_record())
    assert seen[0].payload is None


def test_privacy_ends_with_the_span_and_the_cache_comes_back():
    assert not cache_off()
    with llm_span("contacts.note", private=True):
        assert cache_off() and sd.is_private()
        assert maybe_cached(sd._call_llm_cached) is sd._call_llm_cached.func
    assert not cache_off() and not sd.is_private()
    handle = open_span("gen", private=True)
    assert sd.is_private()
    close_span(handle)
    assert not sd.is_private()


def test_privacy_reaches_threads_started_from_the_span():
    with llm_span("contacts.note", private=True):
        assert async_to_sync(sync_to_async(sd.is_private))()

        async def inner():
            return await asyncio.to_thread(sd.is_private)

        assert async_to_sync(inner)()
    seen = []
    t = threading.Thread(target=lambda: seen.append(sd.is_private()))
    t.start()
    t.join()
    assert seen == [False]


def test_async_private_span():
    async def run():
        handle = await aopen_span("contacts.note", private=True)
        inside = sd.is_private()
        await aclose_span(handle)
        return inside, sd.is_private()

    assert async_to_sync(run)() == (True, False)


def test_a_struckdown_call_in_a_private_span_is_costed_but_neither_cached_nor_captured(
    priced_model, settings
):
    settings.STRUCKDOWN_LEDGER_CAPTURE_PAYLOADS = True
    sd.clear_cache()
    model, creds = priced_model.get_llm_and_credentials()
    with patch.object(sd.LLM, "get_pydantic_model", lambda self, c=None: TestModel()):
        with patch.object(
            sd._call_llm_cached, "call", side_effect=AssertionError("cache used")
        ), llm_span("contacts.note", private=True):
            for _ in range(2):
                sd.complete("Say hi.\n[[greeting]]", model=model, credentials=creds)
    calls = list(LLMCall.objects.all())
    assert len(calls) == 2
    assert not any(c.cache_hit for c in calls)
    assert all(c.total_cost is not None for c in calls)
    assert LLMCallPayload.objects.count() == 0
