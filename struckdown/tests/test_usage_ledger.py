"""The usage ledger: one record per provider request, to whoever listens.

pydantic-ai's TestModel serves every call, so these tests are about what
struckdown reports, not what a model says.
"""

import asyncio
import unittest
from unittest.mock import patch

import pytest
from pydantic_ai.models.test import TestModel

import struckdown as sd
from struckdown import ledger
from struckdown.ledger import (CostBreakdown, StoredPricing, UsageRecord,
                               cost_from_price_calc, cost_from_stored,
                               usage_tracking)
from struckdown.llm import set_model_pricing, structured_chat

CREDS = sd.LLMCredentials(api_key="x", base_url="http://example.invalid/v1")
MODEL = sd.LLM(model_name="stub")


def _served_by_test_model():
    return patch.object(sd.LLM, "get_pydantic_model", lambda self, creds=None: TestModel())


def _collector():
    records = []

    def handler(record):
        records.append(record)

    return records, handler


class CostBreakdownTests(unittest.TestCase):
    def test_stored_pricing_charges_cache_reads_at_their_own_rate(self):
        pricing = StoredPricing(
            input_per_mtok=2.0, output_per_mtok=8.0, cache_read_per_mtok=0.2
        )
        cost = cost_from_stored(
            pricing, input_tokens=1_000_000, output_tokens=0, cache_read_tokens=500_000
        )
        # half the prompt at $2, half at $0.20
        self.assertAlmostEqual(cost.input_cost, 1.0 + 0.1)
        self.assertEqual(cost.output_cost, 0.0)
        self.assertEqual(cost.source, "stored")
        self.assertEqual(cost.cache_read_price, 0.2)

    def test_unknown_cache_rate_falls_back_to_the_input_rate(self):
        pricing = StoredPricing(input_per_mtok=2.0, output_per_mtok=8.0)
        cost = cost_from_stored(
            pricing, input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000
        )
        self.assertAlmostEqual(cost.input_cost, 2.0)
        self.assertAlmostEqual(cost.output_cost, 8.0)
        self.assertAlmostEqual(cost.total_cost, 10.0)

    def test_a_price_calculation_keeps_its_split(self):
        calc = type("Calc", (), {"input_price": 0.3, "output_price": 0.7, "total_price": 1.0})()
        cost = cost_from_price_calc(calc, "genai_prices")
        self.assertEqual((cost.input_cost, cost.output_cost), (0.3, 0.7))

    def test_an_old_total_only_calculation_lands_on_the_input_side(self):
        cost = cost_from_price_calc(type("Calc", (), {"total": 0.5})(), "pydantic_ai")
        self.assertEqual((cost.input_cost, cost.output_cost), (0.5, 0.0))

    def test_set_model_pricing_carries_cache_rates(self):
        set_model_pricing(1.0, 2.0, 0.1, 1.25)
        pricing = sd.llm.get_model_pricing()
        self.assertEqual(pricing.cache_read_per_mtok, 0.1)
        self.assertEqual(pricing.cache_write_per_mtok, 1.25)
        set_model_pricing(None, None)
        self.assertIsNone(sd.llm.get_model_pricing())


class HandlerTests(unittest.TestCase):
    def test_context_handler_only_hears_inside_its_block(self):
        records, handler = _collector()
        with usage_tracking(handler):
            ledger.emit(UsageRecord(kind="chat", model_name="m"))
        ledger.emit(UsageRecord(kind="chat", model_name="m"))
        self.assertEqual(len(records), 1)

    def test_registered_handler_hears_everything_until_removed(self):
        records, handler = _collector()
        sd.register_usage_handler(handler)
        try:
            ledger.emit(UsageRecord(kind="chat", model_name="m"))
        finally:
            sd.unregister_usage_handler(handler)
        ledger.emit(UsageRecord(kind="chat", model_name="m"))
        self.assertEqual(len(records), 1)

    def test_a_raising_handler_does_not_fail_the_call(self):
        def bad(record):
            raise RuntimeError("ledger down")

        records, good = _collector()
        with usage_tracking(bad):
            with _served_by_test_model():
                sd.complete("Say hi.\n[[hi]]", model=MODEL, credentials=CREDS)

    def test_an_awaitable_handler_is_awaited_from_async_code(self):
        seen = []

        def handler(record):
            async def write():
                await asyncio.sleep(0)
                seen.append(record)

            return write()

        async def go():
            with usage_tracking(handler):
                await ledger.emit_async(UsageRecord(kind="chat", model_name="m"))

        asyncio.run(go())
        self.assertEqual(len(seen), 1)

    def test_an_awaitable_handler_from_sync_code_still_runs(self):
        seen = []

        def handler(record):
            async def write():
                seen.append(record)

            return write()

        with usage_tracking(handler):
            ledger.emit(UsageRecord(kind="chat", model_name="m"))
        self.assertEqual(len(seen), 1)

    def test_wants_payload_is_read_off_the_handler(self):
        records, handler = _collector()
        self.assertFalse(ledger.wants_payload())
        handler.wants_payload = True
        with usage_tracking(handler):
            self.assertTrue(ledger.wants_payload())

    def test_model_ref_and_slot_are_carried_onto_records(self):
        sd.set_model_ref("row-42")
        try:
            with ledger.slot_context("answer"):
                record = ledger.record_from_response(
                    kind="chat", model_name="m", response=None, cost=None
                )
        finally:
            sd.set_model_ref(None)
        self.assertEqual(record.model_ref, "row-42")
        self.assertEqual(record.slot, "answer")


class FirePointTests(unittest.TestCase):
    """Each call path produces one record per provider request."""

    def setUp(self):
        sd.clear_cache()

    def test_structured_chat_reports_a_fresh_call_then_a_cache_hit(self):
        records, handler = _collector()
        from pydantic import BaseModel

        class Out(BaseModel):
            response: str

        messages = [{"role": "user", "content": "ledger test: say something unique 7c2f"}]
        with usage_tracking(handler), _served_by_test_model():
            structured_chat(messages=messages, return_type=Out, llm=MODEL, credentials=CREDS)
            structured_chat(messages=messages, return_type=Out, llm=MODEL, credentials=CREDS)

        self.assertEqual(len(records), 2)
        fresh, cached = records
        self.assertEqual(fresh.kind, "chat")
        self.assertFalse(fresh.cache_hit)
        self.assertTrue(fresh.ok)
        self.assertEqual(fresh.base_url_host, "example.invalid")
        self.assertGreater(fresh.input_tokens, 0)
        self.assertIsNotNone(fresh.duration_ms)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(cached.total_cost, 0.0)
        self.assertEqual(cached.input_tokens, fresh.input_tokens)

    def test_a_template_run_names_the_slot(self):
        records, handler = _collector()
        with usage_tracking(handler), _served_by_test_model():
            sd.complete("Tell a joke.\n[[joke]]\nRate it.\n[[int:rating]]", model=MODEL, credentials=CREDS)
        self.assertEqual([r.slot for r in records], ["joke", "rating"])

    def test_stored_pricing_prices_the_record(self):
        records, handler = _collector()
        set_model_pricing(2.0, 8.0)
        try:
            with usage_tracking(handler), _served_by_test_model():
                sd.complete("Hi.\n[[hi]]", model=MODEL, credentials=CREDS)
        finally:
            set_model_pricing(None, None)
        (record,) = records
        self.assertEqual(record.cost.source, "stored")
        self.assertEqual(record.cost.input_price, 2.0)
        self.assertAlmostEqual(record.cost.input_cost, record.input_tokens * 2.0 / 1e6)

    def test_a_tool_loop_yields_one_record_per_response(self):
        records, handler = _collector()

        def lookup(term: str) -> str:
            """Look a thing up."""
            return "found"

        async def go():
            with usage_tracking(handler):
                return await sd.complete_async(
                    "Use the tool, then answer.\n[[answer|use_tools=true, max_iter=3]]",
                    model=MODEL,
                    credentials=CREDS,
                    tools=[lookup],
                )

        with _served_by_test_model():
            asyncio.run(go())
        # TestModel calls every tool once, then answers: two requests
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r.slot == "answer" for r in records))
        self.assertIsNone(records[0].duration_ms)
        self.assertIsNotNone(records[-1].duration_ms)

    def test_a_failed_call_is_a_record_too(self):
        records, handler = _collector()
        from pydantic import BaseModel

        class Out(BaseModel):
            response: str

        def explode(self, creds=None):
            raise RuntimeError("no network")

        with usage_tracking(handler), patch.object(sd.LLM, "get_pydantic_model", explode):
            with self.assertRaises(Exception):
                structured_chat(
                    messages=[{"role": "user", "content": "ledger failure 9b1"}],
                    return_type=Out,
                    llm=MODEL,
                    credentials=CREDS,
                )
        (record,) = records
        self.assertFalse(record.ok)
        self.assertTrue(record.error_class)
        self.assertIsNone(record.cost)

    def test_payload_is_only_built_when_asked(self):
        records, handler = _collector()
        with usage_tracking(handler), _served_by_test_model():
            sd.complete("Hi there.\n[[hi]]", model=MODEL, credentials=CREDS)
        self.assertIsNone(records[0].payload)

        records2, handler2 = _collector()
        handler2.wants_payload = True
        sd.clear_cache()
        with usage_tracking(handler2), _served_by_test_model():
            sd.complete("Hi there.\n[[hi]]", model=MODEL, credentials=CREDS)
        payload = records2[0].payload
        self.assertIsNotNone(payload)
        self.assertIn("messages", payload.request)
        self.assertIn("output", payload.response)


class TranscriptionTests(unittest.TestCase):
    def test_transcribe_reports_minutes_and_rate(self):
        records, handler = _collector()

        class FakeRaw:
            text = "hello"
            duration = 90.0
            language = "en"

        class FakeClient:
            class audio:
                class transcriptions:
                    @staticmethod
                    def create(**kwargs):
                        return FakeRaw()

        from struckdown import audio

        with usage_tracking(handler), patch.object(
            audio, "_build_client", lambda model, creds: (FakeClient(), "whisper-1")
        ), patch.object(
            audio,
            "validate_audio_for_transcription",
            lambda a: type("V", (), {"duration_s": 88.0, "size_bytes": 10})(),
        ), patch.object(audio, "_coerce_audio_file", lambda a, v: (b"", "a.mp3")):
            audio.set_audio_pricing(0.006)
            try:
                result = audio.transcribe(b"", model="openai:whisper-1", credentials=CREDS)
            finally:
                audio.set_audio_pricing(None)

        (record,) = records
        self.assertEqual(record.kind, "transcription")
        self.assertEqual(record.audio_seconds, 90.0)
        self.assertAlmostEqual(record.cost.input_cost, 0.006 * 1.5)
        self.assertEqual(record.cost.output_cost, 0.0)
        self.assertAlmostEqual(result.cost, 0.009)
