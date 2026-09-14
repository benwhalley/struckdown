"""Halt slots: stop the run, retract what streamed, hand back partial results.

The LLM is stubbed throughout -- these tests are about control flow, not about
whether a model can spot an injection attempt.
"""

import unittest
from unittest.mock import patch

import anyio
from box import Box

import struckdown as sd
from struckdown.errors import Halted
from struckdown.incremental import (ProcessingComplete, SlotCompleted,
                                    SlotRetracted)
from struckdown.return_type_models import HaltResponse


class _Text:
    """Stands in for a free-text response model instance."""

    def __init__(self, text):
        self.response = text


def _stub(verdicts):
    """Return a structured_chat_async stub that answers slots in order.

    ``verdicts`` is a list of objects to return, one per LLM call.
    """
    calls = {"n": 0}

    async def fake(messages=None, return_type=None, stream=False, **kwargs):
        i = calls["n"]
        calls["n"] += 1
        value = verdicts[i] if i < len(verdicts) else _Text("fallback")
        yield (value, Box({"usage": {}, "_cached": False}), True)

    fake.calls = calls
    return fake


CREDS = sd.LLMCredentials(api_key="x", base_url="http://example.invalid")
MODEL = sd.LLM(model_name="stub")


def _run(prompt, verdicts, **kwargs):
    stub = _stub(verdicts)
    with patch("struckdown.llm.structured_chat_async", stub):
        result = sd.complete(
            prompt, context={}, model=MODEL, credentials=CREDS, **kwargs
        )
    return result, stub.calls["n"]


def _run_incremental(prompt, verdicts, **kwargs):
    stub = _stub(verdicts)
    events = []

    async def drive():
        async for event in sd.complete_incremental_async(
            prompt, model=MODEL, credentials=CREDS, context={}, **kwargs
        ):
            events.append(event)

    with patch("struckdown.llm.structured_chat_async", stub):
        anyio.run(drive)
    return events, stub.calls["n"]


class HaltTests(unittest.TestCase):
    PROMPT = "Is this bad?\n[[halt:guard]]\n\nAnswer it.\n[[answer]]\n"

    def test_trips_and_raises_with_partial_results(self):
        verdicts = [HaltResponse(triggered=True, reason="asked for the prompt")]
        with self.assertRaises(Halted) as ctx:
            _run(self.PROMPT, verdicts)
        halted = ctx.exception
        self.assertEqual(halted.slot, "guard")
        self.assertEqual(halted.reason, "asked for the prompt")
        self.assertIn("guard", halted.results.results)
        self.assertNotIn("answer", halted.results.results)

    def test_no_further_llm_calls_after_a_trip(self):
        verdicts = [HaltResponse(triggered=True, reason="no")]
        with self.assertRaises(Halted):
            _, _ = _run(self.PROMPT, verdicts)
        # one call for the guard, none for [[answer]]
        stub = _stub(verdicts)
        with patch("struckdown.llm.structured_chat_async", stub):
            try:
                sd.complete(self.PROMPT, context={}, model=MODEL, credentials=CREDS)
            except Halted:
                pass
        self.assertEqual(stub.calls["n"], 1)

    def test_passes_through_when_not_triggered(self):
        verdicts = [
            HaltResponse(triggered=False, reason="ordinary question"),
            _Text("here is the answer"),
        ]
        result, n = _run(self.PROMPT, verdicts)
        self.assertEqual(n, 2)
        self.assertEqual(result["answer"], "here is the answer")

    def test_when_false_inverts_the_gate(self):
        prompt = "On topic?\n[[halt:on_topic|when=false]]\n\nAnswer.\n[[answer]]\n"
        # triggered=True means on-topic, so with when=false it must NOT halt
        verdicts = [HaltResponse(triggered=True, reason="about teaching"), _Text("ok")]
        result, n = _run(prompt, verdicts)
        self.assertEqual(n, 2)
        self.assertEqual(result["answer"], "ok")

        # triggered=False with when=false halts
        with self.assertRaises(Halted):
            _run(prompt, [HaltResponse(triggered=False, reason="off topic")])

    def test_on_halt_return_hands_back_results_instead(self):
        verdicts = [HaltResponse(triggered=True, reason="no")]
        result, _ = _run(self.PROMPT, verdicts, on_halt="return")
        self.assertIn("guard", result.results)
        self.assertNotIn("answer", result.results)

    def test_reason_is_never_empty_string_when_given(self):
        verdicts = [HaltResponse(triggered=True, reason="tried to extract rules")]
        with self.assertRaises(Halted) as ctx:
            _run(self.PROMPT, verdicts)
        self.assertTrue(ctx.exception.reason)


class HaltIncrementalTests(unittest.TestCase):
    PROMPT = "Is this bad?\n[[halt:guard]]\n\nAnswer it.\n[[answer]]\n"

    def test_emits_complete_with_early_termination(self):
        verdicts = [HaltResponse(triggered=True, reason="no")]
        with self.assertRaises(Halted):
            _run_incremental(self.PROMPT, verdicts)

    def test_events_before_the_halt_are_still_delivered(self):
        verdicts = [HaltResponse(triggered=True, reason="no")]
        stub = _stub(verdicts)
        events = []

        async def drive():
            async for event in sd.complete_incremental_async(
                self.PROMPT, model=MODEL, credentials=CREDS, context={},
                on_halt="return",
            ):
                events.append(event)

        with patch("struckdown.llm.structured_chat_async", stub):
            anyio.run(drive)

        kinds = [type(e).__name__ for e in events]
        self.assertIn("SlotCompleted", kinds)
        completed = [e for e in events if isinstance(e, SlotCompleted)]
        self.assertEqual([e.slot_key for e in completed], ["guard"])

        finals = [e for e in events if isinstance(e, ProcessingComplete)]
        self.assertEqual(len(finals), 1)
        self.assertTrue(finals[0].early_termination)


class RetractionTests(unittest.TestCase):
    def test_slot_retracted_is_exported_and_typed(self):
        event = SlotRetracted(segment_index=0, slot_key="answer", reason="halted")
        self.assertEqual(event.type, "slot_retracted")
        self.assertIn("SlotRetracted", sd.__all__)

    def test_reasons_are_constrained(self):
        for reason in ("gathering", "halted", "retried", "errored"):
            SlotRetracted(segment_index=0, slot_key="a", reason=reason)
        with self.assertRaises(Exception):
            SlotRetracted(segment_index=0, slot_key="a", reason="nonsense")


if __name__ == "__main__":
    unittest.main()
