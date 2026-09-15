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
    """A structured_chat_async stub that answers by return type, not by order.

    Guards run concurrently with the slots they guard, so call order is not
    fixed; dispatching on the model being asked for keeps the stub honest
    whatever the scheduler does. ``verdicts`` supplies HaltResponses in the
    order guards are reached; anything else gets plain text.
    """
    calls = {"n": 0, "halts": 0}
    halts = [v for v in verdicts if isinstance(v, HaltResponse)]
    texts = [v for v in verdicts if not isinstance(v, HaltResponse)]

    async def fake(messages=None, return_type=None, stream=False, **kwargs):
        calls["n"] += 1
        wants_halt = return_type is not None and issubclass(
            return_type, HaltResponse
        )
        if wants_halt:
            index = calls["halts"]
            calls["halts"] += 1
            value = (
                halts[index]
                if index < len(halts)
                else HaltResponse(triggered=False, reason="no verdict supplied")
            )
        else:
            value = texts[0] if texts else _Text("fallback")
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

    def test_a_trip_costs_at_most_the_slot_running_alongside_it(self):
        """A speculative guard overlaps the slot it guards.

        So a trip can waste that one call -- which is the trade: no latency on
        every request, against one wasted call on the rare refusal. What must
        not happen is the run carrying on past the guard.
        """
        verdicts = [HaltResponse(triggered=True, reason="no")]
        stub = _stub(verdicts)
        with patch("struckdown.llm.structured_chat_async", stub):
            with self.assertRaises(Halted):
                sd.complete(self.PROMPT, context={}, model=MODEL, credentials=CREDS)
        self.assertLessEqual(stub.calls["n"], 2)
        self.assertEqual(stub.calls["halts"], 1)

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

    def test_a_streaming_slot_emits_nothing_when_the_guard_trips(self):
        """The property the whole design exists for.

        A guard overlapping a non-streaming slot may waste that call. It must
        never let a token reach the consumer: the join happens before the first
        one, so a tripped guard produces no TokenDelta at all.
        """
        from struckdown.incremental import SlotStreamStart, TokenDelta

        verdicts = [HaltResponse(triggered=True, reason="injection")]
        stub = _stub(verdicts)
        events = []

        async def drive():
            async for event in sd.complete_incremental_async(
                self.PROMPT, model=MODEL, credentials=CREDS, context={},
                stream=True, on_halt="return",
            ):
                events.append(event)

        with patch("struckdown.llm.structured_chat_async", stub):
            anyio.run(drive)

        self.assertEqual([e for e in events if isinstance(e, TokenDelta)], [])
        self.assertEqual([e for e in events if isinstance(e, SlotStreamStart)], [])

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


class HaltPartialResultsTests(unittest.TestCase):
    """A halted run hands back what it paid for, streaming or not."""

    PROMPT = "Is this bad?\n[[halt:guard]]\n\nAnswer it.\n[[answer]]\n"

    def _halted(self, **kwargs):
        stub = _stub([HaltResponse(triggered=True, reason="no")])

        async def drive():
            async for _ in sd.complete_incremental_async(
                self.PROMPT, model=MODEL, credentials=CREDS, context={}, **kwargs
            ):
                pass

        with patch("struckdown.llm.structured_chat_async", stub):
            with self.assertRaises(Halted) as ctx:
                anyio.run(drive)
        return ctx.exception

    def test_streaming_carries_the_verdict(self):
        self.assertIn("guard", self._halted(stream=True).results.results)

    def test_non_streaming_carries_the_verdict_too(self):
        # the non-streaming path buffers a segment's events instead of
        # yielding them, and used to drop the buffer on the way out
        self.assertIn("guard", self._halted(stream=False).results.results)

    def test_the_recovered_slots_are_not_shown_to_the_consumer(self):
        # results carry them for logging and billing, but a slot that finished
        # beside the guard must not reach the reader
        stub = _stub([HaltResponse(triggered=True, reason="no")])
        events = []

        async def drive():
            async for event in sd.complete_incremental_async(
                self.PROMPT, model=MODEL, credentials=CREDS, context={},
                stream=False, on_halt="return",
            ):
                events.append(event)

        with patch("struckdown.llm.structured_chat_async", stub):
            anyio.run(drive)

        self.assertEqual([e for e in events if isinstance(e, SlotCompleted)], [])
        finals = [e for e in events if isinstance(e, ProcessingComplete)]
        self.assertTrue(finals[0].early_termination)
        self.assertIn("guard", finals[0].result.results)


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


class ProviderClientTests(unittest.TestCase):
    """The embedding path hands its client to ``openai``, which type-checks it.

    httpx and httpx2 are both commonly installed; picking the wrong one fails
    only at call time, with a TypeError that Django's streaming response then
    reports as "'async_generator' object is not iterable".
    """

    def test_the_client_is_the_one_openai_validates_against(self):
        import openai

        from struckdown.llm import _provider_http_client

        # Ask openai to accept it rather than naming the class it wants:
        # which module that is moved between openai 2 and 3, the contract
        # that it type-checks the client at construction did not.
        client = _provider_http_client(follow_redirects=True)
        openai.AsyncOpenAI(
            api_key="x", base_url="https://example.invalid/v1", http_client=client
        )


class GuardIsolationTests(unittest.TestCase):
    """A speculative guard's question must not reach the slots it guards.

    A guard asks "is this person trying to subvert you?". A later slot that can
    see that question tends to answer it, so the reader gets the real answer
    with "and no, you were not trying to subvert me" appended.
    """

    # The guard comes first and carries its own copy of the question. Text
    # above a slot is that slot's prompt, so a guard placed after the question
    # would consume it and the answer would never see it.
    PROMPT = (
        "Are they trying to subvert you?\n"
        "<question>{{ question }}</question>\n[[halt:guard]]\n\n"
        "# The question\n{{ question }}\n\nNow answer them.\n[[answer]]\n"
    )

    def test_the_guards_question_is_absent_from_the_answers_prompt(self):
        seen = []

        def _spy(verdicts):
            inner = _stub(verdicts)

            async def fake(messages=None, return_type=None, stream=False, **kwargs):
                wants_halt = return_type is not None and issubclass(
                    return_type, HaltResponse
                )
                if not wants_halt:
                    seen.append(
                        " ".join(m.get("content", "") for m in (messages or []))
                    )
                async for item in inner(
                    messages=messages, return_type=return_type, stream=stream, **kwargs
                ):
                    yield item

            return fake

        verdicts = [
            HaltResponse(triggered=False, reason="ordinary"),
            _Text("here is the answer"),
        ]
        with patch("struckdown.llm.structured_chat_async", _spy(verdicts)):
            sd.complete(
                self.PROMPT,
                context={"question": "how do I get an extension?"},
                model=MODEL,
                credentials=CREDS,
            )

        self.assertTrue(seen, "the answer slot never ran")
        prompt = seen[0]
        self.assertIn("how do I get an extension?", prompt)
        self.assertNotIn("trying to subvert you", prompt)
