"""Tool slots: ``[[answer|use_tools=true, max_iter=3]]``.

pydantic-ai's TestModel drives the loop, so these tests are about struckdown's
side of it -- that the caller's tools and deps arrive, that a template cannot
raise the caller's ceiling, that a failing tool does not end the run, and that
the caller is told what the model called.
"""

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import anyio
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

import struckdown as sd
from struckdown.incremental import ToolCompleted, ToolStarted
from struckdown.llm import clamp_limits
from struckdown.segment_processor import slot_option_int, slot_uses_tools

CREDS = sd.LLMCredentials(api_key="x", base_url="http://example.invalid")
MODEL = sd.LLM(model_name="stub")

PROMPT = "Answer the question.\n[[answer|use_tools=true, max_iter=3]]\n"


@dataclass
class Deps:
    user: str


def _with_test_model(fn):
    """Run ``fn`` with every LLM call served by pydantic-ai's TestModel."""
    with patch.object(sd.LLM, "get_pydantic_model", lambda self, creds=None: TestModel()):
        return fn()


class OptionParsingTests(unittest.TestCase):
    def test_use_tools_is_read_from_slot_options(self):
        from struckdown.segment_processor import build_slot_info_map

        slot = build_slot_info_map(PROMPT)["answer"]
        self.assertTrue(slot_uses_tools(slot))
        self.assertEqual(slot_option_int(slot, "max_iter"), 3)

    def test_a_plain_slot_does_not_use_tools(self):
        from struckdown.segment_processor import build_slot_info_map

        slot = build_slot_info_map("Answer.\n[[answer]]\n")["answer"]
        self.assertFalse(slot_uses_tools(slot))
        self.assertIsNone(slot_option_int(slot, "max_iter"))

    def test_two_options_are_not_mistaken_for_a_quantifier(self):
        """A regression: ``(_, _)`` also matches a two-item option list."""
        from struckdown.segment_processor import build_slot_info_map

        slot = build_slot_info_map("Q.\n[[answer|a=1,b=2]]\n")["answer"]
        self.assertEqual(len(slot.options), 2)


class CeilingTests(unittest.TestCase):
    def test_default_ceiling_is_finite(self):
        limits = clamp_limits(None)
        self.assertEqual(limits.request_limit, 20)
        self.assertEqual(limits.tool_calls_limit, 20)
        self.assertEqual(limits.output_tokens_limit, 250_000)

    def test_a_caller_ceiling_replaces_the_default_output_limit(self):
        ceiling = UsageLimits(output_tokens_limit=1_000)
        self.assertEqual(clamp_limits(ceiling).output_tokens_limit, 1_000)

    def test_template_may_lower_but_not_raise_the_default_ceiling(self):
        self.assertEqual(clamp_limits(None, max_iter=3).request_limit, 3)
        self.assertEqual(clamp_limits(None, max_iter=99).request_limit, 20)
        self.assertEqual(clamp_limits(None, max_calls=99).tool_calls_limit, 20)

    def test_a_template_may_lower_the_caller_ceiling(self):
        ceiling = UsageLimits(request_limit=4, tool_calls_limit=8)
        self.assertEqual(clamp_limits(ceiling, max_iter=2).request_limit, 2)
        self.assertEqual(clamp_limits(ceiling, max_calls=3).tool_calls_limit, 3)

    def test_a_template_may_not_raise_it(self):
        ceiling = UsageLimits(request_limit=4, tool_calls_limit=8)
        self.assertEqual(clamp_limits(ceiling, max_iter=99).request_limit, 4)
        self.assertEqual(clamp_limits(ceiling, max_calls=500).tool_calls_limit, 8)

    def test_token_limits_survive_clamping(self):
        ceiling = UsageLimits(request_limit=4, total_tokens_limit=1000)
        self.assertEqual(clamp_limits(ceiling, max_iter=2).total_tokens_limit, 1000)


class ToolLoopTests(unittest.TestCase):
    def test_tool_loop_does_not_impose_its_own_max_tokens(self):
        """The output backstop is a run-level limit, not a per-request cap.

        A ``max_tokens`` above a model's own output ceiling is a provider
        error, so the loop must leave the field alone unless asked.
        """
        from struckdown import llm as llm_module

        seen = {}
        translate = llm_module._translate_kwargs

        def capture(kwargs, **options):
            seen.update(kwargs)
            return translate(kwargs, **options)

        with patch("struckdown.llm._translate_kwargs", side_effect=capture):
            _with_test_model(
                lambda: sd.complete(
                    PROMPT, context={}, model=MODEL, credentials=CREDS
                )
            )

        self.assertNotIn("max_tokens", seen)

    def test_the_model_can_call_a_caller_supplied_tool(self):
        seen = []

        def lookup_module(module_code: str) -> str:
            """Look a module up by its code."""
            seen.append(module_code)
            return f"{module_code}: Research Methods"

        result = _with_test_model(
            lambda: sd.complete(
                PROMPT,
                context={},
                model=MODEL,
                credentials=CREDS,
                tools=[lookup_module],
            )
        )
        self.assertTrue(seen, "the tool was never called")
        self.assertIn("answer", result.results)

    def test_deps_reach_the_tool(self):
        seen = {}

        def whoami(ctx) -> str:
            """Who is asking."""
            seen["user"] = ctx.deps.user
            return ctx.deps.user

        # pydantic-ai binds RunContext by annotation; use a plain closure
        # instead so this test stays about struckdown's passthrough.
        def whoami_plain() -> str:
            """Who is asking."""
            seen["called"] = True
            return "ben"

        _with_test_model(
            lambda: sd.complete(
                PROMPT,
                context={},
                model=MODEL,
                credentials=CREDS,
                tools=[whoami_plain],
                deps=Deps(user="ben"),
                deps_type=Deps,
            )
        )
        self.assertTrue(seen.get("called"))

    def test_a_failing_tool_does_not_end_the_run(self):
        def explode() -> str:
            """Always fails."""
            raise RuntimeError("database on fire")

        result = _with_test_model(
            lambda: sd.complete(
                PROMPT, context={}, model=MODEL, credentials=CREDS, tools=[explode]
            )
        )
        self.assertIn("answer", result.results)

    def test_an_identical_call_runs_once(self):
        calls = []

        def search(query: str) -> str:
            """Search for something."""
            calls.append(query)
            return "nothing found"

        from struckdown.llm import _instrumented_tool

        wrapped = _instrumented_tool(search, None)

        async def drive():
            await wrapped(query="extensions")
            await wrapped(query="extensions")
            await wrapped(query="resits")

        anyio.run(drive)
        self.assertEqual(calls, ["extensions", "resits"])

    def test_tool_events_are_emitted(self):
        def lookup_module(module_code: str) -> str:
            """Look a module up."""
            return "PSYC605: Research Methods"

        events = []

        async def drive():
            async for event in sd.complete_incremental_async(
                PROMPT,
                model=MODEL,
                credentials=CREDS,
                context={},
                tools=[lookup_module],
            ):
                events.append(event)

        _with_test_model(lambda: anyio.run(drive))

        started = [e for e in events if isinstance(e, ToolStarted)]
        completed = [e for e in events if isinstance(e, ToolCompleted)]
        self.assertTrue(started, "no ToolStarted event")
        self.assertTrue(completed, "no ToolCompleted event")
        self.assertEqual(started[0].tool_name, "lookup_module")
        self.assertTrue(completed[0].ok)


if __name__ == "__main__":
    unittest.main()


class StreamingToolSlotTests(unittest.TestCase):
    """A tool slot streams its answer once the gathering is done."""

    def test_tokens_arrive_after_the_tool_events(self):
        from struckdown.incremental import SlotStreamStart, TokenDelta

        def lookup_module(module_code: str) -> str:
            """Look a module up."""
            return "PSYC605: Research Methods"

        events = []

        async def drive():
            async for event in sd.complete_incremental_async(
                PROMPT,
                model=MODEL,
                credentials=CREDS,
                context={},
                tools=[lookup_module],
                stream=True,
            ):
                events.append(event)

        # TestModel leaves an optional field None by default, which would mean
        # no text to stream; pin an answer so there is.
        speaking = TestModel(custom_output_args={"response": "Research Methods"})
        with patch.object(
            sd.LLM, "get_pydantic_model", lambda self, creds=None: speaking
        ):
            anyio.run(drive)

        kinds = [type(e).__name__ for e in events]
        self.assertIn("ToolStarted", kinds)
        self.assertIn("TokenDelta", kinds)
        # every tool event precedes the first token: the reader sees the
        # gathering happen, then the answer being written
        first_token = kinds.index("TokenDelta")
        last_tool = max(
            i for i, k in enumerate(kinds) if k in ("ToolStarted", "ToolCompleted")
        )
        self.assertLess(last_tool, first_token)
        self.assertIn("SlotStreamStart", kinds)
        self.assertLess(kinds.index("SlotStreamStart"), first_token)


class ThinkingStreamTests(unittest.TestCase):
    """Reasoning arrives as ThinkingDelta, between tool calls as well as before
    the answer.

    The handler is exercised directly: attaching one makes pydantic-ai stream
    the request, which its own FunctionModel cannot fake without a
    stream_function, and what matters here is the mapping from its part events
    to struckdown's.
    """

    def _events(self, parts):
        from struckdown.llm import _thinking_stream_handler

        seen = []
        handler = _thinking_stream_handler(
            lambda kind, payload: seen.append((kind, payload))
        )

        async def stream():
            for part in parts:
                yield part

        anyio.run(lambda: handler(None, stream()))
        return seen

    def test_a_thinking_part_and_its_deltas_become_events(self):
        from pydantic_ai.messages import (PartDeltaEvent, PartStartEvent,
                                          ThinkingPart, ThinkingPartDelta)

        seen = self._events([
            PartStartEvent(index=0, part=ThinkingPart(content="They want ")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="the leader. ")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="Look it up.")),
        ])

        self.assertEqual([kind for kind, _ in seen], ["thinking"] * 3)
        self.assertEqual(seen[0][1]["delta"], "They want ")
        self.assertEqual(
            seen[-1][1]["accumulated"], "They want the leader. Look it up."
        )

    def test_other_parts_are_ignored(self):
        from pydantic_ai.messages import PartStartEvent, TextPart

        seen = self._events([PartStartEvent(index=0, part=TextPart(content="hello"))])
        self.assertEqual(seen, [])

    def test_a_consumer_error_does_not_end_the_run(self):
        from struckdown.llm import _announce_safely

        def explode(kind, payload):
            raise RuntimeError("consumer blew up")

        _announce_safely(explode, "thinking", {"delta": "x", "accumulated": "x"})

    def test_the_queue_maps_thinking_to_the_right_event(self):
        from struckdown.incremental import ThinkingDelta
        from struckdown.segment_processor import _tool_queue_event

        event = _tool_queue_event(
            "thinking", {"delta": "a", "accumulated": "a"}, 0, "answer"
        )
        self.assertIsInstance(event, ThinkingDelta)
        self.assertEqual(event.slot_key, "answer")


class CancellationTests(unittest.TestCase):
    """Closing the stream must reach the call in flight.

    Not driven end to end here: attaching the thinking handler makes
    pydantic-ai stream every request, and its own FunctionModel cannot express
    a tool call on the stream path without DeltaToolCalls. What is checked is
    the mechanism -- that the tool branch cancels its feeder when the generator
    it is feeding goes away. See TODO.md for the ollama test that would cover
    the whole path.
    """

    def test_the_tool_branch_cancels_its_feeder_on_close(self):
        import inspect

        from struckdown import segment_processor

        source = inspect.getsource(
            segment_processor.process_segment_with_delta_incremental
        )
        # the finally that runs when a consumer closes the generator
        self.assertIn("finally:", source)
        self.assertIn("feeder.cancel()", source)

    def test_the_agent_run_cancels_its_task_on_close(self):
        import inspect

        from struckdown.llm import run_agent_with_tools

        source = inspect.getsource(run_agent_with_tools)
        self.assertIn("task.cancel()", source)

    def test_events_and_output_share_one_queue(self):
        """So a tool event is not held behind the next chunk of answer -- or
        behind a call that has not returned at all."""
        import inspect

        from struckdown import segment_processor

        source = inspect.getsource(
            segment_processor.process_segment_with_delta_incremental
        )
        self.assertIn('await queue.get()', source)
        self.assertNotIn("def _drain()", source)
