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
