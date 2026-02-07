"""Tests for chatter/chatter_async with list contexts -- the parallel batch processing layer."""

import asyncio
import time
from unittest.mock import MagicMock, patch

from struckdown import (
    LLM,
    ChatterResult,
    SegmentResult,
    chatter,
    chatter_async,
)


def _make_mock_chat(delay=0, fail_for_contexts=None):
    """
    Build a mock for structured_chat that returns a response derived from
    the context variable 'name', so we can verify each context was routed
    correctly.
    """
    fail_for_contexts = fail_for_contexts or set()
    calls = []

    def mock_structured_chat(*args, **kwargs):
        messages = kwargs.get("messages", [])
        user_text = messages[-1]["content"] if messages else ""
        calls.append(user_text)

        for name in fail_for_contexts:
            if name in user_text:
                raise ValueError(f"Deliberate failure for {name}")

        if delay:
            time.sleep(delay)

        mock_res = MagicMock()
        mock_res.response = f"reply:{user_text.strip()}"
        mock_res.model_dump.return_value = {"response": mock_res.response}
        mock_com = {"_run_id": "test", "usage": {}}
        return mock_res, mock_com

    return mock_structured_chat, calls


# -- result ordering ----------------------------------------------------------


def test_results_preserve_input_order():
    """Results list matches the order of the input contexts, not completion order."""
    mock_chat, _ = _make_mock_chat()
    contexts = [{"name": f"ctx_{i}"} for i in range(6)]

    async def run():
        with patch("struckdown.llm.structured_chat", side_effect=mock_chat):
            return await chatter_async(
                "hello {{name}} [[reply]]",
                contexts,
                model=LLM(),
            )

    results = asyncio.run(run())

    assert len(results) == 6
    for i, r in enumerate(results):
        assert isinstance(r, ChatterResult)
        assert f"ctx_{i}" in r.response


# -- per-context isolation ----------------------------------------------------


def test_each_context_receives_its_own_variables():
    """Each context dict is passed independently to the internal processor."""
    mock_chat, calls = _make_mock_chat()
    contexts = [{"name": "alice"}, {"name": "bob"}, {"name": "carol"}]

    async def run():
        with patch("struckdown.llm.structured_chat", side_effect=mock_chat):
            return await chatter_async(
                "greet {{name}} [[reply]]",
                contexts,
                model=LLM(),
            )

    results = asyncio.run(run())

    combined = " ".join(calls)
    assert "alice" in combined
    assert "bob" in combined
    assert "carol" in combined

    assert "alice" in results[0].response
    assert "bob" in results[1].response
    assert "carol" in results[2].response


# -- error handling -----------------------------------------------------------


def test_single_failure_does_not_block_others():
    """One context raising should not prevent other contexts from completing."""
    mock_chat, _ = _make_mock_chat(fail_for_contexts={"bob"})
    contexts = [{"name": "alice"}, {"name": "bob"}, {"name": "carol"}]

    async def run():
        with patch("struckdown.llm.structured_chat", side_effect=mock_chat):
            return await chatter_async(
                "greet {{name}} [[reply]]",
                contexts,
                model=LLM(),
            )

    results = asyncio.run(run())

    assert isinstance(results[0], ChatterResult)
    assert "alice" in results[0].response

    assert isinstance(results[1], (Exception, ValueError))

    assert isinstance(results[2], ChatterResult)
    assert "carol" in results[2].response


def test_all_failures_returns_all_exceptions():
    """If every context fails, the result list is all exceptions."""
    mock_chat, _ = _make_mock_chat(fail_for_contexts={"a", "b", "c"})
    contexts = [{"name": "a"}, {"name": "b"}, {"name": "c"}]

    async def run():
        with patch("struckdown.llm.structured_chat", side_effect=mock_chat):
            return await chatter_async(
                "greet {{name}} [[reply]]",
                contexts,
                model=LLM(),
            )

    results = asyncio.run(run())

    assert len(results) == 3
    for r in results:
        assert isinstance(r, Exception)


# -- on_complete callback -----------------------------------------------------


def test_on_complete_called_for_each_context():
    """on_complete receives (index, result) for every context."""
    mock_chat, _ = _make_mock_chat()
    contexts = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    completions = []

    async def run():
        with patch("struckdown.llm.structured_chat", side_effect=mock_chat):
            return await chatter_async(
                "greet {{name}} [[reply]]",
                contexts,
                model=LLM(),
                on_complete=lambda idx, res: completions.append((idx, res)),
            )

    asyncio.run(run())

    indices = sorted(idx for idx, _ in completions)
    assert indices == [0, 1, 2]
    for _, res in completions:
        assert isinstance(res, ChatterResult)


def test_on_complete_called_for_errors_too():
    """on_complete should fire even when a context raises."""
    mock_chat, _ = _make_mock_chat(fail_for_contexts={"b"})
    contexts = [{"name": "a"}, {"name": "b"}]
    completions = []

    async def run():
        with patch("struckdown.llm.structured_chat", side_effect=mock_chat):
            return await chatter_async(
                "greet {{name}} [[reply]]",
                contexts,
                model=LLM(),
                on_complete=lambda idx, res: completions.append((idx, type(res))),
            )

    asyncio.run(run())

    completions.sort(key=lambda x: x[0])
    assert completions[0] == (0, ChatterResult)
    assert completions[1] == (1, ValueError)


# -- concurrency control ------------------------------------------------------


def test_max_concurrent_limits_parallelism():
    """With max_concurrent=1, contexts should run one at a time."""
    from struckdown import _chatter_single_async

    timestamps = {}
    mock_chat, _ = _make_mock_chat()

    async def recording_chatter_async(
        multipart_prompt, context, model=None, credentials=None, **kw
    ):
        name = context["name"]
        timestamps[name] = {"start": time.monotonic()}
        await asyncio.sleep(0.05)
        with patch("struckdown.llm.structured_chat", side_effect=mock_chat):
            result = await _chatter_single_async(
                multipart_prompt, model=model, credentials=credentials, context=context, **kw
            )
        timestamps[name]["end"] = time.monotonic()
        return result

    contexts = [{"name": "first"}, {"name": "second"}, {"name": "third"}]

    async def run():
        with patch("struckdown._chatter_single_async", side_effect=recording_chatter_async):
            return await chatter_async(
                "greet {{name}} [[reply]]",
                contexts,
                model=LLM(),
                max_concurrent=1,
            )

    asyncio.run(run())

    # sort by start time and verify no overlap
    ordered = sorted(timestamps.items(), key=lambda x: x[1]["start"])
    for i in range(len(ordered) - 1):
        earlier = ordered[i][1]
        later = ordered[i + 1][1]
        assert earlier["end"] <= later["start"] + 0.01, (
            f"{ordered[i][0]} ended at {earlier['end']:.3f} but "
            f"{ordered[i + 1][0]} started at {later['start']:.3f}"
        )


# -- empty input --------------------------------------------------------------


def test_empty_contexts_returns_empty_list():
    async def run():
        return await chatter_async("hello [[reply]]", [], model=LLM())

    results = asyncio.run(run())
    assert results == []


# -- kwargs forwarding --------------------------------------------------------


def test_kwargs_forwarded_to_internal_processor():
    """Extra kwargs like extra_kwargs should reach the internal processor."""
    received_kwargs = {}

    async def spy_chatter_async(
        multipart_prompt, context, model=None, credentials=None, extra_kwargs=None, **kwargs
    ):
        received_kwargs["multipart_prompt"] = multipart_prompt
        received_kwargs["context"] = context
        received_kwargs["extra_kwargs"] = extra_kwargs
        r = ChatterResult()
        r["reply"] = SegmentResult(name="reply", output="ok", prompt="")
        return r

    async def run():
        with patch("struckdown._chatter_single_async", side_effect=spy_chatter_async):
            return await chatter_async(
                "hello [[reply]]",
                [{"name": "test"}],
                model=LLM(),
                extra_kwargs={"temperature": 0.5},
            )

    asyncio.run(run())

    assert received_kwargs["extra_kwargs"] == {"temperature": 0.5}
    assert received_kwargs["context"] == {"name": "test"}
    assert received_kwargs["multipart_prompt"] == "hello [[reply]]"


# -- sync wrapper -------------------------------------------------------------


def test_chatter_sync_with_list():
    """The sync chatter with list should produce the same results."""
    mock_chat, _ = _make_mock_chat()
    contexts = [{"name": "x"}, {"name": "y"}]

    with patch("struckdown.llm.structured_chat", side_effect=mock_chat):
        results = chatter(
            "greet {{name}} [[reply]]",
            contexts,
            model=LLM(),
        )

    assert len(results) == 2
    assert isinstance(results[0], ChatterResult)
    assert isinstance(results[1], ChatterResult)
    assert "x" in results[0].response
    assert "y" in results[1].response
