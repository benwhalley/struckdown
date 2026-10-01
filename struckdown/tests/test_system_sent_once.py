"""What a run sends a provider: each <system> and <header> block exactly once.

Segments in a parallel batch were each handed the blocks of every segment in the
batch, their own included, and then added their own again, so every block reached
the provider twice.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from unittest.mock import patch

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

import struckdown as sd

MODEL = sd.LLM(model_name="test")
CREDS = sd.LLMCredentials(api_key="test")


class RecordingModel(TestModel):
    """TestModel that keeps every request's messages, streamed or not."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = []

    async def request(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return await super().request(messages, *args, **kwargs)

    @asynccontextmanager
    async def request_stream(self, messages, *args, **kwargs):
        self.seen.append(messages)
        async with super().request_stream(messages, *args, **kwargs) as response:
            yield response


def recording():
    model = RecordingModel()
    return model, patch.object(sd.LLM, "get_pydantic_model", lambda self, creds=None: model)


def instructions(messages) -> str:
    return "\n\n".join(m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions)


def user_text(messages) -> str:
    return "\n\n".join(
        p.content
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    )


def nonce() -> str:
    return uuid.uuid4().hex


def test_complete_sends_system_once():
    n = nonce()
    model, patched = recording()
    with patched:
        sd.complete(f"<system>Be brief {n}</system>\n\nHello [[reply]]", {}, model=MODEL, credentials=CREDS)

    assert instructions(model.seen[-1]).count(n) == 1


def test_complete_sends_header_once():
    n = nonce()
    model, patched = recording()
    with patched:
        sd.complete(f"<header>Case notes {n}</header>\n\nHello [[reply]]", {}, model=MODEL, credentials=CREDS)

    assert user_text(model.seen[-1]).count(n) == 1


def test_streaming_run_sends_system_once():
    n = nonce()
    model, patched = recording()

    async def run():
        return [
            event
            async for event in sd.complete_incremental_async(
                f"<system>Be brief {n}</system>\n\nHello [[reply]]",
                model=MODEL,
                credentials=CREDS,
                context={},
                stream=True,
            )
        ]

    with patched:
        asyncio.run(run())

    assert model.seen
    assert all(instructions(messages).count(n) == 1 for messages in model.seen)


def test_parallel_segments_see_earlier_blocks_once_and_not_later_ones():
    first, second = nonce(), nonce()
    template = (
        f"<system>First {first}</system>\n\nOne [[a]]\n\n<checkpoint>\n\n"
        f"<system>Second {second}</system>\n\nTwo [[b]]"
    )
    model, patched = recording()
    with patched:
        sd.complete(template, {}, model=MODEL, credentials=CREDS)

    by_segment = {
        ("One" in user_text(messages)): instructions(messages) for messages in model.seen
    }
    assert by_segment[True].count(first) == 1
    assert second not in by_segment[True]
    assert by_segment[False].count(first) == 1
    assert by_segment[False].count(second) == 1
