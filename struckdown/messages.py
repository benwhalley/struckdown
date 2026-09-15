"""Turning struckdown's message list into what pydantic-ai takes, and back.

Struckdown builds an OpenAI-shaped list -- ``{"role": ..., "content": ...}`` --
as it walks a template: ``<system>`` becomes a system message, ``<user>`` and
``<assistant>`` tags become their own turns, a slot's output is appended as an
assistant turn, and an action returning a :class:`~struckdown.actions.MessageList`
contributes several at once. That list is the conversation.

pydantic-ai wants the same thing in three pieces: ``instructions`` for the
operator's text, ``message_history`` for what has already been said, and
``user_prompt`` for the turn being answered now. This module is the translation,
and :func:`to_openai_messages` is the way back -- what a caller stores between
turns so the next one can carry it.

Going through dicts rather than round-tripping pydantic-ai's own types is
deliberate: the stored shape is the one every provider already speaks, it reads
in an admin JSON field, and it does not change when pydantic-ai's internals do.
"""

from __future__ import annotations

import json
from typing import Any

# Roles a stored message may carry. "tool" is here so a conversation can keep
# the gathering that produced an answer, not only the answer.
ROLES = ("system", "user", "assistant", "tool")


def split_for_agent(messages: list[dict]) -> tuple[str, list, str]:
    """``(instructions, message_history, user_prompt)`` for ``Agent.run``.

    System messages become instructions, joined in order -- ``<system>`` blocks
    accumulate across checkpoints and all of them are still in force. They are
    passed as pydantic-ai *instructions* rather than as a system message in the
    history, because instructions are not replayed out of a supplied history: a
    resumed run gets the current template's text rather than a stale copy of it.

    The trailing user message, if there is one, becomes the prompt for this
    turn. Everything before it is history. A list that ends in an assistant
    message leaves the prompt empty, which is how prefill is expressed.
    """
    instructions = "\n\n".join(
        m.get("content") or "" for m in messages if m.get("role") == "system"
    )
    rest = [m for m in messages if m.get("role") != "system"]

    prompt = ""
    if rest and rest[-1].get("role") == "user" and not rest[-1].get("tool_calls"):
        prompt = rest.pop().get("content") or ""

    return instructions, to_pydantic_messages(rest), prompt


def to_pydantic_messages(messages: list[dict]) -> list:
    """OpenAI-shaped dicts as pydantic-ai ``ModelRequest``/``ModelResponse``.

    Consecutive messages of the same kind are merged into one request or one
    response, because that is how a provider returned them: an assistant turn
    that called two tools is a single response with two ``ToolCallPart``s, and
    the two results that come back are one request with two ``ToolReturnPart``s.
    Splitting them into four messages is a shape no provider ever produces, and
    some reject it.
    """
    from pydantic_ai.messages import (ModelRequest, ModelResponse, SystemPromptPart,
                                      TextPart, ToolCallPart, ToolReturnPart,
                                      UserPromptPart)

    out: list = []
    request_parts: list = []
    response_parts: list = []

    def _flush_request():
        if request_parts:
            out.append(ModelRequest(parts=list(request_parts)))
            request_parts.clear()

    def _flush_response():
        if response_parts:
            out.append(ModelResponse(parts=list(response_parts)))
            response_parts.clear()

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "assistant":
            _flush_request()
            if content:
                response_parts.append(TextPart(content=content))
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                response_parts.append(
                    ToolCallPart(
                        tool_name=function.get("name") or "",
                        args=function.get("arguments") or {},
                        tool_call_id=call.get("id") or "",
                    )
                )
        elif role == "tool":
            _flush_response()
            request_parts.append(
                ToolReturnPart(
                    tool_name=message.get("name") or "",
                    content=content,
                    tool_call_id=message.get("tool_call_id") or "",
                )
            )
        elif role == "system":
            _flush_response()
            request_parts.append(SystemPromptPart(content=content))
        else:
            _flush_response()
            request_parts.append(UserPromptPart(content=content))

    _flush_request()
    _flush_response()
    return out


def to_openai_messages(messages: list) -> list[dict]:
    """pydantic-ai messages as storable dicts: the inverse of the above.

    What a caller keeps between turns. ``result.all_messages()`` after a tool
    run is the calls, the results and the answer in order, so passing that
    through here and storing it is all a conversation needs to carry its own
    gathering forward.

    Reasoning is dropped. It is bulky, it is display material rather than
    conversation, and a provider that validates reasoning chains will reject a
    stored copy of one anyway.
    """
    from pydantic_ai.messages import (ModelRequest, SystemPromptPart, TextPart,
                                      ToolCallPart, ToolReturnPart, UserPromptPart)

    out: list[dict] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    out.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    out.append({"role": "user", "content": _as_text(part.content)})
                elif isinstance(part, ToolReturnPart):
                    out.append(
                        {
                            "role": "tool",
                            "name": part.tool_name,
                            "tool_call_id": part.tool_call_id,
                            "content": _as_text(part.content),
                        }
                    )
            continue

        text = "".join(p.content for p in message.parts if isinstance(p, TextPart))
        calls = [
            {
                "id": part.tool_call_id,
                "type": "function",
                "function": {"name": part.tool_name, "arguments": _as_args(part.args)},
            }
            for part in message.parts
            if isinstance(part, ToolCallPart)
        ]
        if text or calls:
            row: dict = {"role": "assistant", "content": text}
            if calls:
                row["tool_calls"] = calls
            out.append(row)
    return out


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _as_args(args: Any) -> str:
    """Tool arguments as the JSON string the stored shape uses."""
    return args if isinstance(args, str) else json.dumps(args or {}, default=str)
