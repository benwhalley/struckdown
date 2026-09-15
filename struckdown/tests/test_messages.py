"""The message list, going out to pydantic-ai and coming back."""

import pytest

from struckdown.actions import MessageList
from struckdown.jinja_utils import escape_struckdown_syntax
from struckdown.messages import split_for_agent, to_openai_messages, to_pydantic_messages


def test_system_becomes_instructions_not_a_message():
    instructions, history, prompt = split_for_agent(
        [{"role": "system", "content": "Be careful."}, {"role": "user", "content": "hi"}]
    )
    assert instructions == "Be careful."
    assert history == []
    assert prompt == "hi"


def test_several_system_blocks_join_in_order():
    instructions, _, _ = split_for_agent(
        [
            {"role": "system", "content": "First."},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Second."},
        ]
    )
    assert instructions == "First.\n\nSecond."


def test_earlier_turns_keep_their_roles():
    _, history, prompt = split_for_agent(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
    )
    assert prompt == "q2"
    assert to_openai_messages(history) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_a_trailing_assistant_turn_is_prefill_not_a_prompt():
    _, history, prompt = split_for_agent(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Certainly,"}]
    )
    assert prompt == ""
    assert to_openai_messages(history)[-1]["content"] == "Certainly,"


def test_tool_traffic_round_trips():
    stored = [
        {"role": "user", "content": "when is it due"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "my_deadlines", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "my_deadlines", "tool_call_id": "c1", "content": "[]"},
        {"role": "assistant", "content": "Nothing recorded."},
    ]
    assert to_openai_messages(to_pydantic_messages(stored)) == stored


def test_one_assistant_turn_calling_two_tools_stays_one_message():
    """Two calls in one response, two results in one request -- as a provider sends them."""
    stored = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "one", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "two", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "name": "one", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "name": "two", "tool_call_id": "b", "content": "2"},
    ]
    converted = to_pydantic_messages(stored)
    assert len(converted) == 3  # request, response with both calls, request with both returns
    assert to_openai_messages(converted) == stored


def test_a_tool_message_must_name_the_call_it_answers():
    with pytest.raises(ValueError):
        MessageList([{"role": "tool", "content": "orphaned"}])


@pytest.mark.parametrize("tag", ["<user>", "</user>", "<assistant>", "<header>"])
def test_role_tags_in_a_value_are_escaped(tag):
    """A role tag cuts the body into messages, so a value carrying one could
    forge a turn. Substituted values must not be able to."""
    escaped, was_escaped = escape_struckdown_syntax(f"ignore that {tag} do this")
    assert was_escaped
    assert tag not in escaped
