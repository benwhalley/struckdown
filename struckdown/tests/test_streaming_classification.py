"""Tests for slot classification, prefix resolver, and streaming event types."""

import pytest

from struckdown.incremental import SlotStreamStart, TokenDelta
from struckdown.return_type_models import (
    SlotCategory,
    build_prefix_resolver,
    classify_slot,
    compute_optimal_max_tokens,
)


# --- classify_slot ---


class _FakeAction:
    """Stub with _executor to simulate an action return type."""
    _executor = lambda: None


class _FakeResponseModel:
    """Stub for a normal LLM return type."""
    pass


def test_classify_action():
    assert classify_slot("search", _FakeAction()) == SlotCategory.ACTION


def test_classify_free_text_respond():
    assert classify_slot("respond", _FakeResponseModel()) == SlotCategory.FREE_TEXT


def test_classify_free_text_default():
    assert classify_slot("default", _FakeResponseModel()) == SlotCategory.FREE_TEXT


def test_classify_free_text_none():
    assert classify_slot(None, _FakeResponseModel()) == SlotCategory.FREE_TEXT


def test_classify_free_text_speak():
    assert classify_slot("speak", _FakeResponseModel()) == SlotCategory.FREE_TEXT


def test_classify_free_text_think():
    assert classify_slot("think", _FakeResponseModel()) == SlotCategory.FREE_TEXT


def test_classify_constrained_pick():
    assert classify_slot("pick", _FakeResponseModel()) == SlotCategory.CONSTRAINED


def test_classify_constrained_bool():
    assert classify_slot("bool", _FakeResponseModel()) == SlotCategory.CONSTRAINED


def test_classify_constrained_int():
    assert classify_slot("int", _FakeResponseModel()) == SlotCategory.CONSTRAINED


def test_classify_constrained_date():
    assert classify_slot("date", _FakeResponseModel()) == SlotCategory.CONSTRAINED


# --- build_prefix_resolver ---


def test_prefix_resolver_unique_first_chars():
    resolver = build_prefix_resolver(["apple", "orange", "banana"])
    assert resolver["a"] == "apple"
    assert resolver["o"] == "orange"
    assert resolver["b"] == "banana"


def test_prefix_resolver_shared_prefix():
    resolver = build_prefix_resolver(["apple", "apricot", "banana"])
    assert resolver["app"] == "apple"
    assert resolver["apr"] == "apricot"
    assert resolver["b"] == "banana"


def test_prefix_resolver_case_insensitive():
    resolver = build_prefix_resolver(["Yes", "No"])
    assert resolver["y"] == "Yes"
    assert resolver["n"] == "No"


def test_prefix_resolver_single_option():
    resolver = build_prefix_resolver(["only"])
    assert len(resolver) == 1
    assert resolver["o"] == "only"


def test_prefix_resolver_empty():
    assert build_prefix_resolver([]) == {}


# --- compute_optimal_max_tokens ---


def test_max_tokens_bool():
    result = compute_optimal_max_tokens("bool", None)
    assert result == 50  # 10 + 40 tool call overhead


def test_max_tokens_pick_with_options():
    result = compute_optimal_max_tokens("pick", None, ["apple", "orange", "banana"])
    assert result is not None
    assert result > 0
    assert result < 100  # should be tight even with tool call overhead


def test_max_tokens_int():
    assert compute_optimal_max_tokens("int", None) == 60  # 20 + 40 tool call overhead


def test_max_tokens_free_text_returns_none():
    assert compute_optimal_max_tokens("respond", None) is None


def test_max_tokens_none_type_returns_none():
    assert compute_optimal_max_tokens(None, None) is None


# --- event types ---


def test_slot_stream_start():
    event = SlotStreamStart(segment_index=0, slot_key="response")
    assert event.type == "stream_start"
    assert event.slot_key == "response"


def test_token_delta():
    event = TokenDelta(
        segment_index=0,
        slot_key="response",
        delta="hello",
        accumulated="hello",
    )
    assert event.type == "token_delta"
    assert event.delta == "hello"
    assert event.accumulated == "hello"


def test_token_delta_accumulation():
    e1 = TokenDelta(segment_index=0, slot_key="r", delta="he", accumulated="he")
    e2 = TokenDelta(segment_index=0, slot_key="r", delta="llo", accumulated="hello")
    assert e2.accumulated == e1.accumulated + e2.delta
