"""Reasoning-effort negotiation for model generations with different vocabularies."""

import logging

import pytest
from pydantic_ai import ModelHTTPError

from struckdown import reasoning
from struckdown.llm import _retry_effort, _translate_kwargs

SOL_ERROR = (
    "Unsupported value: 'reasoning_effort' does not support 'minimal' with this model. "
    "Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    path = tmp_path / "reasoning_effort.json"
    monkeypatch.setattr(reasoning, "_cache_path", lambda: path)
    reasoning._learned.clear()
    reasoning._loaded = False
    yield
    reasoning._learned.clear()
    reasoning._loaded = False


def test_off_is_translated_to_pydantic_ai_false():
    settings = _translate_kwargs({"thinking": "off"}, model_name="gpt-5.6-sol")
    assert settings["thinking"] is False
    assert "openai_reasoning_effort" not in settings


def test_learn_from_error_uses_cheapest_supported_value(caplog):
    with caplog.at_level(logging.INFO):
        replacement = reasoning.learn_from_error("gpt-5.6-sol", SOL_ERROR)

    assert replacement == "none"
    assert reasoning.learned("gpt-5.6-sol") == "none"
    assert (
        "reasoning_effort='minimal' rejected for model=gpt-5.6-sol; "
        "using 'none' from now on"
    ) in caplog.text


def test_learned_value_is_applied_to_later_calls():
    reasoning.remember("GPT-5.6-SOL", "none")
    settings = _translate_kwargs({"thinking": "off"}, model_name="gpt-5.6-sol")
    assert settings["openai_reasoning_effort"] == "none"


def test_negotiation_is_persisted_between_processes():
    reasoning.remember("gpt-5.6-sol", "none")
    reasoning._learned.clear()
    reasoning._loaded = False
    assert reasoning.learned("gpt-5.6-sol") == "none"


def test_unrelated_error_is_not_retried():
    error = ModelHTTPError(429, "gpt-5.6-sol", {"error": "rate limit"})
    settings = _translate_kwargs({"thinking": "off"}, model_name="gpt-5.6-sol")
    assert not _retry_effort(error, "gpt-5.6-sol", settings)


def test_rejected_effort_updates_settings_for_one_retry():
    error = ModelHTTPError(400, "gpt-5.6-sol", {"error": SOL_ERROR})
    settings = _translate_kwargs({"thinking": "off"}, model_name="gpt-5.6-sol")

    assert _retry_effort(error, "gpt-5.6-sol", settings)
    assert settings["openai_reasoning_effort"] == "none"


def test_no_retry_when_provider_offers_only_rejected_value():
    message = (
        "Unsupported value: 'reasoning_effort' does not support 'minimal'. "
        "Supported values are: 'minimal'."
    )
    error = ModelHTTPError(400, "gpt-5", {"error": message})
    settings = _translate_kwargs({"thinking": "off"}, model_name="gpt-5")
    assert not _retry_effort(error, "gpt-5", settings)
