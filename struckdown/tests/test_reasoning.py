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


# what Azure behind LiteLLM sent for thinking=minimal on gpt-5.4: the parameter is
# "reasoning.effort", the wording differs, and str(exc) shows the body's repr, so the
# quotes arrive backslash-escaped
AZURE_BODY = {
    "message": (
        "litellm.BadRequestError: AzureException BadRequestError - {\n"
        '  "error": {\n'
        "    \"message\": \"Unsupported value: 'minimal' is not supported with the "
        "'gpt-5.4-2026-03-05' model. Supported values are: 'none', 'low', 'medium', "
        "'high', and 'xhigh'.\",\n"
        '    "type": "invalid_request_error",\n'
        '    "param": "reasoning.effort",\n'
        '    "code": "unsupported_value"\n  }\n}'
    ),
    "type": "invalid_request_error",
}


def test_a_refused_level_is_retried_at_the_nearest_supported_one():
    error = ModelHTTPError(400, "gpt-5.4", AZURE_BODY)
    settings = _translate_kwargs({"thinking": "minimal"}, model_name="gpt-5.4")

    assert _retry_effort(error, "gpt-5.4", settings)
    assert settings["thinking"] == "low"


def test_the_replacement_level_is_used_from_then_on():
    error = ModelHTTPError(400, "gpt-5.4", AZURE_BODY)
    _retry_effort(error, "gpt-5.4", _translate_kwargs({"thinking": "minimal"}, model_name="gpt-5.4"))

    assert _translate_kwargs({"thinking": "minimal"}, model_name="GPT-5.4")["thinking"] == "low"
    # only that level, and only that model
    assert _translate_kwargs({"thinking": "high"}, model_name="gpt-5.4")["thinking"] == "high"
    assert _translate_kwargs({"thinking": "minimal"}, model_name="gpt-5")["thinking"] == "minimal"


@pytest.mark.parametrize(
    "level, offered, expected",
    [
        ("minimal", ["none", "low", "medium"], "low"),  # never turned off
        ("xhigh", ["low", "medium", "high"], "high"),
        ("medium", ["low", "high"], "high"),  # a tie goes up
        ("minimal", ["none", "minimal"], None),  # nothing else on offer
    ],
)
def test_nearest_level(level, offered, expected):
    assert reasoning.nearest_level(level, offered) == expected


def test_the_azure_wording_also_drives_the_off_negotiation():
    error = ModelHTTPError(400, "gpt-5.4", AZURE_BODY)
    settings = _translate_kwargs({"thinking": "off"}, model_name="gpt-5.4")

    assert _retry_effort(error, "gpt-5.4", settings)
    assert settings["openai_reasoning_effort"] == "none"


# an OpenAI-style body whose wording the text patterns do not know (no
# "unsupported" anywhere, so str(exc) does not match either), but whose "param"
# still names the effort
def _reworded(param="reasoning_effort"):
    return {
        "error": {
            "message": "The requested reasoning level is not available for this model.",
            "type": "invalid_request_error",
            "param": param,
            "code": "invalid_value",
        }
    }


def test_a_reworded_refusal_is_recognised_by_its_param():
    error = ModelHTTPError(400, "gpt-5.4", _reworded("reasoning.effort"))
    settings = _translate_kwargs({"thinking": "minimal"}, model_name="gpt-5.4")

    assert _retry_effort(error, "gpt-5.4", settings)
    assert settings["thinking"] == "low"


def test_a_refused_off_without_a_list_falls_back_to_safe_off():
    error = ModelHTTPError(400, "gpt-5.4", _reworded())
    settings = _translate_kwargs({"thinking": "off"}, model_name="gpt-5.4")
    settings["openai_reasoning_effort"] = "none"

    assert _retry_effort(error, "gpt-5.4", settings)
    assert settings["openai_reasoning_effort"] == reasoning.SAFE_OFF


def test_an_unresolvable_refusal_is_not_retried_and_is_logged(caplog):
    error = ModelHTTPError(400, "gpt-5.4", _reworded())
    settings = _translate_kwargs({"thinking": "medium"}, model_name="gpt-5.4")

    with caplog.at_level(logging.WARNING, logger="struckdown.reasoning"):
        assert not _retry_effort(error, "gpt-5.4", settings)
    assert "reasoning level is not available" in caplog.text


def test_an_error_about_another_param_is_left_alone(caplog):
    error = ModelHTTPError(400, "gpt-5.4", _reworded("temperature"))
    settings = _translate_kwargs({"thinking": "minimal"}, model_name="gpt-5.4")

    with caplog.at_level(logging.WARNING, logger="struckdown.reasoning"):
        assert not _retry_effort(error, "gpt-5.4", settings)
    assert caplog.text == ""
