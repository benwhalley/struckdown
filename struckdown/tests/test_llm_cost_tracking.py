"""Tests for LLM completion cost tracking."""

import pytest

from struckdown.llm import _build_completion_dict, _calc_cost_from_usage


class FakeUsage:
    input_tokens = 324
    output_tokens = 45
    cache_read_tokens = 0
    cache_write_tokens = 0


class FakeModelResponse:
    usage = FakeUsage()
    model_name = "gpt-5-mini"
    provider_name = "azure"
    provider_url = None


def test_calc_cost_uses_pydantic_ai_response_cost_when_available():
    class ResponseWithCost(FakeModelResponse):
        def cost(self):
            return type("PriceCalc", (), {"total_price": 0.1234})()

    cost = _calc_cost_from_usage(ResponseWithCost(), "azure:gpt-5-mini")

    assert cost == pytest.approx(0.1234)


def test_calc_cost_normalises_provider_prefixed_model_names():
    response = FakeModelResponse()

    cost = _calc_cost_from_usage(response, "azure:gpt-5-mini")

    assert cost is not None
    assert cost > 0


def test_build_completion_dict_fallback_uses_model_name_for_cost_lookup():
    response = FakeModelResponse()

    completion = _build_completion_dict(
        response,
        [{"role": "user", "content": "joke please"}],
        model_name="azure:gpt-5-mini",
    )

    assert completion["_hidden_params"]["response_cost"] is not None
    assert completion["_hidden_params"]["response_cost"] > 0
