"""Model settings sent to the provider."""


def test_reasoning_model_does_not_receive_temperature():
    """Reasoning models ignore sampling params; pydantic-ai warns if we send them."""
    from pydantic_ai.settings import ModelSettings

    from struckdown.llm import LLM, LLMCredentials, _without_sampling_params

    cred = LLMCredentials(api_key="x", base_url="https://proxy.example/v1")
    settings = ModelSettings(temperature=0.7, max_tokens=100)

    reasoning = LLM(model_name="gpt-5.6-sol").get_pydantic_model(cred)
    assert "temperature" not in _without_sampling_params(reasoning, settings)
    assert _without_sampling_params(reasoning, settings)["max_tokens"] == 100

    plain = LLM(model_name="gpt-4.1-mini").get_pydantic_model(cred)
    assert _without_sampling_params(plain, settings)["temperature"] == 0.7
