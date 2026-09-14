"""Token counts behind an OpenAI-compatible proxy.

Both payloads are real, captured from live calls to ``qwen/qwen3.8-27b``. The
OpenRouter one carries ``cost``, ``is_byok`` and ``cost_details`` beside the
counts; some genai-prices versions return nothing for it.
"""

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_ai.usage import RequestUsage

from struckdown.usage import CountingOpenAIChatModel, CountingStreamedResponse, counts_from

OPENROUTER_USAGE = {
    "prompt_tokens": 122,
    "completion_tokens": 18,
    "total_tokens": 140,
    "cost": 5.672e-05,
    "is_byok": False,
    "prompt_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 0},
    "cost_details": {"upstream_inference_cost": 5.672e-05},
    "completion_tokens_details": {"reasoning_tokens": 16},
}

PLAIN_USAGE = {"prompt_tokens": 122, "completion_tokens": 18, "total_tokens": 140}

OPENROUTER = "https://openrouter.ai/api/v1"


def _completion(usage):
    return ChatCompletion.model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen/qwen3.8-27b",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"},
                }
            ],
            "usage": usage,
        }
    )


def _model(base_url=OPENROUTER):
    """A model object only; nothing here makes a request."""
    return CountingOpenAIChatModel(
        "qwen/qwen3.8-27b",
        provider=OpenAIProvider(api_key="not-a-real-key", base_url=base_url),
    )


@pytest.mark.parametrize(
    "base_url",
    [OPENROUTER, "https://litellm.example.net", "https://api.openai.com/v1"],
)
def test_a_proxys_own_fields_do_not_cost_the_token_counts(base_url):
    """The counts reach the caller whichever host the proxy claims to be."""
    mapped = _model(base_url)._map_usage(_completion(OPENROUTER_USAGE))

    assert mapped.input_tokens == 122
    assert mapped.output_tokens == 18


def test_the_cached_prefix_is_counted_too():
    """cached_tokens is how a caller tells a cache hit from a full-price call."""
    mapped = _model()._map_usage(_completion(OPENROUTER_USAGE))

    assert mapped.cache_read_tokens == 64


def test_the_library_alone_reads_a_plain_payload():
    """pydantic-ai reads an ordinary payload correctly, so the fallback is not
    covering for it generally."""
    plain = OpenAIChatModel(
        "qwen/qwen3.8-27b",
        provider=OpenAIProvider(api_key="not-a-real-key", base_url=OPENROUTER),
    )

    mapped = plain._map_usage(_completion(PLAIN_USAGE))

    assert (mapped.input_tokens, mapped.output_tokens) == (122, 18)


def test_counts_are_recovered_when_the_library_finds_none():
    """An empty RequestUsage is what pydantic-ai 1.61 with genai-prices 0.1.4
    returned for this response. Asserted directly, so the test does not depend
    on which versions the host resolved."""
    empty = RequestUsage(details={"reasoning_tokens": 16})

    filled = counts_from(empty, _completion(OPENROUTER_USAGE).usage)

    assert (filled.input_tokens, filled.output_tokens) == (122, 18)
    assert filled.cache_read_tokens == 64
    assert filled.details == {"reasoning_tokens": 16}


def test_counts_the_library_did_find_are_never_overwritten():
    counted = RequestUsage(input_tokens=7, output_tokens=3)

    assert counts_from(counted, _completion(OPENROUTER_USAGE).usage) is counted


def test_the_details_the_library_did_extract_survive():
    """reasoning_tokens is most of what a reasoning model is charged for."""
    mapped = _model()._map_usage(_completion(OPENROUTER_USAGE))

    assert mapped.details.get("reasoning_tokens") == 16


def _chunk(usage):
    return ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "qwen/qwen3.8-27b",
            "choices": [],
            "usage": usage,
        }
    )


def test_a_streamed_run_counts_its_final_chunk():
    streamed = CountingStreamedResponse.__new__(CountingStreamedResponse)
    streamed._provider_name = "openai"
    streamed._provider_url = OPENROUTER
    streamed._model_name = "qwen/qwen3.8-27b"

    mapped = streamed._map_usage(_chunk(OPENROUTER_USAGE))

    assert (mapped.input_tokens, mapped.output_tokens) == (122, 18)


def test_a_chunk_carrying_no_usage_counts_nothing():
    """Usage arrives on the last chunk only."""
    streamed = CountingStreamedResponse.__new__(CountingStreamedResponse)
    streamed._provider_name = "openai"
    streamed._provider_url = OPENROUTER
    streamed._model_name = "qwen/qwen3.8-27b"

    mapped = streamed._map_usage(_chunk(None))

    assert (mapped.input_tokens, mapped.output_tokens) == (0, 0)


def test_the_proxy_path_builds_a_counting_model():
    """Everything above is inert unless get_pydantic_model returns this."""
    import struckdown as sd

    llm = sd.LLM(model_name="qwen/qwen3.8-27b")
    credentials = sd.LLMCredentials(
        api_key="not-a-real-key", base_url=OPENROUTER
    )

    assert isinstance(llm.get_pydantic_model(credentials), CountingOpenAIChatModel)
