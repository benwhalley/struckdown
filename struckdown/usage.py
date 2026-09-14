"""Token counts that survive an OpenAI-compatible proxy.

pydantic-ai extracts usage through genai-prices, which returns an empty
``RequestUsage`` when it cannot parse the payload: no exception, no warning.
Whether it parses depends on the versions the application resolved. One live
OpenRouter response counted 122 in / 19 out under pydantic-ai 1.77 with
genai-prices 0.0.56, and nothing at all under 1.61 with 0.1.4.

The counts are plain integers on the response, so read them when the library
finds none. This only fills in zeros; counts it did find are returned
unchanged.
"""

from dataclasses import replace

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIStreamedResponse


def counts_from(mapped, raw):
    """``mapped``, with the counts it is missing taken from ``raw``.

    ``raw`` is the provider's usage object, absent on every streamed chunk but
    the last.
    """
    if raw is None or mapped.input_tokens or mapped.output_tokens:
        return mapped
    details = getattr(raw, "prompt_tokens_details", None)
    return replace(
        mapped,
        input_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        output_tokens=getattr(raw, "completion_tokens", 0) or 0,
        cache_read_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
    )


class CountingStreamedResponse(OpenAIStreamedResponse):
    """The streaming half: usage arrives on the final chunk."""

    def _map_usage(self, response):
        return counts_from(super()._map_usage(response), response.usage)


class CountingOpenAIChatModel(OpenAIChatModel):
    """An ``OpenAIChatModel`` that counts tokens itself when it has to.

    ``_map_usage`` and ``_streamed_response_cls`` are pydantic-ai extension
    points, so this subclasses rather than patches.
    """

    def _map_usage(self, response):
        return counts_from(super()._map_usage(response), response.usage)

    @property
    def _streamed_response_cls(self):
        return CountingStreamedResponse
