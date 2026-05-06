"""Pluggable price sources and cost estimation helpers.

Price sources look up per-million-token pricing for models from external
databases (genai-prices library, OpenRouter API, etc.).

Estimation helpers convert text lengths into approximate token counts and
costs without requiring a tokenizer.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Protocol, Union

logger = logging.getLogger(__name__)


@dataclass
class PriceLookup:
    """Result of a price source lookup."""

    input_cost_per_mtok: Decimal
    output_cost_per_mtok: Decimal
    source: str


class PriceSource(Protocol):
    """Interface for model pricing backends."""

    name: str

    def lookup(
        self, model_name: str, provider: Optional[str] = None
    ) -> Optional[PriceLookup]: ...


def _extract_genai_prices(prices) -> Optional[tuple[Decimal, Decimal]]:
    """Extract per-mtok costs from a genai-prices model object.

    Handles the .base attribute on tiered pricing structures.
    """
    if not prices:
        return None
    input_mtok = prices.input_mtok
    output_mtok = prices.output_mtok
    if hasattr(input_mtok, "base"):
        input_mtok = input_mtok.base
    if hasattr(output_mtok, "base"):
        output_mtok = output_mtok.base
    if input_mtok is None:
        return None
    return (
        Decimal(str(input_mtok)),
        Decimal(str(output_mtok)) if output_mtok is not None else Decimal("0"),
    )


class GenAIPriceSource:
    """Look up pricing from the genai-prices library (local data snapshot)."""

    name = "genai-prices"

    def lookup(
        self, model_name: str, provider: Optional[str] = None
    ) -> Optional[PriceLookup]:
        try:
            from genai_prices import Usage, calc_price
            from genai_prices.data_snapshot import get_snapshot

            # strip provider prefix for bare name lookup
            bare = model_name.split(":", 1)[1] if ":" in model_name else model_name

            # try calc_price first
            try:
                result = calc_price(
                    Usage(input_tokens=1_000_000, output_tokens=1_000_000), bare
                )
                if result and result.model and result.model.prices:
                    costs = _extract_genai_prices(result.model.prices)
                    if costs:
                        return PriceLookup(
                            input_cost_per_mtok=costs[0],
                            output_cost_per_mtok=costs[1],
                            source=self.name,
                        )
            except Exception:
                pass

            # fallback: scan providers, prefer matching provider
            snap = get_snapshot()
            provider_id = provider if provider and provider != "openai-compatible" else None

            if provider_id:
                for p in snap.providers:
                    if p.id and p.id.lower() == provider_id.lower():
                        model = p.find_model(bare)
                        if model and model.prices:
                            costs = _extract_genai_prices(model.prices)
                            if costs:
                                return PriceLookup(
                                    input_cost_per_mtok=costs[0],
                                    output_cost_per_mtok=costs[1],
                                    source=self.name,
                                )
                            break

            # try all providers
            for p in snap.providers:
                model = p.find_model(bare)
                if model and model.prices:
                    costs = _extract_genai_prices(model.prices)
                    if costs:
                        return PriceLookup(
                            input_cost_per_mtok=costs[0],
                            output_cost_per_mtok=costs[1],
                            source=self.name,
                        )

            return None
        except ImportError:
            logger.warning("genai-prices not installed, cannot look up prices")
            return None
        except Exception as e:
            logger.warning(f"Error looking up prices from genai-prices: {e}")
            return None


def _fetch_openrouter_models(_cache_day: int) -> dict:
    """Fetch the full model list from OpenRouter. Returns {id: pricing_dict}.

    The ``_cache_day`` arg rolls over daily, giving joblib a 1-day TTL.
    """
    import json
    import urllib.request

    resp = urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=15)
    data = json.loads(resp.read())
    return {m["id"]: m.get("pricing", {}) for m in data.get("data", [])}


from struckdown.cache import memory as _memory

_fetch_openrouter_models_cached = _memory.cache(_fetch_openrouter_models)


class OpenRouterPriceSource:
    """Look up pricing from the OpenRouter public API.

    The full model list is cached to disk by joblib for 1 day, and held
    in memory for the process lifetime after first load.
    """

    name = "openrouter"

    def __init__(self):
        self._models: Optional[dict] = None

    def _ensure_loaded(self) -> dict:
        if self._models is not None:
            return self._models
        import time

        day = int(time.time() // 86400)
        try:
            self._models = _fetch_openrouter_models_cached(_cache_day=day)
        except Exception as e:
            logger.warning(f"Error fetching OpenRouter model list: {e}")
            self._models = {}
        return self._models

    def lookup(
        self, model_name: str, provider: Optional[str] = None
    ) -> Optional[PriceLookup]:
        models = self._ensure_loaded()
        bare = model_name.split(":", 1)[1] if ":" in model_name else model_name

        pricing = models.get(model_name) or models.get(bare)
        if pricing is None:
            return None

        prompt_per_token = pricing.get("prompt")
        if prompt_per_token is None:
            return None

        completion_per_token = pricing.get("completion")
        return PriceLookup(
            input_cost_per_mtok=Decimal(str(prompt_per_token)) * 1_000_000,
            output_cost_per_mtok=(
                Decimal(str(completion_per_token)) * 1_000_000
                if completion_per_token
                else Decimal("0")
            ),
            source=self.name,
        )


# -- registry mapping source names to instances --

PRICE_SOURCES: Dict[str, PriceSource] = {
    "genai_prices": GenAIPriceSource(),
    "openrouter": OpenRouterPriceSource(),
}


def get_price_source(name: str) -> PriceSource:
    """Get a registered price source by name. Defaults to genai-prices."""
    return PRICE_SOURCES.get(name, PRICE_SOURCES["genai_prices"])


# -- estimation helpers --

DEFAULT_TOKENS_PER_WORD = 1.3


def estimate_tokens(
    text: Union[str, list[str]],
    tokens_per_word: float = DEFAULT_TOKENS_PER_WORD,
) -> int:
    """Estimate token count from text using a word-count heuristic.

    Default ratio of 1.3 tokens per word is a reasonable approximation for
    English text across GPT-4, Claude, and similar models.
    """
    if isinstance(text, str):
        word_count = len(text.split())
    else:
        word_count = sum(len(t.split()) for t in text)
    return int(word_count * tokens_per_word)


@dataclass
class CostEstimate:
    """Estimated cost for processing text with a model."""

    prompt_tokens: int
    completion_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    tokens_per_word: float
    completion_ratio: float


def estimate_cost(
    text: Union[str, list[str]],
    input_cost_per_mtok: Union[Decimal, float],
    output_cost_per_mtok: Union[Decimal, float],
    tokens_per_word: float = DEFAULT_TOKENS_PER_WORD,
    completion_ratio: float = 0.2,
) -> CostEstimate:
    """Estimate cost for processing text, given per-mtok pricing.

    Args:
        text: Input text or list of texts.
        input_cost_per_mtok: Input cost per million tokens (USD).
        output_cost_per_mtok: Output cost per million tokens (USD).
        tokens_per_word: Ratio of tokens to words (default 1.3).
        completion_ratio: Estimated completion tokens as fraction of prompt tokens.
    """
    prompt_tokens = estimate_tokens(text, tokens_per_word)
    completion_tokens = int(prompt_tokens * completion_ratio)

    in_cost = Decimal(str(input_cost_per_mtok))
    out_cost = Decimal(str(output_cost_per_mtok))

    input_cost = Decimal(prompt_tokens) * in_cost / Decimal("1000000")
    output_cost = Decimal(completion_tokens) * out_cost / Decimal("1000000")

    return CostEstimate(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        tokens_per_word=tokens_per_word,
        completion_ratio=completion_ratio,
    )


def estimate_cost_for_model(
    text: Union[str, list[str]],
    model_name: str,
    price_source: Union[str, PriceSource, None] = None,
    provider: Optional[str] = None,
    tokens_per_word: float = DEFAULT_TOKENS_PER_WORD,
    completion_ratio: float = 0.2,
) -> Optional[CostEstimate]:
    """Estimate cost for text given a model name, looking up pricing automatically.

    Convenience wrapper that combines price lookup with estimation.
    Returns None if pricing cannot be found.
    """
    if price_source is None:
        source = PRICE_SOURCES["genai_prices"]
    elif isinstance(price_source, str):
        source = get_price_source(price_source)
    else:
        source = price_source

    pricing = source.lookup(model_name, provider=provider)
    if pricing is None:
        return None

    return estimate_cost(
        text,
        input_cost_per_mtok=pricing.input_cost_per_mtok,
        output_cost_per_mtok=pricing.output_cost_per_mtok,
        tokens_per_word=tokens_per_word,
        completion_ratio=completion_ratio,
    )
