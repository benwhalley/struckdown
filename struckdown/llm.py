"""LLM client, credentials, and API interaction for struckdown.

Uses pydantic-ai for structured LLM calls and embeddings.
"""

import json
import logging
import os
import time
import traceback
import warnings
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models import Model as PydanticAIModel, infer_model

try:
    from pydantic_ai.models import parse_model_id
except ImportError:
    def parse_model_id(model_id: str) -> tuple[str, str]:
        """Shim for pydantic-ai>=1.0 which removed parse_model_id."""
        if ":" in model_id:
            provider, name = model_id.split(":", 1)
            return provider, name
        return "openai", model_id
from pydantic_ai.output import PromptedOutput
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from struckdown.usage import CountingOpenAIChatModel

def _is_known_provider(name: str) -> bool:
    """Is this a pydantic-ai provider, or just the first half of a model name?

    ``openai:gpt-4o`` splits; ``qwen3:8b`` is one Ollama tag and must not.
    """
    try:
        from pydantic_ai.providers import infer_provider_class
    except ImportError:  # older pydantic-ai: assume it is a provider
        return True
    try:
        infer_provider_class(name)
    except Exception:
        return False
    return True


# Module-level flag for API request logging
_debug_api_requests = False


def enable_api_debug():
    """Enable logging of full API requests (messages + tools schema)."""
    global _debug_api_requests
    _debug_api_requests = True


def _strip_null_bytes(obj):
    """Recursively strip null bytes from LLM responses.

    PostgreSQL cannot store \\x00 in text/JSON columns, and LLMs occasionally
    produce these in their output.
    """
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    elif isinstance(obj, dict):
        return {_strip_null_bytes(k): _strip_null_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_strip_null_bytes(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(_strip_null_bytes(i) for i in obj)
    else:
        return obj


def disable_api_debug():
    """Disable API request logging."""
    global _debug_api_requests
    _debug_api_requests = False


def _log_error_details(error: Exception, model_name: str, context: str = ""):
    """Log full error details when debug is enabled."""
    if not _debug_api_requests:
        return
    import sys

    print("\n" + "=" * 80, file=sys.stderr)
    print(f"LLM ERROR DEBUG {context}", file=sys.stderr)
    print(f"Model: {model_name}", file=sys.stderr)
    print(f"Error type: {type(error).__name__}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    print(f"Full error:\n{error}", file=sys.stderr)
    if hasattr(error, "__cause__") and error.__cause__:
        print("-" * 80, file=sys.stderr)
        print(f"Caused by: {type(error.__cause__).__name__}", file=sys.stderr)
        print(f"{error.__cause__}", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)


import anyio
import tenacity
from box import Box
from decouple import config as env_config

# HTTP status code categories for error classification
CACHEABLE_STATUS_CODES = frozenset()  # determined by response body, not status alone
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
FATAL_STATUS_CODES = frozenset({401, 403, 404})

import numpy as np
from more_itertools import chunked
from pydantic import BaseModel, Field

from . import reasoning
from .cache import hash_return_type, memory


class EmbeddingResult(np.ndarray):
    """
    Numpy array subclass that carries embedding cost metadata.

    Behaves exactly like np.ndarray for all operations,
    but also exposes .cost, .tokens, .model, .cached attributes.

    Example:
        embedding = get_embedding(["hello"])[0]
        similarity = np.dot(embedding, other)  # works as normal
        print(embedding.cost)   # 0.00002 or None if unknown
        print(embedding.tokens) # 5
        print(embedding.cached) # False
    """

    def __new__(
        cls,
        array,
        cost: Optional[float] = None,
        tokens: int = 0,
        model: str = "",
        cached: bool = False,
    ):
        obj = np.asarray(array).view(cls)
        obj.cost = cost
        obj.tokens = tokens
        obj.model = model
        obj.cached = cached
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.cost = getattr(obj, "cost", None)
        self.tokens = getattr(obj, "tokens", 0)
        self.model = getattr(obj, "model", "")
        self.cached = getattr(obj, "cached", False)

    def __reduce__(self):
        # support pickling by including metadata
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self.cost, self.tokens, self.model, self.cached)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # restore metadata when unpickling
        self.cost = state[-4]
        self.tokens = state[-3]
        self.model = state[-2]
        self.cached = state[-1]
        super().__setstate__(state[:-4])


class EmbeddingResultList(list):
    """
    List of EmbeddingResult objects with aggregate cost information.

    Behaves exactly like a list for iteration and indexing,
    but also exposes .total_cost, .total_tokens, .model, .cached_count.

    Example:
        results = get_embedding(["hello", "world"])
        for emb in results:        # iterate as normal
            process(emb)
        print(results.total_cost)  # 0.00004 or None if unknown
        print(results[0].cost)     # individual cost
    """

    def __init__(self, embeddings: list, model: str = ""):
        super().__init__(embeddings)
        self._model = model

    @property
    def model(self) -> str:
        """Model used for embeddings."""
        return self._model

    @property
    def has_unknown_costs(self) -> bool:
        """True if any embedding has unknown (None) cost."""
        return any(
            getattr(e, "cost", None) is None
            for e in self
            if not getattr(e, "cached", False)
        )

    @property
    def total_cost(self) -> Optional[float]:
        """Total USD cost across all embeddings. None if any costs unknown."""
        if self.has_unknown_costs:
            return None
        return sum(getattr(e, "cost", 0.0) or 0.0 for e in self)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all embeddings."""
        return sum(getattr(e, "tokens", 0) for e in self)

    @property
    def cached_count(self) -> int:
        """Number of embeddings retrieved from cache."""
        return sum(1 for e in self if getattr(e, "cached", False))

    @property
    def fresh_count(self) -> int:
        """Number of embeddings from fresh API calls."""
        return len(self) - self.cached_count

    @property
    def fresh_cost(self) -> Optional[float]:
        """Cost from fresh API calls only (excludes cached). None if any unknown."""
        if self.has_unknown_costs:
            return None
        return sum(
            getattr(e, "cost", 0.0) or 0.0
            for e in self
            if not getattr(e, "cached", False)
        )

    @property
    def cached_tokens(self) -> int:
        """Total tokens from cached embeddings (what they cost when originally computed)."""
        return sum(
            getattr(e, "tokens", 0)
            for e in self
            if getattr(e, "cached", False)
        )

    @property
    def cached_cost_estimate(self) -> float:
        """Estimated cost for cached embeddings (what they would have cost if not cached)."""
        return sum(
            getattr(e, "cost", 0.0) or 0.0
            for e in self
            if getattr(e, "cached", False)
        )

    @property
    def total_cost_estimate(self) -> Optional[float]:
        """Total estimated cost (fresh + cached estimates).

        Returns None if fresh costs are unknown, otherwise returns
        fresh_cost + cached_cost_estimate.
        """
        fresh = self.fresh_cost
        if fresh is None:
            return None
        return fresh + self.cached_cost_estimate

    def __repr__(self) -> str:
        fresh = self.fresh_cost
        cached = self.cached_cost_estimate
        return (
            f"EmbeddingResultList({len(self)} embeddings, "
            f"fresh=${fresh or 0:.4f}, "
            f"cached_estimate=${cached:.4f}, "
            f"cache_hits={self.cached_count})"
        )


from .embedding_cache import (CachedEmbedding, clear_embedding_cache,
                              get_cached_embeddings, get_cached_pair_scores,
                              store_embeddings, store_pair_scores)
from .errors import AuthError
from .errors import BadRequestError as SDBadRequestError
from .errors import ConnectionError as SDConnectionError
from .errors import ContentFilterError, ContextWindowError, LLMError
from .errors import RateLimitError as SDRateLimitError
from .ledger import (CostBreakdown, ResponseStandIn, StoredPricing, UsagePayload,
                     cost_from_price_calc, cost_from_stored, deferred_usage, emit,
                     emit_async, flush_usage, flush_usage_async,
                     record_from_response, wants_payload)
from .ledger import now as _utcnow
from .messages import split_for_agent, to_openai_messages


def _classify_http_error_body(error_msg: str) -> Optional[str]:
    """Classify an HTTP error by inspecting the error message body.

    Returns a category string or None if unclassified.
    """
    msg = error_msg.lower()
    # content policy / safety filter
    if any(kw in msg for kw in (
        "content_filter", "content_policy", "contentfilter",
        "content policy", "responsible ai", "safety",
    )):
        return "content_filter"
    # context window / token limit
    if any(kw in msg for kw in (
        "context_length", "context_window", "maximum context",
        "token limit", "too many tokens", "max_tokens",
        "reduce the length", "input too long",
    )):
        return "context_window"
    return None


def _list_provider_models(model_name: str) -> Optional[list[str]]:
    """Try to fetch available model names from the provider's API.

    Returns a list of model ID strings, or None if listing is not possible.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        provider_prefix, _ = parse_model_id(model_name)

    credentials = LLMCredentials.from_env()
    api_key = credentials.api_key_for_provider(provider_prefix)
    if not api_key:
        return None

    endpoints: dict[str, tuple[str, str]] = {
        "openai": ("https://api.openai.com/v1/models", "Bearer"),
        "openai-chat": ("https://api.openai.com/v1/models", "Bearer"),
        "mistral": ("https://api.mistral.ai/v1/models", "Bearer"),
    }
    # google uses query param auth
    google_prefixes = ("google", "google-gla", "google-vertex")

    try:
        import httpx
        if provider_prefix in google_prefixes:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            resp = httpx.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"].removeprefix("models/") for m in data.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", [])]
            return sorted(models)
        elif provider_prefix in endpoints:
            url, auth_scheme = endpoints[provider_prefix]
            resp = httpx.get(url, headers={"Authorization": f"{auth_scheme} {api_key}"}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return sorted(m["id"] for m in data.get("data", []))
        elif credentials.base_url:
            url = credentials.base_url.rstrip("/") + "/models"
            resp = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return sorted(m["id"] for m in data.get("data", []))
    except Exception as e:
        logger.debug(f"Failed to list models for {provider_prefix}: {e}")
    return None


def _make_struckdown_error(
    error: Exception, prompt: str, model_name: str
) -> LLMError:
    """Map a pydantic-ai or other exception to the appropriate struckdown error subclass."""
    if isinstance(error, ModelHTTPError):
        status = error.status_code

        # check body for specific error types (content filter, context window)
        body_class = _classify_http_error_body(str(error))
        if body_class == "content_filter":
            return ContentFilterError(error, prompt, model_name)
        if body_class == "context_window":
            return ContextWindowError(error, prompt, model_name)

        # classify by HTTP status code
        if status in (429, 408):
            return SDRateLimitError(error, prompt, model_name)
        if status in (401, 403):
            return AuthError(error, prompt, model_name)
        if status in (500, 502, 503, 504):
            return SDConnectionError(error, prompt, model_name)
        if status in (400, 422):
            return SDBadRequestError(error, prompt, model_name)
        if status == 404:
            # model not found -- try to list available models
            models = _list_provider_models(model_name)
            if models:
                hint = "\n  Available models:\n    " + "\n    ".join(models[:30])
                if len(models) > 30:
                    hint += f"\n    ... and {len(models) - 30} more"
                wrapped = Exception(f"{error}{hint}")
                return SDBadRequestError(wrapped, prompt, model_name)
            return SDBadRequestError(error, prompt, model_name)

        return LLMError(error, prompt, model_name)

    if isinstance(error, UnexpectedModelBehavior):
        return SDBadRequestError(error, prompt, model_name)

    # fallback for any other exception
    return LLMError(error, prompt, model_name)


def _make_cached_error(
    error_class: str, error_msg: str, prompt: str, model_name: str
) -> LLMError:
    """Create the appropriate error subclass for a cached error."""
    wrapped_error = Exception(error_msg)

    # Map cached error class name to struckdown error type
    if error_class in ("ContentPolicyViolationError", "ContentFilterError"):
        return ContentFilterError(wrapped_error, prompt, model_name)
    if error_class in ("ContextWindowExceededError", "ContextWindowError"):
        return ContextWindowError(wrapped_error, prompt, model_name)
    if error_class in ("RateLimitError", "Timeout"):
        return SDRateLimitError(wrapped_error, prompt, model_name)
    if error_class in ("AuthenticationError", "PermissionDeniedError", "AuthError"):
        return AuthError(wrapped_error, prompt, model_name)
    if error_class in ("APIConnectionError", "ConnectionError"):
        return SDConnectionError(wrapped_error, prompt, model_name)

    # Default to base LLMError
    return LLMError(wrapped_error, prompt, model_name)


logger = logging.getLogger(__name__)

# Marker to detect cache misses - set to True inside _call_llm_cached (only runs on miss)
_cache_miss_marker: ContextVar[bool] = ContextVar("cache_miss", default=False)

# Per-call pricing override from ModelSpec. When set, _calc_cost_from_usage()
# uses these values instead of pydantic-ai or genai-prices.
_model_pricing: ContextVar[Optional[StoredPricing]] = ContextVar(
    "model_pricing", default=None
)


def set_model_pricing(
    input_cost_per_mtok: Optional[float],
    output_cost_per_mtok: Optional[float],
    cache_read_cost_per_mtok: Optional[float] = None,
    cache_write_cost_per_mtok: Optional[float] = None,
) -> None:
    """Set pricing for the current context (used by cost calculation).

    When both input and output values are provided, _calc_cost_from_usage() uses
    them instead of pydantic-ai or genai-prices. The cache rates are optional;
    a missing one is charged at the input rate. Pass None to clear.
    """
    if input_cost_per_mtok is not None and output_cost_per_mtok is not None:
        _model_pricing.set(
            StoredPricing(
                input_per_mtok=float(input_cost_per_mtok),
                output_per_mtok=float(output_cost_per_mtok),
                cache_read_per_mtok=(
                    float(cache_read_cost_per_mtok)
                    if cache_read_cost_per_mtok is not None
                    else None
                ),
                cache_write_per_mtok=(
                    float(cache_write_cost_per_mtok)
                    if cache_write_cost_per_mtok is not None
                    else None
                ),
            )
        )
    else:
        _model_pricing.set(None)


def get_model_pricing() -> Optional[StoredPricing]:
    return _model_pricing.get()

# Shared concurrency control for all LLM calls
# This limits concurrent API calls across templates AND within together blocks
MAX_LLM_CONCURRENCY = env_config("SD_MAX_CONCURRENCY", default=20, cast=int)
_llm_semaphore = None

# Separate concurrency limit for embedding requests (typically lower to avoid overwhelming servers)
MAX_EMBEDDING_CONCURRENCY = env_config("SD_EMBEDDING_CONCURRENCY", default=3, cast=int)
_embedding_semaphore = None

# Embedding API timeout - increased from 60s to handle large batches under load
EMBEDDING_TIMEOUT = env_config("SD_EMBEDDING_TIMEOUT", default=120, cast=int)


def get_llm_semaphore() -> anyio.Semaphore:
    """Get the shared LLM concurrency semaphore (lazy initialization)."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = anyio.Semaphore(MAX_LLM_CONCURRENCY)
    return _llm_semaphore


def get_embedding_semaphore() -> anyio.Semaphore:
    """Get the embedding concurrency semaphore (lazy initialization).

    Embeddings use a separate, lower concurrency limit because embedding
    endpoints (especially self-hosted ones like LiteLLM) can be overwhelmed
    by many concurrent batch requests.
    """
    global _embedding_semaphore
    if _embedding_semaphore is None:
        _embedding_semaphore = anyio.Semaphore(MAX_EMBEDDING_CONCURRENCY)
    return _embedding_semaphore


def set_llm_concurrency(n: int) -> None:
    """Set the maximum concurrency for LLM calls. Replaces the semaphore."""
    global _llm_semaphore
    _llm_semaphore = anyio.Semaphore(n)


# ANSI colour codes for terminal output
class LC:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    ORANGE = "\033[38;5;208m"
    RESET = "\033[0m"


from .model_spec import PROVIDERS

_PROVIDER_KEY_ENV_VARS: dict[str, str] = {
    k: v.key_env_var for k, v in PROVIDERS.items() if v.key_env_var
}

_PROVIDER_ENDPOINT_ENV_VARS: dict[str, str] = {
    k: v.endpoint_env_var for k, v in PROVIDERS.items() if v.endpoint_env_var
}


class LLMCredentials(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __repr_args__(self):
        # mask api_key so secrets don't leak into logs/tracebacks, but still
        # show whether it's set -- a fully empty repr was a debugging footgun
        yield "api_key", _obfuscate_key(self.api_key) if self.api_key else None
        yield "base_url", self.base_url

    @classmethod
    def from_env(cls) -> "LLMCredentials":
        """Build credentials from LLM_API_KEY / LLM_API_BASE env vars.

        Library and Django callers should pass credentials explicitly;
        env vars are intended for CLI use only.
        """
        return cls(
            api_key=env_config("LLM_API_KEY", None),
            base_url=env_config("LLM_API_BASE", None),
        )

    def api_key_for_provider(self, provider_name: str) -> Optional[str]:
        """Return the best API key for a given provider.

        Checks provider-specific env vars (e.g. MISTRAL_API_KEY) first,
        then falls back to the generic LLM_API_KEY.
        """
        env_var = _PROVIDER_KEY_ENV_VARS.get(provider_name)
        if env_var:
            provider_key = os.environ.get(env_var)
            if provider_key:
                return provider_key
        return self.api_key


KNOWN_MODEL_SETTINGS = frozenset({
    "temperature", "max_tokens", "seed", "top_p", "timeout",
    "thinking", "presence_penalty", "frequency_penalty",
    # provider-specific knobs, passed through untouched for callers that
    # want to bypass the unified `thinking` setting
    "openai_reasoning_effort", "anthropic_thinking", "extra_body",
})

# internal keys consumed elsewhere in the pipeline, not model settings
_INTERNAL_KEYS = frozenset({"stream_debounce_ms", "model"})


def _translate_kwargs(
    extra_kwargs: Optional[dict], strict: bool = False, model_name: str = ""
) -> ModelSettings:
    """Map struckdown's extra_kwargs to pydantic-ai ModelSettings.

    Unknown params are logged as warnings by default. With strict=True,
    a ValueError is raised instead -- useful for catching typos or
    params unsupported by the current provider.

    ``thinking`` is translated into pydantic-ai's vocabulary here (see
    :mod:`struckdown.reasoning`): ``"off"`` is ours, not theirs, and
    "as little as possible" is a different word on each model generation.
    """
    if not extra_kwargs:
        return ModelSettings()
    settings: dict = {}
    dropped: list = []
    for key, value in extra_kwargs.items():
        if key in KNOWN_MODEL_SETTINGS:
            settings[key] = value
        elif key in _INTERNAL_KEYS:
            pass
        else:
            dropped.append(key)
    if dropped:
        msg = f"Dropped unsupported LLM parameters: {', '.join(dropped)}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    if "thinking" in settings:
        settings["thinking"] = reasoning.resolve_thinking(settings["thinking"])
        _apply_off_effort(settings, model_name)
    return ModelSettings(**settings)


def _apply_off_effort(settings: dict, model_name: str) -> None:
    """Name the "don't think" value explicitly for OpenAI-style models.

    pydantic-ai maps ``thinking=False`` to ``reasoning_effort='none'`` from
    a static table; whether a given model takes ``none`` or ``minimal``
    depends on its generation, and behind a proxy the name it is asked
    under may not even be the model's own. The provider-specific key wins
    over the unified one, so once a model has told us what it accepts, say
    it outright.
    """
    if settings.get("thinking") is not False:
        return
    if "openai_reasoning_effort" in settings:
        return  # caller was explicit; leave it alone
    if reasoning.learned(model_name):
        settings["openai_reasoning_effort"] = reasoning.off_effort(model_name)


def _retry_effort(exc: Exception, model_name: str, settings: ModelSettings) -> bool:
    """Rewrite ``settings`` in place for one retry, or return False.

    True means the model rejected the reasoning effort and named what it
    would take instead -- so the same call is worth making again.
    """
    if settings.get("thinking") is not False:
        return False
    tried = settings.get("openai_reasoning_effort") or reasoning.rejected_value(str(exc))
    replacement = reasoning.learn_from_error(model_name, str(exc), tried)
    if replacement is None:
        return False
    settings["openai_reasoning_effort"] = replacement
    return True


def _merge_llm_config_defaults(extra_kwargs: Optional[dict], return_type) -> dict:
    """Merge return_type.llm_config defaults into extra_kwargs.

    Explicit kwargs take priority over llm_config defaults.
    """
    call_kwargs = dict(extra_kwargs) if extra_kwargs else {}
    if hasattr(return_type, "llm_config") and return_type.llm_config:
        cfg = return_type.llm_config
        if cfg.temperature is not None and "temperature" not in call_kwargs:
            call_kwargs["temperature"] = cfg.temperature
        if cfg.seed is not None and "seed" not in call_kwargs:
            call_kwargs["seed"] = cfg.seed
        if cfg.thinking is not None and "thinking" not in call_kwargs:
            call_kwargs["thinking"] = cfg.thinking
    return call_kwargs


def _obfuscate_key(key: Optional[str]) -> str:
    """Show first 4 and last 4 chars of an API key, masking the rest."""
    if not key:
        return "(none)"
    if len(key) <= 10:
        return key[:2] + "***" + key[-2:]
    return key[:4] + "***" + key[-4:]


def _print_routing_debug(llm: "LLM", credentials: "LLMCredentials"):
    """Print debug info about how the LLM call will be routed."""
    import sys
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        provider_prefix, bare_name = parse_model_id(llm.model_name)

    provider_env_var = _PROVIDER_KEY_ENV_VARS.get(provider_prefix)
    has_provider_key = bool(provider_env_var and os.environ.get(provider_env_var))
    resolved_key = credentials.api_key_for_provider(provider_prefix)
    endpoint_var = _PROVIDER_ENDPOINT_ENV_VARS.get(provider_prefix)
    endpoint_val = os.environ.get(endpoint_var) if endpoint_var else None

    if credentials.base_url and not has_provider_key:
        print(f"[DEBUG] Mode: proxy", file=sys.stderr)
        print(f"[DEBUG] Model: {bare_name} (from {llm.model_name})", file=sys.stderr)
        print(f"[DEBUG] Endpoint: {credentials.base_url}", file=sys.stderr)
        print(f"[DEBUG] API key: {_obfuscate_key(credentials.api_key)} (LLM_API_KEY)", file=sys.stderr)
    elif resolved_key:
        key_source = provider_env_var if has_provider_key else "LLM_API_KEY"
        print(f"[DEBUG] Mode: native provider", file=sys.stderr)
        print(f"[DEBUG] Provider: {provider_prefix}", file=sys.stderr)
        print(f"[DEBUG] Model: {llm.model_name}", file=sys.stderr)
        print(f"[DEBUG] API key: {_obfuscate_key(resolved_key)} ({key_source})", file=sys.stderr)
        if endpoint_val:
            print(f"[DEBUG] Endpoint: {endpoint_val} ({endpoint_var})", file=sys.stderr)
    else:
        print(f"[DEBUG] Mode: native provider (env var auto-detection)", file=sys.stderr)
        print(f"[DEBUG] Model: {llm.model_name}", file=sys.stderr)


def _create_provider_with_credentials(provider_name: str, credentials: "LLMCredentials"):
    """Create a pydantic-ai provider, injecting explicit api_key from credentials.

    Uses provider-specific env vars (e.g. MISTRAL_API_KEY) when available,
    falling back to the generic LLM_API_KEY.
    """
    api_key = credentials.api_key_for_provider(provider_name)
    if provider_name in ("openai", "openai-chat"):
        return OpenAIProvider(api_key=api_key)
    elif provider_name == "anthropic":
        from pydantic_ai.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key=api_key)
    elif provider_name in ("google", "google-gla", "google-vertex"):
        from pydantic_ai.providers.google import GoogleProvider
        return GoogleProvider(api_key=api_key)
    elif provider_name == "mistral":
        from pydantic_ai.providers.mistral import MistralProvider
        return MistralProvider(api_key=api_key)
    elif provider_name == "ollama":
        return OpenAIProvider(
            base_url=credentials.base_url or "http://localhost:11434/v1",
            api_key=api_key or "ollama",
        )
    elif provider_name == "azure":
        from pydantic_ai.providers.azure import AzureProvider
        endpoint = credentials.base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise Exception(
                "Azure requires AZURE_OPENAI_ENDPOINT (or LLM_API_BASE) to be set"
            )
        api_version = os.environ.get("OPENAI_API_VERSION", "2024-10-21")
        return AzureProvider(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    else:
        # unknown provider -- try OpenAI-compatible
        kwargs = {"api_key": api_key}
        if credentials.base_url:
            kwargs["base_url"] = credentials.base_url
        return OpenAIProvider(**kwargs)


class LLM(BaseModel):
    model_name: Optional[str] = Field(
        default_factory=lambda: env_config("DEFAULT_LLM", "gpt-4.1-mini"),
        exclude=True,
    )

    def get_pydantic_model(self, credentials: Optional[LLMCredentials] = None) -> PydanticAIModel:
        """Create a pydantic-ai model instance for this LLM + credentials.

        Supports two modes:

        1. **Proxy mode** (when credentials.base_url is set and is not a direct
           Azure endpoint): all requests are sent through an OpenAI-compatible
           proxy (LiteLLM, Helicone, etc.). The provider prefix is stripped from
           the model name (e.g. ``openai:gpt-4o`` -> ``gpt-4o``). To use a
           prefixed model name with the proxy (e.g. LiteLLM's ``azure/gpt-5-mini``),
           write it without the colon -- bare names pass through unchanged.

        2. **Native provider mode** (no base_url, or base_url is a real Azure
           endpoint): uses pydantic-ai's infer_model() with the
           ``provider:model_name`` convention (e.g. ``anthropic:claude-sonnet-4-20250514``).
           If credentials.api_key is set, it is injected into the provider;
           otherwise pydantic-ai reads the standard env vars (OPENAI_API_KEY,
           ANTHROPIC_API_KEY, etc.).
        """
        import httpx

        if credentials is None:
            raise ValueError(
                "credentials are required: pass credentials=... explicitly, "
                "or use LLMCredentials.from_env() for env-based config"
            )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            provider_prefix, bare_name = parse_model_id(self.model_name)

        # A colon does not always separate a provider from a model. Ollama
        # writes its tags that way -- `qwen3:8b`, `llama3.2:1b` -- so splitting
        # blindly sends `8b` to the server and gets "model not found". Only
        # treat the prefix as a provider when it actually is one.
        if provider_prefix and not _is_known_provider(provider_prefix):
            provider_prefix, bare_name = "openai", self.model_name

        # An explicit base_url is a deliberate routing choice: send everything
        # through it as an OpenAI-compatible proxy (LiteLLM, Helicone, etc.).
        # The exception is a real Azure endpoint, which uses a different wire
        # protocol and needs AzureProvider. Detect that by hostname.
        base_url = credentials.base_url
        is_azure_endpoint = bool(
            base_url and ".openai.azure.com" in base_url.lower()
        )

        if base_url and not is_azure_endpoint:
            if not credentials.api_key:
                raise Exception(
                    "api_key is required when base_url is set (proxy mode)"
                )
            # Send bare_name to the proxy. If the user wrote a bare model name
            # (no `:` prefix), bare_name == self.model_name, so e.g. an explicit
            # "azure/gpt-5-mini" passes through unchanged for LiteLLM-style
            # routing. With "openai:gpt-4o", the prefix is stripped to "gpt-4o".
            http_client = _provider_http_client(follow_redirects=True)
            provider = OpenAIProvider(
                api_key=credentials.api_key,
                base_url=base_url,
                http_client=http_client,
            )
            # Subclassed to keep the token counts; see struckdown.usage.
            return CountingOpenAIChatModel(bare_name, provider=provider)

        # Native provider mode: pydantic-ai's provider:model convention,
        # used for direct provider access (no base_url) or Azure direct.
        resolved_key = credentials.api_key_for_provider(provider_prefix)
        if resolved_key:
            def provider_factory(provider_name: str):
                return _create_provider_with_credentials(provider_name, credentials)
            return infer_model(self.model_name, provider_factory=provider_factory)

        # no explicit credentials -- let pydantic-ai read env vars
        return infer_model(self.model_name)


# Marker for cached errors - stored in cache, re-raised on cache hit
CACHED_ERROR_MARKER = "__struckdown_cached_error__"


def _create_cached_error(error: Exception, model_name: str) -> dict:
    """Create a cacheable error dict."""
    return {
        CACHED_ERROR_MARKER: True,
        "error_class": type(error).__name__,
        "error_message": str(error)[:2000],
        "model_name": model_name,
    }


def _is_cacheable_error(error: Exception) -> bool:
    """Check if an error is deterministic and safe to cache."""
    if isinstance(error, ModelHTTPError):
        body_class = _classify_http_error_body(str(error))
        return body_class in ("content_filter", "context_window")
    return False


def _is_transient_http_error(error: Exception) -> bool:
    """Check if an error is transient (might succeed on retry)."""
    if isinstance(error, ModelHTTPError):
        return error.status_code in TRANSIENT_STATUS_CODES
    return False


def _is_fatal_http_error(error: Exception) -> bool:
    """Check if an error requires config/credential changes."""
    if isinstance(error, ModelHTTPError):
        return error.status_code in FATAL_STATUS_CODES
    return False


def _build_pydantic_ai_agent(
    return_type,
    llm: "LLM",
    credentials: "LLMCredentials",
    mode: str = "tool",
    instructions: str = "",
) -> Agent:
    """Build a pydantic-ai Agent for structured output extraction.

    Args:
        mode: "tool" (default) uses tool calling; "prompted" injects the JSON
              schema into the prompt text instead -- works with models that
              don't reliably follow tool calls.
        instructions: the template's ``<system>`` text, sent as instructions
              rather than as a message. See :func:`struckdown.messages.split_for_agent`.
    """
    model = llm.get_pydantic_model(credentials)
    kwargs = {"retries": 2, "instructions": instructions or None}
    if mode == "prompted":
        return Agent(model, output_type=PromptedOutput(return_type), **kwargs)
    return Agent(model, output_type=return_type, **kwargs)


def _prompt_repr(messages: List[Dict[str, str]]) -> str:
    """The turn a caller sees quoted back in an error: the last thing asked."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def _build_completion_dict(
    model_response,
    messages: List[Dict[str, str]],
    cost: Optional[float] = None,
    model_name: str = "",
) -> dict:
    """Build a completion dict compatible with StruckdownResult consumers.

    The dict structure matches what downstream code expects:
    - usage.prompt_tokens, usage.completion_tokens
    - usage.prompt_tokens_details.cached_tokens, cache_creation_tokens
    - _hidden_params.response_cost
    - _request_messages
    """
    usage = getattr(model_response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    cache_read = getattr(usage, "cache_read_tokens", 0) if usage else 0
    cache_write = getattr(usage, "cache_write_tokens", 0) if usage else 0

    response_cost = cost
    if response_cost is None and model_response is not None:
        response_cost = _calc_cost_from_usage(model_response, model_name)
    if isinstance(response_cost, CostBreakdown):
        response_cost = response_cost.total_cost

    # extract thinking content from model response (empty string -> None)
    thinking = None
    if model_response is not None:
        thinking = getattr(model_response, "thinking", None) or None

    return {
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cache_read or 0,
                "cache_creation_tokens": cache_write or 0,
            },
        },
        "_hidden_params": {
            "response_cost": response_cost,
        },
        "_request_messages": messages,
        "_thinking": thinking,
    }


def _extract_total_price(price_calc) -> Optional[float]:
    """Extract a numeric total from a price calculation object."""
    if price_calc is None:
        return None
    total = getattr(price_calc, "total_price", None)
    if total is None:
        total = getattr(price_calc, "total", None)
    return float(total) if total is not None else None


def _normalise_pricing_model(model_name: str) -> tuple[str, Optional[str]]:
    """Return (model_ref, provider_id) suitable for genai-prices."""
    if not model_name:
        return "", None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            provider_id, bare_name = parse_model_id(model_name)
            return bare_name, provider_id
        except Exception:
            return model_name, None


def _calc_cost_from_usage(model_response, model_name: str) -> Optional[CostBreakdown]:
    """Compute USD cost from a model response's usage data.

    Priority:
    1. ModelSpec pricing (via _model_pricing context var) -- caller-supplied per-mtok rates
    2. pydantic-ai's built-in .cost() method
    3. genai-prices library fallback

    Returns a :class:`CostBreakdown` (input and output sides, with the prices
    used) or None when no route can price the call. ``.total_cost`` is the old
    single figure.
    """
    usage = getattr(model_response, "usage", None)
    if not usage:
        return None

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    # 1. Check for caller-supplied pricing from ModelSpec
    pricing = _model_pricing.get()
    if pricing is not None:
        return cost_from_stored(
            pricing,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_write_tokens", 0) or 0,
        )

    # 2. PydanticAI already knows how to price a ModelResponse using usage plus
    # provider metadata; prefer that path over our local reconstruction.
    response_cost = getattr(model_response, "cost", None)
    if callable(response_cost):
        try:
            breakdown = cost_from_price_calc(response_cost(), "pydantic_ai")
            if breakdown is not None:
                return breakdown
        except Exception:
            pass

    # 3. Fall back to genai-prices library
    try:
        import genai_prices

        model_ref, provider_id = _normalise_pricing_model(model_name)
        response_model_name = getattr(model_response, "model_name", "") or ""
        if not model_ref and response_model_name:
            model_ref, provider_id = _normalise_pricing_model(response_model_name)

        provider_name = getattr(model_response, "provider_name", None)
        provider_url = getattr(model_response, "provider_url", None)

        if not model_ref:
            return None

        gp_usage = genai_prices.Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_write_tokens", 0) or 0,
        )
        price = genai_prices.calc_price(
            gp_usage,
            model_ref,
            provider_id=provider_name or provider_id,
            provider_api_url=provider_url,
        )
        return cost_from_price_calc(price, "genai_prices")
    except Exception:
        return None


def _store_in_cache(
    messages, model_name, max_retries, extra_kwargs, return_type, result, strict_params,
):
    """Manually store a result in joblib's cache (used after streaming)."""
    from struckdown import __version__

    try:
        cache_args = dict(
            messages=messages,
            model_name=model_name,
            max_retries=max_retries,
            max_tokens=None,
            extra_kwargs=extra_kwargs or {},
            return_type_hash=hash_return_type(return_type),
            cache_version=__version__,
        )
        # ignored params still needed for args_id computation exclusion
        kwargs = {
            **cache_args,
            "return_type": return_type,
            "llm": None,
            "credentials": None,
            "strict_params": strict_params,
        }
        args_id = _call_llm_cached._get_args_id(**kwargs)
        call_id = (_call_llm_cached.func_id, args_id)
        _call_llm_cached.store_backend.dump_item(call_id, result)
        _call_llm_cached._persist_input(0.0, call_id, (), kwargs)
        logger.debug(f"Cached streaming result for {model_name}")
    except Exception as e:
        logger.debug(f"Failed to cache streaming result: {e}")


def _provider_http_client(**kwargs):
    """AsyncClient for an OpenAI-compatible provider.

    pydantic-ai deprecated httpx clients in favour of httpx2, and depends on it
    from v2.42. But an OpenAI-compatible provider hands the client to the
    ``openai`` package, which type-checks it against whichever module *it* was
    built on and refuses the other with a TypeError. Having both installed is
    normal, so the choice has to follow openai rather than a guess:

        Invalid `http_client` argument; Expected an instance of
        `httpx.AsyncClient` but got <class 'httpx2.AsyncClient'>

    Ask openai what it imports, and only fall back to guessing if that private
    attribute ever moves.
    """
    try:
        import openai._base_client as _openai_base

        return _openai_base.httpx.AsyncClient(**kwargs)
    except (ImportError, AttributeError):
        pass
    try:
        import httpx2 as client_module
    except ImportError:
        import httpx as client_module
    return client_module.AsyncClient(**kwargs)


# sampling knobs a reasoning model ignores while it is reasoning
_SAMPLING_SETTINGS = ("temperature", "top_p", "presence_penalty", "frequency_penalty")


def _profile_flag(profile, key: str) -> bool:
    """Read a model-profile flag. pydantic-ai exposes the profile as a dataclass
    in v1 and as a plain dict in v2; unknown flags read as False."""
    if isinstance(profile, dict):
        return bool(profile.get(key, False))
    return bool(getattr(profile, key, False))


def _reasoning_is_active(profile, settings: ModelSettings) -> bool:
    """Whether reasoning will be on for this request. Explicit effort wins,
    then the unified ``thinking`` setting, then the model's own default."""
    effort = settings.get("openai_reasoning_effort")
    if effort is not None:
        return effort != "none"
    thinking = settings.get("thinking")
    if thinking is not None:
        return thinking is not False
    return _profile_flag(profile, "openai_reasoning_enabled_by_default")


def _without_sampling_params(model: PydanticAIModel, settings: ModelSettings) -> ModelSettings:
    """Drop sampling params a reasoning model would ignore.

    pydantic-ai drops them itself and warns each time it does. The temperature
    is usually struckdown's per-return-type default rather than anything the
    caller chose, so that warning is noise on every call. Where the profile
    says nothing, send the settings unchanged and let pydantic-ai decide.
    """
    profile = getattr(model, "profile", None)
    if not _profile_flag(profile, "openai_supports_reasoning"):
        return settings
    if _profile_flag(
        profile, "openai_supports_reasoning_effort_none"
    ) and not _reasoning_is_active(profile, settings):
        return settings
    dropped = [k for k in _SAMPLING_SETTINGS if k in settings]
    if not dropped:
        return settings
    logger.debug(
        f"{model.model_name} is reasoning; not sending {', '.join(dropped)}"
    )
    return ModelSettings(
        **{k: v for k, v in settings.items() if k not in _SAMPLING_SETTINGS}
    )


def _record_for_completion(
    model_name,
    credentials,
    com_dict: dict,
    messages,
    *,
    started_at,
    duration_ms,
    cache_hit: bool,
):
    """The usage record for a finished ``structured_chat`` call.

    Built from the completion dict rather than the response, because on a cache
    hit the dict is all there is. The cost is re-derived from the token counts
    when the dict carries no breakdown (a cache hit made before this version, or
    a call priced by a route that returned only a total).
    """
    response = ResponseStandIn(com_dict)
    cost = None
    if not cache_hit:
        cost = com_dict.get("_cost_breakdown")
        if cost is None:
            cost = _calc_cost_from_usage(response, model_name)
    else:
        cost = CostBreakdown(input_cost=0.0, output_cost=0.0, source="cache")
    payload = None
    if wants_payload():
        payload = UsagePayload(
            request={"messages": messages},
            response={"output": com_dict.get("_output")},
        )
    return record_from_response(
        kind="chat",
        model_name=model_name,
        response=response,
        cost=cost,
        credentials=credentials,
        started_at=started_at,
        duration_ms=duration_ms,
        cache_hit=cache_hit,
        payload=payload,
    )


def _failure_record(
    model_name, credentials, error_class, started_at, duration_ms, cache_hit=False, kind="chat"
):
    return record_from_response(
        kind=kind,
        model_name=model_name,
        response=None,
        cost=None,
        credentials=credentials,
        started_at=started_at,
        duration_ms=duration_ms,
        cache_hit=cache_hit,
        ok=False,
        error_class=error_class,
    )


def _emit_failure(model_name, credentials, error_class, started_at, duration_ms, cache_hit=False):
    emit(_failure_record(model_name, credentials, error_class, started_at, duration_ms, cache_hit))


def _records_for_run(result, model_name, credentials, messages, *, started_at, elapsed_ms):
    """One usage record per model response in an agent run.

    A tool loop is several provider requests, each with its own usage; summing
    them (which the completion dict does) would make a five-round loop look
    like one enormous call. Only the last record carries the run's duration:
    pydantic-ai does not time individual requests.
    """
    # a response without usage (a provider that reported none) has nothing to
    # bill and would only be a zero row
    responses = [
        m
        for m in result.all_messages()
        if hasattr(m, "usage")
        and (
            (getattr(m.usage, "input_tokens", 0) or 0)
            or (getattr(m.usage, "output_tokens", 0) or 0)
        )
    ]
    records = []
    for i, response in enumerate(responses):
        payload = None
        if wants_payload():
            payload = UsagePayload(
                request={"messages": messages} if i == 0 else None,
                response={"parts": to_openai_messages([response])},
            )
        records.append(
            record_from_response(
                kind="chat",
                model_name=model_name,
                response=response,
                cost=_calc_cost_from_usage(response, model_name),
                credentials=credentials,
                started_at=started_at,
                duration_ms=elapsed_ms if i == len(responses) - 1 else None,
                payload=payload,
            )
        )
    return records


def _run_agent_sync(
    agent: Agent, user_prompt: str, settings: ModelSettings, history=None
):
    """Run a pydantic-ai agent synchronously. Returns (output, model_response, cost)."""
    settings = _without_sampling_params(agent.model, settings)
    result = agent.run_sync(
        user_prompt, model_settings=settings, message_history=history or None
    )
    # get the last model response for usage
    last_response = None
    for msg in reversed(result.all_messages()):
        if hasattr(msg, "usage"):
            last_response = msg
            break
    cost = _calc_cost_from_usage(last_response, agent.model.model_name)
    return result.output, last_response, cost


@memory.cache(ignore=["return_type", "llm", "credentials", "strict_params"])
def _call_llm_cached(
    messages: List[Dict[str, str]],
    model_name: str,
    max_retries: int,
    max_tokens: Optional[int],
    extra_kwargs: Optional[dict],
    return_type_hash: str,
    return_type,
    llm,
    credentials,
    cache_version: str,
    strict_params: bool = False,
):
    """Cache the raw response dict from the LLM.

    Uses pydantic-ai Agent.run_sync() for structured output extraction.
    Returns dicts (not Pydantic models) so they pickle safely.

    Deterministic errors (content policy, context window) are cached to avoid
    repeated failed API calls. Change seed to force retry.
    """
    mm = str(messages)[:100]
    logger.debug(f"LLM CALL: {mm}")
    logger.debug(f"\n\n{LC.BLUE}Messages: {messages}{LC.RESET}\n\n")

    # build model settings from extra_kwargs + return_type llm_config
    call_kwargs = _merge_llm_config_defaults(extra_kwargs, return_type)
    if max_tokens and "max_tokens" not in call_kwargs:
        call_kwargs["max_tokens"] = max_tokens
    settings = _translate_kwargs(call_kwargs, strict=strict_params, model_name=model_name)

    if _debug_api_requests:
        import sys
        print("\n" + "=" * 80, file=sys.stderr)
        print("API REQUEST", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(json.dumps({"messages": messages, "model": model_name,
                          "settings": str(settings)}, indent=2, default=str),
              file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)

    instructions, history, user_prompt = split_for_agent(messages)
    prompt_repr = _prompt_repr(messages)

    # try tool mode first, fall back to prompted mode if the model can't
    # handle tool calling (e.g. returns text instead of a tool call)
    effort_retried = False
    for mode in ("tool", "prompted"):
        agent = _build_pydantic_ai_agent(
            return_type, llm, credentials, mode=mode, instructions=instructions
        )
        mode_succeeded = False
        while True:
            try:
                output, model_response, run_cost = _run_agent_sync(
                    agent, user_prompt, settings, history
                )
                mode_succeeded = True
                break
            except ModelHTTPError as e:
                if not effort_retried and _retry_effort(e, model_name, settings):
                    effort_retried = True
                    continue
                _log_error_details(e, model_name, f"(HTTP {e.status_code})")
                if _is_cacheable_error(e):
                    logger.debug(
                        f"Cacheable error for model {model_name} (will be cached): "
                        f"{type(e).__name__}"
                    )
                    return _create_cached_error(e, model_name), None
                raise _make_struckdown_error(e, prompt_repr, model_name) from e
            except UnexpectedModelBehavior as e:
                if mode == "tool":
                    logger.info(
                        f"Tool-mode structured output failed for {model_name}, "
                        f"retrying with prompted mode: {e}"
                    )
                    break
                _log_error_details(e, model_name, "(unexpected model behavior, prompted mode)")
                raise _make_struckdown_error(e, prompt_repr, model_name) from e
            except Exception as e:
                _log_error_details(e, model_name, "(unknown)")
                if _is_cacheable_error(e):
                    return _create_cached_error(e, model_name), None
                full_traceback = traceback.format_exc()
                logger.debug(f"Unknown error calling LLM {model_name}: {e}\n{full_traceback}")
                raise _make_struckdown_error(e, prompt_repr, model_name) from e
        if mode_succeeded:
            break

    logger.debug(f"\n\n{LC.GREEN}Response: {output}{LC.RESET}\n")

    com_dict = _build_completion_dict(
        model_response,
        messages,
        cost=run_cost,
        model_name=agent.model.model_name,
    )
    com_dict["_cost_breakdown"] = run_cost
    com_dict["_provider_response_id"] = str(
        getattr(model_response, "provider_response_id", "") or ""
    )
    com_dict["_finish_reason"] = str(getattr(model_response, "finish_reason", "") or "")

    # mark that we made a fresh API call (this code only runs on cache miss)
    _cache_miss_marker.set(True)

    res_dict = output.model_dump() if hasattr(output, "model_dump") else output
    res_dict = _strip_null_bytes(res_dict)
    com_dict["_output"] = res_dict
    return res_dict, com_dict


def structured_chat(
    prompt=None,
    messages=None,
    return_type=None,
    llm: LLM = None,
    credentials: LLMCredentials = None,
    max_retries=3,
    max_tokens=None,
    extra_kwargs=None,
    strict_params: bool = False,
):
    """Make a structured LLM call via pydantic-ai, returning (response, completion).

    Args:
        prompt: (Deprecated) Single prompt string. Use messages parameter instead.
        messages: List of message dicts with 'role' and 'content' keys.
        return_type: Pydantic model for response structure
        llm: LLM configuration
        credentials: API credentials
        max_retries: Number of retry attempts (handled by caller, not pydantic-ai)
        max_tokens: Maximum tokens in response
        extra_kwargs: Additional LLM parameters
        strict_params: Raise ValueError on unsupported params instead of warning
    """
    import time

    from struckdown import __version__

    if llm is None:
        llm = LLM()
    if credentials is None:
        raise ValueError(
            "credentials are required: pass credentials=... explicitly, "
            "or use LLMCredentials.from_env() for env-based config"
        )

    if prompt is not None and messages is None:
        messages = [{"role": "user", "content": prompt}]
    elif messages is None:
        raise ValueError("Either prompt or messages must be provided")

    # Extract a hint for logging from the last user message
    call_hint = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")[:50].replace("\n", " ")
            call_hint = (
                f" ({content}...)"
                if len(msg.get("content", "")) > 50
                else f" ({content})"
            )
            break

    logger.debug(
        f"Using model {llm.model_name}, max_retries {max_retries}, max_tokens: {max_tokens}"
    )
    logger.debug(f"LLM kwargs: {extra_kwargs}")

    if _debug_api_requests:
        _print_routing_debug(llm, credentials)

    start_time = time.monotonic()
    logger.debug(f"{LC.CYAN}LLM CALL START{call_hint}{LC.RESET}")

    # Warn user if call takes longer than threshold
    import threading

    SLOW_CALL_THRESHOLD = 45  # seconds

    def _slow_call_warning(hint, cancel_event):
        if cancel_event.wait(SLOW_CALL_THRESHOLD):
            return  # call completed before threshold
        logger.warning(
            f"LLM call still in progress after {SLOW_CALL_THRESHOLD}s...{hint}"
        )

    cancel_event = threading.Event()
    warning_thread = threading.Thread(
        target=_slow_call_warning, args=(call_hint, cancel_event), daemon=True
    )
    warning_thread.start()

    # Reset cache miss marker before call
    _cache_miss_marker.set(False)
    started_at = _utcnow()

    try:
        res_dict, com_dict = _call_llm_cached(
            messages=messages,
            model_name=llm.model_name,
            max_retries=max_retries,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs or {},
            return_type_hash=hash_return_type(return_type),
            return_type=return_type,
            llm=llm,
            credentials=credentials,
            cache_version=__version__,
            strict_params=strict_params,
        )
    except LLMError as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            f"{LC.RED}LLM CALL FAILED [{elapsed_ms:.0f}ms]{call_hint}: {e}{LC.RESET}"
        )
        _emit_failure(llm.model_name, credentials, type(e).__name__, started_at, elapsed_ms)
        raise
    except Exception as e:
        # not an LLMError: a provider client that would not build, a bug. Still
        # a request that was attempted, so still a row.
        elapsed_ms = (time.monotonic() - start_time) * 1000
        _emit_failure(llm.model_name, credentials, type(e).__name__, started_at, elapsed_ms)
        raise
    finally:
        cancel_event.set()  # stop the warning thread

    # check for cached error response
    if isinstance(res_dict, dict) and res_dict.get(CACHED_ERROR_MARKER):
        elapsed_ms = (time.monotonic() - start_time) * 1000
        error_class = res_dict.get("error_class", "CachedError")
        error_msg = res_dict.get("error_message", "Unknown cached error")
        logger.debug(
            f"{LC.RED}LLM CALL FAILED (cached) [{elapsed_ms:.0f}ms]{call_hint}: {error_class}{LC.RESET}"
        )
        _emit_failure(
            llm.model_name, credentials, error_class, started_at, elapsed_ms, cache_hit=True
        )
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise _make_cached_error(error_class, error_msg, prompt_repr, llm.model_name)

    res = return_type.model_validate(res_dict)

    # Determine if this was a cache hit or fresh call using the marker
    was_cached = not _cache_miss_marker.get()
    com_dict["_cached"] = was_cached
    com = Box(com_dict)

    elapsed_ms = (time.monotonic() - start_time) * 1000
    emit(
        _record_for_completion(
            llm.model_name,
            credentials,
            com_dict,
            messages,
            started_at=started_at,
            duration_ms=elapsed_ms,
            cache_hit=was_cached,
        )
    )
    cache_status = " (cached)" if was_cached else ""
    logger.debug(was_cached and "Cache hit" or "")
    logger.debug(
        f"{LC.GREEN}LLM CALL DONE [{elapsed_ms:.0f}ms]{cache_status}{call_hint}{LC.RESET}"
    )

    logger.debug(
        f"{LC.PURPLE}Response type: {type(res)}; {len(str(res))} chars produced{LC.RESET}\n\n"
    )
    return res, com


async def structured_chat_async(
    messages=None,
    return_type=None,
    llm: LLM = None,
    credentials: LLMCredentials = None,
    max_retries=3,
    extra_kwargs=None,
    stream=False,
    strict_params: bool = False,
) -> AsyncGenerator[Tuple, None]:
    """Async structured chat. Yields (partial_or_result, completion, is_final) tuples.

    When stream=False: runs the existing sync structured_chat in a thread,
    yields a single (result, completion, True) tuple.

    When stream=True: uses pydantic-ai's Agent.run_stream() to stream partial
    Pydantic models, yielding (partial, None, False) for each chunk,
    then (final_result, completion, True) at the end.
    """
    if llm is None:
        llm = LLM()
    if credentials is None:
        raise ValueError(
            "credentials are required: pass credentials=... explicitly, "
            "or use LLMCredentials.from_env() for env-based config"
        )

    if not stream:
        # non-streaming: wrap existing sync path in a thread. Its usage records
        # are dispatched here, on the loop, not from the worker thread.
        with deferred_usage() as pending:
            res, com = await anyio.to_thread.run_sync(
                lambda: structured_chat(
                    messages=messages,
                    return_type=return_type,
                    llm=llm,
                    credentials=credentials,
                    max_retries=max_retries,
                    extra_kwargs=extra_kwargs,
                    strict_params=strict_params,
                ),
                abandon_on_cancel=True,
            )
        await flush_usage_async(pending)
        yield (res, com, True)
        return

    if _debug_api_requests:
        _print_routing_debug(llm, credentials)

    # streaming path: check if result is already cached (without computing)
    from struckdown import __version__

    def _probe_cache():
        """Check if the result exists in cache without making an API call."""
        cache_args = dict(
            messages=messages,
            model_name=llm.model_name,
            max_retries=max_retries,
            max_tokens=None,
            extra_kwargs=extra_kwargs or {},
            return_type_hash=hash_return_type(return_type),
            cache_version=__version__,
        )
        # joblib's check_call_in_cache returns True if cached
        if _call_llm_cached.check_call_in_cache(
            **cache_args,
            return_type=return_type,
            llm=llm,
            credentials=credentials,
        ):
            # cache hit -- fetch the cached result
            _cache_miss_marker.set(False)
            res_dict, com_dict = _call_llm_cached(
                **cache_args,
                return_type=return_type,
                llm=llm,
                credentials=credentials,
                strict_params=strict_params,
            )
            return res_dict, com_dict
        return None, None

    try:
        res_dict, com_dict = await anyio.to_thread.run_sync(
            _probe_cache, abandon_on_cancel=True,
        )
        if res_dict is not None:
            # cache hit -- return immediately without streaming
            if isinstance(res_dict, dict) and res_dict.get(CACHED_ERROR_MARKER):
                error_class = res_dict.get("error_class", "CachedError")
                error_msg = res_dict.get("error_message", "Unknown cached error")
                prompt_repr = next(
                    (m["content"] for m in messages if m["role"] == "user"), ""
                )
                raise _make_cached_error(
                    error_class, error_msg, prompt_repr, llm.model_name
                )
            res = return_type.model_validate(res_dict)
            com_dict["_cached"] = True
            # emit intermediate yield so streaming consumers display the text
            yield (res, None, False)
            yield (res, Box(com_dict), True)
            return
    except LLMError:
        raise

    # cache miss -- stream via pydantic-ai Agent.run_stream()
    call_kwargs = _merge_llm_config_defaults(extra_kwargs, return_type)
    settings = _translate_kwargs(
        call_kwargs, strict=strict_params, model_name=llm.model_name
    )

    instructions, history, user_prompt = split_for_agent(messages)
    prompt_repr = _prompt_repr(messages)

    # debounce: configurable via extra_kwargs, default 200ms for responsive feel
    debounce_s = (extra_kwargs or {}).get("stream_debounce_ms", 200) / 1000.0

    stream_started_at = _utcnow()
    stream_started = time.monotonic()

    # try tool mode first, fall back to prompted mode
    streaming_failed = False
    effort_retried = False
    for mode in ("tool", "prompted"):
        agent = _build_pydantic_ai_agent(
            return_type, llm, credentials, mode=mode, instructions=instructions
        )
        mode_succeeded = False
        while True:
            prev_text = ""
            final_output = None
            last_response = None
            stream_cost = None
            try:
                async with agent.run_stream(
                    user_prompt,
                    model_settings=_without_sampling_params(agent.model, settings),
                    message_history=history or None,
                ) as stream:
                    async for partial in stream.stream_output(debounce_by=debounce_s):
                        final_output = partial
                        current = getattr(partial, "response", None)
                        if current is not None and str(current) != prev_text:
                            yield (partial, None, False)
                            prev_text = str(current)
                    # get the final output after stream completes
                    if final_output is None:
                        final_output = await stream.get_output()
                    # extract model response for usage info
                    for msg in reversed(stream.all_messages()):
                        if hasattr(msg, "usage"):
                            last_response = msg
                            break
                    stream_cost = _calc_cost_from_usage(last_response, llm.model_name)
                mode_succeeded = True
                break
            except UnexpectedModelBehavior as e:
                if mode == "tool":
                    logger.info(
                        f"Tool-mode streaming failed for {llm.model_name}, "
                        f"retrying with prompted mode: {e}"
                    )
                    break
                # prompted mode also failed -- fall back to non-streaming
                logger.info(
                    f"Streaming failed for {llm.model_name} in both modes, "
                    f"falling back to non-streaming: {e}"
                )
                streaming_failed = True
                break
            except ModelHTTPError as e:
                if not effort_retried and _retry_effort(e, llm.model_name, settings):
                    effort_retried = True
                    continue
                await emit_async(
                    _failure_record(
                        llm.model_name, credentials, type(e).__name__, stream_started_at,
                        (time.monotonic() - stream_started) * 1000,
                    )
                )
                raise _make_struckdown_error(e, prompt_repr, llm.model_name) from e
            except Exception as e:
                await emit_async(
                    _failure_record(
                        llm.model_name, credentials, type(e).__name__, stream_started_at,
                        (time.monotonic() - stream_started) * 1000,
                    )
                )
                raise _make_struckdown_error(e, prompt_repr, llm.model_name) from e
        if streaming_failed or mode_succeeded:
            break

    # non-streaming fallback: use run_sync which supports retries
    if streaming_failed:
        with deferred_usage() as pending:
            res_dict, com_dict = await anyio.to_thread.run_sync(
                lambda: structured_chat(
                    messages=messages,
                    return_type=return_type,
                    llm=llm,
                    credentials=credentials,
                    max_retries=3,
                    extra_kwargs=extra_kwargs,
                    strict_params=strict_params,
                )
            )
        await flush_usage_async(pending)
        com_dict["_cached"] = False
        # reconstruct the output as a pydantic model for consistency
        if hasattr(return_type, "model_validate"):
            final_output = return_type.model_validate(res_dict)
        else:
            final_output = res_dict
        # yield the complete text as an intermediate so streaming consumers
        # (e.g. the CLI) display it before the final yield
        yield (final_output, None, False)
        yield (final_output, Box(com_dict), True)
        return

    if final_output is None:
        raise LLMError(
            Exception("No response received from streaming"),
            next((m["content"] for m in messages if m["role"] == "user"), ""),
            llm.model_name,
        )

    com_dict = _build_completion_dict(
        last_response,
        messages,
        cost=stream_cost,
        model_name=llm.model_name,
    )
    com_dict["_cached"] = False
    com_dict["_cost_breakdown"] = stream_cost

    # store the streamed result into joblib's cache so subsequent calls hit cache
    res_dict = final_output.model_dump() if hasattr(final_output, "model_dump") else final_output
    res_dict = _strip_null_bytes(res_dict)
    com_dict["_output"] = res_dict
    await emit_async(
        record_from_response(
            kind="chat",
            model_name=llm.model_name,
            response=last_response,
            cost=stream_cost,
            credentials=credentials,
            started_at=stream_started_at,
            duration_ms=(time.monotonic() - stream_started) * 1000,
            payload=(
                UsagePayload(request={"messages": messages}, response={"output": res_dict})
                if wants_payload()
                else None
            ),
        )
    )
    _store_in_cache(
        messages=messages,
        model_name=llm.model_name,
        max_retries=max_retries,
        extra_kwargs=extra_kwargs,
        return_type=return_type,
        result=(res_dict, com_dict),
        strict_params=strict_params,
    )

    yield (final_output, Box(com_dict), True)


# Singleton cache for local embedding models
_local_embedding_models = {}

# Singleton cache for cross-encoder models
_cross_encoder_models = {}


def _get_best_device() -> str:
    """Auto-detect the best available device for local models.

    Respects PYTORCH_MPS_DISABLE and CUDA_VISIBLE_DEVICES env vars.
    """
    import os

    import torch

    # Check if MPS is explicitly disabled (for forked processes on macOS)
    mps_disabled = os.environ.get("PYTORCH_MPS_DISABLE", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if torch.cuda.is_available():
        return "cuda"
    elif not mps_disabled and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_local_embedding(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
    device: str = None,
) -> List[List[float]]:
    """Get embeddings using local sentence-transformers model.

    Args:
        texts: List of texts to embed
        model_name: HuggingFace model name (e.g., "all-MiniLM-L6-v2")
        batch_size: Batch size for encoding (default: 32)
        device: Device to use ("cpu", "cuda", "mps"). None for auto-detect.

    Returns:
        List of embedding vectors

    Raises:
        ImportError: If sentence-transformers is not installed
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for local embeddings. "
            "Install with: pip install struckdown[local]"
        )

    if device is None:
        device = _get_best_device()

    cache_key = (model_name, device)
    if cache_key not in _local_embedding_models:
        logger.info(f"Loading local embedding model: {model_name} on {device}")
        _local_embedding_models[cache_key] = SentenceTransformer(
            model_name, device=device
        )

    model = _local_embedding_models[cache_key]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
        batch_size=batch_size,
    )
    return embeddings.tolist()


def _is_transient_embedding_error(exception: BaseException) -> bool:
    """Check if an embedding exception is transient and should be retried."""
    if isinstance(exception, ModelHTTPError):
        return exception.status_code in TRANSIENT_STATUS_CODES
    return False


@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_transient_embedding_error),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=120),
    stop=tenacity.stop_after_attempt(7),
    before_sleep=lambda retry_state: logger.info(
        f"Embedding API error ({type(retry_state.outcome.exception()).__name__}), "
        f"retrying in {retry_state.next_action.sleep:.1f}s (attempt {retry_state.attempt_number}/7)"
    ),
    reraise=True,
)
async def _get_api_embedding_batch_async(
    batch: List[str],
    model_name: str,
    dimensions: Optional[int],
    api_key: str,
    base_url: Optional[str],
    timeout: int = EMBEDDING_TIMEOUT,
) -> Tuple[List[List[float]], int, Optional[float]]:
    """Get embeddings for a single batch via pydantic-ai async API call.

    Retries transient errors (rate limits, timeouts, 5xx) with exponential backoff.

    Returns:
        Tuple of (embeddings, total_tokens, cost). cost is None if unknown.
    """
    import httpx
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel

    logger.debug(
        f"API embedding batch: {len(batch)} texts, model={model_name}, dims={dimensions}"
    )

    http_client = _provider_http_client(follow_redirects=True, timeout=timeout)
    provider = OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
    )
    embedding_model = OpenAIEmbeddingModel(model_name, provider=provider)

    settings = {"dimensions": dimensions} if dimensions else {}
    started_at = _utcnow()
    started = time.monotonic()
    try:
        result = await embedding_model.embed(
            list(map(str, batch)),
            input_type="document",
            settings=settings,
        )
    except Exception as e:
        await emit_async(
            record_from_response(
                kind="embedding",
                model_name=model_name,
                response=None,
                cost=None,
                credentials=LLMCredentials(api_key=api_key, base_url=base_url or None),
                started_at=started_at,
                duration_ms=(time.monotonic() - started) * 1000,
                ok=False,
                error_class=type(e).__name__,
            )
        )
        raise

    logger.debug(f"API embedding batch complete: {len(result.embeddings)} embeddings")

    # extract token count and cost from result
    total_tokens = result.usage.total_tokens if result.usage else 0

    breakdown = _embedding_cost(result, total_tokens)
    cost = breakdown.total_cost if breakdown is not None else None

    usage_stand_in = ResponseStandIn({"usage": {"prompt_tokens": total_tokens}})
    await emit_async(
        record_from_response(
            kind="embedding",
            model_name=model_name,
            response=usage_stand_in,
            cost=breakdown,
            credentials=LLMCredentials(api_key=api_key, base_url=base_url or None),
            started_at=started_at,
            duration_ms=(time.monotonic() - started) * 1000,
            payload=(
                UsagePayload(request={"texts": len(batch)}, response=None)
                if wants_payload()
                else None
            ),
        )
    )

    return result.embeddings, total_tokens, cost


def _embedding_cost(result, total_tokens: int) -> Optional[CostBreakdown]:
    """Price an embedding batch: the caller's stored rate first, else pydantic-ai's."""
    pricing = _model_pricing.get()
    if pricing is not None:
        return cost_from_stored(pricing, input_tokens=total_tokens, output_tokens=0)
    try:
        price = result.cost()
    except Exception as e:
        logger.debug(f"Could not calculate embedding cost: {e}")
        return None
    if price is None:
        return None
    total = getattr(price, "total", None)
    if total is None:
        total = getattr(price, "total_price", None)
    if total is None:
        return None
    return CostBreakdown(input_cost=float(total), output_cost=0.0, source="pydantic_ai")


# Type alias for progress callback: receives count of items just completed
ProgressCallback = Callable[[int], None]

# Type alias for embedding cost callback: receives (fresh_cost, fresh_tokens, fresh_count)
# Called after embeddings complete with cost incurred from fresh API calls
EmbeddingCostCallback = Callable[[float, int, int], None]


async def _compute_embeddings_async(
    texts: List[str],
    model_name: str,
    dimensions: Optional[int],
    api_key: str,
    base_url: Optional[str],
    batch_size: int,
    progress_callback: Optional[ProgressCallback] = None,
    base_progress: int = 0,
    max_tokens_per_batch: Optional[int] = None,
) -> Tuple[List[List[float]], int, Optional[float]]:
    """Compute embeddings for texts via concurrent async API calls with progress.

    Caches embeddings incrementally as each batch completes, so partial progress
    is preserved if the process is interrupted.

    Args:
        texts: List of texts to embed
        model_name: Model name for API
        dimensions: Embedding dimensions
        api_key: API key
        base_url: API base URL
        batch_size: Texts per batch (max)
        progress_callback: Optional callback(n) called with cumulative count completed
        base_progress: Starting progress count (e.g., from cached items)
        max_tokens_per_batch: Maximum tokens per batch (prevents context window errors)

    Returns:
        Tuple of (embeddings, total_tokens, total_cost). total_cost is None if unknown.
    """
    import asyncio

    # Use token-aware batching to prevent context window errors
    if max_tokens_per_batch is None:
        max_tokens_per_batch = MAX_EMBEDDING_TOKENS_PER_BATCH
    batches = _create_token_aware_batches(texts, batch_size, max_tokens_per_batch)
    logger.debug(
        f"Created {len(batches)} embedding batches "
        f"(max {batch_size} texts, max ~{max_tokens_per_batch} tokens per batch)"
    )

    if len(batches) == 1:
        embeddings, tokens, cost = await _get_api_embedding_batch_async(
            batches[0], model_name, dimensions, api_key, base_url
        )
        # cache with per-embedding cost metadata
        n_emb = len(embeddings)
        per_emb_tokens = [tokens // n_emb] * n_emb if n_emb > 0 else []
        per_emb_cost = [cost / n_emb] * n_emb if cost is not None and n_emb > 0 else None
        store_embeddings(
            batches[0], embeddings, model_name, dimensions,
            tokens_per_embedding=per_emb_tokens,
            costs_per_embedding=per_emb_cost,
        )
        if progress_callback:
            progress_callback(base_progress + len(batches[0]))
        return embeddings, tokens, cost

    logger.debug(
        f"Computing {len(texts)} embeddings in {len(batches)} batches concurrently"
    )

    # track progress and costs across concurrent batches
    completed_count = 0
    total_tokens = 0
    total_cost: Optional[float] = 0.0
    has_unknown_cost = False
    progress_lock = asyncio.Lock()

    # Use dedicated embedding semaphore (lower concurrency than LLM calls)
    sem = get_embedding_semaphore()

    async def process_batch(batch_idx: int, batch: List[str]) -> tuple:
        nonlocal completed_count, total_tokens, total_cost, has_unknown_cost
        async with sem:
            embeddings, tokens, cost = await _get_api_embedding_batch_async(
                batch, model_name, dimensions, api_key, base_url
            )
            # cache this batch immediately with per-embedding cost metadata
            n_emb = len(embeddings)
            per_emb_tokens = [tokens // n_emb] * n_emb if n_emb > 0 else []
            per_emb_cost = (
                [cost / n_emb] * n_emb if cost is not None and n_emb > 0 else None
            )
            store_embeddings(
                batch, embeddings, model_name, dimensions,
                tokens_per_embedding=per_emb_tokens,
                costs_per_embedding=per_emb_cost,
            )
            async with progress_lock:
                completed_count += len(batch)
                total_tokens += tokens
                if cost is None:
                    has_unknown_cost = True
                elif total_cost is not None:
                    total_cost += cost
                if progress_callback:
                    progress_callback(base_progress + completed_count)
            return batch_idx, embeddings, tokens, cost

    # run all batches concurrently, semaphore limits concurrency
    results = await asyncio.gather(
        *[process_batch(idx, batch) for idx, batch in enumerate(batches)]
    )

    # reassemble in original order
    results_ordered = sorted(results, key=lambda x: x[0])
    all_embeddings = []
    for _, batch_embeddings, _, _ in results_ordered:
        all_embeddings.extend(batch_embeddings)

    return all_embeddings, total_tokens, None if has_unknown_cost else total_cost


DEFAULT_EMBEDDING_BATCH_SIZE = env_config(
    "SD_EMBEDDING_BATCH_SIZE", default=100, cast=int
)
# Maximum tokens per embedding batch (leave margin below 8192 for safety)
MAX_EMBEDDING_TOKENS_PER_BATCH = env_config(
    "SD_EMBEDDING_MAX_TOKENS", default=7000, cast=int
)


def _estimate_tokens(text: str) -> int:
    """Estimate token count for a text (rough approximation: ~4 chars per token)."""
    return len(text) // 4 + 1


def _create_token_aware_batches(
    texts: List[str],
    max_texts_per_batch: int,
    max_tokens_per_batch: int,
) -> List[List[str]]:
    """Create batches respecting both text count and token limits.

    Args:
        texts: List of texts to batch
        max_texts_per_batch: Maximum texts per batch
        max_tokens_per_batch: Maximum estimated tokens per batch

    Returns:
        List of batches, each containing texts that fit within limits
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        text_tokens = _estimate_tokens(text)

        # If this single text exceeds the limit, it gets its own batch
        # (the API will handle truncation or error)
        if text_tokens > max_tokens_per_batch:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([text])
            continue

        # Check if adding this text would exceed limits
        would_exceed_tokens = current_tokens + text_tokens > max_tokens_per_batch
        would_exceed_count = len(current_batch) >= max_texts_per_batch

        if would_exceed_tokens or would_exceed_count:
            if current_batch:
                batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


async def get_embedding_async(
    texts: List[str],
    model: Optional[str] = None,
    credentials: Optional[LLMCredentials] = None,
    dimensions: Optional[int] = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    max_tokens_per_batch: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cost_callback: Optional[EmbeddingCostCallback] = None,
) -> EmbeddingResultList:
    """
    Async version of get_embedding. Use this when calling from an async context.

    Get embeddings for texts using local or API models.
    Uses per-string caching via diskcache to avoid redundant API calls.

    Args:
        texts: List of texts to embed
        model: Model name. Use "local/model-name" for sentence-transformers
               (e.g., "local/all-MiniLM-L6-v2"), or API model name
               (e.g., "text-embedding-3-large"). Defaults to DEFAULT_EMBEDDING_MODEL env var.
        credentials: Optional LLMCredentials for API calls.
        dimensions: Embedding dimensions (API models only, model-specific)
        batch_size: Texts per batch (default: 100)
        max_tokens_per_batch: Maximum tokens per batch to avoid context window errors
                             (default: SD_EMBEDDING_MAX_TOKENS env var or 7000)
        progress_callback: Optional callback(n) called after each batch with count of items completed
        cost_callback: Optional callback(fresh_cost, fresh_tokens, fresh_count) called after
                      embeddings complete with cost info from fresh (non-cached) API calls.
                      Use this to track budget incrementally.

    Returns:
        EmbeddingResultList containing EmbeddingResult objects.
        Each embedding has .cost, .tokens, .model, .cached attributes.
        The list has .total_cost, .total_tokens, .cached_count properties.
    """
    if not texts:
        return EmbeddingResultList([], model="")

    # Handle tuple input for backward compatibility
    if isinstance(texts, tuple):
        texts = list(texts)

    # Default model from env
    if model is None:
        model = env_config("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large")

    is_local = model.startswith("local/")

    # Check cache first -- avoids loading local model if everything is cached
    cached, missing = get_cached_embeddings(texts, model, dimensions)

    # Track which indices are cached vs fresh
    cached_indices = set(cached.keys())

    # If all cached, return immediately (no model loading needed)
    if not missing:
        logger.debug(f"All {len(texts)} embeddings found in cache")
        if progress_callback:
            progress_callback(len(texts))
        # use stored cost metadata from cache (what it would have cost)
        results = [
            EmbeddingResult(
                cached[i].embedding,
                cost=cached[i].cost,
                tokens=cached[i].tokens,
                model=model,
                cached=True,
            )
            for i in range(len(texts))
        ]
        return EmbeddingResultList(results, model=model)

    # Report cached items as progress
    n_cached = len(texts) - len(missing)
    if n_cached > 0 and progress_callback:
        progress_callback(n_cached)

    # Compute missing embeddings
    missing_texts = [text for _, text in missing]
    logger.debug(
        f"Computing {len(missing_texts)} missing embeddings ({n_cached} cached)"
    )

    total_tokens = 0
    total_cost: Optional[float] = None

    if is_local:
        local_model_name = model[6:]  # strip "local/"
        logger.debug(f"Using local embeddings: {local_model_name}")
        missing_embeddings = _get_local_embedding(
            missing_texts, local_model_name, batch_size=batch_size
        )
        # Cache the computed local embeddings
        store_embeddings(missing_texts, missing_embeddings, model, dimensions)
        if progress_callback:
            progress_callback(len(texts))
        # local models have zero API cost
        total_tokens = 0
        total_cost = 0.0
    else:
        # API embeddings
        logger.debug(f"Using API embeddings: {model}")

        if credentials is None:
            raise ValueError(
                "credentials are required for API embeddings: "
                "pass credentials=... explicitly, or use LLMCredentials.from_env()"
            )

        missing_embeddings, total_tokens, total_cost = await _compute_embeddings_async(
            missing_texts,
            model,
            dimensions,
            credentials.api_key,
            credentials.base_url,
            batch_size,
            progress_callback,
            base_progress=n_cached,
            max_tokens_per_batch=max_tokens_per_batch,
        )
        # Note: API embeddings are cached incrementally in _compute_embeddings_async

    # Calculate per-embedding cost (distribute evenly among fresh embeddings)
    n_fresh = len(missing)
    if total_cost is None:
        per_embedding_cost = 0.0
    elif n_fresh > 0:
        per_embedding_cost = total_cost / n_fresh
    else:
        per_embedding_cost = 0.0
    per_embedding_tokens = total_tokens // n_fresh if n_fresh > 0 else 0

    # Merge fresh embeddings into cached dict as CachedEmbedding objects
    for (idx, _), emb in zip(missing, missing_embeddings):
        cached[idx] = CachedEmbedding(
            embedding=emb,
            tokens=per_embedding_tokens,
            cost=per_embedding_cost,
        )

    # Build result list with cost metadata
    results = []
    for i in range(len(texts)):
        is_cached = i in cached_indices
        entry = cached[i]
        results.append(
            EmbeddingResult(
                entry.embedding,
                cost=entry.cost,
                tokens=entry.tokens,
                model=model,
                cached=is_cached,
            )
        )

    # Notify cost callback with fresh API costs (excludes cached)
    if cost_callback and n_fresh > 0 and total_cost is not None:
        cost_callback(total_cost, total_tokens, n_fresh)

    return EmbeddingResultList(results, model=model)


def get_embedding(texts: List[str], **kwargs) -> EmbeddingResultList:
    """
    Get embeddings for texts using local or API models (sync version).

    For async contexts, use get_embedding_async() instead.

    Args:
        texts: List of texts to embed
        **kwargs: Forwarded to get_embedding_async (model, credentials, dimensions, etc.)

    Returns:
        EmbeddingResultList containing EmbeddingResult objects.
        Each embedding has .cost, .tokens, .model, .cached attributes.
        The list has .total_cost, .total_tokens, .cached_count properties.

    Raises:
        RuntimeError: If called from within a running async event loop.

    Examples:
        # Local embeddings (fast, free)
        results = get_embedding(texts, model="local/all-MiniLM-L6-v2")

        # API embeddings (better quality)
        results = get_embedding(texts, model="text-embedding-3-large")
        print(results.total_cost)  # total USD cost
        print(results[0].tokens)   # tokens for first embedding

        # API embeddings with custom credentials
        get_embedding(texts, model="text-embedding-3-large", credentials=my_creds)
    """
    import asyncio

    # Check if we're in an async context
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "get_embedding() cannot be called from an async context. "
            "Use 'await get_embedding_async(...)' instead."
        )
    except RuntimeError as e:
        if "no running event loop" not in str(e):
            raise

    with deferred_usage() as pending:
        result = asyncio.run(get_embedding_async(texts, **kwargs))
    flush_usage(pending)
    return result


# --- Cross-encoder similarity ---

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/stsb-roberta-large"


def get_cross_encoder_scores(
    pairs: List[Tuple[str, str]],
    model: str = DEFAULT_CROSS_ENCODER_MODEL,
    normalise_with: Optional[Callable[[List[float], str], List[float]]] = None,
    batch_size: int = 32,
    device: str = None,
    show_progress: bool = True,
) -> List[float]:
    """
    Compute semantic similarity scores for text pairs using a cross-encoder.

    Cross-encoders process both texts together through attention, providing
    more accurate similarity judgments than comparing independent embeddings.

    Uses per-pair caching via diskcache to avoid redundant model calls.
    Note: Scores are cached BEFORE normalisation is applied, so changing
    normalise_with will still use cached raw scores.

    Args:
        pairs: List of (text_a, text_b) tuples to score
        model: Cross-encoder model name from HuggingFace
               (default: "cross-encoder/stsb-roberta-large")
        normalise_with: Optional function(scores, model_name) -> normalised_scores.
                        If None, returns raw model scores.
        batch_size: Batch size for encoding (default: 32)
        device: Device to use ("cpu", "cuda", "mps"). None for auto-detect.
        show_progress: Show progress bar for large batches

    Returns:
        List of similarity scores (one per pair)

    Examples:
        # Raw scores (no normalisation)
        scores = get_cross_encoder_scores(pairs)

        # STS-B models output [0, 5], normalise to [0, 1]
        scores = get_cross_encoder_scores(pairs, normalise_with=lambda s, m: [x / 5 for x in s])
    """
    if not pairs:
        return []

    # Check cache first -- avoids loading model if everything is cached
    cached, missing = get_cached_pair_scores(pairs, model)

    # If all cached, return immediately (no model loading needed)
    if not missing:
        logger.debug(f"All {len(pairs)} pair scores found in cache")
        scores = [cached[i] for i in range(len(pairs))]
        if normalise_with is not None:
            scores = normalise_with(scores, model)
        return scores

    # Need to compute some scores, so load the model
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for cross-encoder scoring. "
            "Install with: pip install sentence-transformers"
        )

    if device is None:
        device = _get_best_device()

    # Load or retrieve cached model
    model_cache_key = (model, device)
    if model_cache_key not in _cross_encoder_models:
        logger.info(f"Loading cross-encoder model: {model} on {device}")
        _cross_encoder_models[model_cache_key] = CrossEncoder(model, device=device)

    encoder = _cross_encoder_models[model_cache_key]

    # Score only missing pairs
    missing_pairs = [pair for _, pair in missing]
    n_cached = len(pairs) - len(missing)
    logger.debug(
        f"Computing {len(missing_pairs)} missing pair scores ({n_cached} cached)"
    )

    missing_scores = encoder.predict(
        missing_pairs,
        batch_size=batch_size,
        show_progress_bar=show_progress and len(missing_pairs) > batch_size,
    )
    missing_scores = list(missing_scores)

    # Cache the computed scores (raw, before normalisation)
    store_pair_scores(missing_pairs, missing_scores, model)

    # Merge cached and computed scores in original order
    for (idx, _), score in zip(missing, missing_scores):
        cached[idx] = score

    scores = [cached[i] for i in range(len(pairs))]

    if normalise_with is not None:
        scores = normalise_with(scores, model)

    return scores


# ---------------------------------------------------------------------------
# Tool loops
#
# A slot marked ``use_tools=true`` hands control of the next few round trips to
# the model: it chooses tools, reads what they return, and answers when it has
# enough. pydantic-ai runs that loop; struckdown's job is to hand it the tools
# the caller supplied, keep it inside the caller's budget, and report what
# happened as ordinary slot events.
#
# The caller's UsageLimits is a CEILING. A template may lower it -- prompts are
# editable in the admin on some deployments, so a template must never be able
# to raise a cap.
# ---------------------------------------------------------------------------


DEFAULT_TOOL_REQUEST_LIMIT = 20
DEFAULT_TOOL_CALLS_LIMIT = 20

# A backstop against a runaway loop, not a length budget: a quarter of a
# million output tokens is far more than any real answer, so it should never
# bind on legitimate work. Deliberately a run-level ``output_tokens_limit``
# rather than a per-request ``max_tokens``: pydantic-ai enforces it locally and
# raises ``UsageLimitExceeded``, whereas a ``max_tokens`` above a model's own
# output ceiling is rejected by the provider, which would turn a safety net
# into a hard failure on every small model.
DEFAULT_TOOL_OUTPUT_TOKENS = 250_000


def clamp_limits(ceiling, max_iter=None, max_calls=None):
    """Return UsageLimits no looser than ``ceiling``.

    ``max_iter`` / ``max_calls`` come from the template. Either is applied only
    where it is *lower* than the caller's own limit; a template asking for more
    gets the ceiling and a warning. When the caller supplies no ceiling, finite
    defaults prevent a model from running an unbounded tool loop.
    """
    from pydantic_ai.usage import UsageLimits

    if ceiling is None:
        ceiling = UsageLimits(
            request_limit=DEFAULT_TOOL_REQUEST_LIMIT,
            tool_calls_limit=DEFAULT_TOOL_CALLS_LIMIT,
            output_tokens_limit=DEFAULT_TOOL_OUTPUT_TOKENS,
        )

    ceiling_requests = getattr(ceiling, "request_limit", None)
    ceiling_calls = getattr(ceiling, "tool_calls_limit", None)

    def _lower(name, wanted, cap):
        if wanted is None:
            return cap
        if cap is not None and wanted > cap:
            logger.warning(
                f"template asked for {name}={wanted} but the caller's ceiling is "
                f"{cap}; using {cap}"
            )
            return cap
        return wanted

    requests = _lower("max_iter", max_iter, ceiling_requests)
    calls = _lower("max_calls", max_calls, ceiling_calls)

    kwargs = {}
    if requests is not None:
        kwargs["request_limit"] = requests
    if calls is not None:
        kwargs["tool_calls_limit"] = calls
    for field in ("input_tokens_limit", "output_tokens_limit", "total_tokens_limit"):
        value = getattr(ceiling, field, None) if ceiling else None
        if value is not None:
            kwargs[field] = value
    return UsageLimits(**kwargs) if kwargs else None


def _tool_toolset(tools, max_iter):
    """The caller's tools as a toolset, empty on the last round.

    Emptying the list is how "send the final request without tools" is
    expressed to pydantic-ai. Only the caller's tools are in here, so the
    output tool -- which is what produces the answer -- is untouched: the model
    can still finish, it just cannot start more gathering.

    A toolset rather than ``tools=``/``prepare_tools=`` because pydantic-ai v2
    dropped the ``prepare_tools`` argument; ``.prepared()`` is the same hook by
    another name, and is available across v1 and v2 alike. ``max_retries``
    matches the agent's own ``retries``, which ``tools=`` used to apply for us.
    """
    from pydantic_ai.toolsets import FunctionToolset

    toolset = FunctionToolset(tools, max_retries=2)
    if max_iter is None:
        return toolset

    async def prepare(ctx, tool_defs):
        return [] if ctx.run_step >= max_iter else tool_defs

    return toolset.prepared(prepare)



def _thinking_stream_handler(announce):
    """A pydantic-ai event handler that reports reasoning as it arrives.

    Reasoning models emit their thinking as ordinary streamed parts, so the
    text is available between tool calls rather than only at the end. What
    reaches ``announce`` is a display copy; the parts pydantic-ai round-trips
    back to the provider are untouched, which matters because Anthropic rejects
    a reasoning chain whose blocks have been altered.
    """
    accumulated = {"text": ""}

    async def handler(ctx, stream):
        from pydantic_ai.messages import (PartDeltaEvent, PartStartEvent,
                                          ThinkingPart, ThinkingPartDelta)

        async for event in stream:
            chunk = ""
            if isinstance(event, PartStartEvent) and isinstance(
                event.part, ThinkingPart
            ):
                chunk = event.part.content or ""
            elif isinstance(event, PartDeltaEvent) and isinstance(
                event.delta, ThinkingPartDelta
            ):
                chunk = event.delta.content_delta or ""
            if not chunk:
                continue
            accumulated["text"] += chunk
            announce(
                "thinking",
                {"delta": chunk, "accumulated": accumulated["text"]},
            )

    return handler


async def run_agent_with_tools(
    messages,
    return_type,
    llm: "LLM",
    credentials: "LLMCredentials",
    *,
    tools=None,
    deps=None,
    deps_type=None,
    limits=None,
    max_iter=None,
    max_calls=None,
    extra_kwargs=None,
    strict_params: bool = False,
    stream: bool = False,
    on_tool_event=None,
):
    """Run one tool-using slot, yielding ``(partial, completion, is_final)``.

    Same contract as :func:`structured_chat_async`, so the segment processor
    handles a tool slot and an ordinary one the same way.

    ``on_tool_event`` is called with ``("started"|"completed", payload)`` as the
    loop goes, so the caller can turn them into slot events without this
    function knowing anything about struckdown's event types.

    Tool results are live reads, so nothing here is cached: the response cache
    keys on messages alone and would serve yesterday's database.
    """
    import time as _time

    from pydantic_ai import Agent

    model = llm.get_pydantic_model(credentials)
    call_kwargs = _merge_llm_config_defaults(extra_kwargs, return_type)
    settings = _translate_kwargs(
        call_kwargs, strict=strict_params, model_name=llm.model_name
    )

    wrapped = [_instrumented_tool(t, on_tool_event) for t in (tools or [])]

    instructions, history, user_prompt = split_for_agent(messages)

    agent_kwargs = {
        "output_type": return_type,
        "retries": 2,
        "instructions": instructions or None,
    }
    if deps_type is not None:
        agent_kwargs["deps_type"] = deps_type
    agent_kwargs["toolsets"] = [_tool_toolset(wrapped, max_iter)]

    agent = Agent(model, **agent_kwargs)

    started = _time.monotonic()
    run_kwargs = {"usage_limits": clamp_limits(limits, max_iter, max_calls)}
    if history:
        run_kwargs["message_history"] = history
    if deps is not None:
        run_kwargs["deps"] = deps
    if settings:
        run_kwargs["model_settings"] = settings

    if on_tool_event is not None:
        run_kwargs["event_stream_handler"] = _thinking_stream_handler(
            lambda kind, payload: _announce_safely(on_tool_event, kind, payload)
        )

    started_at = _utcnow()

    if not stream:
        result = await agent.run(user_prompt, **run_kwargs)
        elapsed_ms = (_time.monotonic() - started) * 1000
        completion = _completion_dict_for_run(
            result, llm.model_name, messages, elapsed_ms
        )
        for record in _records_for_run(
            result, llm.model_name, credentials, messages,
            started_at=started_at, elapsed_ms=elapsed_ms,
        ):
            await emit_async(record)
        yield (result.output, Box(completion), True)
        return

    # Streaming: the tool rounds happen inside run_stream, and only the final
    # answer is streamed. A consumer therefore sees tool events during the
    # gathering and tokens once the model starts writing -- which is the order
    # a reader experiences it in.
    #
    # The run is driven by its own task rather than yielded from inside the
    # `async with`. A generator that suspends inside that block is resumed by
    # whoever consumes it, which is a different step of a different generator,
    # and pydantic-ai's graph then reports itself as already running. Pushing
    # through a queue keeps the context manager on one task from open to close.
    import asyncio as _asyncio

    out: "_asyncio.Queue" = _asyncio.Queue()
    sentinel = object()

    async def _drive():
        try:
            async with agent.run_stream(user_prompt, **run_kwargs) as result:
                async for partial in result.stream_output(debounce_by=None):
                    await out.put((partial, None, False))
                # the streamed response's usage is complete only once the
                # output has been read, so gather that before counting
                output = await result.get_output()
                elapsed = (_time.monotonic() - started) * 1000
                completion = _completion_dict_for_run(
                    result, llm.model_name, messages, elapsed
                )
                for record in _records_for_run(
                    result, llm.model_name, credentials, messages,
                    started_at=started_at, elapsed_ms=elapsed,
                ):
                    await emit_async(record)
                await out.put((output, Box(completion), True))
        except BaseException as exc:  # re-raised on the consumer's side
            await out.put(exc)
        finally:
            await out.put(sentinel)

    task = _asyncio.ensure_future(_drive())
    try:
        while True:
            item = await out.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        if not task.done():
            task.cancel()


def _announce_safely(on_tool_event, kind, payload):
    """Report an event without letting a consumer's error end the run."""
    try:
        on_tool_event(kind, payload)
    except Exception:
        logger.exception("tool event callback raised")


def _instrumented_tool(tool, on_tool_event):
    """Wrap a tool so it reports itself and cannot end the run by raising.

    Three behaviours the hand-written loops had and pydantic-ai does not:

    * an identical call inside one run returns the first result, because a
      model that is unsure tends to ask the same thing twice in one breath;
    * a tool that raises becomes an error string the model can read and work
      around, rather than an exception that ends the turn;
    * every call is announced before it runs and reported after.
    """
    import functools
    import inspect
    import json
    import time as _time

    cache: Dict[str, Any] = {}

    original = tool
    signature = inspect.signature(original)
    is_async = inspect.iscoroutinefunction(original)

    def _key(kwargs):
        return json.dumps(kwargs, sort_keys=True, default=str)

    def _announce(kind, payload):
        if on_tool_event is not None:
            try:
                on_tool_event(kind, payload)
            except Exception:
                logger.exception("tool event callback raised")

    @functools.wraps(original)
    async def wrapper(*args, **kwargs):
        name = getattr(original, "__name__", "tool")
        key = _key(kwargs)
        _announce("started", {"tool_name": name, "arguments": dict(kwargs)})
        started = _time.monotonic()

        if key in cache:
            output = cache[key]
            _announce(
                "completed",
                {
                    "tool_name": name,
                    "arguments": dict(kwargs),
                    "output": output,
                    "ok": True,
                    "elapsed_ms": 0.0,
                    "was_cached": True,
                },
            )
            return output

        try:
            output = original(*args, **kwargs)
            if inspect.isawaitable(output):
                output = await output
            cache[key] = output
            ok, error = True, None
        except Exception as exc:  # a tool bug must not end the run
            logger.exception(f"tool {name} failed")
            output = f"{type(exc).__name__}: {exc}"
            ok, error = False, str(exc)

        _announce(
            "completed",
            {
                "tool_name": name,
                "arguments": dict(kwargs),
                "output": output,
                "ok": ok,
                "error": error,
                "elapsed_ms": (_time.monotonic() - started) * 1000,
                "was_cached": False,
            },
        )
        return output

    wrapper.__signature__ = signature
    if not is_async:
        wrapper.__doc__ = original.__doc__
    return wrapper


def _new_messages(result):
    """Messages this run produced, excluding any history it was given.

    ``new_messages()`` where the result has it; a streamed run exposes the same
    thing and both are safe to call more than once.
    """
    getter = getattr(result, "new_messages", None)
    return getter() if callable(getter) else result.all_messages()


def _run_usage(result):
    """Usage for a whole run. A method in pydantic-ai v1, a property in v2."""
    usage = result.usage
    return usage() if callable(usage) else usage


def _completion_dict_for_run(result, model_name, messages, elapsed_ms):
    """Usage and reasoning for a whole agent run, not one request.

    A tool loop is several model responses. Their token counts are summed --
    reading only the last would under-report the most expensive slot type there
    is -- and every ThinkingPart is kept in order, so a caller can show what the
    model was reasoning about between calls rather than only at the end.
    """
    usage = _run_usage(result)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    thinking_steps = []
    for message in result.all_messages():
        for part in getattr(message, "parts", []) or []:
            if type(part).__name__ == "ThinkingPart" and getattr(part, "content", ""):
                thinking_steps.append(part.content)

    # the run's cost is the sum over its responses; None if any is unpriced
    costs = [
        _calc_cost_from_usage(m, model_name)
        for m in result.all_messages()
        if hasattr(m, "usage")
    ]
    run_cost = (
        sum(c.total_cost for c in costs) if costs and all(c is not None for c in costs) else None
    )

    return {
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "prompt_tokens_details": {
                "cached_tokens": getattr(usage, "cache_read_tokens", 0) or 0,
                "cache_creation_tokens": getattr(usage, "cache_write_tokens", 0) or 0,
            },
        },
        "_hidden_params": {"response_cost": run_cost},
        "_request_messages": messages,
        "_thinking": "\n\n".join(thinking_steps) or None,
        "_thinking_steps": thinking_steps,
        "_all_messages": result.all_messages(),
        # This run only, in the storable shape: the prompt, the calls it made,
        # what came back and the answer -- without the history it was handed,
        # which the caller already has. What a conversation keeps between
        # turns, so the next one carries its own gathering.
        "_messages": to_openai_messages(_new_messages(result)),
        "_elapsed_ms": elapsed_ms,
        "_cached": False,
    }
