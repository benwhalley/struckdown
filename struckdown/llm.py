"""LLM client, credentials, and API interaction for struckdown."""

import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

import instructor
import litellm
from instructor.core.hooks import HookName

# Configure litellm to drop unsupported params rather than error
litellm.drop_params = True
# litellm._turn_on_debug()

# Module-level flag for API request logging
_debug_api_requests = False

def enable_api_debug():
    """Enable logging of full API requests (messages + tools schema)."""
    global _debug_api_requests
    _debug_api_requests = True


def disable_api_debug():
    """Disable API request logging."""
    global _debug_api_requests
    _debug_api_requests = False


import anyio
from box import Box
from decouple import config as env_config
from litellm.exceptions import (APIConnectionError, APIError,
                                APIResponseValidationError,
                                AuthenticationError, BadRequestError,
                                BudgetExceededError,
                                ContentPolicyViolationError,
                                ContextWindowExceededError,
                                InternalServerError, NotFoundError,
                                PermissionDeniedError, RateLimitError,
                                ServiceUnavailableError, Timeout,
                                UnprocessableEntityError,
                                UnsupportedParamsError)
from more_itertools import chunked
from pydantic import BaseModel, Field

from .cache import hash_return_type, memory
from .errors import StruckdownLLMError
from .results import get_run_id

logger = logging.getLogger(__name__)

# Shared concurrency control for all LLM calls
# This limits concurrent API calls across templates AND within together blocks
MAX_LLM_CONCURRENCY = env_config("SD_MAX_CONCURRENCY", default=20, cast=int)
_llm_semaphore = None


def get_llm_semaphore() -> anyio.Semaphore:
    """Get the shared LLM concurrency semaphore (lazy initialization)."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = anyio.Semaphore(MAX_LLM_CONCURRENCY)
    return _llm_semaphore


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


class LLMCredentials(BaseModel):
    api_key: Optional[str] = Field(
        default_factory=lambda: env_config("LLM_API_KEY", None), repr=False
    )
    base_url: Optional[str] = Field(
        default_factory=lambda: env_config("LLM_API_BASE", None), repr=False
    )


class LLM(BaseModel):
    model_name: Optional[str] = Field(
        default_factory=lambda: env_config("DEFAULT_LLM", "gpt-4.1-mini"),
        exclude=True,
    )

    def client(self, credentials: LLMCredentials = None):
        if credentials is None:
            credentials = LLMCredentials()

        if not credentials.api_key or not credentials.base_url:
            raise Exception("Set LLM_API_KEY and LLM_API_BASE environment variables")

        # Create OpenAI-compatible instructor client (works with litellm proxies)
        # Use JSON mode for broad compatibility (uses response_format: json_object)
        litellm.api_key = credentials.api_key
        litellm.api_base = credentials.base_url
        litellm.drop_params = True
        litellm.suppress_debug_info = True
        client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.TOOLS)

        # Truncate validation errors in retry messages to save tokens
        def truncate_validation_errors(**kwargs):
            max_chars = 2000
            for msg in kwargs.get("messages", []):
                content = msg.get("content", "")
                if "validation error" in content.lower() and len(content) > max_chars:
                    msg["content"] = content[:max_chars] + "\n\n... (truncated)"

        client.on(HookName.COMPLETION_KWARGS, truncate_validation_errors)

        # Attach debug hook if enabled
        if _debug_api_requests:

            def log_api_request(**kwargs):
                """Log the full API request as JSON."""
                import sys

                print("\n" + "=" * 80, file=sys.stderr)
                print("API REQUEST", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                print(json.dumps(kwargs, indent=2, default=str), file=sys.stderr)
                print("=" * 80 + "\n", file=sys.stderr)

            client.on(HookName.COMPLETION_KWARGS, log_api_request)

        return client


@memory.cache(ignore=["return_type", "llm", "credentials"])
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
):
    """
    Cache the raw completion dict from the LLM.
    This is the expensive API call we want to cache.
    Returns dicts (not Pydantic models) so they pickle safely.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        cache_version: Version string included in cache key (typically struckdown version)
    """
    mm = str(messages)[:100]
    logger.debug(f"LLM CALL: {mm}")
    logger.debug(f"\n\n{LC.BLUE}Messages: {messages}{LC.RESET}\n\n")
    try:
        call_kwargs = extra_kwargs.copy() if extra_kwargs else {}
        call_kwargs["drop_params"] = True
        res, com = llm.client(credentials).chat.completions.create_with_completion(
            model=model_name,
            response_model=return_type,
            messages=messages,
            max_retries=max_retries,
            **call_kwargs,
        )
    except ContentPolicyViolationError as e:
        logger.warning(f"Content policy violation for model {model_name}: {e}")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise StruckdownLLMError(e, prompt_repr, model_name) from e
    except ContextWindowExceededError as e:
        logger.warning(f"Context window exceeded for model {model_name}: {e}")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise StruckdownLLMError(e, prompt_repr, model_name) from e
    except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
        logger.error(f"Fatal API error for model {model_name}: {e}")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise StruckdownLLMError(e, prompt_repr, model_name) from e
    except (
        BadRequestError,
        UnsupportedParamsError,
        APIResponseValidationError,
        BudgetExceededError,
    ) as e:
        logger.error(f"Bad request error for model {model_name}: {e}")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise StruckdownLLMError(e, prompt_repr, model_name) from e
    except (
        RateLimitError,
        Timeout,
        UnprocessableEntityError,
        APIConnectionError,
        APIError,
        ServiceUnavailableError,
        InternalServerError,
    ) as e:
        logger.warning(f"Retryable API error for model {model_name}: {e}")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise StruckdownLLMError(e, prompt_repr, model_name) from e
    except Exception as e:
        full_traceback = traceback.format_exc()
        logger.warning(f"Unknown error calling LLM {model_name}: {e}\n{full_traceback}")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise StruckdownLLMError(e, prompt_repr, model_name) from e

    logger.debug(f"\n\n{LC.GREEN}Response: {res}{LC.RESET}\n")

    # for safe pickling
    com_dict = com.model_dump()

    if hasattr(com, "_hidden_params"):
        com_dict["_hidden_params"] = com._hidden_params
        logger.debug(
            f"Preserved _hidden_params with response_cost: {com._hidden_params.get('response_cost') if com._hidden_params else None}"
        )
    else:
        logger.debug("No _hidden_params attribute on completion object")

    # mark with current run ID for cache detection
    com_dict["_run_id"] = get_run_id()
    # store the actual messages sent to the API
    com_dict["_request_messages"] = messages
    return res.model_dump(), com_dict


def structured_chat(
    prompt=None,
    messages=None,
    return_type=None,
    llm: LLM = None,
    credentials: LLMCredentials = None,
    max_retries=3,
    max_tokens=None,
    extra_kwargs=None,
):
    """
    Use instructor to make a tool call to an LLM, returning the `response` field, and a completion object.

    Args:
        prompt: (Deprecated) Single prompt string. Use messages parameter instead.
        messages: List of message dicts with 'role' and 'content' keys.
        return_type: Pydantic model for response structure
        llm: LLM configuration
        credentials: API credentials
        max_retries: Number of retry attempts
        max_tokens: Maximum tokens in response
        extra_kwargs: Additional LLM parameters
    """
    import time

    from struckdown import __version__

    if llm is None:
        llm = LLM()
    if credentials is None:
        credentials = LLMCredentials()

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

    start_time = time.monotonic()
    logger.debug(f"{LC.CYAN}LLM CALL START{call_hint}{LC.RESET}")

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
        )
    except StruckdownLLMError as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.warning(
            f"{LC.RED}LLM CALL FAILED [{elapsed_ms:.0f}ms]{call_hint}: {e}{LC.RESET}"
        )
        raise

    res = return_type.model_validate(res_dict)
    com = Box(com_dict)

    elapsed_ms = (time.monotonic() - start_time) * 1000
    was_cached = com.get("_run_id") != get_run_id()
    cache_status = " (cached)" if was_cached else ""
    logger.debug(was_cached and "Cache hit" or "")
    logger.debug(
        f"{LC.GREEN}LLM CALL DONE [{elapsed_ms:.0f}ms]{cache_status}{call_hint}{LC.RESET}"
    )

    logger.debug(
        f"{LC.PURPLE}Response type: {type(res)}; {len(str(res))} tokens produced{LC.RESET}\n\n"
    )
    return res, com


# Singleton cache for local embedding models
_local_embedding_models = {}


def _get_local_embedding(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
) -> List[List[float]]:
    """Get embeddings using local sentence-transformers model.

    Args:
        texts: List of texts to embed
        model_name: HuggingFace model name (e.g., "all-MiniLM-L6-v2")
        batch_size: Batch size for encoding (default: 32)

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

    if model_name not in _local_embedding_models:
        logger.info(f"Loading local embedding model: {model_name}")
        _local_embedding_models[model_name] = SentenceTransformer(model_name)

    model = _local_embedding_models[model_name]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
        batch_size=batch_size,
    )
    return embeddings.tolist()


def _get_api_embedding_batch(
    batch: List[str],
    model_name: str,
    dimensions: Optional[int],
    api_key: str,
    base_url: Optional[str],
    timeout: int = 60,
) -> List[List[float]]:
    """Get embeddings for a single batch via API (uncached)."""
    logger.debug(f"API embedding batch: {len(batch)} texts, model={model_name}, dims={dimensions}")
    try:
        response = litellm.embedding(
            model=model_name,
            input=list(map(str, batch)),
            dimensions=dimensions,
            api_key=api_key,
            api_base=base_url,
            timeout=timeout,
        )
    except Exception as e:
        raise Exception(f"Error getting embeddings: {e}")
    logger.debug(f"API embedding batch complete: {len(response['data'])} embeddings")
    return [item["embedding"] for item in response["data"]]


@memory.cache
def _get_api_embedding_batch_cached(
    batch: Tuple[str, ...],
    model_name: str,
    dimensions: Optional[int],
) -> List[List[float]]:
    """Get embeddings for a single batch via API (cached).

    Uses tuple for batch to enable joblib caching.
    Credentials fetched fresh each call (not part of cache key).
    """
    logger.debug(f"Cache miss for embedding batch: {len(batch)} texts")
    credentials = LLMCredentials()
    return _get_api_embedding_batch(
        list(batch),
        model_name,
        dimensions,
        credentials.api_key,
        credentials.base_url,
    )


def get_embedding(
    texts: List[str],
    model: Optional[str] = None,
    credentials: Optional[LLMCredentials] = None,
    dimensions: Optional[int] = None,
    batch_size: int = 100,
    max_workers: int = 20,
) -> List[List[float]]:
    """
    Get embeddings for texts using local or API models.

    Supports local sentence-transformers models (prefix with "local/") or
    API-based models via litellm.

    Args:
        texts: List of texts to embed
        model: Model name. Use "local/model-name" for sentence-transformers
               (e.g., "local/all-MiniLM-L6-v2"), or API model name
               (e.g., "text-embedding-3-large"). Defaults to DEFAULT_EMBEDDING_MODEL env var.
        credentials: Optional LLMCredentials for API calls. If provided, bypasses cache.
        dimensions: Embedding dimensions (API models only, model-specific)
        batch_size: Texts per batch (default: 100)
        max_workers: Max parallel API requests (default: 20)

    Returns:
        List of embedding vectors

    Examples:
        # Local embeddings (fast, free)
        get_embedding(texts, model="local/all-MiniLM-L6-v2")

        # API embeddings (better quality)
        get_embedding(texts, model="text-embedding-3-large")

        # API embeddings with custom credentials
        get_embedding(texts, model="text-embedding-3-large", credentials=my_creds)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not texts:
        return []

    # Handle tuple input for backward compatibility
    if isinstance(texts, tuple):
        texts = list(texts)

    # Default model from env
    if model is None:
        model = env_config("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large")

    # Check for local/ prefix
    if model.startswith("local/"):
        local_model_name = model[6:]  # strip "local/"
        logger.debug(f"Using local embeddings: {local_model_name}, {len(texts)} texts")
        return _get_local_embedding(texts, local_model_name, batch_size=batch_size)

    # API embeddings with parallel batching
    logger.debug(f"Using API embeddings: {model}, {len(texts)} texts")

    # Default dimensions for known models (only large model supports 3072)
    if dimensions is None and "text-embedding-3-large" in model:
        dimensions = 3072

    batches = list(chunked(texts, batch_size))

    # Choose batch function: cached (default) or uncached (when credentials provided)
    if credentials is not None:
        # Custom credentials - bypass cache
        def get_batch(batch):
            return _get_api_embedding_batch(
                list(batch), model, dimensions, credentials.api_key, credentials.base_url
            )
    else:
        # Default credentials - use cache
        def get_batch(batch):
            return _get_api_embedding_batch_cached(tuple(batch), model, dimensions)

    # Single batch - no parallelism needed
    if len(batches) == 1:
        return get_batch(batches[0])

    # Multiple batches - process in parallel
    logger.debug(f"Processing {len(batches)} batches with {max_workers} workers")
    results: List[Optional[List[List[float]]]] = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(get_batch, batch): idx
            for idx, batch in enumerate(batches)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    # Flatten in order
    embeddings = []
    for batch_result in results:
        embeddings.extend(batch_result)

    return embeddings
