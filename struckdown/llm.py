"""LLM client, credentials, and API interaction for struckdown."""

import json
import logging
import os
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import instructor
import litellm
from instructor.core.hooks import HookName
from instructor.core.exceptions import InstructorRetryException

# Suppress instructor's hook error warnings (we intentionally raise to stop retries)
warnings.filterwarnings(
    "ignore",
    message="Error in completion:error handler:",
    category=UserWarning,
    module="instructor.core.hooks"
)

# Suppress litellm's async cleanup warning -- their atexit handler doesn't properly handle
# all client types (e.g. openai.AsyncOpenAI) and runs after asyncio.run() closes the loop.
# The clients get cleaned up by GC at process exit anyway.
warnings.filterwarnings(
    "ignore",
    message="coroutine 'close_litellm_async_clients' was never awaited",
    category=RuntimeWarning,
)

# Configure litellm to drop unsupported params rather than error
litellm.drop_params = True
# Suppress litellm's verbose error messages (errors still saved in CSV output)
litellm.set_verbose = False
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)

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

# Error categories for caching and retry behavior
# Cacheable: deterministic errors - same input will always fail, safe to cache
CACHEABLE_ERRORS = (
    ContentPolicyViolationError,
    ContextWindowExceededError,
)

# Transient: temporary errors - might succeed on retry, never cache
TRANSIENT_ERRORS = (
    RateLimitError,
    Timeout,
    APIConnectionError,
    ServiceUnavailableError,
    InternalServerError,
)

# Fatal: require config/credential changes, don't cache (user should fix and retry)
FATAL_ERRORS = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BudgetExceededError,
)

from more_itertools import chunked
from pydantic import BaseModel, Field

from .cache import hash_return_type, memory
from .embedding_cache import (
    clear_embedding_cache,
    get_cached_embeddings,
    get_cached_pair_scores,
    store_embeddings,
    store_pair_scores,
)
from .errors import (
    LLMError,
    ContentFilterError,
    RateLimitError as SDRateLimitError,
    ContextWindowError,
    AuthError,
    BadRequestError as SDBadRequestError,
    ConnectionError as SDConnectionError,
)


def _make_struckdown_error(litellm_error: Exception, prompt: str, model_name: str) -> LLMError:
    """Map a litellm exception to the appropriate struckdown error subclass."""
    error_type = type(litellm_error).__name__

    # Content policy violations
    if isinstance(litellm_error, ContentPolicyViolationError):
        return ContentFilterError(litellm_error, prompt, model_name)

    # Context window errors
    if isinstance(litellm_error, ContextWindowExceededError):
        return ContextWindowError(litellm_error, prompt, model_name)

    # Rate limit errors
    if isinstance(litellm_error, (RateLimitError, Timeout)):
        return SDRateLimitError(litellm_error, prompt, model_name)

    # Auth errors
    if isinstance(litellm_error, (AuthenticationError, PermissionDeniedError)):
        return AuthError(litellm_error, prompt, model_name)

    # Connection errors
    if isinstance(litellm_error, (APIConnectionError, ServiceUnavailableError, InternalServerError)):
        return SDConnectionError(litellm_error, prompt, model_name)

    # Bad request errors
    if isinstance(litellm_error, (BadRequestError, UnsupportedParamsError, APIResponseValidationError)):
        return SDBadRequestError(litellm_error, prompt, model_name)

    # Default to base LLMError
    return LLMError(litellm_error, prompt, model_name)


def _make_cached_error(error_class: str, error_msg: str, prompt: str, model_name: str) -> LLMError:
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
        
        client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)

        # Truncate validation errors in retry messages to save tokens
        def truncate_validation_errors(**kwargs):
            max_chars = 2000
            for msg in kwargs.get("messages", []):
                content = msg.get("content", "")
                if "validation error" in content.lower() and len(content) > max_chars:
                    msg["content"] = content[:max_chars] + "\n\n... (truncated)"

        client.on(HookName.COMPLETION_KWARGS, truncate_validation_errors)

        # Log errors/retries so users see when calls fail and retry
        # Re-raise non-retryable errors to stop retry loop
        def log_completion_error(error, **kwargs):
            # don't retry content policy violations - they will always fail
            if isinstance(error, ContentPolicyViolationError):
                logger.debug("Content policy violation - not retrying")
                raise error
            logger.debug(f"LLM error, retrying: {type(error).__name__}")

        client.on(HookName.COMPLETION_ERROR, log_completion_error)

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

    Deterministic errors (content policy, context window) are cached to avoid
    repeated failed API calls. Change seed to force retry.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        cache_version: Version string included in cache key (typically struckdown version)
    """
    mm = str(messages)[:100]
    logger.debug(f"LLM CALL: {mm}")
    logger.debug(f"\n\n{LC.BLUE}Messages: {messages}{LC.RESET}\n\n")
    try:
        call_kwargs = extra_kwargs.copy() if extra_kwargs else {}
        res, com = llm.client(credentials).chat.completions.create_with_completion(
            model=model_name,
            response_model=return_type,
            messages=messages,
            max_retries=max_retries,
            **call_kwargs,
        )
    except CACHEABLE_ERRORS as e:
        # deterministic errors - cache them to avoid repeated failures
        logger.debug(f"Cacheable error for model {model_name} (will be cached): {type(e).__name__}")
        _log_error_details(e, model_name, "(cacheable)")
        return _create_cached_error(e, model_name), None
    except FATAL_ERRORS as e:
        # fatal errors - don't cache, user needs to fix config
        logger.debug(f"Fatal API error for model {model_name}: {e}")
        _log_error_details(e, model_name, "(fatal)")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise _make_struckdown_error(e, prompt_repr, model_name) from e
    except TRANSIENT_ERRORS as e:
        # transient errors - don't cache, might succeed on retry
        logger.debug(f"Transient API error for model {model_name}: {e}")
        _log_error_details(e, model_name, "(transient)")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise _make_struckdown_error(e, prompt_repr, model_name) from e
    except (
        BadRequestError,
        UnsupportedParamsError,
        APIResponseValidationError,
        UnprocessableEntityError,
        APIError,
    ) as e:
        # these may wrap cacheable errors - check before deciding
        _log_error_details(e, model_name, "(bad request / API error)")
        if isinstance(e, CACHEABLE_ERRORS):
            logger.debug(f"Cacheable error for model {model_name} (will be cached): {type(e).__name__}")
            return _create_cached_error(e, model_name), None
        logger.debug(f"Bad request error for model {model_name}: {e}")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise _make_struckdown_error(e, prompt_repr, model_name) from e
    except InstructorRetryException as e:
        # instructor wraps LLM errors after retries; check if cacheable
        _log_error_details(e, model_name, "(instructor retry exhausted)")
        if isinstance(e, CACHEABLE_ERRORS):
            logger.debug(f"Cacheable error for model {model_name} (will be cached): wrapped")
            return _create_cached_error(e, model_name), None
        logger.debug(f"LLM call failed after retries for model {model_name}: {e}")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise _make_struckdown_error(e, prompt_repr, model_name) from e
    except Exception as e:
        # catch-all: check if this is a cacheable error in disguise
        _log_error_details(e, model_name, "(unknown)")
        if isinstance(e, CACHEABLE_ERRORS):
            logger.debug(f"Cacheable error for model {model_name} (will be cached): {type(e).__name__}")
            return _create_cached_error(e, model_name), None
        full_traceback = traceback.format_exc()
        logger.debug(f"Unknown error calling LLM {model_name}: {e}\n{full_traceback}")
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise _make_struckdown_error(e, prompt_repr, model_name) from e

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

    # Warn user if call takes longer than threshold
    import threading

    SLOW_CALL_THRESHOLD = 45  # seconds

    def _slow_call_warning(hint, cancel_event):
        if cancel_event.wait(SLOW_CALL_THRESHOLD):
            return  # call completed before threshold
        logger.warning(f"LLM call still in progress after {SLOW_CALL_THRESHOLD}s...{hint}")

    cancel_event = threading.Event()
    warning_thread = threading.Thread(
        target=_slow_call_warning, args=(call_hint, cancel_event), daemon=True
    )
    warning_thread.start()

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
    except LLMError as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            f"{LC.RED}LLM CALL FAILED [{elapsed_ms:.0f}ms]{call_hint}: {e}{LC.RESET}"
        )
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
        prompt_repr = next((m["content"] for m in messages if m["role"] == "user"), "")
        raise _make_cached_error(error_class, error_msg, prompt_repr, llm.model_name)

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

# Singleton cache for cross-encoder models
_cross_encoder_models = {}


def _get_best_device() -> str:
    """Auto-detect the best available device for local models."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
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
        _local_embedding_models[cache_key] = SentenceTransformer(model_name, device=device)

    model = _local_embedding_models[cache_key]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
        batch_size=batch_size,
    )
    return embeddings.tolist()


async def _get_api_embedding_batch_async(
    batch: List[str],
    model_name: str,
    dimensions: Optional[int],
    api_key: str,
    base_url: Optional[str],
    timeout: int = 60,
) -> List[List[float]]:
    """Get embeddings for a single batch via async API call."""
    logger.debug(f"API embedding batch: {len(batch)} texts, model={model_name}, dims={dimensions}")
    response = await litellm.aembedding(
        model=model_name,
        input=list(map(str, batch)),
        dimensions=dimensions,
        api_key=api_key,
        api_base=base_url,
        timeout=timeout,
    )
    logger.debug(f"API embedding batch complete: {len(response['data'])} embeddings")
    return [item["embedding"] for item in response["data"]]


# Type alias for progress callback: receives count of items just completed
ProgressCallback = Callable[[int], None]


async def _compute_embeddings_async(
    texts: List[str],
    model_name: str,
    dimensions: Optional[int],
    api_key: str,
    base_url: Optional[str],
    batch_size: int,
    progress_callback: Optional[ProgressCallback] = None,
    base_progress: int = 0,
) -> List[List[float]]:
    """Compute embeddings for texts via concurrent async API calls with progress.

    Caches embeddings incrementally as each batch completes, so partial progress
    is preserved if the process is interrupted.

    Args:
        texts: List of texts to embed
        model_name: Model name for API
        dimensions: Embedding dimensions
        api_key: API key
        base_url: API base URL
        batch_size: Texts per batch
        progress_callback: Optional callback(n) called with cumulative count completed
        base_progress: Starting progress count (e.g., from cached items)
    """
    import asyncio

    batches = list(chunked(texts, batch_size))
    # track which texts are in each batch for incremental caching
    batch_texts = list(chunked(texts, batch_size))

    if len(batches) == 1:
        result = await _get_api_embedding_batch_async(
            batches[0], model_name, dimensions, api_key, base_url
        )
        # cache immediately
        store_embeddings(batches[0], result, model_name, dimensions)
        if progress_callback:
            progress_callback(base_progress + len(batches[0]))
        return result

    logger.debug(f"Computing {len(texts)} embeddings in {len(batches)} batches concurrently")

    # track progress across concurrent batches
    completed_count = 0
    progress_lock = asyncio.Lock()

    sem = get_llm_semaphore()

    async def process_batch(batch_idx: int, batch: List[str]) -> tuple:
        nonlocal completed_count
        async with sem:
            result = await _get_api_embedding_batch_async(
                batch, model_name, dimensions, api_key, base_url
            )
            # cache this batch immediately so partial progress is preserved
            store_embeddings(batch, result, model_name, dimensions)
            if progress_callback:
                async with progress_lock:
                    completed_count += len(batch)
                    progress_callback(base_progress + completed_count)
            return batch_idx, result

    # run all batches concurrently, semaphore limits concurrency
    results = await asyncio.gather(*[
        process_batch(idx, batch) for idx, batch in enumerate(batches)
    ])

    # reassemble in original order
    results_ordered = sorted(results, key=lambda x: x[0])
    all_embeddings = []
    for _, batch_result in results_ordered:
        all_embeddings.extend(batch_result)

    return all_embeddings


DEFAULT_EMBEDDING_BATCH_SIZE = env_config("SD_EMBEDDING_BATCH_SIZE", default=100, cast=int)


async def get_embedding_async(
    texts: List[str],
    model: Optional[str] = None,
    credentials: Optional[LLMCredentials] = None,
    dimensions: Optional[int] = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[List[float]]:
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
        progress_callback: Optional callback(n) called after each batch with count of items completed

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    # Handle tuple input for backward compatibility
    if isinstance(texts, tuple):
        texts = list(texts)

    # Default model from env
    if model is None:
        model = env_config("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large")

    is_local = model.startswith("local/")

    # Check cache first -- avoids loading local model if everything is cached
    cached, missing = get_cached_embeddings(texts, model, dimensions)

    # If all cached, return immediately (no model loading needed)
    if not missing:
        logger.debug(f"All {len(texts)} embeddings found in cache")
        if progress_callback:
            progress_callback(len(texts))
        return [cached[i] for i in range(len(texts))]

    # Report cached items as progress
    n_cached = len(texts) - len(missing)
    if n_cached > 0 and progress_callback:
        progress_callback(n_cached)

    # Compute missing embeddings
    missing_texts = [text for _, text in missing]
    logger.debug(f"Computing {len(missing_texts)} missing embeddings ({n_cached} cached)")

    if is_local:
        local_model_name = model[6:]  # strip "local/"
        logger.debug(f"Using local embeddings: {local_model_name}")
        missing_embeddings = _get_local_embedding(missing_texts, local_model_name, batch_size=batch_size)
        # Cache the computed local embeddings
        store_embeddings(missing_texts, missing_embeddings, model, dimensions)
        if progress_callback:
            progress_callback(len(texts))
    else:
        # API embeddings
        logger.debug(f"Using API embeddings: {model}")

        # Get credentials (use default if not provided)
        if credentials is None:
            credentials = LLMCredentials()

        missing_embeddings = await _compute_embeddings_async(
            missing_texts,
            model,
            dimensions,
            credentials.api_key,
            credentials.base_url,
            batch_size,
            progress_callback,
            base_progress=n_cached,
        )
        # Note: API embeddings are cached incrementally in _compute_embeddings_async

    # Merge cached and computed embeddings in original order
    for (idx, _), emb in zip(missing, missing_embeddings):
        cached[idx] = emb

    return [cached[i] for i in range(len(texts))]


def get_embedding(texts: List[str], **kwargs) -> List[List[float]]:
    """
    Get embeddings for texts using local or API models (sync version).

    For async contexts, use get_embedding_async() instead.

    Args:
        texts: List of texts to embed
        **kwargs: Forwarded to get_embedding_async (model, credentials, dimensions, etc.)

    Returns:
        List of embedding vectors

    Raises:
        RuntimeError: If called from within a running async event loop.

    Examples:
        # Local embeddings (fast, free)
        get_embedding(texts, model="local/all-MiniLM-L6-v2")

        # API embeddings (better quality)
        get_embedding(texts, model="text-embedding-3-large")

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

    async def _run_with_cleanup():
        try:
            return await get_embedding_async(texts, **kwargs)
        finally:
            # clean up litellm's async clients before asyncio.run() closes the event loop
            await litellm.close_litellm_async_clients()

    return asyncio.run(_run_with_cleanup())


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
    logger.debug(f"Computing {len(missing_pairs)} missing pair scores ({n_cached} cached)")

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
