"""LLM client, credentials, and API interaction for struckdown."""

import json
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from .embedding_cache import clear_embedding_cache, get_cached_embeddings, store_embeddings
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
        # litellm._turn_on_debug()
        
        client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.TOOLS)

        # Truncate validation errors in retry messages to save tokens
        def truncate_validation_errors(**kwargs):
            max_chars = 2000
            for msg in kwargs.get("messages", []):
                content = msg.get("content", "")
                if "validation error" in content.lower() and len(content) > max_chars:
                    msg["content"] = content[:max_chars] + "\n\n... (truncated)"

        client.on(HookName.COMPLETION_KWARGS, truncate_validation_errors)

        # Log errors/retries so users see when calls fail and retry
        def log_completion_error(error, **kwargs):
            logger.warning(f"LLM error, retrying: {type(error).__name__}")

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
    except StruckdownLLMError as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.warning(
            f"{LC.RED}LLM CALL FAILED [{elapsed_ms:.0f}ms]{call_hint}: {e}{LC.RESET}"
        )
        raise
    finally:
        cancel_event.set()  # stop the warning thread

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

    # Check for local/ prefix
    if model.startswith("local/"):
        local_model_name = model[6:]  # strip "local/"
        logger.debug(f"Using local embeddings: {local_model_name}, {len(texts)} texts")
        return _get_local_embedding(texts, local_model_name, batch_size=batch_size)

    # API embeddings
    logger.debug(f"Using API embeddings: {model}, {len(texts)} texts")

    # Default dimensions for known models (only large model supports 3072)
    if dimensions is None and "text-embedding-3-large" in model:
        dimensions = 3072

    # Get credentials (use default if not provided)
    if credentials is None:
        credentials = LLMCredentials()

    # Check cache for existing embeddings
    cached, missing = get_cached_embeddings(texts, model, dimensions)

    # If all cached, return immediately (still call progress callback)
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
    logger.debug(f"Computing {len(missing_texts)} missing embeddings")

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

    # Note: embeddings are cached incrementally in _compute_embeddings_async
    # as each batch completes, so partial progress is preserved on interruption

    # Merge cached and computed embeddings in original order
    for (idx, _), emb in zip(missing, missing_embeddings):
        cached[idx] = emb

    return [cached[i] for i in range(len(texts))]


def get_embedding(
    texts: List[str],
    model: Optional[str] = None,
    credentials: Optional[LLMCredentials] = None,
    dimensions: Optional[int] = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    max_workers: int = MAX_LLM_CONCURRENCY,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[List[float]]:
    """
    Get embeddings for texts using local or API models (sync version).

    For async contexts, use get_embedding_async() instead.

    Supports local sentence-transformers models (prefix with "local/") or
    API-based models via litellm.

    Uses per-string caching via diskcache to avoid redundant API calls.

    Args:
        texts: List of texts to embed
        model: Model name. Use "local/model-name" for sentence-transformers
               (e.g., "local/all-MiniLM-L6-v2"), or API model name
               (e.g., "text-embedding-3-large"). Defaults to DEFAULT_EMBEDDING_MODEL env var.
        credentials: Optional LLMCredentials for API calls.
        dimensions: Embedding dimensions (API models only, model-specific)
        batch_size: Texts per batch (default: 100)
        max_workers: Unused, kept for backward compatibility.
        progress_callback: Optional callback(n) called after each batch with count of items completed

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

    return asyncio.run(
        get_embedding_async(texts, model, credentials, dimensions, batch_size, progress_callback)
    )
