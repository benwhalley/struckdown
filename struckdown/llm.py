"""LLM client, credentials, and API interaction for struckdown."""

import json
import logging
import traceback
from typing import Any, Dict, List, Optional

import instructor
import litellm
from instructor.core.hooks import HookName

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
from box import Box
from decouple import config as env_config
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    AuthenticationError,
    BadRequestError,
    BudgetExceededError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
    UnsupportedParamsError,
)
from more_itertools import chunked
from pydantic import BaseModel, Field

import anyio

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
        litellm.api_key = credentials.api_key
        litellm.api_base = credentials.base_url
        litellm.drop_params = True
        client = instructor.from_litellm(litellm.completion)

        # Attach debug hook if enabled
        if _debug_api_requests:
            def log_api_request(**kwargs):
                """Log the full API request as JSON."""
                import sys
                print("\n" + "="*80, file=sys.stderr)
                print("API REQUEST", file=sys.stderr)
                print("="*80, file=sys.stderr)
                print(json.dumps(kwargs, indent=2, default=str), file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)

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
    logger.info(f"\n\n{LC.BLUE}Messages: {messages}{LC.RESET}\n\n")
    try:
        res, com = llm.client(credentials).chat.completions.create_with_completion(
            model=model_name,
            response_model=return_type,
            messages=messages,
            **(extra_kwargs if extra_kwargs else {}),
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

    logger.info(f"\n\n{LC.GREEN}Response: {res}{LC.RESET}\n")

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
            call_hint = f" ({content}...)" if len(msg.get("content", "")) > 50 else f" ({content})"
            break

    logger.debug(
        f"Using model {llm.model_name}, max_retries {max_retries}, max_tokens: {max_tokens}"
    )
    logger.debug(f"LLM kwargs: {extra_kwargs}")

    start_time = time.monotonic()
    logger.info(f"{LC.CYAN}LLM CALL START{call_hint}{LC.RESET}")

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

        res = return_type.model_validate(res_dict)
        com = Box(com_dict)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        was_cached = com.get("_run_id") != get_run_id()
        cache_status = " (cached)" if was_cached else ""
        logger.info(f"{LC.GREEN}LLM CALL DONE [{elapsed_ms:.0f}ms]{cache_status}{call_hint}{LC.RESET}")

        logger.debug(
            f"{LC.PURPLE}Response type: {type(res)}; {len(str(res))} tokens produced{LC.RESET}\n\n"
        )
        return res, com

    except (EOFError, Exception) as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.warning(f"{LC.RED}LLM CALL FAILED [{elapsed_ms:.0f}ms]{call_hint}: {e}{LC.RESET}")
        raise e


def get_embedding(
    texts: List[str],
    llm: LLM = None,
    credentials: LLMCredentials = None,
    dimensions: Optional[int] = 3072,
    batch_size: int = 500,
) -> List[List[float]]:
    """
    Get embeddings for a list of texts using litellm directly.
    """
    if llm is None:
        llm = LLM(model_name=env_config("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large"))
    if credentials is None:
        credentials = LLMCredentials()

    api_key = credentials.api_key
    base_url = credentials.base_url

    embeddings = []
    for batch in chunked(texts, batch_size):
        logger.debug(f"Getting batch of embeddings:\n{texts}")
        try:
            response = litellm.embedding(
                model=llm.model_name,
                input=list(map(str, batch)),
                dimensions=dimensions,
                api_key=api_key,
                api_base=base_url,
            )
        except Exception as e:
            raise Exception(f"Error getting embeddings: {e}")

        embeddings.extend(item["embedding"] for item in response["data"])

    return embeddings
