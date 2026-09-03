"""Turn ``thinking=`` into a reasoning setting the model will actually accept.

struckdown speaks one vocabulary (``off``/``minimal``/``low``/``medium``/
``high``/``xhigh``); pydantic-ai speaks another (``False`` plus the same
levels) and translates that per provider. Two things fall through the gap:

* ``off`` is not a pydantic-ai level at all, so it reached
  ``OPENAI_REASONING_EFFORT_MAP`` as an unknown key;
* the map is static, and which value means "as little as possible" depends
  on the model generation. gpt-5.0 and the o-series take ``minimal``;
  gpt-5.6 rejects it -- *Supported values are: 'none', 'low', 'medium',
  'high', 'xhigh'* -- and wants ``none``. pydantic-ai decides that from a
  prefix list (``gpt-5.1`` ... ``gpt-5.4``), which cannot know a model
  released after it, and behind a proxy the model is an alias anyway.

So the value is negotiated rather than predicted. On a 400 naming
``reasoning_effort``, :func:`learn_from_error` reads the supported set out
of the error, remembers the cheapest workable value for that model and the
call is retried once. Only the first call after a model switch pays for it.
"""

import json
import logging
import re
import threading
from typing import Optional, Union

logger = logging.getLogger(__name__)

# What struckdown's ``thinking=`` accepts, mapped to a pydantic-ai
# ThinkingLevel (bool | ThinkingEffort). Levels pass through untouched.
_ALIASES = {
    "off": False,
    "false": False,
    "no": False,
    "on": True,
    "true": True,
    "yes": True,
}

# Values that mean "as little reasoning as possible", cheapest first. Which
# of them a model accepts is what gets negotiated.
OFF_CANDIDATES = ("none", "minimal", "low")

# Accepted by every reasoning generation so far. Used only when a call site
# cannot retry.
SAFE_OFF = "low"

_SUPPORTED_RE = re.compile(r"[Ss]upported values are:\s*(.+?)(?:\.|$)")
_QUOTED_RE = re.compile(r"['\"]([a-z]+)['\"]")
_REJECTED_RE = re.compile(
    r"does not support[^'\"]*['\"]([a-z]+)['\"]", re.IGNORECASE
)

_learned: dict = {}
_lock = threading.Lock()
_loaded = False


def resolve_thinking(value) -> Optional[Union[bool, str]]:
    """struckdown's ``thinking=`` value as a pydantic-ai ThinkingLevel.

    ``None`` means "say nothing", leaving the model's own default alone.
    """
    if value is None or isinstance(value, bool):
        return value
    return _ALIASES.get(str(value).strip().lower(), value)


def _cache_path():
    from struckdown.cache import get_cache_dir

    cache_dir = get_cache_dir()
    return cache_dir / "reasoning_effort.json" if cache_dir else None


def _load() -> None:
    """Read the learned map once per process.

    Kept on disk beside the response cache so a fresh worker doesn't have to
    spend a failed request rediscovering what the last one learned.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    path = _cache_path()
    if path and path.exists():
        try:
            _learned.update(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            logger.debug(f"Could not read {path}: {exc}")


def learned(model_name: str) -> Optional[str]:
    """The effort value this model was last seen to accept, if any."""
    with _lock:
        _load()
        return _learned.get((model_name or "").lower())


def remember(model_name: str, effort: str) -> None:
    with _lock:
        _load()
        _learned[(model_name or "").lower()] = effort
        path = _cache_path()
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(_learned, indent=2, sort_keys=True))
            except OSError as exc:
                logger.debug(f"Could not write {path}: {exc}")


def forget(model_name: Optional[str] = None) -> None:
    """Drop what was learned -- one model, or all of them."""
    with _lock:
        _load()
        if model_name is None:
            _learned.clear()
        else:
            _learned.pop((model_name or "").lower(), None)
        path = _cache_path()
        if path and path.exists():
            try:
                path.write_text(json.dumps(_learned, indent=2, sort_keys=True))
            except OSError as exc:
                logger.debug(f"Could not write {path}: {exc}")


def off_effort(model_name: str, *, negotiable: bool = False) -> str:
    """Lowest reasoning effort to ask ``model_name`` for.

    ``negotiable=True`` means the caller will retry on rejection, so it gets
    the optimistic value; everyone else gets the one that always works.
    """
    return learned(model_name) or (OFF_CANDIDATES[0] if negotiable else SAFE_OFF)


def supported_values(message: str) -> list:
    """Effort names quoted in a provider's "supported values" complaint."""
    match = _SUPPORTED_RE.search(message or "")
    return _QUOTED_RE.findall(match.group(1)) if match else []


def rejected_value(message: str) -> Optional[str]:
    """The effort value named as invalid in a provider error, if present."""
    match = _REJECTED_RE.search(message or "")
    return match.group(1).lower() if match else None


def rejects_effort(message: str) -> bool:
    """Does this error blame ``reasoning_effort``?"""
    text = (message or "").lower()
    return "reasoning_effort" in text and ("unsupported" in text or "supported values" in text)


def learn_from_error(model_name: str, message: str, tried=None) -> Optional[str]:
    """Pick and remember the cheapest replacement offered by the provider."""
    if not rejects_effort(message):
        return None

    tried = rejected_value(message) or tried
    offered = supported_values(message)
    if not offered:
        return None
    choices = [candidate for candidate in OFF_CANDIDATES if candidate in offered]
    replacement = next((candidate for candidate in choices if candidate != tried), None)
    if replacement is None:
        return None

    remember(model_name, replacement)
    logger.info(
        f"reasoning_effort={tried!r} rejected for model={model_name}; "
        f"using {replacement!r} from now on"
    )
    return replacement
