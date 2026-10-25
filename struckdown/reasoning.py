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

A named level is negotiated the same way. pydantic-ai drops ``minimal`` only
for gpt-5.6, so it still sends it to gpt-5.4, which refuses it.
:func:`learn_level_from_error` picks the nearest level the model offers
(``minimal`` -> ``low``, ``xhigh`` -> ``high``; never ``none``, which would
turn thinking off) and remembers it for that model and level.
"""

import json
import logging
import re
import threading
from typing import Any, Optional, Union

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

# Effort levels, least reasoning first: "nearest" is measured along this.
LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh")

# What to try when a provider refuses a level without saying what it takes:
# the ends of the scale are the levels that some generations lack.
LEVEL_FALLBACK = {"minimal": "low", "xhigh": "high"}

# The parameter an OpenAI-style error body names in its "param" field.
EFFORT_PARAMS = ("reasoning_effort", "reasoning.effort")

# a quote, optionally backslash-escaped: the error usually arrives as the repr of
# the response body, where 'minimal' is printed as \'minimal\'
_Q = r"\\?['\"]"
_SUPPORTED_RE = re.compile(r"[Ss]upported values are:\s*(.+?)(?:\.|$)")
_QUOTED_RE = re.compile(_Q + r"([a-z]+)" + _Q)
_REJECTED_RES = (
    # "'reasoning_effort' does not support 'minimal'"
    re.compile(r"does not support[^'\"]*" + _Q + r"([a-z]+)" + _Q, re.IGNORECASE),
    # "Unsupported value: 'minimal' is not supported with the ... model"
    re.compile(r"unsupported value:\s*" + _Q + r"([a-z]+)" + _Q, re.IGNORECASE),
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
    for pattern in _REJECTED_RES:
        match = pattern.search(message or "")
        if match:
            return match.group(1).lower()
    return None


def rejects_effort(message: str) -> bool:
    """Does this error blame the reasoning effort?

    Chat Completions calls the parameter ``reasoning_effort``; the Responses
    API (and Azure behind LiteLLM) ``reasoning.effort``.
    """
    text = (message or "").lower()
    named = "reasoning_effort" in text or "reasoning.effort" in text
    return named and ("unsupported" in text or "supported values" in text)


def _names_effort_param(body: Any) -> bool:
    """Does a structured error body anywhere give ``param`` as the effort?"""
    if isinstance(body, dict):
        return body.get("param") in EFFORT_PARAMS or any(
            _names_effort_param(v) for v in body.values()
        )
    if isinstance(body, list):
        return any(_names_effort_param(v) for v in body)
    return False


def blames_effort(message: str, body: Any = None) -> bool:
    """Does this error blame the reasoning effort, by its wording or its ``param``?"""
    return rejects_effort(message) or _names_effort_param(body)


def _unresolved(model_name: str, asked, message: str) -> None:
    logger.warning(
        f"reasoning effort {asked!r} refused by model={model_name} and no replacement "
        f"could be read from the error: {(message or '')[:500]}"
    )


def learn_from_error(
    model_name: str, message: str, tried=None, body: Any = None
) -> Optional[str]:
    """Pick and remember the cheapest replacement offered by the provider.

    An error that blames the effort without listing what the model takes falls
    back to ``SAFE_OFF``.
    """
    if not blames_effort(message, body):
        return None

    tried = rejected_value(message) or tried
    offered = supported_values(message) or [SAFE_OFF]
    choices = [candidate for candidate in OFF_CANDIDATES if candidate in offered]
    replacement = next((candidate for candidate in choices if candidate != tried), None)
    if replacement is None:
        _unresolved(model_name, tried, message)
        return None

    remember(model_name, replacement)
    logger.info(
        f"reasoning_effort={tried!r} rejected for model={model_name}; "
        f"using {replacement!r} from now on"
    )
    return replacement


def _level_key(model_name: str, level: str) -> str:
    return f"{(model_name or '').lower()}@{level}"


def learned_level(model_name: str, level: str) -> Optional[str]:
    """The level this model was last seen to accept in place of ``level``."""
    with _lock:
        _load()
        return _learned.get(_level_key(model_name, level))


def nearest_level(level: str, offered) -> Optional[str]:
    """The offered level closest to ``level``, a tie going to more reasoning.

    Never ``none``: a caller who asked for some thinking should not have it
    turned off.
    """
    if level not in LEVELS:
        return None
    usable = [v for v in offered if v in LEVELS and v not in ("none", level)]
    if not usable:
        return None
    asked = LEVELS.index(level)
    return min(usable, key=lambda v: (abs(LEVELS.index(v) - asked), -LEVELS.index(v)))


def learn_level_from_error(
    model_name: str, message: str, level: str, body: Any = None
) -> Optional[str]:
    """Pick and remember the nearest level to ``level`` that the model offers.

    An error that blames the effort without listing what the model takes falls
    back to ``LEVEL_FALLBACK``.
    """
    if not blames_effort(message, body):
        return None
    offered = supported_values(message)
    replacement = nearest_level(level, offered) if offered else LEVEL_FALLBACK.get(level)
    if replacement is None:
        _unresolved(model_name, level, message)
        return None
    remember(_level_key(model_name, level), replacement)
    logger.info(
        f"thinking={level!r} rejected for model={model_name}; "
        f"using {replacement!r} from now on"
    )
    return replacement
