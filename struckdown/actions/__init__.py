"""Struckdown actions module.

Actions are custom functions that execute instead of LLM calls.
They are invoked via template syntax: [[@action_name:varname|param1=value1]]

Built-in actions:
    @set        - Set a variable without LLM call
    @break      - Early termination
    @fetch      - Fetch URL content as markdown
    @markdownify - Convert HTML to markdown
    @search     - Web search via DuckDuckGo
    @timestamp  - Get current timestamp

Example usage in templates:
    [[@set:greeting|Hello, World!]]
    [[@fetch:page|url="https://example.com"]]
    [[@search:results|query="python tutorials"]]
"""

import asyncio
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, get_type_hints

from asgiref.sync import sync_to_async

import markdownify as markdownify_lib
import requests
import validators
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field, ValidationError, create_model
from readability import Document

from struckdown.return_type_models import LLMConfig, ResponseModel

MessageRole = Literal["user", "assistant", "system"]


class MessageList(list):
    """Return type for actions that emit multiple messages with different roles.

    When an action returns a MessageList, each message is added to the
    conversation history with its specified role, instead of a single message
    with the default role.

    Example:
        @Actions.register('turns', default_save=False)
        def turns_action(context: dict, filter_type: str = "all") -> MessageList:
            turns = get_turns_from_context(context)
            messages = [
                {"role": "user" if t.is_user else "assistant", "content": t.text}
                for t in turns
            ]
            return MessageList(messages)
    """

    def __init__(self, messages: list[dict]):
        super().__init__(messages)
        for msg in messages:
            if not isinstance(msg, dict):
                raise TypeError(f"MessageList items must be dicts, got {type(msg)}")
            if "content" not in msg:
                raise ValueError("MessageList items must have 'content' key")
            if msg.get("role") not in ("user", "assistant", "system"):
                raise ValueError(f"Invalid role: {msg.get('role')}")

    def __str__(self) -> str:
        """Concatenate all message contents for string representation."""
        return "\n\n".join(m["content"] or "" for m in self)

    def __repr__(self) -> str:
        return f"MessageList({list(self)})"


# optional playwright support
try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# optional magic support for content-type sniffing
try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter Resolution
# =============================================================================


def resolve_option_value(opt, context: dict):
    """Resolve an OptionValue by looking up variable references in context.

    Args:
        opt: OptionValue namedtuple with (key, value, is_variable_ref)
        context: Context dict with extracted variables

    Returns:
        Resolved value (from context if variable_ref, otherwise literal value)
    """
    if not opt.is_variable_ref:
        return opt.value

    # variable reference - look up in context
    var_name = opt.value
    if var_name in context:
        return context[var_name]

    # fallback: warning + treat as literal
    logger.warning(
        f"Variable '{var_name}' not found in context, treating as literal string. "
        f"Available: {list(context.keys())}"
    )
    return var_name


# =============================================================================
# Parameter Coercion
# =============================================================================


def _coerce_params(func: Callable, action_name: str, rendered_params: dict) -> dict:
    """Coerce parameters to match function signature types via Pydantic."""
    try:
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        field_defs = {}

        for param_name, param in sig.parameters.items():
            if param_name == "context":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            param_type = type_hints.get(param_name, str)
            if param.default is inspect.Parameter.empty:
                field_defs[param_name] = (param_type, ...)
            else:
                field_defs[param_name] = (param_type, param.default)

        if not field_defs:
            return rendered_params

        from pydantic import ConfigDict

        CoercionModel = create_model(
            f"{action_name.title()}Params",
            __config__=ConfigDict(extra="allow"),
            **field_defs,
        )
        try:
            return CoercionModel(**rendered_params).model_dump()
        except ValidationError as ve:
            raise ValueError(f"Parameter validation failed for '{action_name}': {ve}") from ve

    except Exception as e:
        logger.debug(f"Type coercion failed for '{action_name}': {e}")
        return rendered_params


# =============================================================================
# Registry
# =============================================================================

ErrorStrategy = Literal["propagate", "return_empty", "return_default"]


class Actions:
    """Registry for custom function-based actions.

    Actions registered here bypass LLM calls and execute custom logic instead.
    Typical use cases: RAG retrieval, database queries, API calls.

    Example:
        @Actions.register('expertise', on_error='return_empty')
        def expertise_search(context, query, n=3):
            # Search database
            return result_text

        # Template usage:
        # [[@expertise:guidance|query="insomnia",n=5]]
    """

    # Registry stores: (func, on_error, default_save, return_type, default, allow_remote_use, role)
    _registry: dict[
        str, tuple[Callable, ErrorStrategy, bool, type | None, Any, bool, MessageRole]
    ] = {}

    @classmethod
    def register(
        cls,
        action_name: str,
        on_error: ErrorStrategy = "propagate",
        default_save: bool = True,
        return_type: type | None = None,
        default: Any = "",
        allow_remote_use: bool = True,
        role: MessageRole = "user",
    ):
        """Decorator to register a function as a custom action.

        Args:
            action_name: Name used in templates, e.g., 'expertise' for [[@expertise:var|...]]
            on_error: How to handle exceptions:
                - 'propagate': Re-raise exception (default)
                - 'return_empty': Return empty string on error
                - 'return_default': Return custom default value on error
            default_save: Whether to save result to context when no variable name specified.
                - True: [[@action]] saves result to context with key 'action' (default)
                - False: [[@action]] doesn't save result, only includes in prompt
                - Note: [[@action:var]] always saves regardless of this setting
            return_type: Pydantic model type returned by this action.
                - Enables automatic deserialization when loading from JSON
                - Example: return_type=FoundEvidenceSet for evidence search
            default: Default value to return when on_error='return_default' (default: "")
                - Only used when on_error='return_default'
                - Can be any type, but should be serializable to string
            allow_remote_use: Whether this action is allowed in remote/hosted mode.
                - True: Action can be used in remote playground (default)
                - False: Action is blocked in remote mode (e.g., fetch, search)
            role: Message role when action output is added to chat history (default: "user")
                - "user": Action output is context/information for the LLM (most common)
                - "assistant": Action output appears as prior LLM response
                - "system": Action output added as system message
                - Ignored if action returns MessageList (which specifies roles per-message)

        Returns:
            Decorator function

        Example:
            @Actions.register('memory', on_error='return_empty', default_save=True, return_type=MemoryResult)
            def search_memory(context, query, n=3):
                return results

            @Actions.register('search', on_error='return_default', default='No results found', allow_remote_use=False)
            def search_db(context, query):
                return database.search(query)

            @Actions.register('turns', on_error='return_empty', default_save=False)
            def get_turns(context, filter_type='all'):
                return formatted_turns  # Output to prompt but don't save
        """

        def decorator(func: Callable) -> Callable:
            cls._registry[action_name] = (
                func,
                on_error,
                default_save,
                return_type,
                default,
                allow_remote_use,
                role,
            )
            logger.debug(
                f"Registered action '{action_name}' with function {func.__name__}, default_save={default_save}, return_type={return_type}, default={default}, allow_remote_use={allow_remote_use}, role={role}"
            )
            return func

        return decorator

    @classmethod
    def create_action_model(
        cls,
        action_name: str,
        options: Optional[list[str]],
        quantifier: Optional[tuple],
        required_prefix: bool,
    ):
        """Create a ResponseModel with custom executor for a registered action.

        This is called by struckdown during template parsing when it encounters
        [[@action_name:varname|options]] syntax.

        Args:
            action_name: The action name from template (e.g., 'expertise')
            options: Parsed options from template (e.g., ['query=x', 'n=3'])
            quantifier: Not used for custom actions
            required_prefix: Not used for custom actions

        Returns:
            ResponseModel class with _executor and _is_function attributes,
            or None if action_name not registered
        """
        if action_name not in cls._registry:
            return None

        (
            func,
            on_error,
            default_save,
            return_type,
            default_value,
            allow_remote_use,
            role,
        ) = cls._registry[action_name]

        # create response model
        class ActionResult(ResponseModel):
            """Result from custom action"""

            response: Any = Field(
                default="", description=f"Result from {action_name} action"
            )

        async def executor(context: dict, rendered_prompt: str, **kwargs):
            """Async executor that calls the registered function.

            Automatically wraps sync functions with sync_to_async for safe
            execution in async contexts (e.g., Django ORM operations).
            """
            from struckdown.parsing import OptionValue

            def ensure_option_value(opt):
                if isinstance(opt, str):
                    if "=" in opt:
                        key, value = opt.split("=", 1)
                        key, value = key.strip(), value.strip()
                    else:
                        key, value = None, opt.strip()
                    is_var_ref = value.startswith("{{") and value.endswith("}}")
                    if is_var_ref:
                        value = value[2:-2].strip()
                    return OptionValue(key=key, value=value, is_variable_ref=is_var_ref)
                return opt

            normalized_options = [ensure_option_value(opt) for opt in (options or [])]

            sig = inspect.signature(func)
            param_names = [n for n in sig.parameters.keys() if n != "context"]

            positional_opts = [opt for opt in normalized_options if opt.key is None]
            keyed_opts = [opt for opt in normalized_options if opt.key is not None]

            rendered_params = {}
            for i, opt in enumerate(positional_opts):
                if i < len(param_names):
                    rendered_params[param_names[i]] = resolve_option_value(opt, context)
                else:
                    logger.warning(
                        f"Action '{action_name}': too many positional args "
                        f"(expected {len(param_names)}, got {len(positional_opts)})"
                    )
                    break

            for opt in keyed_opts:
                rendered_params[opt.key] = resolve_option_value(opt, context)

            logger.debug(f"Executor '{action_name}': params={rendered_params}")

            # type coercion via pydantic
            coerced_params = _coerce_params(func, action_name, rendered_params)

            # call function, wrapping sync functions automatically
            try:
                if asyncio.iscoroutinefunction(func):
                    result_text = await func(context=context, **coerced_params)
                else:
                    result_text = await sync_to_async(func, thread_sensitive=True)(
                        context=context, **coerced_params
                    )
                action_result = ActionResult(response=result_text)
                action_result._resolved_params = coerced_params
                return action_result, None

            except Exception as e:
                if on_error == "propagate":
                    raise
                logger.warning(f"Action '{action_name}' failed: {e}")
                fallback = "" if on_error == "return_empty" else default_value
                return ActionResult(response=fallback), None

        # attach executor and metadata to response model
        ActionResult._executor = executor
        ActionResult._is_function = True
        ActionResult._default_save = default_save
        ActionResult._role = role

        return ActionResult

    @classmethod
    def list_registered(cls) -> list[str]:
        """List all registered action names.

        Returns:
            List of action names
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, action_name: str) -> bool:
        """Check if an action is registered.

        Args:
            action_name: Action name to check

        Returns:
            True if registered, False otherwise
        """
        return action_name in cls._registry

    @classmethod
    def get_default_save(cls, action_name: str) -> bool:
        """Get the default_save setting for a registered action.

        Args:
            action_name: Action name to check

        Returns:
            default_save value (True if not found, for backward compatibility)
        """
        if action_name not in cls._registry:
            return True  # default to saving if action not found
        return cls._registry[action_name][2]  # third element is default_save

    @classmethod
    def get_return_type(cls, action_name: str) -> type | None:
        """Get the return type for a registered action.

        Args:
            action_name: Action name to check

        Returns:
            Return type (Pydantic model class) or None if not specified
        """
        if action_name not in cls._registry:
            return None
        return cls._registry[action_name][3]  # fourth element is return_type

    @classmethod
    def get_registered_types(cls) -> list[type]:
        """Get all unique return types registered across all actions.

        Returns:
            List of Pydantic model classes registered as return types
        """
        types = []
        for (
            func,
            on_error,
            default_save,
            return_type,
            default_value,
            allow_remote_use,
            role,
        ) in cls._registry.values():
            if return_type is not None and return_type not in types:
                types.append(return_type)
        return types

    @classmethod
    def get_role(cls, action_name: str) -> MessageRole:
        """Get the message role for a registered action.

        Args:
            action_name: Action name to check

        Returns:
            Role string ("user", "assistant", or "system"). Defaults to "user".
        """
        if action_name not in cls._registry:
            return "user"
        return cls._registry[action_name][6]  # seventh element is role

    @classmethod
    def is_allowed_remote(cls, action_name: str) -> bool:
        """Check if an action is allowed in remote/hosted mode.

        Args:
            action_name: Action name to check

        Returns:
            True if allowed in remote mode, False otherwise
        """
        if action_name not in cls._registry:
            return False  # unknown actions not allowed in remote
        return cls._registry[action_name][5]  # sixth element is allow_remote_use

    @classmethod
    def get_remote_allowed_actions(cls) -> list[str]:
        """Get list of action names allowed in remote mode.

        Returns:
            List of action names with allow_remote_use=True
        """
        return [
            name
            for name, reg in cls._registry.items()
            if reg[5]  # sixth element is allow_remote_use
        ]


# =============================================================================
# Loader functions
# =============================================================================


def load_action_file(path: Path) -> list[str]:
    """Load a Python file and register any @Actions.register decorated functions.

    Args:
        path: Path to Python file containing action definitions

    Returns:
        List of registered action names from this file

    Raises:
        ImportError: If the file cannot be imported
        FileNotFoundError: If the file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Action file not found: {path}")

    if not path.suffix == ".py":
        logger.warning(f"Skipping non-Python file: {path}")
        return []

    # track which actions exist before loading
    actions_before = set(Actions.list_registered())

    # import the module
    module_name = f"struckdown_actions_{path.stem}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # execute the module (this triggers @Actions.register decorators)
        spec.loader.exec_module(module)

        # find newly registered actions
        actions_after = set(Actions.list_registered())
        new_actions = actions_after - actions_before

        if new_actions:
            logger.info(f"Loaded actions from {path}: {sorted(new_actions)}")
        else:
            logger.debug(f"No new actions registered from {path}")

        return sorted(new_actions)

    except Exception as e:
        logger.error(f"Failed to load actions from {path}: {e}")
        raise


def load_action_directory(directory: Path) -> list[str]:
    """Load all Python files from a directory.

    Args:
        directory: Directory to scan for .py files

    Returns:
        List of all registered action names
    """
    if not directory.is_dir():
        logger.warning(f"Not a directory: {directory}")
        return []

    all_actions = []

    for py_file in sorted(directory.glob("*.py")):
        # skip __init__.py and __pycache__
        if py_file.name.startswith("_"):
            continue

        try:
            actions = load_action_file(py_file)
            all_actions.extend(actions)
        except Exception as e:
            logger.error(f"Error loading {py_file}: {e}")

    return all_actions


def load_actions(paths: list[Path]) -> list[str]:
    """Load actions from files or directories.

    Args:
        paths: List of Python files or directories to load

    Returns:
        List of registered action names
    """
    all_actions = []

    for path in paths:
        if path.is_dir():
            actions = load_action_directory(path)
            all_actions.extend(actions)
        elif path.is_file():
            actions = load_action_file(path)
            all_actions.extend(actions)
        else:
            logger.warning(f"Path not found: {path}")

    return all_actions


def discover_actions(
    template_path: Path | None = None,
    cwd: Path | None = None,
) -> list[str]:
    """Auto-discover and load actions from conventional locations.

    Discovery order:
    1. actions/ relative to template file
    2. actions/ in current working directory

    Note: Built-in actions (set, break, fetch, search, etc.) are registered
    automatically when struckdown.actions is imported -- they don't need
    discovery. This function is for user-defined custom actions.

    Args:
        template_path: Path to the .sd template file
        cwd: Current working directory (defaults to Path.cwd())

    Returns:
        List of registered action names
    """
    if cwd is None:
        cwd = Path.cwd()

    all_actions = []
    locations_searched = []

    # Note: We skip the package's own actions/ directory because:
    # - Built-in actions use relative imports (from . import Actions)
    # - They're already registered via imports in struckdown/actions/__init__.py
    # - Dynamic loading via importlib fails with "relative import with no known parent"

    # 1. actions/ relative to template
    if template_path and template_path.parent.is_dir():
        actions_dir = template_path.parent / "actions"
        if actions_dir.is_dir() and actions_dir not in locations_searched:
            actions = load_action_directory(actions_dir)
            all_actions.extend(actions)
            locations_searched.append(actions_dir)

    # 2. actions/ in cwd
    cwd_actions = cwd / "actions"
    if cwd_actions.is_dir() and cwd_actions not in locations_searched:
        actions = load_action_directory(cwd_actions)
        all_actions.extend(actions)
        locations_searched.append(cwd_actions)

    if locations_searched:
        logger.debug(f"Searched for actions in: {locations_searched}")

    return all_actions


# backwards compatibility aliases
load_tools_file = load_action_file
load_tools_directory = load_action_directory
load_tools = load_actions
discover_tools = discover_actions


# =============================================================================
# Utils -- URL fetching, HTML conversion
# =============================================================================

DEFAULT_MAX_CHARS = 32000 * 2
DEFAULT_TIMEOUT = int(os.environ.get("STRUCKDOWN_WEB_FETCH_TIMEOUT", "8"))
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:132.0) Gecko/20100101 Firefox/132.0"


def sniff_content_type(content: bytes | str) -> str:
    """Sniff content type from content using magic library if available.

    Args:
        content: Content to sniff (bytes or str)

    Returns:
        MIME type string (e.g. "text/html") or empty string if unable to detect
    """
    if not MAGIC_AVAILABLE:
        return ""
    try:
        if isinstance(content, str):
            content = content.encode("utf-8", errors="ignore")
        return magic.from_buffer(content, mime=True).lower()
    except Exception:
        return ""


def normalise_url(s: str) -> str:
    """Normalise URL, adding https:// if no scheme provided."""
    s = s.strip()
    if not s:
        return s

    # if no scheme, add https://
    if not s.startswith(("http://", "https://")):
        s = "https://" + s

    return s


def is_url(s: str) -> bool:
    """Check if string looks like a URL (must have http:// or https:// scheme)."""
    s = s.strip()
    if not s.startswith(("http://", "https://")):
        return False
    return validators.url(s) is True


def fetch_url_playwright(
    url: str, timeout: int = DEFAULT_TIMEOUT, user_agent: str | None = None
) -> tuple[str, str]:
    """Fetch content from URL using Playwright (headless browser).

    Requires playwright optional dependency:
        pip install struckdown[playwright]
        playwright install chromium

    Returns:
        Tuple of (content, content_type)
    """
    from struckdown.errors import StruckdownFetchError

    if not PLAYWRIGHT_AVAILABLE:
        raise StruckdownFetchError(
            url,
            "Playwright not installed. Install with: pip install struckdown[playwright] && playwright install chromium",
        )

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=user_agent or DEFAULT_USER_AGENT,
            )
            page = context.new_page()
            response = page.goto(url, timeout=timeout * 1000)  # playwright uses ms
            page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            html = page.content()
            ctype = response.headers.get("content-type", "").lower() if response else ""
            # fallback to magic sniffing if header missing or ambiguous
            if not ctype or ctype in ("application/octet-stream", "text/plain"):
                sniffed = sniff_content_type(html)
                if sniffed:
                    ctype = sniffed
            browser.close()
            return html, ctype
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            raise StruckdownFetchError(url, f"request timed out after {timeout}s")
        raise StruckdownFetchError(url, error_msg)


def fetch_url(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    user_agent: str | None = None,
    playwright: bool = False,
) -> tuple[str, str]:
    """Fetch content from URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        user_agent: Custom user agent string
        playwright: If True, use Playwright; if False, use requests with
            automatic fallback to Playwright on 403/401 errors

    Returns:
        Tuple of (content, content_type)
    """
    from struckdown.errors import StruckdownFetchError

    # use playwright directly if requested
    if playwright:
        return fetch_url_playwright(url, timeout, user_agent)

    # try requests first
    headers = {"User-Agent": user_agent or DEFAULT_USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        logger.debug(f"Fetched content:\n{response.text}")
        ctype = response.headers.get("Content-Type", "").lower()
        # fallback to magic sniffing if header missing or ambiguous
        if not ctype or ctype in ("application/octet-stream", "text/plain"):
            sniffed = sniff_content_type(response.content)
            if sniffed:
                ctype = sniffed
        return response.text, ctype
    except requests.exceptions.ConnectionError:
        raise StruckdownFetchError(url, "could not connect (check the URL is correct)")
    except requests.exceptions.Timeout:
        raise StruckdownFetchError(url, f"request timed out after {timeout}s")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        # fallback to playwright on 403/401 (bot detection)
        if status in (401, 403) and PLAYWRIGHT_AVAILABLE:
            logger.info(f"HTTP {status} from {url}, retrying with Playwright...")
            return fetch_url_playwright(url, timeout, user_agent)
        raise StruckdownFetchError(url, f"HTTP {status}")
    except requests.exceptions.RequestException as e:
        raise StruckdownFetchError(url, str(e))


def extract_readable(html: str) -> str:
    """Extract main content from HTML using readability."""
    doc = Document(html)
    return doc.summary()


def html_to_markdown(html: str) -> str:
    """Convert HTML to markdown."""
    mymd = markdownify_lib.markdownify(html, heading_style="ATX")
    logger.info(f"Converted HTML to markdown:")
    logger.debug(mymd)
    return mymd


def truncate_content(content: str, max_chars: int) -> str:
    """Truncate content to max_chars if specified."""
    if max_chars <= 0:
        return content
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[... content truncated ...]"


def fetch_and_parse(
    url: str,
    raw: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    max_chars: int = DEFAULT_MAX_CHARS,
    playwright: bool = False,
) -> str:
    """
    Fetch URL and return content.

    Args:
        url: URL to fetch (with or without scheme)
        raw: If True, return raw content; if False and HTML, extract readable as markdown
        timeout: Request timeout in seconds
        max_chars: Maximum characters to return (0 = no limit)
        playwright: If True, use Playwright browser; if False, use requests
            with automatic fallback to Playwright on 403/401 errors

    Returns:
        Content string (raw content, or cleaned markdown if HTML)
    """
    url = normalise_url(url)

    if not is_url(url):
        raise ValueError(f"Invalid URL: {url}")

    content, ctype = fetch_url(url, timeout=timeout, playwright=playwright)

    # only process as HTML if content-type indicates HTML and raw=False
    if raw or "html" not in ctype:
        return truncate_content(content, max_chars)

    readable_html = extract_readable(content)
    markdown = html_to_markdown(readable_html)
    return truncate_content(markdown, max_chars)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # main registry
    "Actions",
    "MessageList",
    # loader functions
    "load_action_file",
    "load_action_directory",
    "load_actions",
    "discover_actions",
    # backwards compat aliases
    "load_tools_file",
    "load_tools_directory",
    "load_tools",
    "discover_tools",
    # utilities
    "normalise_url",
    "is_url",
    "fetch_url",
    "fetch_url_playwright",
    "extract_readable",
    "html_to_markdown",
    "truncate_content",
    "fetch_and_parse",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_USER_AGENT",
    "PLAYWRIGHT_AVAILABLE",
    "MAGIC_AVAILABLE",
    "sniff_content_type",
]


# =============================================================================
# Import built-in actions to register them
# (the @Actions.register decorators run on import)
# =============================================================================

from . import (break_, evidence, fetch, history, markdownify, search, set_,
               timestamp)

# =============================================================================
# Built-in noop action for unknown/unregistered actions
# =============================================================================


@Actions.register("noop", on_error="return_empty", default_save=False)
def noop_action(**kwargs) -> str:
    """No-op action that returns empty string. Used for unknown actions."""
    return ""
