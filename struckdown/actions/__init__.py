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

import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, get_type_hints

import markdownify as markdownify_lib
import requests
import validators
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field, ValidationError, create_model
from readability import Document

from struckdown.return_type_models import LLMConfig, ResponseModel

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

    # Registry stores: (func, on_error, default_save, return_type, default, allow_remote_use)
    _registry: dict[str, tuple[Callable, ErrorStrategy, bool, type | None, Any, bool]] = {}

    @classmethod
    def register(
        cls,
        action_name: str,
        on_error: ErrorStrategy = "propagate",
        default_save: bool = True,
        return_type: type | None = None,
        default: Any = "",
        allow_remote_use: bool = True,
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
            )
            logger.debug(
                f"Registered action '{action_name}' with function {func.__name__}, default_save={default_save}, return_type={return_type}, default={default}, allow_remote_use={allow_remote_use}"
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

        func, on_error, default_save, return_type, default_value, allow_remote_use = cls._registry[
            action_name
        ]

        # create response model
        class ActionResult(ResponseModel):
            """Result from custom action"""

            response: Any = Field(
                default="", description=f"Result from {action_name} action"
            )

        def executor(context: dict, rendered_prompt: str, **kwargs):
            """Generic executor that calls the registered function.

            Args:
                context: Accumulated context dict (all extracted variables)
                rendered_prompt: Rendered prompt text (not used for actions)
                **kwargs: Additional parameters from struckdown

            Returns:
                (ActionResult, None): Result and completion object
            """
            # parse options to dict, handling both positional and keyword arguments
            # Example: [[@evidence|"CBT",3,types="techniques"]]
            #   positional: ["CBT", 3]
            #   keyword: {types: "techniques"}

            positional_args = []
            keyword_args = {}

            for opt in options or []:
                if "=" in opt:
                    # keyword argument
                    key, value = opt.split("=", 1)
                    keyword_args[key.strip()] = value.strip()
                else:
                    # positional argument
                    positional_args.append(opt.strip())

            # get function signature to map positional args to parameter names
            sig = inspect.signature(func)
            param_names = [
                name
                for name in sig.parameters.keys()
                if name != "context"  # skip context parameter
            ]

            # map positional args to parameter names
            params = {}
            for i, value in enumerate(positional_args):
                if i < len(param_names):
                    params[param_names[i]] = value
                else:
                    logger.warning(
                        f"Action '{action_name}' received too many positional args. "
                        f"Expected at most {len(param_names)}, got {len(positional_args)}"
                    )
                    break

            # merge keyword args (they override positional if there's a conflict)
            params.update(keyword_args)

            # debug logging to understand context state
            logger.debug(f"Executor called for action '{action_name}'")
            logger.debug(f"Context keys available: {list(context.keys())}")
            logger.debug(
                f"Positional args: {positional_args}, Keyword args: {keyword_args}"
            )
            logger.debug(f"Mapped params to render: {params}")

            # render Jinja2 variables in parameter values
            # this allows: query={{extracted_var}} in templates
            rendered_params = {}
            for k, v in params.items():
                try:
                    # try rendering as Jinja2 template with StrictUndefined
                    # this will error if variables are missing instead of silently rendering to ''
                    rendered_value = Template(str(v), undefined=StrictUndefined).render(
                        **context
                    )
                    logger.debug(f"Rendered '{k}': '{v}' → '{rendered_value}'")
                    rendered_params[k] = rendered_value
                except Exception as e:
                    # fallback: try variable lookup, or use literal value
                    logger.warning(
                        f"Jinja2 rendering failed for action '{action_name}' parameter '{k}={v}': {e}. "
                        f"Available context keys: {list(context.keys())}. "
                        f"Keeping unresolved value."
                    )
                    # Keep the original value (with template syntax) so it's visible that it failed
                    rendered_params[k] = v

            # automatic type coercion based on function signature
            try:
                # get type hints from the function
                type_hints = get_type_hints(func)

                # build field definitions for Pydantic model (skip 'context' parameter)
                # (reuse sig from positional arg parsing above)
                field_defs = {}

                for param_name, param in sig.parameters.items():
                    if param_name == "context":
                        continue  # context is always dict, passed separately

                    # skip *args and **kwargs - they're catch-alls, not validated params
                    if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                        continue

                    # get type hint for this parameter
                    param_type = type_hints.get(
                        param_name, str
                    )  # default to str if no hint

                    # get default value if specified
                    if param.default is inspect.Parameter.empty:
                        # no default - required field
                        field_defs[param_name] = (param_type, ...)
                    else:
                        # has default - optional field
                        field_defs[param_name] = (param_type, param.default)

                # create temporary Pydantic model for validation/coercion
                if field_defs:
                    CoercionModel = create_model(
                        f"{action_name.title()}Params", **field_defs
                    )

                    # validate and coerce the parameters
                    try:
                        coerced = CoercionModel(**rendered_params)
                        coerced_params = coerced.model_dump()
                    except ValidationError as ve:
                        # provide helpful error message
                        error_msg = f"Parameter validation failed for action '{action_name}': {ve}"
                        logger.error(error_msg)
                        raise ValueError(error_msg) from ve
                else:
                    # no type hints - use rendered params as-is
                    coerced_params = rendered_params

            except Exception as e:
                # if type introspection fails, fall back to uncoerced params
                logger.debug(
                    f"Type coercion failed for action '{action_name}': {e}. Using uncoerced params."
                )
                coerced_params = rendered_params

            # call the registered function
            try:
                result_text = func(context=context, **coerced_params)

                # Create action result and attach resolved params for display
                action_result = ActionResult(response=result_text)
                action_result._resolved_params = coerced_params
                return action_result, None

            except Exception as e:
                if on_error == "propagate":
                    raise
                elif on_error == "return_empty":
                    logger.warning(
                        f"Action '{action_name}' failed with error: {e}. Returning empty string."
                    )
                    return ActionResult(response=""), None
                elif on_error == "return_default":
                    logger.warning(
                        f"Action '{action_name}' failed with error: {e}. Returning default value: {default_value!r}"
                    )
                    return ActionResult(response=default_value), None

        # attach executor and metadata to response model
        ActionResult._executor = executor
        ActionResult._is_function = True
        ActionResult._default_save = default_save

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
        ) in cls._registry.values():
            if return_type is not None and return_type not in types:
                types.append(return_type)
        return types

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
            name for name, reg in cls._registry.items()
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

DEFAULT_MAX_CHARS = 32000*2
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
            logger.info(
                f"HTTP {status} from {url}, retrying with Playwright..."
            )
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

from . import set_
from . import break_
from . import fetch
from . import markdownify
from . import search
from . import timestamp
