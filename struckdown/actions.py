"""
Custom action registry for struckdown.

Allows plugins to register functions that execute instead of LLM calls.
Functions are invoked via template syntax: [[action_name:varname|param1=value1,param2=value2]]
"""

import inspect
import logging
import re
from typing import Any, Callable, Literal, Optional, get_type_hints

from jinja2 import Template, StrictUndefined
from pydantic import BaseModel, Field, ValidationError, create_model

from struckdown.return_type_models import LLMConfig, ResponseModel

logger = logging.getLogger(__name__)

ErrorStrategy = Literal["propagate", "return_empty", "log_and_continue"]


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
        # [[expertise:guidance|query="insomnia",n=5]]
    """

    _registry: dict[str, tuple[Callable, ErrorStrategy, bool, type | None]] = {}

    @classmethod
    def register(cls, action_name: str, on_error: ErrorStrategy = "propagate", default_save: bool = True, return_type: type | None = None):
        """Decorator to register a function as a custom action.

        Args:
            action_name: Name used in templates, e.g., 'expertise' for [[expertise:var|...]]
            on_error: How to handle exceptions:
                - 'propagate': Re-raise exception (default)
                - 'return_empty': Return empty string on error
                - 'log_and_continue': Log error and return empty string
            default_save: Whether to save result to context when no variable name specified.
                - True: [[@action]] saves result to context with key 'action' (default)
                - False: [[@action]] doesn't save result, only includes in prompt
                - Note: [[@action:var]] always saves regardless of this setting
            return_type: Pydantic model type returned by this action.
                - Enables automatic deserialization when loading from JSON
                - Example: return_type=FoundEvidenceSet for evidence search

        Returns:
            Decorator function

        Example:
            @Actions.register('memory', on_error='return_empty', default_save=True, return_type=MemoryResult)
            def search_memory(context, query, n=3):
                return results

            @Actions.register('turns', on_error='return_empty', default_save=False)
            def get_turns(context, filter_type='all'):
                return formatted_turns  # Output to prompt but don't save
        """

        def decorator(func: Callable) -> Callable:
            cls._registry[action_name] = (func, on_error, default_save, return_type)
            logger.debug(f"Registered action '{action_name}' with function {func.__name__}, default_save={default_save}, return_type={return_type}")
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
        [[action_name:varname|options]] syntax.

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

        func, on_error, default_save, return_type = cls._registry[action_name]

        # create response model
        class ActionResult(ResponseModel):
            """Result from custom action"""

            response: Any = Field(default="", description=f"Result from {action_name} action")

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
                name for name in sig.parameters.keys()
                if name != 'context'  # skip context parameter
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
            logger.debug(f"Positional args: {positional_args}, Keyword args: {keyword_args}")
            logger.debug(f"Mapped params to render: {params}")

            # render Jinja2 variables in parameter values
            # this allows: query={{extracted_var}} in templates
            rendered_params = {}
            for k, v in params.items():
                try:
                    # try rendering as Jinja2 template with StrictUndefined
                    # this will error if variables are missing instead of silently rendering to ''
                    rendered_value = Template(str(v), undefined=StrictUndefined).render(**context)
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
                    if param_name == 'context':
                        continue  # context is always dict, passed separately

                    # get type hint for this parameter
                    param_type = type_hints.get(param_name, str)  # default to str if no hint

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
                        f'{action_name.title()}Params',
                        **field_defs
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
                logger.debug(f"Type coercion failed for action '{action_name}': {e}. Using uncoerced params.")
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
                elif on_error == "log_and_continue":
                    logger.error(
                        f"Action '{action_name}' failed with error: {e}. Continuing with empty result.",
                        exc_info=True,
                    )
                    return ActionResult(response=""), None

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
        for func, on_error, default_save, return_type in cls._registry.values():
            if return_type is not None and return_type not in types:
                types.append(return_type)
        return types
