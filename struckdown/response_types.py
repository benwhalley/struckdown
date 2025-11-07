"""Response type registry for struckdown.

Allows registration of custom response types (Pydantic models or factory functions)
that can be used in templates via [[type:var]] syntax.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Union

logger = logging.getLogger(__name__)


class ResponseTypes:
    """Global registry for response types (built-in and custom).

    Response types are Pydantic ResponseModel subclasses or factory functions
    that return such models. They define the structure of LLM responses.

    Built-in types are registered in return_type_models.py.
    Plugins can register custom types via decorator.

    Example:
        @ResponseTypes.register('sentiment')
        class SentimentResponse(ResponseModel):
            score: float = Field(ge=-1, le=1)
            label: str

        # Use in template:
        # What is the sentiment? [[sentiment:result]]
    """

    _registry: Dict[str, Union[type, Callable]] = {}

    @classmethod
    def register(cls, type_name: str):
        """Decorator to register a response type.

        Args:
            type_name: Name used in templates, e.g., 'code' for [[code:var]]

        Returns:
            Decorator function

        Example:
            @ResponseTypes.register('code')
            class Code(ResponseModel):
                name: str
                description: str

            @ResponseTypes.register('pick')
            def selection_response_model(options, quantifier, required_prefix):
                # Factory function that returns a ResponseModel
                ...
        """

        def decorator(model_or_factory: Union[type, Callable]) -> Union[type, Callable]:
            cls._registry[type_name] = model_or_factory
            logger.debug(
                f"Registered response type '{type_name}' with {model_or_factory}"
            )
            return model_or_factory

        return decorator

    @classmethod
    def get(cls, type_name: str, default: Any = None) -> Union[type, Callable, None]:
        """Get a registered response type by name.

        Args:
            type_name: The type name to look up
            default: Value to return if not found

        Returns:
            The registered response type (model class or factory function)
        """
        return cls._registry.get(type_name, default)

    @classmethod
    def get_lookup(cls) -> Dict[str, Union[type, Callable]]:
        """Build ACTION_LOOKUP dict from registry.

        Returns a defaultdict where unknown keys return the 'default' type.
        This maintains backward compatibility with the old ACTION_LOOKUP dict.

        Returns:
            Dict mapping type names to ResponseModel classes/factories
        """
        default_type = cls._registry.get("default")
        if default_type is None:
            raise RuntimeError(
                "No 'default' response type registered. "
                "This should be registered in return_type_models.py"
            )

        lookup = defaultdict(lambda: default_type)
        lookup.update(cls._registry)
        return lookup

    @classmethod
    def list_registered(cls) -> List[str]:
        """List all registered response type names.

        Returns:
            List of type names
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, type_name: str) -> bool:
        """Check if a response type is registered.

        Args:
            type_name: Type name to check

        Returns:
            True if registered, False otherwise
        """
        return type_name in cls._registry
