"""Shared utilities for creating dynamic Pydantic models."""

from pydantic import Field, create_model

from struckdown.return_type_models import ResponseModel


def create_list_model(
    model: type,
    type_name: str,
    quantifier: tuple,
    module_name: str = "struckdown.dynamic_types",
) -> type:
    """Create a list wrapper model for quantified types.

    Creates a MultiXResponse model with a 'response' field containing
    a list of the original model type. This wrapper is required because
    LLM structured output APIs (e.g., OpenAI) require root-level objects,
    not arrays.

    Args:
        model: The base Pydantic model to wrap in a list
        type_name: Name of the type (e.g., "theme") for generating model name
        quantifier: Tuple of (min_items, max_items) constraints
        module_name: Module name for the dynamic model

    Returns:
        A new Pydantic model class with a 'response' field containing list[model]
    """
    min_items, max_items = quantifier

    field_kwargs = {}
    if min_items is not None:
        field_kwargs["min_length"] = min_items
    if max_items is not None:
        field_kwargs["max_length"] = max_items

    # Build description
    if min_items == max_items:
        desc = f"Exactly {min_items}"
    elif max_items is None:
        desc = f"At least {min_items}" if min_items > 0 else "Any number of"
    elif min_items == 0:
        desc = f"Up to {max_items}"
    else:
        desc = f"Between {min_items} and {max_items}"

    description = f"{desc} {type_name} items."
    capitalized_name = type_name[0].upper() + type_name[1:] if type_name else type_name

    list_model = create_model(
        f"Multi{capitalized_name}Response",
        __base__=ResponseModel,
        __module__=module_name,
        response=(list[model], Field(..., description=description, **field_kwargs)),
    )

    # Copy LLM config from base model if available
    if hasattr(model, "llm_config"):
        list_model.llm_config = model.llm_config

    # Add post_process method that iterates through list items
    def list_post_process(self, context):
        """Post-process each item in the response list."""
        for item in self.response:
            if hasattr(item, "post_process"):
                item.post_process(context)

    list_model.post_process = list_post_process

    return list_model
