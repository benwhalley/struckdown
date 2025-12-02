from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from .validation import parse_options
from .response_types import ResponseTypes

# Standard RTs for template syntax


class LLMConfig(BaseModel):
    """Configuration for LLM API calls with validation.

    This class defines all valid LLM parameters with their types and constraints.
    Use this for both defaults and runtime parameter validation.
    """

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    model: Optional[str] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    seed: Optional[int] = Field(default=None, ge=0)  # For reproducible outputs
    # top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # top_k: Optional[int] = Field(default=None, gt=0)
    # frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    # presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)

    model_config = ConfigDict(extra="forbid")  # Reject unknown parameters


class ResponseModel(BaseModel):
    """Base class for all Struckdown response models with LLM parameter defaults.

    Subclasses should set llm_config to customize LLM call parameters.
    Subclasses can set _capture_from_context to auto-inject context values.
    """

    # Class-level config for LLM parameters (not part of response schema)
    # Subclasses override this to set their own defaults
    llm_config: LLMConfig = LLMConfig(model=None)

    # Class-level list of field names to auto-populate from context
    # Example: _capture_from_context = ["source_id", "item_index"]
    _capture_from_context: List[str] = []

    def post_process(self, context: Dict[str, Any]) -> None:
        """Override in subclasses for model-specific post-processing.

        Called automatically after LLM extraction with accumulated context.
        Default implementation does nothing.
        """
        pass

    @classmethod
    def customize_schema_for_context(
        cls, schema: dict, context: Dict[str, Any]
    ) -> dict:
        """Override to customize schema based on context.

        Called by struckdown before passing schema to LLM.
        Subclasses can inspect context and modify schema accordingly.

        Args:
            schema: Base JSON schema for this model
            context: Accumulated template context

        Returns:
            Modified schema (or original if no changes needed)
        """
        return schema


@ResponseTypes.register("poem")
class PoeticalResponse(ResponseModel):
    """A spoken response, continuing the previous conversation, in 16th C style."""

    response: str = Field(
        ...,
        description="A response, continuing the previous conversation but always in POETRICAL form - often a haiku.",
    )


PoeticalResponse.llm_config = LLMConfig(temperature=1.5, model=None)  # High creativity


@ResponseTypes.register("chunked_conversation")
class ChunkedConversationResponse(ResponseModel):
    """A spoken response, continuing the previous conversation, in 16th C style."""

    response: List["ConversationSegment"] = Field(
        ...,
        description="Returns a list of ConversationSegments, each with a description",
    )


ChunkedConversationResponse.llm_config = LLMConfig(temperature=0.3, model=None)


class ConversationSegment(BaseModel):
    description: str = Field(description="A short description of the segment.")
    start: int
    end: int


class DefaultResponse(ResponseModel):
    """Respond to the context intelligently and concisely."""

    response: str = Field(
        ...,
        description="An intelligent completion that responds to the context in a concise manner.",
    )


DefaultResponse.llm_config = LLMConfig(temperature=0.7, model=None)


class ExtractedResponse(ResponseModel):
    """Extract information from the context verbatim."""

    response: str = Field(
        ...,
        description="Text extracted verbatim. Copy text exactly as it appears in the context. Never paraphrase or summarize. Never include any additional information.",
    )


ExtractedResponse.llm_config = LLMConfig(
    temperature=0.0, model=None
)  # deterministic extraction


class SpokenResponse(ResponseModel):
    """A spoken response, continuing the previous conversation naturally and fluently."""

    response: str = Field(
        ...,
        description="A spoken response, continuing the previous conversation. Don't label the speaker or use quotes, just produce the words spoken.",
    )


SpokenResponse.llm_config = LLMConfig(
    temperature=0.8, model=None
)  # More natural variation


class InternalThoughtsResponse(ResponseModel):
    """A response containing plans, thoughts and step-by-step reasoning to solve a task."""

    response: str = Field(
        ...,
        description="Your thoughts. Never a spoken response, yet -- just careful step by step thinking, planning and reasoning, written in super-concise note form. Always on topic and relevant to the task at hand.",
    )


InternalThoughtsResponse.llm_config = LLMConfig(
    temperature=0.5, model=None
)  # Moderate creativity for reasoning


@ResponseTypes.register("pick")
def selection_response_model(valid_options, quantifier=None, required_prefix=False):
    """Factory to produce a pydantic model with specific options.

    Args:
        valid_options: List of valid option strings. May include 'required=true' key-value to make field required.
        quantifier: Optional tuple of (min_items, max_items) where None means unlimited
                   Examples: (1, 3) = 1 to 3 items, (0, None) = 0 or more, (2, 2) = exactly 2
        required_prefix: Boolean indicating if ! prefix was used (takes precedence over required= option)
    """

    if not valid_options:
        raise ValueError("valid_options must be a non-empty list of strings")

    opts = parse_options(valid_options)
    is_required = opts.required or required_prefix
    selection_options = opts.positional  # non-keyword options are the actual choices

    if not selection_options:
        raise ValueError("valid_options must contain at least one selectable option")

    literals = Literal[tuple(selection_options)]

    if quantifier:
        # Multiple selection mode
        min_items, max_items = quantifier

        # Build field kwargs dynamically
        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        # Build description based on constraints
        if min_items == max_items:
            constraint_desc = f"exactly {min_items}"
        elif max_items is None:
            constraint_desc = (
                f"at least {min_items}" if min_items > 0 else "any number of"
            )
        elif min_items == 0:
            constraint_desc = f"up to {max_items}"
        else:
            constraint_desc = f"between {min_items} and {max_items}"

        # Format options list for display -- quote each option for clarity
        options_display = ", ".join(f"'{opt}'" for opt in selection_options)
        description = f"A list of {constraint_desc} selections from these options: {options_display}. Each selection must exactly match one of the provided options. No discussion or explanation."

        model = create_model(
            "MultiSelectionResponse",
            __base__=ResponseModel,
            response=(
                List[literals],
                Field(..., description=description, **field_kwargs),
            ),
            __module__=__name__,
        )
        model.llm_config = LLMConfig(
            temperature=0.0, model=None
        )  # Deterministic selection
        return model
    else:
        # Single selection mode
        if is_required:
            # Required field - must return one of the options
            # Format options list for display -- quote each option for clarity
            options_display = ", ".join(f"'{opt}'" for opt in selection_options)
            model = create_model(
                "SelectionResponse",
                __base__=ResponseModel,
                response=(
                    literals,
                    Field(
                        ...,
                        description=f"A selection from these options: {options_display}. No discussion or explanation, just the text of the choice, exactly matching one of the options provided.",
                    ),
                ),
                __module__=__name__,
            )
        else:
            # Optional field - can return None if no valid selection can be made
            # Format options list for display -- quote each option for clarity
            options_display = ", ".join(f"'{opt}'" for opt in selection_options)
            model = create_model(
                "SelectionResponse",
                __base__=ResponseModel,
                response=(
                    Union[literals, None],
                    Field(
                        default=None,
                        description=f"A selection from these options: {options_display}, or null if no valid selection can be made. No discussion or explanation, just the text of the choice exactly matching one of the options provided, or null.",
                    ),
                ),
                __module__=__name__,
            )
        model.llm_config = LLMConfig(
            temperature=0.0, model=None
        )  # Deterministic selection
        return model


class BooleanResponse(ResponseModel):
    """A boolean response, making a True/False decision based on the question posed."""

    response: bool = Field(
        ...,
        description="Returns true or false only, based on the question/statement posed.",
    )


BooleanResponse.llm_config = LLMConfig(
    temperature=0.0, model=None
)  # Deterministic boolean


class IntegerResponse(ResponseModel):
    """An integer response, based on the question posed."""

    response: int = Field(
        ...,
        description="Valid integers only.",
    )


IntegerResponse.llm_config = LLMConfig(
    temperature=0.0, model=None
)  # Deterministic integers


@ResponseTypes.register("default")
@ResponseTypes.register("respond")
def default_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory to produce default text response models with optional list support.

    Args:
        options: Optional list of options (currently unused for default type)
        quantifier: Optional tuple of (min_items, max_items) for list responses
                   None means single value. Examples: (1, 3) = 1 to 3 items, (0, None) = 0 or more
        required_prefix: Boolean indicating if ! prefix was used

    Returns:
        Pydantic model with str or List[str] response field
    """
    opts = parse_options(options)
    is_required = opts.required or required_prefix

    if quantifier:
        # Multiple items mode - return a list
        min_items, max_items = quantifier

        # Build field kwargs dynamically
        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        # Build description based on constraints
        if min_items == max_items:
            constraint_desc = f"exactly {min_items}"
        elif max_items is None:
            constraint_desc = (
                f"at least {min_items}" if min_items > 0 else "any number of"
            )
        elif min_items == 0:
            constraint_desc = f"up to {max_items}"
        else:
            constraint_desc = f"between {min_items} and {max_items}"

        list_description = f"Generate {constraint_desc} intelligent completions that respond to the context in a concise manner. Each item should be a separate, distinct response."

        model = create_model(
            "MultiDefaultResponse",
            __base__=ResponseModel,
            response=(
                List[str],
                Field(
                    default_factory=list, description=list_description, **field_kwargs
                ),
            ),
            __module__=__name__,
        )
        model.llm_config = LLMConfig(temperature=0.7, model=None)
        return model
    else:
        # Single item mode
        if is_required:
            return DefaultResponse
        else:
            # Optional field - can return None
            model = create_model(
                "OptionalDefaultResponse",
                __base__=ResponseModel,
                response=(
                    Union[str, None],
                    Field(
                        default=None,
                        description="An intelligent completion that responds to the context in a concise manner, or null if no response is appropriate.",
                    ),
                ),
                __module__=__name__,
            )
            model.llm_config = LLMConfig(temperature=0.7, model=None)
            return model


@ResponseTypes.register("int")
def integer_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory to produce integer response models with optional list support.

    Args:
        options: Optional list that may contain 'min=X', 'max=Y' constraints, or 'required=true'
        quantifier: Optional tuple of (min_items, max_items) for list responses
                   None means single value
        required_prefix: Boolean indicating if ! prefix was used

    Returns:
        Pydantic model with int or List[int] response field
    """
    opts = parse_options(options)
    is_required = opts.required or required_prefix
    min_val = int(opts.ge) if opts.ge is not None else None
    max_val = int(opts.le) if opts.le is not None else None

    # Build description with constraints
    constraint_desc = ""
    if min_val is not None and max_val is not None:
        constraint_desc = f" Range: {min_val} to {max_val}."
    elif min_val is not None:
        constraint_desc = f" Minimum: {min_val}."
    elif max_val is not None:
        constraint_desc = f" Maximum: {max_val}."

    if quantifier:
        # Multiple items mode
        min_items, max_items = quantifier

        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        if min_items == max_items:
            quant_desc = f"exactly {min_items}"
        elif max_items is None:
            quant_desc = f"at least {min_items}" if min_items > 0 else "any number of"
        elif min_items == 0:
            quant_desc = f"up to {max_items}"
        else:
            quant_desc = f"between {min_items} and {max_items}"

        list_description = f"Return {quant_desc} valid integers.{constraint_desc}"

        model = create_model(
            "MultiIntegerResponse",
            __base__=ResponseModel,
            response=(
                List[int],
                Field(
                    default_factory=list, description=list_description, **field_kwargs
                ),
            ),
            __module__=__name__,
        )
        model.llm_config = LLMConfig(temperature=0.0, model=None)
        return model
    else:
        # Single item mode
        if is_required:
            return IntegerResponse
        else:
            model = create_model(
                "OptionalIntegerResponse",
                __base__=ResponseModel,
                response=(
                    Union[int, None],
                    Field(
                        default=None,
                        description=f"A valid integer, or null if no integer can be determined.{constraint_desc}",
                    ),
                ),
                __module__=__name__,
            )
            model.llm_config = LLMConfig(temperature=0.0, model=None)
            return model


@ResponseTypes.register("boolean")
@ResponseTypes.register("bool")
@ResponseTypes.register("decide")
def boolean_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory to produce boolean response models with optional list support.

    Args:
        options: Optional list (may contain 'required=true')
        quantifier: Optional tuple of (min_items, max_items) for list responses
        required_prefix: Boolean indicating if ! prefix was used

    Returns:
        Pydantic model with bool or List[bool] response field
    """
    opts = parse_options(options)
    is_required = opts.required or required_prefix

    if quantifier:
        # Multiple items mode
        min_items, max_items = quantifier

        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        if min_items == max_items:
            quant_desc = f"exactly {min_items}"
        elif max_items is None:
            quant_desc = f"at least {min_items}" if min_items > 0 else "any number of"
        elif min_items == 0:
            quant_desc = f"up to {max_items}"
        else:
            quant_desc = f"between {min_items} and {max_items}"

        list_description = f"Return {quant_desc} boolean values (true or false) based on the questions posed."

        model = create_model(
            "MultiBooleanResponse",
            __base__=ResponseModel,
            response=(
                List[bool],
                Field(
                    default_factory=list, description=list_description, **field_kwargs
                ),
            ),
            __module__=__name__,
        )
        model.llm_config = LLMConfig(temperature=0.0, model=None)
        return model
    else:
        # Single item mode
        if is_required:
            return BooleanResponse
        else:
            model = create_model(
                "OptionalBooleanResponse",
                __base__=ResponseModel,
                response=(
                    Union[bool, None],
                    Field(
                        default=None,
                        description="Returns true, false, or null if the question cannot be answered.",
                    ),
                ),
                __module__=__name__,
            )
            model.llm_config = LLMConfig(temperature=0.0, model=None)
            return model


@ResponseTypes.register("extract")
def extracted_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory to produce extracted text response models with optional list support.

    Args:
        options: Optional list (may contain 'required=true')
        quantifier: Optional tuple of (min_items, max_items) for list responses
        required_prefix: Boolean indicating if ! prefix was used

    Returns:
        Pydantic model with str or List[str] response field for verbatim extraction
    """
    opts = parse_options(options)
    is_required = opts.required or required_prefix

    if quantifier:
        # Multiple items mode
        min_items, max_items = quantifier

        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        if min_items == max_items:
            quant_desc = f"exactly {min_items}"
        elif max_items is None:
            quant_desc = f"at least {min_items}" if min_items > 0 else "any number of"
        elif min_items == 0:
            quant_desc = f"up to {max_items}"
        else:
            quant_desc = f"between {min_items} and {max_items}"

        list_description = f"Extract {quant_desc} pieces of text verbatim from the context. Copy text exactly as it appears. Never paraphrase or summarize."

        model = create_model(
            "MultiExtractedResponse",
            __base__=ResponseModel,
            response=(
                List[str],
                Field(
                    default_factory=list, description=list_description, **field_kwargs
                ),
            ),
            __module__=__name__,
        )
        model.llm_config = LLMConfig(temperature=0.0, model=None)
        return model
    else:
        # Single item mode
        if is_required:
            return ExtractedResponse
        else:
            model = create_model(
                "OptionalExtractedResponse",
                __base__=ResponseModel,
                response=(
                    Union[str, None],
                    Field(
                        default=None,
                        description="Text extracted verbatim, or null if no text found. Copy text exactly as it appears in the context.",
                    ),
                ),
                __module__=__name__,
            )
            model.llm_config = LLMConfig(temperature=0.0, model=None)
            return model


@ResponseTypes.register("think")
def internal_thoughts_response_model(
    options=None, quantifier=None, required_prefix=False
):
    """Factory to produce internal thoughts response models with optional list support.

    Args:
        options: Optional list (may contain 'required=true')
        quantifier: Optional tuple of (min_items, max_items) for list responses
        required_prefix: Boolean indicating if ! prefix was used

    Returns:
        Pydantic model with str or List[str] response field for reasoning
    """
    opts = parse_options(options)
    is_required = opts.required or required_prefix

    if quantifier:
        # Multiple items mode
        min_items, max_items = quantifier

        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        if min_items == max_items:
            quant_desc = f"exactly {min_items}"
        elif max_items is None:
            quant_desc = f"at least {min_items}" if min_items > 0 else "any number of"
        elif min_items == 0:
            quant_desc = f"up to {max_items}"
        else:
            quant_desc = f"between {min_items} and {max_items}"

        list_description = f"Generate {quant_desc} sets of internal thoughts. Each should contain plans, thoughts and step-by-step reasoning in super-concise note form."

        model = create_model(
            "MultiInternalThoughtsResponse",
            __base__=ResponseModel,
            response=(
                List[str],
                Field(
                    default_factory=list, description=list_description, **field_kwargs
                ),
            ),
            __module__=__name__,
        )
        model.llm_config = LLMConfig(temperature=0.5, model=None)
        return model
    else:
        # Single item mode
        if is_required:
            return InternalThoughtsResponse
        else:
            model = create_model(
                "OptionalInternalThoughtsResponse",
                __base__=ResponseModel,
                response=(
                    Union[str, None],
                    Field(
                        default=None,
                        description="Your thoughts, plans and step-by-step reasoning in concise note form, or null if no reasoning needed.",
                    ),
                ),
                __module__=__name__,
            )
            model.llm_config = LLMConfig(temperature=0.5, model=None)
            return model


@ResponseTypes.register("speak")
def spoken_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory to produce spoken response models with optional list support.

    Args:
        options: Optional list (may contain 'required=true')
        quantifier: Optional tuple of (min_items, max_items) for list responses
        required_prefix: Boolean indicating if ! prefix was used

    Returns:
        Pydantic model with str or List[str] response field for spoken dialogue
    """
    opts = parse_options(options)
    is_required = opts.required or required_prefix

    if quantifier:
        # Multiple items mode
        min_items, max_items = quantifier

        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        if min_items == max_items:
            quant_desc = f"exactly {min_items}"
        elif max_items is None:
            quant_desc = f"at least {min_items}" if min_items > 0 else "any number of"
        elif min_items == 0:
            quant_desc = f"up to {max_items}"
        else:
            quant_desc = f"between {min_items} and {max_items}"

        list_description = f"Generate {quant_desc} spoken responses, continuing the conversation naturally. Don't label speakers or use quotes, just produce the words spoken."

        model = create_model(
            "MultiSpokenResponse",
            __base__=ResponseModel,
            response=(
                List[str],
                Field(
                    default_factory=list, description=list_description, **field_kwargs
                ),
            ),
            __module__=__name__,
        )
        model.llm_config = LLMConfig(temperature=0.8, model=None)
        return model
    else:
        # Single item mode
        if is_required:
            return SpokenResponse
        else:
            model = create_model(
                "OptionalSpokenResponse",
                __base__=ResponseModel,
                response=(
                    Union[str, None],
                    Field(
                        default=None,
                        description="A spoken response continuing the conversation, or null if no response is appropriate.",
                    ),
                ),
                __module__=__name__,
            )
            model.llm_config = LLMConfig(temperature=0.8, model=None)
            return model


@ResponseTypes.register("number")
def number_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory to produce numeric response models for int/float values.

    Args:
        options: List that may contain 'min=X', 'max=Y' constraints, or 'required=true'
                 Examples: ["min=0"], ["max=100"], ["min=0,max=100"], ["min=-10.5", "max=99.9"], ["required=true"]
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple numbers
                   None means single value. Examples: (1, 3) = 1 to 3 items, (0, None) = 0 or more
        required_prefix: Boolean indicating if ! prefix was used

    Returns:
        Pydantic model with Union[int, float] field that can be validated post-extraction
    """
    opts = parse_options(options)
    is_required = opts.required or required_prefix
    min_val = opts.ge
    max_val = opts.le

    # Build description with constraints as hints
    constraint_desc = ""
    if min_val is not None and max_val is not None:
        constraint_desc = f" Suggested range: {min_val} to {max_val}."
    elif min_val is not None:
        constraint_desc = f" Suggested minimum: {min_val}."
    elif max_val is not None:
        constraint_desc = f" Suggested maximum: {max_val}."

    if quantifier:
        # Multiple extraction mode - return a list
        min_items, max_items = quantifier

        # Build field kwargs dynamically
        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        # Build description based on constraints
        if min_items == max_items:
            quant_desc = f"exactly {min_items}"
        elif max_items is None:
            quant_desc = f"at least {min_items}" if min_items > 0 else "any number of"
        elif min_items == 0:
            quant_desc = f"up to {max_items}"
        else:
            quant_desc = f"between {min_items} and {max_items}"

        list_description = f"Extract {quant_desc} numeric values (integers or decimals) from the text.{constraint_desc} Return empty list if no numbers found."

        model = create_model(
            "MultiNumberResponse",
            __base__=ResponseModel,
            response=(
                List[Union[int, float]],
                Field(
                    default_factory=list, description=list_description, **field_kwargs
                ),
            ),
            __module__=__name__,
        )
        model.llm_config = LLMConfig(
            temperature=0.1, model=None
        )  # Slightly flexible for extraction
        return model
    else:
        # Single extraction mode
        if is_required:
            # Required field - must return a number
            single_description = f"Extract a single numeric value (integer or decimal) from the text.{constraint_desc}"

            model = create_model(
                "NumberResponse",
                __base__=ResponseModel,
                response=(
                    Union[int, float],
                    Field(..., description=single_description),
                ),
                __module__=__name__,
            )
        else:
            # Optional field - can return None
            single_description = f"Extract a single numeric value (integer or decimal) from the text.{constraint_desc} Return null if no number found."

            model = create_model(
                "NumberResponse",
                __base__=ResponseModel,
                response=(
                    Union[int, float, None],
                    Field(default=None, description=single_description),
                ),
                __module__=__name__,
            )
        model.llm_config = LLMConfig(
            temperature=0.1, model=None
        )  # Slightly flexible for extraction
        return model


@ResponseTypes.register("date_rule")
class DateRuleResponse(ResponseModel):
    """Returns RRULE parameters for generating recurring dates.

    Convert natural language date patterns into RRULE parameters.
    """

    freq: str = Field(
        ..., description='Frequency: one of "DAILY", "WEEKLY", "MONTHLY", "YEARLY"'
    )
    dtstart: str = Field(
        ...,
        description="Starting date in ISO format (YYYY-MM-DD) or datetime with timezone. If no year is specified in the input, use the current year from the temporal context. For relative expressions like 'next month' or ambiguous month references like 'September' (without year), choose the year that makes most sense given the current date.",
    )
    count: Optional[int] = Field(
        None, description="Number of occurrences (e.g., 5 for 'first 5 Tuesdays')"
    )
    until: Optional[str] = Field(
        None, description="End date in ISO format (YYYY-MM-DD)"
    )
    interval: Optional[int] = Field(
        None,
        description="Interval between occurrences (e.g., 2 for 'every other week')",
    )
    byweekday: Optional[List[int]] = Field(
        None, description="List of weekdays as integers 0-6 (Monday=0, Sunday=6)"
    )
    bymonth: Optional[List[int]] = Field(
        None, description="List of months as integers 1-12"
    )
    bymonthday: Optional[List[int]] = Field(
        None, description="List of days of month as integers 1-31"
    )
    bysetpos: Optional[List[int]] = Field(
        None, description="Position in set (e.g., [1] for 'first', [-1] for 'last')"
    )


DateRuleResponse.llm_config = LLMConfig(
    temperature=0.0, model=None
)  # Deterministic rule parsing


def temporal_response_model(
    field_type,
    type_name,
    description,
    required=False,
    quantifier=None,
    required_prefix=False,
):
    """Factory to produce temporal response models.

    Args:
        field_type: The Python type (date, datetime, time, timedelta)
        type_name: The name of the type for the model
        description: Field description
        required: If True, field cannot be None and must be provided. If False (default), field is optional.
        quantifier: Optional tuple of (min_items, max_items) for list responses.
                   None means single value. Examples: (1, 3) = 1 to 3 items, (0, None) = 0 or more
        required_prefix: Boolean indicating if ! prefix was used (takes precedence over required param)
    """
    # ! prefix takes precedence over required parameter
    is_required = required_prefix or required
    if quantifier:
        # Multiple extraction mode - return a list
        min_items, max_items = quantifier

        # Build field kwargs dynamically
        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        # Build description based on constraints
        if min_items == max_items:
            constraint_desc = f"exactly {min_items}"
        elif max_items is None:
            constraint_desc = (
                f"at least {min_items}" if min_items > 0 else "any number of"
            )
        elif min_items == 0:
            constraint_desc = f"up to {max_items}"
        else:
            constraint_desc = f"between {min_items} and {max_items}"

        list_description = f"""Extract {constraint_desc} {type_name.lower()} values. You have TWO OPTIONS:

Option 1 - EXPLICIT {type_name.upper()}S: If you can determine specific {type_name.lower()}s directly from the text
(e.g., "Jan 15, 2024", "tomorrow", "3pm", "2 hours"), return them as a list of {type_name.lower()} values.

Option 2 - RECURRING PATTERN: If the input describes a recurring pattern that requires date/time arithmetic
(e.g., "every Tuesday", "first 2 Tuesdays in October", "all weekdays in March", "last Friday of each month"),
return a list containing a SINGLE STRING that describes the pattern exactly as stated in the input.

Examples of patterns that should return a string:
- "every Tuesday in October"
- "first 2 Tuesdays in October 2025"
- "all Mondays and Wednesdays in December"
- "last Friday of each month for 6 months"
- "every other week starting Monday"

When in doubt, if the pattern involves repetition or requires calculating multiple occurrences, return the pattern string.

{description} Return empty list if no {type_name.lower()}s or patterns found."""

        model = create_model(
            f"Multi{type_name}Response",
            __base__=ResponseModel,
            response=(
                Union[List[field_type], List[str]],  # Accept dates OR pattern strings
                Field(
                    default_factory=list, description=list_description, **field_kwargs
                ),
            ),
            __module__=__name__,
        )
        model.llm_config = LLMConfig(
            temperature=0.1, model=None
        )  # Low temp for temporal extraction
        return model
    else:
        # Single extraction mode
        if is_required:
            # Required field: accept field_type or str (for ISO strings/patterns), but NOT None
            single_description = f"""Extract a single {type_name.lower()} value. You have TWO OPTIONS:

Option 1 - EXPLICIT {type_name.upper()}: If you can determine a specific {type_name.lower()} directly from the text
(e.g., "Jan 15, 2024", "tomorrow", "3pm"), return it as a {type_name.lower()} value.

Option 2 - RECURRING PATTERN: If the input describes a recurring pattern
(e.g., "every Tuesday", "first 2 Tuesdays in October"), return a STRING that describes the pattern exactly as stated.
The pattern will be expanded automatically and the first occurrence will be used.

{description}"""

            # For required fields, use plain Union without BeforeValidator
            # BeforeValidator causes Pydantic to add None to the union automatically
            model = create_model(
                f"{type_name}Response",
                __base__=ResponseModel,
                response=(
                    Union[
                        field_type, str
                    ],  # Accept field_type or pattern string, NO None
                    Field(
                        ...,  # Required field - no default
                        description=single_description,
                    ),
                ),
                __module__=__name__,
            )
            model.llm_config = LLMConfig(
                temperature=0.1, model=None
            )  # Low temp for temporal extraction
            return model
        else:
            # Optional field: accept field_type, str, or None
            single_description = f"""Extract a single {type_name.lower()} value. You have TWO OPTIONS:

Option 1 - EXPLICIT {type_name.upper()}: If you can determine a specific {type_name.lower()} directly from the text
(e.g., "Jan 15, 2024", "tomorrow", "3pm"), return it as a {type_name.lower()} value.

Option 2 - RECURRING PATTERN: If the input describes a recurring pattern
(e.g., "every Tuesday", "first 2 Tuesdays in October"), return a STRING that describes the pattern exactly as stated.
The pattern will be expanded automatically and the first occurrence will be used.

{description} Return null if no {type_name.lower()} or pattern found."""

            # Optional field accepts field_type, pattern strings, or None
            # Pydantic handles ISO string conversion natively
            model = create_model(
                f"{type_name}Response",
                __base__=ResponseModel,
                response=(
                    Union[
                        field_type, str, None
                    ],  # Accept field_type, pattern string, or None
                    Field(default=None, description=single_description),
                ),
                __module__=__name__,
            )
            model.llm_config = LLMConfig(
                temperature=0.1, model=None
            )  # Low temp for temporal extraction
            return model


@ResponseTypes.register("date")
def date_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory for date response models.

    Args:
        options: List that may contain 'required=true' key-value or bare 'required'
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple dates
        required_prefix: Boolean indicating if ! prefix was used
    """
    opts = parse_options(options)
    required = opts.is_required_flag_set()

    return temporal_response_model(
        date,
        "Date",
        "A date value. For relative dates (e.g., 'next Tuesday', 'tomorrow'), use the current date/time context provided. Return dates in ISO format (YYYY-MM-DD).",
        required=required,
        quantifier=quantifier,
        required_prefix=required_prefix,
    )


@ResponseTypes.register("datetime")
def datetime_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory for datetime response models.

    Args:
        options: List that may contain 'required=true' key-value or bare 'required'
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple datetimes
        required_prefix: Boolean indicating if ! prefix was used
    """
    opts = parse_options(options)
    required = opts.is_required_flag_set()

    return temporal_response_model(
        datetime,
        "Datetime",
        "A datetime value with timezone awareness. For relative datetimes (e.g., 'next Tuesday at 3pm'), use the current date/time context provided. Return in ISO format with timezone.",
        required=required,
        quantifier=quantifier,
        required_prefix=required_prefix,
    )


@ResponseTypes.register("time")
def time_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory for time response models.

    Args:
        options: List that may contain 'required=true' key-value or bare 'required'
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple times
        required_prefix: Boolean indicating if ! prefix was used
    """
    opts = parse_options(options)
    required = opts.is_required_flag_set()

    return temporal_response_model(
        time,
        "Time",
        "A time value. For times without explicit AM/PM, use context clues. Return in HH:MM:SS format.",
        required=required,
        quantifier=quantifier,
        required_prefix=required_prefix,
    )


@ResponseTypes.register("duration")
def duration_response_model(options=None, quantifier=None, required_prefix=False):
    """Factory for duration response models.

    Args:
        options: List that may contain 'required=true' key-value or bare 'required'
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple durations
        required_prefix: Boolean indicating if ! prefix was used
    """
    opts = parse_options(options)
    required = opts.is_required_flag_set()

    return temporal_response_model(
        timedelta,
        "Duration",
        "A duration expressed as a timedelta. Extract from phrases like '2 hours', '3 days', '1 week and 2 days', '30 minutes'. Return the total duration in days and seconds.",
        required=required,
        quantifier=quantifier,
        required_prefix=required_prefix,
    )


# --- Job and JobGroup schemas for API output ---


class JobSchema(BaseModel):
    id: int
    context: dict
    result: Optional[dict] = None
    completed: Optional[str] = None
    cancelled: bool
    tool_id: Optional[int] = None
    source_file: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class JobGroupSchema(BaseModel):
    uuid: str
    tool_id: int
    tool_name: str
    model_name: str
    credentials_label: Optional[str] = None
    owner_id: int
    include_source: bool
    jobs: List[JobSchema]
    status: str
    created: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# Build ACTION_LOOKUP from registered types
# This maintains backward compatibility with code that imports ACTION_LOOKUP
ACTION_LOOKUP = ResponseTypes.get_lookup()
