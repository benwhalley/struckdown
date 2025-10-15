from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, create_model

# Standard RTs for template syntax


# Validators for temporal types to handle string inputs
def validate_date_input(v):
    """Convert ISO date strings to date objects."""
    if isinstance(v, str):
        # Try to parse as ISO date string
        if 'T' in v:
            # Has time component, parse as datetime and extract date
            return datetime.fromisoformat(v.replace('Z', '+00:00')).date()
        else:
            # Just a date
            return date.fromisoformat(v)
    return v


def validate_datetime_input(v):
    """Convert ISO datetime strings to datetime objects."""
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace('Z', '+00:00'))
    return v


def validate_time_input(v):
    """Convert time strings to time objects."""
    if isinstance(v, str):
        return time.fromisoformat(v)
    return v


def validate_timedelta_input(v):
    """Convert numeric seconds to timedelta objects."""
    if isinstance(v, (int, float)):
        return timedelta(seconds=v)
    return v


class DefaultResponse(BaseModel):
    """Respond to the context intelligently and concisely."""

    response: str = Field(
        ...,
        description="An intelligent completion that responds to the context in a concise manner.",
    )


class ExtractedResponse(BaseModel):
    """Extract information from the context verbatim."""

    response: str = Field(
        ...,
        description="Text extracted from the context verbatim. Copy text exactly as it appears in the context. Never paraphrase or summarize. Never include any additional information.",
    )


class SpokenResponse(BaseModel):
    """A spoken response, continuing the previous conversation naturally and fluently."""

    response: str = Field(
        ...,
        description="A spoken response, continuing the previous conversation. Don't label the speaker or use quotes, just produce the words spoken.",
    )


class InternalThoughtsResponse(BaseModel):
    """A response containing plans, thoughts and step-by-step reasoning to solve a task."""

    response: str = Field(
        ...,
        description="Your thoughts. Never a spoken response, yet -- just careful step by step thinking, planning and reasoning, written in super-concise note form. Always on topic and relevant to the task at hand.",
    )


class PoeticalResponse(BaseModel):
    """A spoken response, continuing the previous conversation, in 16th C style."""

    response: str = Field(
        ...,
        description="A response, continuing the previous conversation but always in POETRICAL form - often a haiku.",
    )


class ConversationSegment(BaseModel):
    description: str = Field(description="A short description of the segment.")
    start: int
    end: int


class ChunkedConversationResponse(BaseModel):
    """A spoken response, continuing the previous conversation, in 16th C style."""

    response: List[ConversationSegment] = Field(
        ...,
        description="Returns a list of ConversationSegments, each with a description",
    )


def selection_response_model(valid_options, quantifier=None):
    """Factory to produce a pydantic model with specific options required.

    Args:
        valid_options: List of valid option strings
        quantifier: Optional tuple of (min_items, max_items) where None means unlimited
                   Examples: (1, 3) = 1 to 3 items, (0, None) = 0 or more, (2, 2) = exactly 2
    """

    if not valid_options:
        raise ValueError("valid_options must be a non-empty list of strings")

    literals = Literal[tuple(valid_options)]

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

        description = f"A list of {constraint_desc} selections from the options. Each selection must exactly match one of the provided options. No discussion or explanation."

        return create_model(
            "MultiSelectionResponse",
            response=(
                List[literals],
                Field(..., description=description, **field_kwargs),
            ),
            __module__=__name__,
        )
    else:
        # Single selection mode (backward compatible)
        return create_model(
            "SelectionResponse",
            response=(
                literals,
                Field(
                    ...,
                    description="A selection from one of the options. No discussion or explanation, just the text of the choice, exactly matching one of the options provided.",
                ),
            ),
            __module__=__name__,
        )


class BooleanResponse(BaseModel):
    """A boolean response, making a True/False decision based on the question posed."""

    response: bool = Field(
        ...,
        description="Returns true or false only, based on the question/statement posed.",
    )


class IntegerResponse(BaseModel):
    """An integer response, based on the question posed."""

    response: int = Field(
        ...,
        description="Valid integers only.",
    )


def number_response_model(options=None, quantifier=None):
    """Factory to produce numeric response models for int/float values.

    Args:
        options: List that may contain 'min=X' and/or 'max=Y' constraints
                 Examples: ["min=0"], ["max=100"], ["min=0,max=100"], ["min=-10.5", "max=99.9"]
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple numbers
                   None means single value. Examples: (1, 3) = 1 to 3 items, (0, None) = 0 or more

    Returns:
        Pydantic model with Union[int, float] field that can be validated post-extraction
    """
    # Parse min/max constraints from options
    min_val = None
    max_val = None

    if options:
        for opt in options:
            # Handle both "min=0,max=100" and separate ["min=0", "max=100"]
            if "min=" in opt or "max=" in opt:
                parts = opt.split(",")
                for part in parts:
                    part = part.strip()
                    if part.startswith("min="):
                        try:
                            min_val = float(part.split("=", 1)[1])
                        except (ValueError, IndexError):
                            pass
                    elif part.startswith("max="):
                        try:
                            max_val = float(part.split("=", 1)[1])
                        except (ValueError, IndexError):
                            pass

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
            quant_desc = (
                f"at least {min_items}" if min_items > 0 else "any number of"
            )
        elif min_items == 0:
            quant_desc = f"up to {max_items}"
        else:
            quant_desc = f"between {min_items} and {max_items}"

        list_description = f"Extract {quant_desc} numeric values (integers or decimals) from the text.{constraint_desc} Return empty list if no numbers found."

        return create_model(
            "MultiNumberResponse",
            response=(
                List[Union[int, float]],
                Field(default_factory=list, description=list_description, **field_kwargs),
            ),
            __module__=__name__,
        )
    else:
        # Single extraction mode - return optional single value
        single_description = f"Extract a single numeric value (integer or decimal) from the text.{constraint_desc} Return null if no number found."

        return create_model(
            "NumberResponse",
            response=(
                Union[int, float, None],
                Field(
                    default=None,
                    description=single_description
                ),
            ),
            __module__=__name__,
        )


class DateRuleResponse(BaseModel):
    """Returns RRULE parameters for generating recurring dates.

    Convert natural language date patterns into RRULE parameters.
    """

    freq: str = Field(
        ...,
        description='Frequency: one of "DAILY", "WEEKLY", "MONTHLY", "YEARLY"'
    )
    dtstart: str = Field(
        ...,
        description="Starting date in ISO format (YYYY-MM-DD) or datetime with timezone. If no year is specified in the input, use the current year from the temporal context. For relative expressions like 'next month' or ambiguous month references like 'September' (without year), choose the year that makes most sense given the current date."
    )
    count: Optional[int] = Field(
        None,
        description="Number of occurrences (e.g., 5 for 'first 5 Tuesdays')"
    )
    until: Optional[str] = Field(
        None,
        description="End date in ISO format (YYYY-MM-DD)"
    )
    interval: Optional[int] = Field(
        None,
        description="Interval between occurrences (e.g., 2 for 'every other week')"
    )
    byweekday: Optional[List[int]] = Field(
        None,
        description="List of weekdays as integers 0-6 (Monday=0, Sunday=6)"
    )
    bymonth: Optional[List[int]] = Field(
        None,
        description="List of months as integers 1-12"
    )
    bymonthday: Optional[List[int]] = Field(
        None,
        description="List of days of month as integers 1-31"
    )
    bysetpos: Optional[List[int]] = Field(
        None,
        description="Position in set (e.g., [1] for 'first', [-1] for 'last')"
    )


def temporal_response_model(field_type, type_name, description, required=False, quantifier=None):
    """Factory to produce temporal response models.

    Args:
        field_type: The Python type (date, datetime, time, timedelta)
        type_name: The name of the type for the model
        description: Field description
        required: If True, field cannot be None and must be provided. If False (default), field is optional.
        quantifier: Optional tuple of (min_items, max_items) for list responses.
                   None means single value. Examples: (1, 3) = 1 to 3 items, (0, None) = 0 or more
    """
    # Map field types to their validators
    validator_map = {
        date: validate_date_input,
        datetime: validate_datetime_input,
        time: validate_time_input,
        timedelta: validate_timedelta_input,
    }
    validator = validator_map.get(field_type)
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

        return create_model(
            f"Multi{type_name}Response",
            response=(
                Union[List[field_type], List[str]],  # Accept dates OR pattern strings
                Field(default_factory=list, description=list_description, **field_kwargs),
            ),
            __module__=__name__,
        )
    else:
        # Single extraction mode
        if required:
            # Required field: accept field_type or str (for ISO strings/patterns), but NOT None
            single_description = f"""Extract a single {type_name.lower()} value. You have TWO OPTIONS:

Option 1 - EXPLICIT {type_name.upper()}: If you can determine a specific {type_name.lower()} directly from the text
(e.g., "Jan 15, 2024", "tomorrow", "3pm"), return it as a {type_name.lower()} value.

Option 2 - RECURRING PATTERN: If the input describes a recurring pattern
(e.g., "every Tuesday", "first 2 Tuesdays in October"), return a STRING that describes the pattern exactly as stated.
The pattern will be expanded automatically and the first occurrence will be used.

{description}"""

            # Use Annotated with BeforeValidator to convert strings to proper types
            if validator:
                validated_type = Annotated[Union[field_type, str], BeforeValidator(validator)]
            else:
                validated_type = Union[field_type, str]

            return create_model(
                f"{type_name}Response",
                response=(
                    validated_type,  # Accept date or pattern string, but NOT None
                    Field(
                        ...,  # Required field
                        description=single_description
                    ),
                ),
                __module__=__name__,
            )
        else:
            # Optional field: accept field_type, str, or None
            single_description = f"""Extract a single {type_name.lower()} value. You have TWO OPTIONS:

Option 1 - EXPLICIT {type_name.upper()}: If you can determine a specific {type_name.lower()} directly from the text
(e.g., "Jan 15, 2024", "tomorrow", "3pm"), return it as a {type_name.lower()} value.

Option 2 - RECURRING PATTERN: If the input describes a recurring pattern
(e.g., "every Tuesday", "first 2 Tuesdays in October"), return a STRING that describes the pattern exactly as stated.
The pattern will be expanded automatically and the first occurrence will be used.

{description} Return null if no {type_name.lower()} or pattern found."""

            # Use Annotated with BeforeValidator to convert strings to proper types
            if validator:
                validated_type = Annotated[Union[field_type, str, None], BeforeValidator(validator)]
            else:
                validated_type = Union[field_type, str, None]

            return create_model(
                f"{type_name}Response",
                response=(
                    validated_type,  # Accept date, pattern string, or None
                    Field(
                        default=None,
                        description=single_description
                    ),
                ),
                __module__=__name__,
            )


def date_response_model(options=None, quantifier=None):
    """Factory for date response models.

    Args:
        options: List that may contain 'required' flag
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple dates
    """
    required = options and "required" in options
    return temporal_response_model(
        date,
        "Date",
        "A date value. For relative dates (e.g., 'next Tuesday', 'tomorrow'), use the current date/time context provided. Return dates in ISO format (YYYY-MM-DD).",
        required=required,
        quantifier=quantifier
    )


def datetime_response_model(options=None, quantifier=None):
    """Factory for datetime response models.

    Args:
        options: List that may contain 'required' flag
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple datetimes
    """
    required = options and "required" in options
    return temporal_response_model(
        datetime,
        "Datetime",
        "A datetime value with timezone awareness. For relative datetimes (e.g., 'next Tuesday at 3pm'), use the current date/time context provided. Return in ISO format with timezone.",
        required=required,
        quantifier=quantifier
    )


def time_response_model(options=None, quantifier=None):
    """Factory for time response models.

    Args:
        options: List that may contain 'required' flag
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple times
    """
    required = options and "required" in options
    return temporal_response_model(
        time,
        "Time",
        "A time value. For times without explicit AM/PM, use context clues. Return in HH:MM:SS format.",
        required=required,
        quantifier=quantifier
    )


def duration_response_model(options=None, quantifier=None):
    """Factory for duration response models.

    Args:
        options: List that may contain 'required' flag
        quantifier: Optional tuple of (min_items, max_items) for extracting multiple durations
    """
    required = options and "required" in options
    return temporal_response_model(
        timedelta,
        "Duration",
        "A duration expressed as a timedelta. Extract from phrases like '2 hours', '3 days', '1 week and 2 days', '30 minutes'. Return the total duration in days and seconds.",
        required=required,
        quantifier=quantifier
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


ACTION_LOOKUP = defaultdict(lambda: DefaultResponse)
ACTION_LOOKUP.update(
    {
        "default": DefaultResponse,
        "respond": DefaultResponse,
        "extract": ExtractedResponse,
        "think": InternalThoughtsResponse,
        "speak": SpokenResponse,
        "number": number_response_model,
        "int": IntegerResponse,
        "pick": selection_response_model,
        "decide": BooleanResponse,
        "boolean": BooleanResponse,
        "bool": BooleanResponse,
        "poem": PoeticalResponse,
        "chunked_conversation": ChunkedConversationResponse,
        "date": date_response_model,
        "datetime": datetime_response_model,
        "time": time_response_model,
        "duration": duration_response_model,
        "date_rule": DateRuleResponse,
    }
)
