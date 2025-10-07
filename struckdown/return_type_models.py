from collections import defaultdict
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

# Standard RTs for template syntax


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
        "number": IntegerResponse,
        "int": IntegerResponse,
        "pick": selection_response_model,
        "decide": BooleanResponse,
        "boolean": BooleanResponse,
        "bool": BooleanResponse,
        "poem": PoeticalResponse,
        "chunked_conversation": ChunkedConversationResponse,
    }
)
