"""Result classes and run tracking for struckdown."""

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional

from box import Box
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


# Run ID for cache detection - uses contextvars for thread/async safety
# Works correctly in long-running processes (Django, Jupyter) by scoping to logical runs
# Fresh API calls will have current run ID, cached calls will have old/missing IDs
_run_id_var: ContextVar[Optional[str]] = ContextVar("run_id", default=None)


def get_run_id() -> str:
    """Get current run ID, auto-initializing if needed.

    Returns:
        Current run ID (auto-generated if not set)
    """
    run_id = _run_id_var.get()
    if run_id is None:
        run_id = str(uuid.uuid4())
        _run_id_var.set(run_id)
    return run_id


def new_run() -> str:
    """Start a new logical run with a fresh ID.

    Call this at the start of CLI commands or Django views to ensure
    cache detection works correctly in long-running processes.

    Returns:
        New run ID

    Example:
        # In Django view:
        from struckdown import new_run
        def my_view(request):
            new_run()  # Fresh run ID for this request
            result = chatter(...)
            ...
    """
    run_id = str(uuid.uuid4())
    _run_id_var.set(run_id)
    return run_id


# Progress callback for per-API-call updates
_progress_callback: ContextVar[Optional[Callable[[], None]]] = ContextVar(
    '_progress_callback', default=None
)


def get_progress_callback() -> Optional[Callable[[], None]]:
    """Get the current progress callback, if any."""
    return _progress_callback.get()


@contextmanager
def progress_tracking(on_api_call: Callable[[], None]):
    """Context manager for tracking individual API calls.

    Allows callers to receive a callback after each LLM completion,
    enabling real-time progress updates without changing the chatter() signature.

    Usage:
        def on_call():
            print("API call completed!")

        with progress_tracking(on_api_call=on_call):
            result = chatter(...)  # on_call() fires after each completion
    """
    token = _progress_callback.set(on_api_call)
    try:
        yield
    finally:
        _progress_callback.reset(token)


class SegmentResult(BaseModel):
    name: Optional[str] = Field(
        default=None, description="The slot/variable name for this result"
    )
    prompt: str
    output: Any
    completion: Optional[Any] = Field(default=None, exclude=False)
    action: Optional[str] = Field(
        default=None,
        description="Action type (e.g., 'pick', 'bool', 'int', 'evidence', 'memory')",
    )
    options: Optional[List[str]] = Field(
        default=None, description="Raw template options before variable resolution"
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Resolved action parameters (with variables interpolated)",
    )
    messages: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Full message list sent to LLM (system, user, assistant messages)",
    )
    response_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pydantic model schema (JSON Schema) for the expected response",
    )

    def get_schema_summary(self) -> str:
        """Extract key constraints from response schema.

        Returns a simplified string showing just the important constraints
        like type, enum values, min/max, required fields, etc.

        Returns:
            Formatted string with schema constraints, or message if no schema available
        """
        schema = self.response_schema

        if not schema or "properties" not in schema:
            return "No schema information available"

        # Get the response field schema (the actual constraint)
        response_schema = schema.get("properties", {}).get("response", {})

        if not response_schema:
            return "No response constraints"

        parts = []

        # Get title if available
        title = schema.get("title", "")
        if title:
            parts.append(f"Type: {title}")

        # Handle anyOf (optional types like Optional[bool])
        if "anyOf" in response_schema:
            types = []
            for option in response_schema["anyOf"]:
                if option.get("type") == "null":
                    continue  # Skip null, we'll indicate optionality differently

                # Check for enum
                if "enum" in option:
                    enum_values = ", ".join(f'"{v}"' for v in option["enum"])
                    types.append(f"enum: [{enum_values}]")
                elif "type" in option:
                    types.append(option["type"])

            if types:
                parts.append(f"Type: {' | '.join(types)}")
        elif "enum" in response_schema:
            # Direct enum
            enum_values = ", ".join(f'"{v}"' for v in response_schema["enum"])
            parts.append(f"Enum: [{enum_values}]")
        elif "type" in response_schema:
            parts.append(f"Type: {response_schema['type']}")

        # Get description
        if "description" in response_schema:
            desc = response_schema["description"]
            parts.append(f"Description: {desc}")

        # Get numeric constraints
        if "minimum" in response_schema:
            parts.append(f"Min: {response_schema['minimum']}")
        if "maximum" in response_schema:
            parts.append(f"Max: {response_schema['maximum']}")
        if "exclusiveMinimum" in response_schema:
            parts.append(f"Min (exclusive): {response_schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in response_schema:
            parts.append(f"Max (exclusive): {response_schema['exclusiveMaximum']}")

        # Get string constraints
        if "minLength" in response_schema:
            parts.append(f"Min length: {response_schema['minLength']}")
        if "maxLength" in response_schema:
            parts.append(f"Max length: {response_schema['maxLength']}")
        if "pattern" in response_schema:
            parts.append(f"Pattern: {response_schema['pattern']}")

        # Check if required
        required_fields = schema.get("required", [])
        if "response" in required_fields:
            parts.append("Required: Yes")
        else:
            parts.append("Required: No (can return null)")

        return "\n".join(parts)

    @model_validator(mode="before")
    @classmethod
    def reconstruct_typed_output(cls, data: Any) -> Any:
        """Reconstruct registered Pydantic models from dict during deserialization.

        When SegmentResult is loaded from JSON, action outputs (like FoundEvidenceSet)
        are plain dicts. This validator checks if the action has a registered return_type
        and reconstructs the Pydantic model, preserving computed fields and methods.
        """
        # import here to avoid circular import
        from struckdown.actions import Actions

        if isinstance(data, dict) and "action" in data and "output" in data:
            action_name = data.get("action")
            output = data.get("output")

            # check if this action has a registered return type
            if action_name and isinstance(output, dict):
                return_type = Actions.get_return_type(action_name)
                if return_type is not None:
                    try:
                        # reconstruct the Pydantic model from the dict
                        data["output"] = return_type.model_validate(output)
                    except Exception as e:
                        # if reconstruction fails, log warning but keep the dict
                        logger.warning(
                            f"Failed to reconstruct {return_type.__name__} for action '{action_name}': {e}"
                        )

        return data

    def __str__(self):
        """String representation returns the output's string representation"""
        return str(self.output)

    def __repr__(self):
        """Debug representation shows it's a SegmentResult with the output"""
        return f"SegmentResult(name={self.name!r}, output={self.output!r}, action={self.action!r})"

    def __eq__(self, other):
        """Equality compares the output value"""
        if isinstance(other, SegmentResult):
            return self.output == other.output
        return self.output == other

    def __bool__(self):
        """Boolean evaluation based on output"""
        return bool(self.output)

    def __hash__(self):
        """Make hashable based on output (if output is hashable)"""
        try:
            return hash(self.output)
        except TypeError:
            return hash(id(self))


class ChatterResult(BaseModel):
    type: str = Field(
        default="chatter", description="Discriminator field for union serialization"
    )
    results: Dict[str, SegmentResult] = Field(default_factory=dict)
    interim_results: Dict[str, List[SegmentResult]] = Field(
        default_factory=dict,
        description="Intermediate LLM calls and processing steps for multi-stage extractions (e.g., pattern expansion)",
    )

    def __str__(self):
        # abbreviate first prompt
        first_prompt = ""
        if self.results:
            first_seg = next(iter(self.results.values()))
            first_prompt = first_seg.prompt[:80] + ("..." if len(first_seg.prompt) > 80 else "")

        # show outputs dict
        outputs_repr = {k: str(v.output) for k, v in self.results.items()}

        return f"<ChatterResult: outputs={outputs_repr}, prompt='{first_prompt}'>"

    def __getitem__(self, key):
        return self.results[key].output

    def __setitem__(self, key: str, value: SegmentResult):
        """Set a result, enforcing SegmentResult type and ensuring name is set."""
        if not isinstance(value, SegmentResult):
            raise TypeError(
                f"ChatterResult only accepts SegmentResult values, got {type(value).__name__}"
            )
        if value.name is None:
            value.name = key
        self.results[key] = value

    def update(self, d: Dict[str, SegmentResult]):
        """Update results with a dict of SegmentResults, ensuring names are set."""
        for k, v in d.items():
            if v.name is None:
                v.name = k
            self.results[k] = v

    def keys(self):
        return self.results.keys()

    def __len__(self):
        return len(self.results)

    @property
    def response(self):
        # dict is insertion ordered python > 3.7
        if not self.results:
            return None
        last = self.results.get(next(reversed(self.results)), None)
        return last and last.output or None

    @property
    def outputs(self):
        return Box({k: v.output for k, v in self.results.items()})

    @property
    def total_cost(self) -> float:
        """Total USD cost from all segments (0.0 for cached/unavailable)"""
        return sum(
            (seg.completion._hidden_params.get("response_cost", 0.0) or 0.0)
            for seg in self.results.values()
            if seg.completion and hasattr(seg.completion, "_hidden_params")
        )

    @property
    def prompt_tokens(self) -> int:
        """Total input tokens across all segments"""
        return sum(
            (seg.completion.usage.prompt_tokens or 0)
            for seg in self.results.values()
            if seg.completion
            and hasattr(seg.completion, "usage")
            and seg.completion.usage
        )

    @property
    def completion_tokens(self) -> int:
        """Total output tokens across all segments"""
        return sum(
            (seg.completion.usage.completion_tokens or 0)
            for seg in self.results.values()
            if seg.completion
            and hasattr(seg.completion, "usage")
            and seg.completion.usage
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion)"""
        return self.prompt_tokens + self.completion_tokens

    @property
    def has_unknown_costs(self) -> bool:
        """True if ANY segment has unknown/missing cost data"""
        for seg in self.results.values():
            if not seg.completion:
                continue  # function executor, no cost
            if not hasattr(seg.completion, "_hidden_params"):
                return True  # missing cost data
            cost = seg.completion._hidden_params.get("response_cost")
            if cost is None:
                return True  # explicit None = unknown
        return False

    @property
    def all_costs_unknown(self) -> bool:
        """True if ALL segments with completions have unknown costs"""
        segments_with_completion = [
            seg
            for seg in self.results.values()
            if seg.completion and hasattr(seg.completion, "_hidden_params")
        ]
        if not segments_with_completion:
            return True  # no completions = unknown

        for seg in segments_with_completion:
            cost = seg.completion._hidden_params.get("response_cost")
            if cost is not None:
                return False  # found at least one known cost
        return True

    @property
    def fresh_call_count(self) -> int:
        """Count of segments from fresh API calls (not cached)"""
        current_run_id = get_run_id()
        return sum(
            1
            for seg in self.results.values()
            if seg.completion and seg.completion.get("_run_id") == current_run_id
        )

    @property
    def cached_call_count(self) -> int:
        """Count of segments from cache"""
        current_run_id = get_run_id()
        return sum(
            1
            for seg in self.results.values()
            if seg.completion and seg.completion.get("_run_id") != current_run_id
        )

    @property
    def fresh_cost(self) -> float:
        """Total USD cost from fresh API calls only (excludes cached)"""
        current_run_id = get_run_id()
        return sum(
            (seg.completion._hidden_params.get("response_cost", 0.0) or 0.0)
            for seg in self.results.values()
            if (
                seg.completion
                and hasattr(seg.completion, "_hidden_params")
                and seg.completion.get("_run_id") == current_run_id
            )
        )

    @property
    def cached_cost(self) -> float:
        """Total USD cost from cached calls (original cost when first made)"""
        return self.total_cost - self.fresh_cost

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # treat unknown sub-types as Any
    )


class CostSummary(BaseModel):
    """Aggregated cost summary from multiple ChatterResult objects.

    Consolidates cost tracking logic for display in CLIs.
    """

    total_cost: float = 0.0
    fresh_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    fresh_count: int = 0
    cached_count: int = 0
    has_unknown_costs: bool = False
    all_costs_unknown: bool = False

    @classmethod
    def from_results(cls, results: List["ChatterResult"]) -> "CostSummary":
        """Aggregate cost data from multiple ChatterResult objects.

        Args:
            results: List of ChatterResult objects to aggregate

        Returns:
            CostSummary with aggregated data
        """
        total_cost = 0.0
        fresh_cost = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        fresh_count = 0
        cached_count = 0
        has_unknown_costs = False
        all_costs_unknown = True

        for result in results:
            if result is None:
                continue

            total_cost += result.total_cost
            fresh_cost += result.fresh_cost
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            fresh_count += result.fresh_call_count
            cached_count += result.cached_call_count

            if result.has_unknown_costs:
                has_unknown_costs = True
            if not result.all_costs_unknown:
                all_costs_unknown = False

        return cls(
            total_cost=total_cost,
            fresh_cost=fresh_cost,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            fresh_count=fresh_count,
            cached_count=cached_count,
            has_unknown_costs=has_unknown_costs,
            all_costs_unknown=all_costs_unknown,
        )

    def format_summary(self, include_breakdown: bool = True) -> str:
        """Format cost summary for display.

        Args:
            include_breakdown: Include cache breakdown if available

        Returns:
            Formatted string ready for stderr output
        """
        lines = []

        # main line - handle unknown costs
        if self.all_costs_unknown:
            lines.append(
                f"Total cost: unknown "
                f"({self.total_prompt_tokens:,} in / {self.total_completion_tokens:,} out)"
            )
        elif self.has_unknown_costs:
            lines.append(
                f"Total cost: >=${self.total_cost:.4f} "
                f"({self.total_prompt_tokens:,} in / {self.total_completion_tokens:,} out, some costs unknown)"
            )
        else:
            lines.append(
                f"Total cost: ${self.total_cost:.4f} "
                f"({self.total_prompt_tokens:,} in / {self.total_completion_tokens:,} out)"
            )

            # cache breakdown (only if not unknown and has cached calls)
            if include_breakdown and self.cached_count > 0:
                lines.append(
                    f"  This run: ${self.fresh_cost:.4f} "
                    f"({self.fresh_count} fresh, {self.cached_count} cached)"
                )

        return "\n".join(lines)


class StruckdownEarlyTermination(Exception):
    """Raised when [[end]] or [[@break]] is encountered to stop template execution."""
    def __init__(self, message, partial_results=None):
        super().__init__(message)
        self.partial_results = partial_results if partial_results is not None else ChatterResult()
