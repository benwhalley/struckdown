"""Segment execution logic for struckdown."""

import logging
from collections import OrderedDict
from types import FunctionType
from typing import List, Optional

import anyio
from jinja2 import Environment, StrictUndefined, UndefinedError

from .errors import StruckdownTemplateError
from .jinja_utils import extract_jinja_variables, struckdown_finalize, escape_struckdown_syntax
from .llm import LC, LLM, LLMCredentials, structured_chat
from .results import (
    ChatterResult,
    SegmentResult,
    StruckdownEarlyTermination,
    get_progress_callback,
)
from .return_type_models import LLMConfig
from .temporal_patterns import expand_temporal_pattern
from .validation import parse_options, validate_number_constraints

logger = logging.getLogger(__name__)


async def process_single_segment_(
    segment: OrderedDict, llm: LLM, credentials: Optional[LLMCredentials], context={}, **kwargs
):
    """
    Process a single segment sequentially, building context as we go.
    Builds proper message threads with system, user, and assistant messages.

    Message structure per completion:
    - First completion: [system, user(header + prompt)]
    - Second completion: [system, user(header + prompt1), assistant(response1), user(prompt2)]
    - And so on...

    System and header are re-rendered with accumulated context before each completion.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    results = ChatterResult()
    accumulated_context = context.copy()
    logger.debug(
        f"Initial context keys at segment start: {list(accumulated_context.keys())}"
    )

    # Inject temporal context for date/time extractions
    uses_temporal_types = any(
        prompt_part.action_type in ["date", "datetime", "time", "duration"]
        for prompt_part in segment.values()
    )

    if uses_temporal_types:
        try:
            current_dt = datetime.now().astimezone()
        except Exception:
            current_dt = datetime.now(ZoneInfo("UTC"))

        accumulated_context["_current_date"] = current_dt.date().isoformat()
        accumulated_context["_current_time"] = current_dt.time().isoformat()
        accumulated_context["_current_datetime"] = current_dt.isoformat()
        accumulated_context["_current_timezone"] = str(current_dt.tzinfo)

    # Track message history within this segment
    segment_history = []

    termination_requested = False
    break_message = None
    for idx, (key, prompt_part) in enumerate(segment.items()):
        # Check for break tag (early termination)
        if prompt_part.is_break:
            termination_requested = True
            break_message = prompt_part.break_message
            if break_message:
                accumulated_context["_break_message"] = break_message
            break

        # Check for old-style early termination marker
        if key.lower() == "end" and prompt_part.required_prefix:
            termination_requested = True
            break

        is_first_completion = (idx == 0)

        # Re-render system and header with current accumulated context
        env_with_finalize = Environment(
            undefined=StrictUndefined,
            finalize=struckdown_finalize,
        )

        def render_template_safe(template_str: str, context: dict, source_desc: str) -> str:
            """Render template with user-friendly error on undefined variables."""
            try:
                template = env_with_finalize.from_string(template_str)
                return template.render(**context)
            except UndefinedError as e:
                raise StruckdownTemplateError(
                    message=str(e),
                    original_error=e,
                    line_number=prompt_part.line_number,
                    context_variables=list(context.keys()),
                ) from e

        system_message = ""
        if prompt_part.system_message:
            system_message = render_template_safe(
                prompt_part.system_message, accumulated_context, "system message"
            )

        # Build messages list for this completion
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        user_prompt = prompt_part.text

        # Add temporal context hint if needed
        temporal_hint = ""
        if prompt_part.action_type in ["date", "datetime", "time", "duration", "date_rule"]:
            temporal_hint = f"\n\n--- TEMPORAL CONTEXT (for resolving relative references only) ---\nUse this ONLY to resolve relative temporal expressions like 'tomorrow', 'next week', 'in 3 days', etc.\nDO NOT return these values as your answer. Extract temporal information from the INPUT TEXT above.\nReturn null if no temporal information can be found or interpreted in the input text.\n\nCurrent Date: {accumulated_context.get('_current_date', 'N/A')}\nCurrent Time: {accumulated_context.get('_current_time', 'N/A')}\nTimezone: {accumulated_context.get('_current_timezone', 'N/A')}\n--- END CONTEXT ---"

        rendered_user_content = render_template_safe(
            user_prompt + temporal_hint + "\n\nAlways use the tools/JSON response.\n\n```json\n",
            accumulated_context,
            "user prompt",
        )

        messages.extend(segment_history)
        messages.append({"role": "user", "content": rendered_user_content})

        logger.debug(f"Template context keys: {list(accumulated_context.keys())}")
        logger.debug(f"Built message list with {len(messages)} messages")

        # Determine the appropriate return type
        if isinstance(prompt_part.return_type, FunctionType):
            required_prefix = getattr(prompt_part, "required_prefix", False)

            if prompt_part.action_type in ["date", "datetime", "time", "duration", "number"]:
                rt = prompt_part.return_type(
                    prompt_part.options, prompt_part.quantifier, required_prefix
                )
            else:
                rt = prompt_part.return_type(
                    prompt_part.options, prompt_part.quantifier, required_prefix
                )
        else:
            rt = prompt_part.return_type

        # Build LLM kwargs
        if hasattr(rt, "llm_config") and isinstance(rt.llm_config, LLMConfig):
            llm_config = rt.llm_config.model_copy()
        else:
            llm_config = LLMConfig()

        if hasattr(prompt_part, "llm_kwargs") and prompt_part.llm_kwargs:
            try:
                llm_config = llm_config.model_copy(update=prompt_part.llm_kwargs)
            except Exception as e:
                logger.warning(f"Invalid LLM parameters in prompt: {e}")

        if kwargs:
            current_config = llm_config.model_dump(exclude_none=True)
            for k, v in kwargs.items():
                if k not in current_config:
                    try:
                        llm_config = llm_config.model_copy(update={k: v})
                    except Exception:
                        pass

        llm_kwargs = llm_config.model_dump(exclude_none=True)

        slot_model = llm_kwargs.pop("model", None)
        if slot_model:
            slot_llm = LLM(model_name=slot_model)
        else:
            slot_llm = llm

        is_function_call = getattr(prompt_part, "is_function", False)

        if is_function_call and hasattr(rt, "_executor"):
            logger.debug(
                f"{LC.CYAN}Function call: {key} (action: {prompt_part.action_type}){LC.RESET}"
            )
            logger.debug(
                f"accumulated_context keys before executor: {list(accumulated_context.keys())}"
            )
            res, completion_obj = rt._executor(
                accumulated_context, rendered_user_content, **llm_kwargs
            )
            resolved_params = getattr(res, "_resolved_params", None)
        else:
            res, completion_obj = await anyio.to_thread.run_sync(
                lambda: structured_chat(
                    messages=messages,
                    return_type=rt,
                    llm=slot_llm,
                    credentials=credentials,
                    extra_kwargs=llm_kwargs,
                ),
                abandon_on_cancel=True,
            )
            resolved_params = None

        if hasattr(res, "response"):
            extracted_value = res.response
        else:
            extracted_value = res

        # Auto-inject context values into ResponseModel instances
        def inject_context_recursively(obj):
            capture_fields = getattr(type(obj), "_capture_from_context", [])
            if hasattr(capture_fields, "default"):
                capture_fields = capture_fields.default or []

            if capture_fields:
                for field_name in capture_fields:
                    if field_name in accumulated_context and hasattr(obj, field_name):
                        setattr(obj, field_name, accumulated_context[field_name])

            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in obj.__dict__.items():
                    if isinstance(attr_value, list):
                        for item in attr_value:
                            inject_context_recursively(item)

        if isinstance(extracted_value, list):
            for item in extracted_value:
                inject_context_recursively(item)
        else:
            inject_context_recursively(extracted_value)

        # Call post_process hook
        def call_post_process(obj):
            if hasattr(obj, "post_process") and callable(obj.post_process):
                obj.post_process(accumulated_context)

            if hasattr(obj, "__dict__"):
                for attr_value in obj.__dict__.values():
                    if isinstance(attr_value, list):
                        for item in attr_value:
                            if hasattr(item, "post_process"):
                                call_post_process(item)

        if isinstance(extracted_value, list):
            for item in extracted_value:
                call_post_process(item)
        else:
            call_post_process(extracted_value)

        # Handle date/datetime pattern expansion via RRULE
        if prompt_part.action_type in ["date", "datetime"]:
            pattern_string = None
            is_single_value = not prompt_part.quantifier

            if is_single_value:
                if isinstance(extracted_value, str):
                    pattern_string = extracted_value
            else:
                if isinstance(extracted_value, list) and len(extracted_value) > 0:
                    if isinstance(extracted_value[0], str):
                        pattern_string = extracted_value[0]

            if pattern_string:
                extracted_value, interim_steps = await expand_temporal_pattern(
                    pattern_string=pattern_string,
                    action_type=prompt_part.action_type,
                    is_single_value=is_single_value,
                    quantifier=prompt_part.quantifier,
                    llm=llm,
                    credentials=credentials,
                    accumulated_context=accumulated_context,
                )
                if interim_steps:
                    if key not in results.interim_results:
                        results.interim_results[key] = []
                    results.interim_results[key].extend(interim_steps)

        # Validate required temporal fields
        if prompt_part.action_type in ["date", "datetime", "time", "duration"]:
            if not prompt_part.quantifier:
                is_required = prompt_part.options and "required" in prompt_part.options
                if is_required and extracted_value is None:
                    raise ValueError(
                        f"Required temporal field '{key}' (type: {prompt_part.action_type}) "
                        f"could not be extracted from the input text. "
                        f"Please ensure the input contains valid {prompt_part.action_type} information."
                    )

        # Validate numeric fields
        if prompt_part.action_type == "number":
            opts = parse_options(prompt_part.options)
            extracted_value = validate_number_constraints(
                extracted_value,
                field_name=key,
                min_val=opts.ge,
                max_val=opts.le,
                is_required=opts.required,
            )

        # Capture response schema
        response_schema = None
        if not is_function_call:
            try:
                response_schema = rt.model_json_schema()
            except Exception:
                pass

        results[key] = SegmentResult(
            name=key,
            output=extracted_value,
            completion=completion_obj,
            prompt=rendered_user_content,
            action=prompt_part.action_type,
            options=prompt_part.options if prompt_part.options else None,
            params=resolved_params,
            messages=messages,
            response_schema=response_schema,
        )

        escaped_value, was_escaped = escape_struckdown_syntax(extracted_value, var_name=key)
        accumulated_context[key] = escaped_value
        logger.debug(
            f"Added '{key}' to accumulated_context. Keys now: {list(accumulated_context.keys())}"
        )

        # Fire progress callback
        callback = get_progress_callback()
        if callback:
            callback()

        # Check for [[@break]] action
        if accumulated_context.get('_break_requested'):
            break_msg = accumulated_context.get('_break_message', '')
            logger.info(f"Break action triggered: {break_msg}")
            raise StruckdownEarlyTermination(
                f"Break requested: {break_msg}",
                partial_results=results
            )

        # Add this exchange to segment history
        segment_history.append({"role": "user", "content": rendered_user_content})

        if is_function_call:
            segment_history.append({"role": "user", "content": str(extracted_value)})
        else:
            segment_history.append({"role": "assistant", "content": str(extracted_value)})

    if termination_requested:
        raise StruckdownEarlyTermination(
            f"Execution stopped at [[end]] marker",
            partial_results=results
        )

    return results


class SegmentDependencyGraph:
    """Analyzes dependencies between segments and determines execution order"""

    def __init__(self, segments: List[OrderedDict]):
        self.segments = segments
        self.dependency_graph = {}
        self.segment_vars = {}
        self.build_dependency_graph()

    def get_segment_display_name(self, segment_id: str) -> str:
        """Get a human-readable name for a segment."""
        idx = int(segment_id.split("_")[1])
        if idx < len(self.segments):
            segment = self.segments[idx]
            segment_name = getattr(segment, 'segment_name', None)
            if segment_name:
                return f"{segment_name} ({segment_id})"
        return segment_id

    def build_dependency_graph(self):
        # First pass: identify variables defined in each segment
        for i, segment in enumerate(self.segments):
            segment_id = f"segment_{i}"
            self.segment_vars[segment_id] = set(segment.keys())
            self.dependency_graph[segment_id] = set()

        # Second pass: identify dependencies between segments
        for i, segment in enumerate(self.segments):
            segment_id = f"segment_{i}"
            template_vars = set()
            for prompt_part in segment.values():
                template_vars.update(extract_jinja_variables(prompt_part.text))

                if hasattr(prompt_part, 'system_message') and prompt_part.system_message:
                    template_vars.update(extract_jinja_variables(prompt_part.system_message))

                if prompt_part.options and isinstance(prompt_part.options, (list, tuple)):
                    for option in prompt_part.options:
                        template_vars.update(extract_jinja_variables(option))

            for var in template_vars:
                for j in range(i):
                    dep_segment_id = f"segment_{j}"
                    if var in self.segment_vars[dep_segment_id]:
                        self.dependency_graph[segment_id].add(dep_segment_id)

        # Third pass: handle blocking completions
        for i, segment in enumerate(self.segments):
            segment_id = f"segment_{i}"
            has_blocking = any(
                hasattr(part, 'block') and part.block
                for part in segment.values()
            )
            if has_blocking:
                for j in range(i + 1, len(self.segments)):
                    later_segment_id = f"segment_{j}"
                    self.dependency_graph[later_segment_id].add(segment_id)
                    logger.debug(
                        f"Blocking completion in {self.get_segment_display_name(segment_id)} "
                        f"-> {self.get_segment_display_name(later_segment_id)} depends on it"
                    )

    def get_execution_plan(self) -> List[List[str]]:
        """Returns a list of batches that can be executed in parallel"""
        remaining = set(self.dependency_graph.keys())
        execution_plan = []

        while remaining:
            ready = {
                seg_id
                for seg_id in remaining
                if all(dep not in remaining for dep in self.dependency_graph[seg_id])
            }

            if not ready and remaining:
                logging.warning(f"Circular dependency detected in segments: {remaining}")
                execution_plan.extend([[seg_id] for seg_id in remaining])
                break

            execution_plan.append(list(ready))
            remaining -= ready

        return execution_plan


async def merge_contexts(*contexts):
    """Must be a task to preserve ordering/graph in chatter"""
    merged = {}
    if contexts:
        for c in contexts:
            if isinstance(c, ChatterResult):
                for key, segment_result in c.results.items():
                    merged[key] = segment_result.output
            else:
                merged.update(c)
    return merged
