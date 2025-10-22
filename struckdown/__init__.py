import logging
import re
import traceback
import warnings
from collections import OrderedDict
from types import FunctionType
from typing import Any, Dict, List, Optional

import anyio
import instructor
import litellm
import openai
from box import Box
from decouple import config as env_config
from instructor import from_openai, Mode
from jinja2 import StrictUndefined, Template, Undefined
from more_itertools import chunked
from pydantic import BaseModel, ConfigDict, Field

from struckdown.actions import Actions
from struckdown.response_types import ResponseTypes
from struckdown.cache import clear_cache, memory
from struckdown.parsing import parser, _add_default_completion_if_needed
from struckdown.return_type_models import ACTION_LOOKUP, LLMConfig
from struckdown.temporal_patterns import expand_temporal_pattern
from struckdown.number_validation import parse_number_options, validate_number_constraints


# litellm._turn_on_debug()

# Version - reads from package metadata (set in pyproject.toml)
try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("struckdown")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source in dev mode)
    __version__ = "0.0.0+dev"

# Suppress Pydantic serialization warnings from OpenAI/Anthropic SDK completion objects
# These occur when serializing completion metadata and are benign
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

logger = logging.getLogger(__name__)


class KeepUndefined(Undefined):
    """Custom Undefined class that preserved {{vars}} if they are not defined in context."""

    def __str__(self):
        return f"{{{{ {self._undefined_name} }}}}"


class Example(BaseModel):
    name: str
    age: int


LC = Box(
    {
        "RED": "\033[91m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "BLUE": "\033[94m",
        "PURPLE": "\033[95m",  # sometimes called MAGENTA
        "CYAN": "\033[96m",
        "ORANGE": "\033[38;5;208m",  # extended colour, may not work in all terminals
        "RESET": "\033[0m",
    }
)


class LLMCredentials(BaseModel):
    api_key: Optional[str] = Field(
        default_factory=lambda: env_config("LLM_API_KEY", None), repr=False
    )
    base_url: Optional[str] = Field(
        default_factory=lambda: env_config("LLM_API_BASE", None), repr=False
    )


class LLM(BaseModel):
    model_name: Optional[str] = Field(
        default_factory=lambda: env_config("DEFAULT_LLM", "gpt-4.1-mini"),
        exclude=True,
    )

    def client(self, credentials: LLMCredentials = None):
        if credentials is None:
            credentials = LLMCredentials()

        if not credentials.api_key or not credentials.base_url:
            raise Exception("Set LLM_API_KEY and LLM_API_BASE environment variables")

        # Create OpenAI-compatible instructor client (works with litellm proxies)
        litellm.api_key=credentials.api_key
        litellm.api_base=credentials.base_url
        litellm.drop_params=True
        client = instructor.from_litellm(litellm.completion)
        return client


@memory.cache(ignore=["return_type", "llm", "credentials"])
def _call_llm_cached(
    prompt: str,
    model_name: str,
    max_retries: int,
    max_tokens: Optional[int],
    extra_kwargs: Optional[dict],
    return_type,
    llm,
    credentials,
):
    """
    Cache the raw completion dict from the LLM.
    This is the expensive API call we want to cache.
    Returns dicts (not Pydantic models) so they pickle safely.
    """
    logger.info(f"\n\n{LC.BLUE}Prompt: {prompt}{LC.RESET}\n\n")
    try:
        res, com = llm.client(credentials).chat.completions.create_with_completion(
            model=model_name,
            response_model=return_type,
            messages=[{"role": "user", "content": prompt}],
            **(extra_kwargs if extra_kwargs else {}),
        )
    except Exception as e:
        full_traceback = traceback.format_exc()
        logger.warning(f"Error calling LLM: {e}\n{full_traceback}")
        raise e

    logger.info(f"\n\n{LC.GREEN}Response: {res}{LC.RESET}\n")

    # Serialize to dicts for safe pickling (instructor always returns Pydantic models)
    return res.model_dump(), com.model_dump()


def structured_chat(
    prompt,
    return_type,
    llm: LLM = LLM(),
    credentials=LLMCredentials(),
    max_retries=3,
    max_tokens=None,
    extra_kwargs=None,
):
    """
    Use instructor to make a tool call to an LLM, returning the `response` field, and a completion object.

    Results are cached to disk using joblib.Memory. Cache behavior can be controlled via the
    STRUCKDOWN_CACHE environment variable:
    - Default: ~/.struckdown/cache
    - Disable: Set to "0", "false", or empty string
    - Custom location: Set to any valid directory path

    Cache key includes: prompt, model_name, max_retries, max_tokens, extra_kwargs
    Credentials are NOT included in the cache key (same prompt + model will hit cache regardless of API key).
    """
    logger.debug(
        f"Using model {llm.model_name}, max_retries {max_retries}, max_tokens: {max_tokens}"
    )
    logger.debug(f"LLM kwargs: {extra_kwargs}")
    try:
        res_dict, com_dict = _call_llm_cached(
            prompt=prompt,
            model_name=llm.model_name,
            max_retries=max_retries,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs or {},
            return_type=return_type,
            llm=llm,
            credentials=credentials,
        )

        # Deserialize dicts back to Pydantic models (cached function always returns dicts)
        res = return_type.model_validate(res_dict)
        com = Box(com_dict)

        logger.debug(
            f"{LC.PURPLE}Response type: {type(res)}; {len(str(res))} tokens produced{LC.RESET}\n\n"
        )
        return res, com

    except (EOFError, Exception) as e:
        # If cache fails, log and re-raise
        logger.warning(f"Cache/LLM error: {e}")
        raise e


class SegmentResult(BaseModel):
    prompt: str
    output: Any
    completion: Optional[Any] = Field(default=None, exclude=False)
    action: Optional[str] = Field(default=None, description="Action type (e.g., 'pick', 'bool', 'int')")
    options: Optional[List[str]] = Field(default=None, description="Valid options for pick-type actions")


class ChatterResult(BaseModel):
    type: str = Field(
        default="chatter", description="Discriminator field for union serialization"
    )
    results: Dict[str, SegmentResult] = Field(default_factory=dict)
    interim_results: Dict[str, List[SegmentResult]] = Field(
        default_factory=dict,
        description="Intermediate LLM calls and processing steps for multi-stage extractions (e.g., pattern expansion)"
    )

    def __str__(self):
        return "\n\n".join([f"{k}: {v.output}" for k, v in self.results.items()])

    def __getitem__(self, key):
        return self.results[key].output

    def __setitem__(self, key, value):
        # Assume caller is giving raw output — use None for completion by default
        if isinstance(value, SegmentResult):
            self.results[key] = value
        else:
            self.results[key] = SegmentResult(prompt="", output=value)

    def update(self, d: Dict[str, Any]):
        for k, v in d.items():
            self[k] = v

    def keys(self):
        return self.results.keys()

    def __len__(self):
        return len(self.results)

    @property
    def response(self):
        # dict is insertion ordered python > 3.7
        last = self.results.get(next(reversed(self.results)), None)
        return last and last.output or None

    @property
    def outputs(self):
        return Box({k: v.output for k, v in self.results.items()})

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # treat unknown sub‑types as Any
    )


async def process_single_segment_(
    segment: str, llm: LLM, credentials: Optional[LLMCredentials], context={}, **kwargs
):
    """
    Process a single segment sequentially, building context as we go.
    This is used by both single-segment prompts and as a building block
    for parallel processing.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    results = ChatterResult()
    prompt_parts = []
    accumulated_context = context.copy()

    # Inject temporal context for date/time extractions
    # Check if any prompt parts use temporal response types
    uses_temporal_types = any(
        prompt_part.action_type in ["date", "datetime", "time", "duration"]
        for prompt_part in segment.values()
    )

    if uses_temporal_types:
        # Get current datetime with timezone
        try:
            # Try to get local timezone
            current_dt = datetime.now().astimezone()
        except Exception:
            # Fallback to UTC if timezone detection fails
            current_dt = datetime.now(ZoneInfo("UTC"))

        # Add temporal context that won't conflict with user variables
        accumulated_context["_current_date"] = current_dt.date().isoformat()
        accumulated_context["_current_time"] = current_dt.time().isoformat()
        accumulated_context["_current_datetime"] = current_dt.isoformat()
        accumulated_context["_current_timezone"] = str(current_dt.tzinfo)

    # Extract shared_header from first prompt_part (all parts in segment share same header)
    shared_header = ""
    if segment:
        first_part = next(iter(segment.values()))
        shared_header = (
            first_part.shared_header if hasattr(first_part, "shared_header") else ""
        )

    for key, prompt_part in segment.items():
        # Append the raw text for this prompt part.
        prompt_parts.append(prompt_part.text)
        # Build the prompt for this completion from the parts within this segment.
        try:
            segment_prompt = "\n\n--\n\n".join(filter(bool, map(str, prompt_parts)))
            # Prepend shared_header once for the entire segment
            if shared_header:
                segment_prompt = (
                    f"{shared_header}\n\n{segment_prompt}"
                    if segment_prompt
                    else shared_header
                )
        except Exception as e:
            logger.error(f"Error building segment prompt: {prompt_parts}\n{e}")

        # Add temporal context hint if this is a temporal extraction
        temporal_hint = ""
        if prompt_part.action_type in ["date", "datetime", "time", "duration", "date_rule"]:
            temporal_hint = f"\n\n--- TEMPORAL CONTEXT (for resolving relative references only) ---\nUse this ONLY to resolve relative temporal expressions like 'tomorrow', 'next week', 'in 3 days', etc.\nDO NOT return these values as your answer. Extract temporal information from the INPUT TEXT above.\nReturn null if no temporal information can be found or interpreted in the input text.\n\nCurrent Date: {accumulated_context.get('_current_date', 'N/A')}\nCurrent Time: {accumulated_context.get('_current_time', 'N/A')}\nTimezone: {accumulated_context.get('_current_timezone', 'N/A')}\n--- END CONTEXT ---"

        # Render the prompt template.
        template = Template(
            segment_prompt
            + temporal_hint
            + "\nAlways use the tools/JSON response.\n\n```json\n",
            undefined=StrictUndefined,
        )
        rendered_prompt = template.render(**accumulated_context)

        # Debug the context to see what's available to template tags
        logger.debug(f"Template context keys: {list(accumulated_context.keys())}")

        # Determine the appropriate return type.
        if isinstance(prompt_part.return_type, FunctionType):
            # Get required_prefix flag if available (for ! prefix support)
            required_prefix = getattr(prompt_part, 'required_prefix', False)

            # Factory functions: all now support both options and quantifier
            if prompt_part.action_type in ["date", "datetime", "time", "duration", "number"]:
                # Temporal and numeric types use both options (for constraints/flags) and quantifier (for lists)
                rt = prompt_part.return_type(prompt_part.options, prompt_part.quantifier, required_prefix)
            else:
                # Other factory functions like pick also use both + required_prefix
                rt = prompt_part.return_type(prompt_part.options, prompt_part.quantifier, required_prefix)
        else:
            rt = prompt_part.return_type

        # Build LLM kwargs: start with model defaults, then apply slot-specific overrides
        # Start with the response model's LLM config defaults
        if hasattr(rt, 'llm_config') and isinstance(rt.llm_config, LLMConfig):
            llm_config = rt.llm_config.model_copy()
        else:
            llm_config = LLMConfig()  # Fall back to base defaults

        # Apply slot-specific overrides from [[type:var|temperature=X,model=Y]]
        if hasattr(prompt_part, 'llm_kwargs') and prompt_part.llm_kwargs:
            try:
                # Use Pydantic to validate and convert types automatically
                llm_config = llm_config.model_copy(update=prompt_part.llm_kwargs)
            except Exception as e:
                logger.warning(f"Invalid LLM parameters in prompt: {e}")

        # Merge with any global extra_kwargs (global kwargs take lowest priority)
        # Only add global kwargs that aren't already set
        if kwargs:
            current_config = llm_config.model_dump(exclude_none=True)
            for k, v in kwargs.items():
                if k not in current_config:
                    try:
                        llm_config = llm_config.model_copy(update={k: v})
                    except Exception:
                        # If it's not a valid LLM config param, ignore it
                        pass

        # Convert to dict for API call, excluding None values
        llm_kwargs = llm_config.model_dump(exclude_none=True)

        # Handle model override by creating a new LLM instance if needed
        slot_model = llm_kwargs.pop('model', None)
        if slot_model:
            slot_llm = LLM(model_name=slot_model)
        else:
            slot_llm = llm

        # Check if this is a function call (no LLM) or a completion (LLM)
        is_function_call = getattr(prompt_part, 'is_function', False)

        if is_function_call and hasattr(rt, '_executor'):
            # Execute custom function instead of calling LLM
            logger.debug(f"{LC.CYAN}Function call: {key} (action: {prompt_part.action_type}){LC.RESET}")
            res, completion_obj = rt._executor(
                accumulated_context,
                rendered_prompt,
                **llm_kwargs
            )
        else:
            # Call the LLM via structured_chat.
            res, completion_obj = await anyio.to_thread.run_sync(
                lambda: structured_chat(
                    rendered_prompt,
                    return_type=rt,
                    llm=slot_llm,
                    credentials=credentials,
                    extra_kwargs=llm_kwargs,
                ),
                abandon_on_cancel=True,
            )

        # Extract .response field if it exists (even if None)
        # For temporal types, we need to check the actual response value
        if hasattr(res, "response"):
            extracted_value = res.response
        else:
            extracted_value = res

        # Handle date/datetime pattern expansion via RRULE
        if prompt_part.action_type in ["date", "datetime"]:
            pattern_string = None
            is_single_value = not prompt_part.quantifier

            # Check if we have a pattern string
            if is_single_value:
                # For single values, check if result is a string
                if isinstance(extracted_value, str):
                    pattern_string = extracted_value
            else:
                # For lists, check if first element is a string
                if isinstance(extracted_value, list) and len(extracted_value) > 0:
                    if isinstance(extracted_value[0], str):
                        pattern_string = extracted_value[0]

            if pattern_string:
                # Use the extracted function to handle pattern expansion
                extracted_value, interim_steps = await expand_temporal_pattern(
                    pattern_string=pattern_string,
                    action_type=prompt_part.action_type,
                    is_single_value=is_single_value,
                    quantifier=prompt_part.quantifier,
                    llm=llm,
                    credentials=credentials,
                    accumulated_context=accumulated_context,
                )
                # Store interim steps in results
                if interim_steps:
                    if key not in results.interim_results:
                        results.interim_results[key] = []
                    results.interim_results[key].extend(interim_steps)

        # Validate required temporal fields (only applies to single values, not lists)
        # For lists, quantifier min_length already handles validation
        if prompt_part.action_type in ["date", "datetime", "time", "duration"]:
            if not prompt_part.quantifier:  # Only validate single values
                is_required = prompt_part.options and "required" in prompt_part.options
                if is_required and extracted_value is None:
                    raise ValueError(
                        f"Required temporal field '{key}' (type: {prompt_part.action_type}) "
                        f"could not be extracted from the input text. "
                        f"Please ensure the input contains valid {prompt_part.action_type} information."
                    )

        # Validate numeric fields with min/max constraints
        if prompt_part.action_type == "number":
            min_val, max_val, is_required = parse_number_options(prompt_part.options)
            extracted_value = validate_number_constraints(
                extracted_value,
                field_name=key,
                min_val=min_val,
                max_val=max_val,
                is_required=is_required,
            )

        # Store the completion in both our final results and accumulated context.
        results[key] = SegmentResult(
            output=extracted_value,
            completion=completion_obj,
            prompt=rendered_prompt,
            action=prompt_part.action_type,
            options=prompt_part.options if prompt_part.options else None,
        )

        accumulated_context[key] = extracted_value

        # For this segment, include the result in the prompt parts.
        prompt_parts.append(extracted_value)

    return results


class SegmentDependencyGraph:
    """Analyzes dependencies between segments and determines execution order"""

    def __init__(self, segments: List[OrderedDict]):
        self.segments = segments
        self.dependency_graph = {}  # segment_id -> set of segment_ids it depends on
        self.segment_vars = (
            {}
        )  # segment_id -> set of variable names defined in this segment
        self.build_dependency_graph()

    def build_dependency_graph(self):
        # First pass: identify variables defined in each segment
        for i, segment in enumerate(self.segments):
            segment_id = f"segment_{i}"
            # segment is already an OrderedDict from parser
            self.segment_vars[segment_id] = set(segment.keys())
            self.dependency_graph[segment_id] = set()

        # Second pass: identify dependencies between segments
        for i, segment in enumerate(self.segments):
            segment_id = f"segment_{i}"
            # Find all template variables {{ VAR}} in the segment by examining the text of each prompt part
            template_vars = set()
            for prompt_part in segment.values():
                template_vars.update(
                    re.findall(r"\{\{\s*(\w+)\s*\}\}", prompt_part.text)
                )

            # For each template variable, find which earlier segment defines it
            for var in template_vars:
                for j in range(i):
                    dep_segment_id = f"segment_{j}"
                    if var in self.segment_vars[dep_segment_id]:
                        self.dependency_graph[segment_id].add(dep_segment_id)

    def get_execution_plan(self) -> List[List[str]]:
        """
        Returns a list of batches, where each batch is a list of segment_ids
        that can be executed in parallel
        """
        remaining = set(self.dependency_graph.keys())
        execution_plan = []

        while remaining:
            # segments with no unprocessed dependencies
            ready = {
                seg_id
                for seg_id in remaining
                if all(dep not in remaining for dep in self.dependency_graph[seg_id])
            }

            if not ready and remaining:
                # Circular dependency detected
                logging.warning(
                    f"Circular dependency detected in segments: {remaining}"
                )
                # Fall back to sequential execution for remaining segments
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
                # extract individual variables from ChatterResult
                for key, segment_result in c.results.items():
                    merged[key] = segment_result.output
            else:
                merged.update(c)
    return merged


async def chatter_async(
    multipart_prompt: str,
    model: LLM = LLM(),
    credentials: Optional[LLMCredentials] = None,
    context={},
    extra_kwargs=None,
):
    """
    example:
    chatter("tell a joke [[joke]]")
    """

    logger.debug(f"\n\n{LC.ORANGE}Chatter Prompt: {multipart_prompt}{LC.RESET}\n\n")

    multipart_prompt_prefilled = Template(
        multipart_prompt,
        undefined=KeepUndefined,
    ).render(**context)

    # Add default completion to final segment if needed
    preprocessed = _add_default_completion_if_needed(multipart_prompt_prefilled)

    segments = parser().parse(preprocessed.strip())
    dependency_graph = SegmentDependencyGraph(segments)
    plan = dependency_graph.get_execution_plan()
    if max([len(i) for i in plan]) > 1:
        logger.debug(f"Execution plan includes concurrency: {plan}")

    import anyio
    from anyio import Event

    segment_events = {f"segment_{i}": Event() for i, _ in enumerate(segments)}
    segment_results = {}

    for batch in plan:
        async with anyio.create_task_group() as tg:
            for segment_id in batch:
                i = int(segment_id.split("_")[1])
                segment = segments[i]
                deps = dependency_graph.dependency_graph[segment_id]

                async def run_segment(sid=segment_id, seg=segment, deps=deps):
                    if deps:
                        # wait for dependencies to complete
                        [await segment_events[d].wait() for d in deps]
                        dep_results = [segment_results[d] for d in deps]
                        # merge base context with dependency results
                        resolved_context = await merge_contexts(context, *dep_results)
                    else:
                        resolved_context = await merge_contexts(context)

                    result = await process_single_segment_(
                        seg,
                        model,
                        credentials,
                        resolved_context,
                        **(extra_kwargs or {}),
                    )
                    segment_results[sid] = result
                    segment_events[sid].set()  # signal completion
                    return result

                tg.start_soon(run_segment)

    # Gather results from all segments in original order
    final = ChatterResult()
    for i, segment in enumerate(segments):
        sid = f"segment_{i}"
        result = segment_results[sid]
        final.update(result.results)
        # Merge interim_results from this segment
        for key, interim_steps in result.interim_results.items():
            if key not in final.interim_results:
                final.interim_results[key] = []
            final.interim_results[key].extend(interim_steps)

    logger.debug(f"\n\n{LC.GREEN}Chatter Response: {final.response}{LC.RESET}\n\n")
    return final


def chatter(
    multipart_prompt: str,
    model: LLM = LLM(),
    credentials: Optional[LLMCredentials] = None,
    context={},
    extra_kwargs=None,
):
    return anyio.run(
        chatter_async,
        multipart_prompt,
        model,
        credentials,
        context,
        extra_kwargs,
    )


def get_embedding(
    texts: List[str],
    llm: LLM = LLM(
        model_name=env_config("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large")
    ),
    credentials: LLMCredentials = LLMCredentials(),
    dimensions: Optional[int] = 3072,
    batch_size: int = 500,
) -> List[List[float]]:
    """
    Get embeddings for a list of texts using litellm directly.
    """

    api_key = credentials.api_key
    base_url = credentials.base_url

    embeddings = []
    for batch in chunked(texts, batch_size):
        logger.debug(f"Getting batch of embeddings:\n{texts}")
        try:
            response = litellm.embedding(
                model=llm.model_name,
                input=str(batch),
                dimensions=dimensions,
                api_key=api_key,
                api_base=base_url,
            )
        except Exception as e:
            raise Exception(f"Error getting embeddings: {e}")
            
        embeddings.extend(item["embedding"] for item in response["data"])

    return embeddings
