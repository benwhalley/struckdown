import logging
import re
import traceback
from collections import OrderedDict
from types import FunctionType
from typing import Any, Dict, List, Optional

import anyio
import instructor
import litellm
import openai
from box import Box
from decouple import config as env_config
from instructor import from_provider
from jinja2 import StrictUndefined, Template, Undefined
from more_itertools import chunked
from pydantic import BaseModel, ConfigDict, Field

from struckdown.cache import clear_cache, memory
from struckdown.parsing import parser
from struckdown.return_type_models import ACTION_LOOKUP

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
        default_factory=lambda: env_config("DEFAULT_LLM", "openai/gpt-4.1-mini"),
        exclude=True,
    )

    def client(self, credentials: LLMCredentials = None):
        if credentials is None:
            credentials = LLMCredentials()

        if not credentials.api_key or not credentials.base_url:
            raise Exception("Set LLM_API_KEY and LLM_API_BASE environment variables")

        # Create the instructor client with credentials already set
        client = instructor.from_provider(
            self.model_name, api_key=credentials.api_key, base_url=credentials.base_url
        )

        return client


@memory.cache(ignore=["return_type", "llm", "credentials"])
def _call_llm_cached(
    prompt: str,
    model_name: str,
    max_retries: int,
    max_tokens: Optional[int],
    extra_kwargs_str: str,
    return_type,
    llm,
    credentials,
):
    """
    Cache the raw completion dict from the LLM.
    This is the expensive API call we want to cache.
    Returns dicts (not Pydantic models) so they pickle safely.
    """
    logger.debug(f"\n\n{LC.BLUE}Prompt: {prompt}{LC.RESET}\n\n")
    try:
        res, com = llm.client(credentials).chat.completions.create_with_completion(
            model=model_name.split("/")[-1],
            response_model=return_type,
            messages=[{"role": "user", "content": prompt}],
            **(eval(extra_kwargs_str) if extra_kwargs_str else {}),
        )
    except Exception as e:
        full_traceback = traceback.format_exc()
        logger.warning(f"Error calling LLM: {e}\n{full_traceback}")
        raise e

    logger.debug(f"\n\n{LC.GREEN}Response: {res}{LC.RESET}\n")

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

    Cache key includes: prompt, model_name, max_retries, max_tokens, extra_kwargs_str
    Credentials are NOT included in the cache key (same prompt + model will hit cache regardless of API key).
    """
    logger.debug(
        f"Using model {llm.model_name}, max_retries {max_retries}, max_tokens: {max_tokens}"
    )

    extra_kwargs_str = str(extra_kwargs) if extra_kwargs else ""

    try:
        res_dict, com_dict = _call_llm_cached(
            prompt=prompt,
            model_name=llm.model_name,
            max_retries=max_retries,
            max_tokens=max_tokens,
            extra_kwargs_str=extra_kwargs_str,
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


class ChatterResult(BaseModel):
    type: str = Field(
        default="chatter", description="Discriminator field for union serialization"
    )
    results: Dict[str, SegmentResult] = Field(default_factory=dict)

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

    results = ChatterResult()
    prompt_parts = []
    accumulated_context = context.copy()

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

        # Render the prompt template.
        template = Template(
            segment_prompt + "\nAlways use the tools/JSON response.\n\n```json\n",
            undefined=StrictUndefined,
        )
        rendered_prompt = template.render(**accumulated_context)

        # Debug the context to see what's available to template tags
        logger.debug(f"Template context keys: {list(accumulated_context.keys())}")

        # Determine the appropriate return type.
        if isinstance(prompt_part.return_type, FunctionType):
            # Pass both options and quantifier to factory functions (like selection_response_model)
            rt = prompt_part.return_type(prompt_part.options, prompt_part.quantifier)
        else:
            rt = prompt_part.return_type

        # Call the LLM via structured_chat.
        res, completion_obj = await anyio.to_thread.run_sync(
            lambda: structured_chat(
                rendered_prompt,
                return_type=rt,
                llm=llm,
                credentials=credentials,
                extra_kwargs=kwargs,
            ),
            abandon_on_cancel=True,
        )

        # Only extract .response field if it exists and is not None
        # otherwise just return the object
        if hasattr(res, "response") and res.response is not None:
            res = res.response

        # Store the completion in both our final results and accumulated context.
        results[key] = SegmentResult(
            output=res, completion=completion_obj, prompt=rendered_prompt
        )

        accumulated_context[key] = res

        # For this segment, include the result in the prompt parts.
        prompt_parts.append(res)

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
    action_lookup=ACTION_LOOKUP,
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

    segments = parser(action_lookup=action_lookup).parse(
        multipart_prompt_prefilled.strip()
    )
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

    logger.debug(f"\n\n{LC.GREEN}Chatter Response: {final.response}{LC.RESET}\n\n")
    return final


def chatter(
    multipart_prompt: str,
    model: LLM = LLM(),
    credentials: Optional[LLMCredentials] = None,
    context={},
    action_lookup=ACTION_LOOKUP,
    extra_kwargs=None,
):
    return anyio.run(
        chatter_async,
        multipart_prompt,
        model,
        credentials,
        context,
        action_lookup,
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
        response = litellm.embedding(
            model=llm.model_name,
            input=batch,
            dimensions=dimensions,
            api_key=api_key,
            api_base=base_url,
        )

        embeddings.extend(item["embedding"] for item in response["data"])

    return embeddings
