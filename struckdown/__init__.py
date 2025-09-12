import logging
import re
import traceback
from collections import OrderedDict
from types import FunctionType
from typing import Any, Dict, List, Optional

import anyio
import litellm
import openai
from box import Box
from decouple import config as env_config
import instructor
from instructor import from_provider
from jinja2 import StrictUndefined, Template
from more_itertools import chunked
from pydantic import BaseModel, ConfigDict, Field

from struckdown.parsing import parser
from struckdown.return_type_models import ACTION_LOOKUP

logger = logging.getLogger(__name__)


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
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLM(BaseModel):
    model_name: Optional[str] = Field(
        default_factory=lambda: env_config("DEFAULT_LLM", "openai/gpt-4.1-mini"),
        exclude=True,
    )

    def client(self, credentials: LLMCredentials = None):
        client = instructor.from_provider(self.model_name)
        if credentials:
            client.api_key = credentials.api_key
            client.base_url = credentials.base_url
        return client


def structured_chat(
    prompt,
    return_type,
    llm: LLM = LLM(),
    credentials=None,
    max_retries=3,
    max_tokens=None,
    instructor_args=dict(),
):
    """
    Use instructor to make a tool call to an LLM, returning the `response` field, and a completion object
    """

    logger.info(
        f"Using model {llm.model_name}, max_retries {max_retries}, max_tokens: {max_tokens}"
    )

    logger.debug(f"\n\n{LC.BLUE}Prompt: {prompt}{LC.RESET}\n\n")
    try:
        res, com = llm.client(credentials).chat.completions.create_with_completion(
            model=llm.model_name.split("/")[-1],
            response_model=return_type,
            messages=[{"role": "user", "content": prompt}],
            **instructor_args,
        )
        _msg, _lt, _meta = res, None, com.model_dump()

    except Exception as e:
        print("USING:", credentials)
        full_traceback = traceback.format_exc()
        logger.warning(f"Error calling LLM: {e}\n{full_traceback}")
        raise e

    logger.debug(f"\n\n{LC.GREEN}Response: {res}{LC.RESET}\n")
    logger.info(
        f"{LC.PURPLE}Response type: {type(res)}; {len(str(res))} tokens produced{LC.RESET}\n\n"
    )
    return res, com


class SegmentResult(BaseModel):
    prompt: str
    output: Any
    completion: Optional[Any] = Field(default=None, exclude=False)


class ChatterResult(BaseModel):
    type: str = Field(default="chatter", description="Discriminator field for union serialization")
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
            self.results[key] = SegmentResult(output=value)

    def update(self, d: Dict[str, Any]):
        for k, v in d.items():
            self[k] = v

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

    for key, prompt_part in segment.items():
        # Append the raw text for this prompt part.
        prompt_parts.append(prompt_part.text)
        # Build the prompt for this completion from the parts within this segment.
        try:
            segment_prompt = "\n\n--\n\n".join(filter(bool, map(str, prompt_parts)))
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
            rt = prompt_part.return_type(prompt_part.options)
        else:
            rt = prompt_part.return_type

        # Call the LLM via structured_chat.
        res, completion_obj = await anyio.to_thread.run_sync(
            lambda: structured_chat(
                rendered_prompt, return_type=rt, llm=llm, credentials=credentials, **kwargs
            ),
            abandon_on_cancel=True,
        )

        # Only extract .response field if it exists and is not None
        # otherwise just return the object
        if hasattr(res, "response") and res.response is not None:
            res = res.response

        # Store the completion in both our final results and accumulated context.
        results[key] = SegmentResult(output=res, completion=completion_obj, prompt=rendered_prompt)

        accumulated_context[key] = res

        # For this segment, include the result in the prompt parts.
        prompt_parts.append(res)

    return results


class SegmentDependencyGraph:
    """Analyzes dependencies between segments and determines execution order"""

    def __init__(self, segments: List[OrderedDict]):
        self.segments = segments
        self.dependency_graph = {}  # segment_id -> set of segment_ids it depends on
        self.segment_vars = {}  # segment_id -> set of variable names defined in this segment
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
                template_vars.update(re.findall(r"\{\{\s*(\w+)\s*\}\}", prompt_part.text))

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
                logging.warning(f"Circular dependency detected in segments: {remaining}")
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

    segments = parser(action_lookup=action_lookup).parse(multipart_prompt.strip())
    dependency_graph = SegmentDependencyGraph(segments)
    plan = dependency_graph.get_execution_plan()
    if max([len(i) for i in plan]) > 1:
        logger.info(f"Execution plan includes concurrency: {plan}")

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
                        resolved_context = await merge_contexts(*dep_results)
                    else:
                        resolved_context = await merge_contexts(context)

                    result = await process_single_segment_(
                        seg, model, credentials, resolved_context, **(extra_kwargs or {})
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
        chatter_async, multipart_prompt, model, credentials, context, action_lookup, extra_kwargs
    )


def get_embedding(
    texts: List[str],
    llm: LLM = LLM(model_name="openai/text-embedding-3-large"),
    credentials: Optional[LLMCredentials] = None,
    dimensions: Optional[int] = 3072,
    batch_size: int = 500,
) -> List[List[float]]:
    """
    Get embeddings for a list of texts using litellm directly.
    """

    api_key = credentials.api_key if credentials else env_config("OPENAI_API_KEY", default=None)
    base_url = credentials.base_url if credentials else env_config("OPENAI_API_BASE", default=None)

    embeddings = []
    for batch in chunked(texts, batch_size):
        response = litellm.embedding(
            model=llm.model_name,
            input=batch,
            dimensions=dimensions,
            api_key=api_key,
            api_base=base_url,
        )
        embeddings.extend(item["embedding"] for item in response["data"])

    return embeddings


# get_embedding(["hello world"])
