""""""

import logging
import re
import traceback
from collections import OrderedDict
from types import FunctionType
from typing import Any, Dict, List, Optional

import openai
from box import Box
from decouple import config
from decouple import config as env_config
from instructor import from_openai
from jinja2 import StrictUndefined, Template
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.task_runners import ConcurrentTaskRunner
from pydantic import BaseModel, ConfigDict, Field

CACHE_POLICY = INPUTS + TASK_SOURCE


from .parsing import parser
from .return_type_models import ACTION_LOOKUP

logger = logging.getLogger(__name__)


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
    llm_api_key: str = env_config("LLM_API_KEY")
    llm_base_url: str = env_config("LLM_BASE_URL", None)

    def __str__(self):
        return f"LLMCredentials: {self.llm_api_key[:5]}"


class LLM(BaseModel):
    model_name: str = "gpt-4o-mini"
    temperature: float = 1.0

    def __str__(self):
        return self.model_name

    def client(self, credentials: LLMCredentials):
        return from_openai(
            openai.OpenAI(api_key=credentials.llm_api_key, base_url=credentials.llm_base_url)
        )


@task(name="structured-chat", cache_policy=CACHE_POLICY)
def structured_chat(prompt, llm, credentials, return_type, max_retries=3):
    """
    Use instructor to make a tool call to an LLM, returning the `response` field, and a completion object
    """

    try:
        res, com = llm.client(credentials).chat.completions.create_with_completion(
            model=llm.model_name,
            response_model=return_type,
            messages=[{"role": "user", "content": prompt}],
            max_retries=max_retries,
            temperature=llm.temperature,
        )
        msg, lt, meta = res, None, com.dict()

    except Exception as e:
        full_traceback = traceback.format_exc()
        logger.warning(f"Error calling LLM: {e}\n{full_traceback}")
        raise e

    return res, com


class ChatterResult(OrderedDict):
    @property
    def response(self):
        # return the last item in the dict
        return next(reversed(self.items()))[1]

    @property
    def outputs(self):
        return Box(self)


@task(cache_policy=CACHE_POLICY)
def process_single_segment(segment, model, credentials, context={}, cache=True):
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
        logger.info(f"Template context keys: {list(accumulated_context.keys())}")
        logger.info(f"{LC.BLUE}Prompt: {rendered_prompt}{LC.RESET}")

        # Determine the appropriate return type.
        if isinstance(prompt_part.return_type, FunctionType):
            rt = prompt_part.return_type(prompt_part.options)
        else:
            rt = prompt_part.return_type

        # Call the LLM via structured_chat.
        res_future = structured_chat.submit(rendered_prompt, model, credentials, return_type=rt)
        res, completion_obj = res_future.result()

        # Only extract .response field if it exists and is not None
        # otherwise just return the object
        if hasattr(res, "response") and res.response is not None:
            res = res.response

        logger.info(f"{LC.GREEN}Response: {res}{LC.RESET}")
        logger.info(f"{LC.PURPLE}Response type: {type(res)}{LC.RESET}")

        # Store the completion in both our final results and accumulated context.
        results[key] = res
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


@task(name="merge_contexts", persist_result=False, cache_result_in_memory=True)
def merge_contexts(*contexts):
    """Must be a task to preserve ordering/graph in chatter"""
    merged = {}
    if contexts:
        for c in contexts:
            merged.update(c)
        return merged


@flow(name="chatter")
def chatter(
    multipart_prompt: str,
    model=LLM(model_name="gpt-4o-mini"),
    credentials=LLMCredentials(),
    context={},
    action_lookup=ACTION_LOOKUP,
):
    """
    example:
    chatter("tell a joke [[joke]]")
    """
    segments = parser(action_lookup=action_lookup).parse(multipart_prompt.strip())
    dependency_graph = SegmentDependencyGraph(segments)
    plan = dependency_graph.get_execution_plan()

    segment_futures = {}

    for batch in plan:
        for segment_id in batch:
            i = int(segment_id.split("_")[1])
            segment = segments[i]
            deps = dependency_graph.dependency_graph[segment_id]

            if deps:
                dep_results = [segment_futures[d] for d in deps]
                context_future = merge_contexts.submit(*dep_results)
            else:
                context_future = merge_contexts.submit(context)

            segment_future = process_single_segment.submit(
                segment, model, credentials, context_future
            )
            segment_futures[segment_id] = segment_future

    # Gather results from all segments in original order
    final = ChatterResult()
    for i, segment in enumerate(segments):
        sid = f"segment_{i}"
        result = segment_futures[sid].result()
        final.update(dict(result))
    return final


if False:
    creds = LLMCredentials()
    llm = LLM("gpt-4o-mini")

    AL = ACTION_LOOKUP.copy()

    print(
        chatter(
            """
    pick a number 1 - 11. [[int:number]]""",
            model=llm,
            credentials=creds,
            action_lookup=AL,
        )
    )


# from langfuse.decorators import langfuse_context, observe
# from langfuse.openai import OpenAI  # OpenAI integration with tracing

# langfuse_context.configure(debug=False)


def get_embedding(texts, model_name="text-embedding-3-large", dimensions=3072) -> list:
    """
    get_embedding(["hello", ])
    """
    client = openai.OpenAI(api_key=config("LLM_API_KEY"), base_url=config("LLM_BASE_URL"))
    response = client.embeddings.create(
        input=texts,
        model=model_name,
        dimensions=dimensions,
    )
    return [i.embedding for i in response.data]
