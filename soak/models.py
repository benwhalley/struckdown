"""Data models for qualitative analysis pipelines."""

from decouple import config
from collections.abc import Awaitable
import asyncio
import anyio
import hashlib
import inspect
import itertools
import json
import logging
import math
import os
import re
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import pandas as pd
import tiktoken
import yaml
from box import Box
from chatter import LLM, ChatterResult, LLMCredentials, chatter
from chatter.parsing import parse_syntax
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateSyntaxError,
    meta,
)
from .async_decorators import flow, task
from pydantic import BaseModel, ConfigDict, Field, RootModel, Tag
from soak.chatter_dag import chatter_dag
from typing_extensions import Annotated

from .document_utils import extract_text, get_scrubber, unpack_zip_to_temp_paths_if_needed

if TYPE_CHECKING:
    from .dag import QualitativeAnalysisPipeline

import logging

logger = logging.getLogger(__name__)


MAX_CONCURRENCY = config("MAX_CONCURRENCY", default=10, cast=int)


# cache policy no longer needed with custom decorators
# DAG_CACHE_POLICY = INPUTS + TASK_SOURCE


class Code(BaseModel):
    # slug: str = Field(..., min_length=8, description="Unique identifier for the code")
    name: str = Field(..., min_length=1, description="A short name for the code.")
    description: str = Field(..., min_length=5, description="A description of the code.")
    quotes: List[str] = Field(
        ...,
        min_length=0,
        description="Example quotes from the text which illustrate the code. Choose the best examples.",
    )


class CodeList(BaseModel):
    codes: List[Code] = Field(..., min_length=0)

    def to_markdown(self):
        return "\n\n".join([f"- {i.name}: {i.description}\n{i.quotes}" for i in self.codes])


class Theme(BaseModel):
    name: str = Field(..., min_length=10)
    description: str = Field(..., min_length=10)
    # refer to codes by slug/identifier
    code_names: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of the codes that are part of this theme. Identify them accurately by name",
    )


class Themes(BaseModel):
    themes: List[Theme] = Field(..., min_length=1)

    def to_markdown(self):
        return "\n- ".join([i.name for i in self.themes])


@dataclass
class Document:
    """Represents a source document (transcript)."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualitativeAnalysis(BaseModel):
    codes: Optional[List[Code]] = None
    themes: Optional[List[Theme]] = None
    narrative: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    config: Optional["DAGConfig"] = None
    pipeline: Optional[str] = None

    def name(self):
        return self.sha256()[:8]

    def sha256(self):
        return hashlib.sha256(json.dumps(self.model_dump()).encode()).hexdigest()[:8]

    def __str__(self):
        return f"Themes: {self.themes}\nCodes: {self.codes}"

    def to_html(self, template_path: Optional[str] = None) -> str:
        """Render the analysis as HTML using Jinja2 template from file.

        Args:
            template_path: Path to the HTML template file. If None, uses default template.

        Returns:
            Rendered HTML string.
        """
        if template_path is None:
            # Use default template in soak/templates directory
            template_dir = Path(__file__).parent / "templates"
            template_name = "qualitative_analysis.html"
        else:
            # Use provided template path
            template_path = Path(template_path)
            template_dir = template_path.parent
            template_name = template_path.name

        # Create Jinja2 environment and load template
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_name)

        # Render template with data
        return template.render(analysis=self)


class QualitativeAnalysisComparison(BaseModel):
    results: List["QualitativeAnalysisPipeline"]
    combinations: Dict[str, Tuple["QualitativeAnalysisPipeline", "QualitativeAnalysisPipeline"]]
    statistics: Dict[str, Dict]
    comparison_plots: Dict[str, Dict[str, Any]]  # eg. heatmaps.xxxyyy = List
    additional_plots: Dict[str, Any]
    config: dict

    def by_comparisons(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a nested dict keyed by comparison key, with inner dict containing:
            - 'a', 'b' (the QualitativeAnalysis objects)
            - 'stats'
            - 'plots': { plot_type: plot_object, ... }
        """
        out = {}
        for key, (a, b) in self.combinations.items():
            out[key] = {"a": a, "b": b, "stats": self.statistics.get(key), "plots": {}}

        for plot_type in self.comparison_plots.keys():
            for k in out.keys():
                out[k]["plots"][plot_type] = self.comparison_plots[plot_type][k]

        return out


class DAGConfig(BaseModel):
    document_paths: Optional[List[str]] = []
    documents: List[str] = []
    model_name: str = "gpt-4o-mini"
    temperature: float = 1.0
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words
    extra_context: Dict[str, Any] = {}
    llm_credentials: Optional[LLMCredentials] = None
    scrub_pii: bool = False
    scrubber_model: str = "en_core_web_md"
    scrubber_salt: Optional[str] = Field(default="42", exclude=True)

    def get_model(self):
        return LLM(model_name=self.model_name, temperature=self.temperature)

    def load_documents(self) -> List[str]:
        if hasattr(self, "documents") and self.documents:
            logger.info("Using cached documents")
            return self.documents

        with unpack_zip_to_temp_paths_if_needed(self.document_paths) as dp_:
            self.documents = [extract_text(i) for i in dp_]

        if self.scrub_pii:
            logger.info("Scrubbing PII")
            if self.scrubber_salt == 42:
                logger.warning("Scrubber salt is default, consider setting to a random value")

            scrubber = get_scrubber(model=self.scrubber_model, salt=self.scrubber_salt)
            self.documents = [scrubber.clean(i) for i in self.documents]

        return self.documents


@task
async def run_node(node):
    r = await node.run() if asyncio.iscoroutinefunction(node.run) else node.run()
    logger.info(f"COMPLETED: {node.name}")
    return r


def get_template_variables(template_string: str) -> Set[str]:
    """Extract all variables from a jinja template string.
    get_template_variables('{a} {{b}} c ')
    """
    env = Environment()
    ast = env.parse(template_string)
    return meta.find_undeclared_variables(ast)


def render_strict_template(template_str: str, context: dict) -> str:
    env = Environment(undefined=StrictUndefined)
    template = env.from_string(template_str)
    return template.render(**context)


@dataclass(frozen=True)
class Edge:
    from_node: str
    to_node: str


DAGNodeUnion = Annotated[
    Union["Map", "Reduce", "Transform", "Batch", "Split"], Field(discriminator="type")
]


class DAG(BaseModel):
    model_config = {}

    name: str
    default_context: Dict[str, Any] = {}
    nodes: List["DAGNodeUnion"] = Field(default_factory=list)
    config: DAGConfig = DAGConfig()

    def progress(self):
        last_complete = self.nodes[0]
        return f"Last completed: {last_complete.name}"

    @property
    def edges(self) -> List["Edge"]:
        all_edges = []
        for node in self.nodes:
            for input_ref in node.inputs:
                if input_ref in [i.name for i in self.nodes]:
                    all_edges.append(Edge(from_node=input_ref, to_node=node.name))

        return all_edges

    def to_mermaid(self) -> str:
        """Generate a Mermaid diagram of the DAG structure with shapes by node type."""
        lines = ["flowchart TD"]

        shape_map = {
            "Split": ("(", ")"),  # round edges
            "Map": ("[[", "]]"),  # standard rectangle
            "Reduce": ("{{", "}}"),  # hexagon
            "Transform": (">", "]"),  # circle
            "Batch": ("[[", "]]"),  # subroutine shape
        }

        for node in self.nodes:
            l, r = shape_map.get(node.type, ("[", "]"))  # fallback to rectangle
            label = f"{node.type}: {node.name}"
            lines.append(f"    {node.name}{l}{label}{r}")

        for edge in self.edges:
            lines.append(f"    {edge.from_node} --> {edge.to_node}")

        return "\n".join(lines)

    def get_execution_order(self) -> List[List[str]]:
        """Get the execution order as batches of nodes that can run in parallel."""
        remaining = set([i.name for i in self.nodes])
        execution_order = []

        while remaining:
            # Find nodes with no unprocessed dependencies
            ready = set()
            for node_name in remaining:
                deps = self.get_dependencies_for_node(node_name)
                if all(dep not in remaining for dep in deps):
                    ready.add(node_name)

            if not ready and remaining:
                # Circular dependency detected
                raise ValueError(f"Circular dependency detected in nodes: {remaining}")

            execution_order.append(list(ready))
            remaining -= ready

        return execution_order

    @property
    def nodes_dict(self):
        return {i.name: i for i in self.nodes}

    @flow
    async def run(self):
        self.config.load_documents()
        if not self.config.llm_credentials:
            raise Exception("LLMCredentials must be set for DAG")
        for batch in self.get_execution_order():
            # use anyio structured concurrency - start all tasks in batch concurrently
            async with anyio.create_task_group() as tg:
                for name in batch:
                    tg.start_soon(run_node, self.nodes_dict[name])
            # all tasks in batch complete when task group exits
        return self

    def get_dependencies_for_node(self, node_name: str) -> Set[str]:
        """Get nodes that must complete before a node can run."""

        dependencies = set()

        # set[edge for edge in self.edges if edge.to_node == node_name]
        for edge in self.edges:
            if edge.to_node == node_name:
                dependencies.add(edge.from_node)

        return dependencies

    def add_node(self, node: "DAGNode"):
        # if self.nodes_dict.get(node.name):
        #     raise ValueError(f"Node '{node.name}' already exists in DAG")
        node.dag = self
        self.nodes.append(node)

    def get_required_context_variables(self):
        node_names = [i.name for i in self.nodes]
        all_vars = []
        tmplts = list(
            itertools.chain(*[get_template_variables(i.template) for i in self.nodes if i.template])
        )
        return set(tmplts).difference(node_names)

    def __str__(self):
        return f"DAG: {self.name}"

    def __repr__(self):
        return f"DAG: {self.name}"

    @property
    def context(self) -> Dict[str, Any]:
        """Backward compatibility: return node outputs as dict"""
        results = {v.name: v.output for v in self.nodes if v and v.output is not None}
        conf = self.config.extra_context.copy()
        conf.update(results)
        return conf


OutputUnion = Union[
    str,
    List[str],
    List[List[str]],
    ChatterResult,
    List[ChatterResult],
    List[List[ChatterResult]],
]


class DAGNode(BaseModel):
    model_config = {
        "discriminator": "type",
    }

    type: str = Field(default_factory=lambda self: type(self).__name__, exclude=False)

    dag: Optional["DAG"] = Field(default=None, exclude=True)
    name: str
    inputs: Optional[List[str]] = []
    template_text: Optional[str] = None
    output: Optional[OutputUnion] = Field(default=None)

    def get_model(self):
        return self.dag.config.get_model()

    def validate_template(self):
        try:
            Environment().parse(self.template_text)
            return True
        except TemplateSyntaxError as e:
            raise e
            logger.error(f"Template syntax error: {e}")
            return False

    def run(self, items: List[Any] = None) -> List[Any]:
        logger.info(f"\n\nRunning `{self.name}` ({self.__class__.__name__})\n\n")

    @property
    def context(self) -> Dict[str, Any]:
        ctx = self.dag.default_context.copy()

        # merge in extra_context from config (includes persona, research_question, etc.)
        ctx.update(self.dag.config.extra_context)

        # if there are no inputs, assume we'll be using input documents
        if not self.inputs:
            self.inputs = ["documents"]

        if "documents" in self.inputs:
            ctx["documents"] = self.dag.config.load_documents()

        prev_nodes = {k: self.dag.nodes_dict.get(k) for k in self.inputs}
        prev_output = {k: v.output for k, v in prev_nodes.items() if v and v.output is not None}
        ctx.update(prev_output)

        return ctx

    @property
    def template(self) -> str:
        return self.template_text


class Split(DAGNode):
    type: Literal["Split"] = "Split"

    name: str = "chunks"
    template_text: str = "{{input}}"

    near: List[str] = ["\n\n", "\n", ".", " "]
    chunk_size: int = 20000
    min_split: int = 500
    split_unit: Literal["chars", "tokens"] = "chars"
    encoding_name: str = "cl100k_base"

    @property
    def token_encoder(self):
        return tiktoken.get_encoding(self.encoding_name)

    @property
    def template(self):
        return None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if len(self.inputs) > 1:
            raise ValueError("Split node can only have one input")

        assert self.chunk_size > self.min_split, "chunk_size must be greater than min_split"

    def run(self) -> List[str]:
        super().run()
        input_docs = self.context[self.inputs[0]]
        self.output = list(itertools.chain.from_iterable(map(self.split_document, input_docs)))
        logger.info(f"CREATED {len(self.output)} chunks")
        return self.output

    def split_document(self, doc: str) -> List[str]:
        if self.split_unit == "chars":
            return self._split_by_length(doc, len_fn=len)
        else:
            encode = self.token_encoder.encode
            return self._split_by_length(doc, len_fn=lambda s: len(encode(s)))

    def _split_by_length(self, doc: str, len_fn: Callable[[str], int]) -> List[str]:
        doc_len = len_fn(doc)
        if doc_len <= self.chunk_size:
            return [doc.strip()]

        # Compute ideal chunk target and find separators
        n_chunks = max(1, math.ceil(doc_len / self.chunk_size))
        target = max(self.min_split, math.ceil(doc_len / n_chunks))

        # Find all candidate breakpoints (based on char indices)
        split_points = []
        for sep in self.near:
            split_points += [m.end() for m in re.finditer(re.escape(sep), doc)]
        split_points = sorted(set(p for p in split_points if p >= self.min_split))

        # Greedy chunking using split_unit length function
        chunks, last = [], 0
        for i in range(1, n_chunks):
            ideal = i * target
            candidates = [p for p in split_points if last < p and len_fn(doc[last:p]) <= target]
            if not candidates:
                # fallback to next available split
                candidates = [p for p in split_points if p > last]
            if not candidates:
                break
            best = candidates[-1]
            chunks.append(doc[last:best].strip())
            last = best
        chunks.append(doc[last:].strip())

        return [c for c in chunks if len_fn(c) >= self.min_split or len(chunks) == 1]


def resolve(obj):
    """Synchronously resolve awaitables, recursively in lists/dicts."""

    if isinstance(obj, Awaitable):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(obj)
        else:
            # You are already in a running event loop — can't block
            raise RuntimeError("resolve() cannot be called inside an async context")

    elif isinstance(obj, list):
        return [resolve(x) for x in obj]

    elif isinstance(obj, dict):
        return {k: resolve(v) for k, v in obj.items()}

    else:
        return obj


class ItemsNode(DAGNode):
    """Any note which applies to multiple items at once"""

    def get_items(self) -> List[Dict[str, Any]]:
        """Resolve all inputs, then zip together, combining multiple inputs"""
        # resolve futures now (it's lazy to this point)
        input_data = resolve({k: self.context[k] for k in self.inputs})
        lengths = {k: len(v) if isinstance(v, list) else 1 for k, v in input_data.items()}
        max_len = max(lengths.values())

        for k, v in input_data.items():
            if isinstance(v, list):
                if len(v) != max_len:
                    raise ValueError(
                        f"Length mismatch for input '{k}': expected {max_len}, got {len(v)}"
                    )
            else:
                input_data[k] = [v] * max_len

        zipped = list(zip(*[input_data[k] for k in self.inputs]))

        # make the first input available as {{input}} in any template
        items = []
        for values in zipped:
            item_dict = dict(zip(self.inputs, values))
            if self.inputs:
                item_dict["input"] = item_dict[self.inputs[0]]
            items.append(Box(item_dict))

        return items


@task
async def default_map_task(template, context, model, credentials, **kwargs):
    """Default map task renders the Step template for each input item and calls the LLM."""

    rt = render_strict_template(template, context)
    # call chatter async in the main event loop
    from chatter import chatter
    from soak.chatter_dag import ACTION_LOOKUP
    from soak.models import Code, CodeList, Theme, Themes

    action_lookup = ACTION_LOOKUP.copy()
    action_lookup.update(
        {
            "theme": Theme,
            "code": Code,
            "themes": Themes,
            "codes": CodeList,
        }
    )

    # call chatter as async function within the main event loop
    result = await chatter(
        multipart_prompt=rt,
        context=context,
        model=model,
        credentials=credentials,
        action_lookup=action_lookup,
        **kwargs,
    )
    return result


class Map(ItemsNode):
    model_config = {
        "discriminator": "type",
    }

    type: Literal["Map"] = "Map"
    # fn must accepts **kwargs, return iterable
    task: Callable = Field(default=default_map_task, exclude=True)
    default_template: str = "{{input}} <prompt>: [[output]]"

    @property
    def template(self) -> str:
        return self.template_text or self.default_template

    def validate_template(self):
        try:
            parse_syntax(self.template_text)
            return True
        except Exception as e:
            logger.error(f"Template syntax error: {e}")
            return False

    async def run(self) -> List[Any]:
        super().run()

        # create semaphore to limit concurrency within this Map operation
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        # check if input is BatchList
        input_data = self.context[self.inputs[0]] if self.inputs else None
        if isinstance(input_data, BatchList):
            # flatten BatchList into individual items for processing
            all_items = []
            batch_sizes = []
            for batch in input_data.batches:
                batch_items = [Box({"input": item}) for item in batch]
                all_items.extend(batch_items)
                batch_sizes.append(len(batch))

            # process each item individually (exclude BatchList from context)
            filtered_context = {
                k: v for k, v in self.context.items() if not isinstance(v, BatchList)
            }

            # create helper function to wrap task execution with semaphore
            async def run_task_with_semaphore(item):
                async with semaphore:
                    return await self.task(
                        template=self.template,
                        context={**filtered_context, **item},
                        model=self.get_model(),
                        credentials=self.dag.config.llm_credentials,
                    )

            # create async tasks and await them
            tasks = [asyncio.create_task(run_task_with_semaphore(item)) for item in all_items]

            # await all tasks
            results = await asyncio.gather(*tasks)

            # reconstruct BatchList structure
            reconstructed_batches = []
            result_idx = 0
            for batch_size in batch_sizes:
                batch_results = results[result_idx : result_idx + batch_size]
                reconstructed_batches.append(batch_results)
                result_idx += batch_size

            batch_list_result = BatchList(batches=reconstructed_batches)
            self.output = batch_list_result
            return batch_list_result
        else:
            # original behavior for non-BatchList inputs
            items = self.get_items()

            # create helper function to wrap task execution with semaphore
            async def run_task_with_semaphore(item):
                async with semaphore:
                    return await self.task(
                        template=self.template,
                        context={**self.context, **item},
                        model=self.get_model(),
                        credentials=self.dag.config.llm_credentials,
                    )

            # create async tasks and await them
            tasks = [asyncio.create_task(run_task_with_semaphore(item)) for item in items]
            # await all tasks
            results = await asyncio.gather(*tasks)
            self.output = results
            return results


class Transform(ItemsNode):
    type: Literal["Transform"] = "Transform"
    # TODO: allow for arbitrary python functions instead of chatter?
    # fn: Field(Callable[..., Iterable[Any]],  exclude=True) = None
    default_template: str = "{{input}} <prompt>: [[output]]"

    @property
    def template(self) -> str:
        return self.template_text or self.default_template

    async def run(self):
        super().run()

        items = self.get_items()

        if not isinstance(items, str):
            assert len(items) == 1, "Transform nodes must have exactly one input item"

        rt = render_strict_template(self.template, {**self.context, **items[0]})
        # call chatter async in the main event loop
        from chatter import chatter
        from soak.chatter_dag import ACTION_LOOKUP
        from soak.models import Code, CodeList, Theme, Themes

        action_lookup = ACTION_LOOKUP.copy()
        action_lookup.update(
            {
                "theme": Theme,
                "code": Code,
                "themes": Themes,
                "codes": CodeList,
            }
        )

        # call chatter as async function within the main event loop
        self.output = await chatter(
            multipart_prompt=rt,
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            action_lookup=action_lookup,
        )
        return self.output


class Reduce(ItemsNode):
    type: Literal["Reduce"] = "Reduce"
    default_template: str = "{{input}}"

    @property
    def template(self) -> str:
        return self.template_text or self.default_template

    def get_items(self):
        if len(self.inputs) > 1:
            raise ValueError("Reduce nodes can only have one input")

        input_data = self.dag.context[self.inputs[0]]

        # if input is a BatchList, return it directly for special handling in run()
        if isinstance(input_data, BatchList):
            return input_data

        # otherwise, wrap individual items in the expected format
        items = [{"input": v, self.inputs[0]: v} for v in input_data]
        return items

    def run(self, items=None) -> Any:

        super().run()

        items = items or self.get_items()

        # if items is a BatchList, run on each batch
        if isinstance(items, BatchList):
            self.output = [self.run(items=i) for i in items.batches]
            return self.output

        else:
            # handle both dictionaries and strings
            rendered = []
            for item in items:
                if isinstance(item, dict):
                    context = {**item}
                else:
                    # item is a string, wrap it for template processing
                    context = {"input": item}
                rendered.append(render_strict_template(self.template, context))
            self.output = "\n".join(rendered)
            return self.output


@dataclass
class BatchList(object):
    batches: List[Any]

    def __iter__(self):
        return iter(self.batches)


class Batch(ItemsNode):
    type: Literal["Batch"] = "Batch"
    # batch_fn: Optional[Callable] = None
    batch_size: int = 10
    default_template: Optional[str] = None

    def run(self) -> List[List[Any]]:
        super().run()

        batches_ = self.default_batch(self.get_items(), self.batch_size)
        self.output = BatchList(batches=batches_)
        return self.output

    def default_batch(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Batch items into lists of size batch_size."""
        return list(itertools.batched(items, batch_size))


class QualitativeAnalysisPipeline(DAG):

    name: Optional[str] = None

    def result(self):

        # import pdb; pdb.set_trace()
        return Box(
            {
                "pipeline": self,
                "themes": Themes(**self.nodes_dict.get("themes").output.response).themes,
                "codes": CodeList(**self.nodes_dict.get("codes").output.response).codes,
                "narrative": self.nodes_dict.get("narrative").output.response,
                "detail": self,
            }
        )

    def export(self, file_path=None) -> str:
        """
        Export pipeline to template bundle format string.

        Args:
            file_path: Optional path to save the template bundle to a file

        Returns:
            str: Template bundle content
        """
        raise Exception("Deprecated?")

        # from .specs import pipeline_to_template_bundle

        # bundle = pipeline_to_template_bundle(self)

        # if file_path is not None:
        #     Path(file_path).write_text(bundle)

        # return bundle

    # @classmethod
    # def import_(cls, template_bundle) -> "QualitativeAnalysisPipeline":
    #     """Import pipeline from template bundle format string or Path object."""
    #     from .specs import load_template_bundle

    #     return load_template_bundle(template_bundle)


# Resolve forward references after QualitativeAnalysisPipeline is defined

ItemsNode.model_rebuild(force=True)
DAGNode.model_rebuild(force=True)
DAG.model_rebuild(force=True)
QualitativeAnalysis.model_rebuild(force=True)
