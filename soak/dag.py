import inspect
import itertools
from pathlib import Path
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Set, Union, Annotated

import pandas as pd
import prefect
import tiktoken
import yaml
from chatter.parsing import parse_syntax
from box import Box
from chatter import LLM, LLMCredentials, chatter
from chatter.return_type_models import ACTION_LOOKUP
from jinja2 import Environment, StrictUndefined, Template, meta, TemplateSyntaxError
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import PrefectFuture
from pydantic import BaseModel, ConfigDict, Field
from .document_utils import extract_text, unpack_zip_to_temp_paths_if_needed
from .models import Code, CodeList, QualitativeAnalysis, Theme, Themes

logger = logging.getLogger(__name__)

DAG_CACHE_POLICY = INPUTS + TASK_SOURCE


@task(cache_policy=None)
def run_node(node):
    r = node.run()
    logger.info(f"COMPLETED: {node.name}")
    return r


def chatter_dag(**kwargs):
    """
    # example usage to return Codes
    chatter_dag(multipart_prompt="Software as a service (SaaS /sæs/[1]) is a cloud computing service model where the provider offers use of application software to a client and manages all needed physical and software resources.[2] SaaS is usually accessed via a web application. Unlike other software delivery models, it separates 'the possession and ownership of software from its use'.[3] SaaS use began around 2000, and by 2023 was the main form of software application deployment.\n\n What is the theme of this text [[codes:code]]")
    """

    action_lookup = ACTION_LOOKUP.copy()
    action_lookup.update(
        {
            "theme": Theme,
            "code": Code,
            "themes": Themes,
            "codes": CodeList,
        }
    )
    return chatter(**kwargs, action_lookup=action_lookup)


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


class DAGConfig(BaseModel):
    document_paths: Optional[List[str]] = []
    documents: List[str] = []
    model_name: str = "gpt-4o-mini"
    temperature: float = 1.0
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words
    extra_context: Dict[str, Any] = {}

    def get_model(self):
        return LLM(model_name=self.model_name, temperature=self.temperature)

    def load_documents(self) -> List[str]:
        if hasattr(self, "documents") and self.documents:
            logger.info("Using cached documents")
            return self.documents

        with unpack_zip_to_temp_paths_if_needed(self.document_paths) as dp_:
            self.documents = [extract_text(i) for i in dp_]
        return self.documents


DAGNodeUnion = Annotated[
    Union["Map", "Reduce", "Transform", "Batch", "Split"], Field(discriminator="type")
]


class DAG(BaseModel):
    model_config = {
        "ignored_types": (prefect.flows.Flow,),
    }
    name: str
    default_context: Dict[str, Any] = {}
    nodes: List["DAGNodeUnion"] = Field(default_factory=list)
    config: DAGConfig = DAGConfig()

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
        lines = ["graph TD"]

        shape_map = {
            "Split": ("(", ")"),  # round edges
            "Map": ("[", "]"),  # standard rectangle
            "Reduce": ("{{", "}}"),  # hexagon
            "Transform": ("([", "])"),  # circle
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

    @flow(log_prints=True)
    def run(self):
        self.config.load_documents()
        for batch in self.get_execution_order():
            futures = [run_node.submit(self.nodes_dict[name]) for name in batch]
            # Wait for all in batch to finish
            for f in futures:
                f.result()
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


class DAGNode(BaseModel):
    type: str = Field(default_factory=lambda self: type(self).__name__, exclude=False)

    model_config = {"discriminator": "type"}

    dag: Optional["DAG"] = Field(default=None, exclude=True)
    name: str
    inputs: Optional[List[str]] = []
    template_text: Optional[str] = None
    output: Any = None

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
    if isinstance(obj, PrefectFuture):
        return obj.result()
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


@task(cache_policy=None)
def default_map_task(template, context, model, **kwargs):
    """Default map task renders the Step template for each input item and calls the LLM."""
    rt = render_strict_template(template, context)
    return chatter_dag(multipart_prompt=rt, context=context, model=model, **kwargs)


class Map(ItemsNode):
    type: Literal["Map"] = "Map"
    # fn must accepts **kwargs, return iterable
    task: Callable = staticmethod(default_map_task)
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

    def run(self) -> List[Any]:
        super().run()

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
            futures = [
                self.task.submit(
                    template=self.template,
                    context={**filtered_context, **item},
                    model=self.get_model(),
                )
                for item in all_items
            ]

            # note results are a list of ChatterResult
            results = [f.result() for f in futures]

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
            futures = [
                self.task.submit(
                    template=self.template,
                    context={**self.context, **item},
                    model=self.get_model(),
                )
                for item in items
            ]
            results = [f.result() for f in futures]
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

    def run(self):
        super().run()

        items = self.get_items()

        if not isinstance(items, str):
            assert len(items) == 1, "Transform nodes must have exactly one input item"

        rt = render_strict_template(self.template, {**self.context, **items[0]})
        self.output = chatter_dag(multipart_prompt=rt, model=self.get_model()).response
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
        items = [{"input": v} for v in input_data]
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

    def result(self):
        thms_node = self.nodes_dict.get("themes")
        thms = getattr(thms_node.output, "themes", []) if thms_node and thms_node.output else []

        cds_node = self.nodes_dict.get("codes")
        cds = getattr(cds_node.output, "codes", []) if cds_node and cds_node.output else []

        try:

            return QualitativeAnalysis(
                themes=thms,
                codes=cds,
                details={n.name: n.output for n in self.nodes if n and n.output is not None},
                pipeline=self.export(),
            )
        except Exception as e:
            logger.error(f"Error creating QualitativeAnalysis from pipeline: {e}")
            return self.nodes_dict

    def export(self, file_path=None) -> str:
        """
        Export pipeline to template bundle format string.

        Args:
            file_path: Optional path to save the template bundle to a file

        Returns:
            str: Template bundle content
        """
        from .specs import pipeline_to_template_bundle

        bundle = pipeline_to_template_bundle(self)

        if file_path is not None:
            Path(file_path).write_text(bundle)

        return bundle

    @classmethod
    def import_(cls, template_bundle) -> "QualitativeAnalysisPipeline":
        """Import pipeline from template bundle format string or Path object."""
        from .specs import load_template_bundle

        return load_template_bundle(template_bundle)


# Resolve forward references after QualitativeAnalysisPipeline is defined
from .models import QualitativeAnalysis

ItemsNode.model_rebuild(force=True)
DAGNode.model_rebuild(force=True)
DAG.model_rebuild(force=True)
QualitativeAnalysis.model_rebuild()


# from typing import Annotated, Union
# from pydantic import Field

# from soak.dag import Map, Reduce, Transform, Batch, Split

# TypeAdapter(DAGNodeUnion).validate_python({'type': 'Map', 'name': 'test'})
