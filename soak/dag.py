import inspect
import itertools
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Set, Union

import pandas as pd
import prefect
import tiktoken
import yaml
from box import Box
from chatter import LLM, LLMCredentials, chatter
from chatter.return_type_models import ACTION_LOOKUP
from jinja2 import Environment, StrictUndefined, Template, meta
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import PrefectFuture

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


class DAG(object):
    name: str
    nodes = Dict[str, "DAGNode"]
    context: Dict[str, Any] = {}
    extra_context: Dict[str, Any] = {}
    document_paths: Optional[List[str]] = []

    model_name: str = "gpt-4o-mini"
    temperature: float = 1.0
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words

    def get_model(self):
        return LLM(model_name=self.model_name, temperature=self.temperature)

    @property
    def edges(self) -> List["Edge"]:
        all_edges = []
        for node in self.nodes.values():
            for input_ref in node.inputs:
                if input_ref.startswith("context."):
                    continue
                from_node = input_ref.split(".")[0]
                if from_node in self.nodes:
                    all_edges.append(Edge(from_node=from_node, to_node=node.name))

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

        for node_name, node in self.nodes.items():
            node_type = type(node).__name__
            l, r = shape_map.get(node_type, ("[", "]"))  # fallback to rectangle
            label = f"{node_type}: {node_name}"
            lines.append(f"    {node_name}{l}{label}{r}")

        for edge in self.edges:
            lines.append(f"    {edge.from_node} --> {edge.to_node}")

        return "\n".join(lines)

    def get_execution_order(self) -> List[List[str]]:
        """Get the execution order as batches of nodes that can run in parallel."""
        remaining = set(self.nodes.keys())
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

    def load_documents(self) -> List[str]:
        if hasattr(self, "_documents") and self._documents:
            logger.info("Using cached documents")
            return self._documents

        with unpack_zip_to_temp_paths_if_needed(self.document_paths) as dp_:
            self._documents = [extract_text(i) for i in dp_]
        return self._documents

    @flow(log_prints=True)
    def run(self):
        self.load_documents()
        for batch in self.get_execution_order():
            futures = [run_node.submit(self.nodes[name]) for name in batch]
            # Wait for all in batch to finish
            for f in futures:
                f.result()
        return self

    def __init__(self, name: str, extra_context: Dict[str, Any] = {}, documents=[]):
        self.name = name
        self.extra_context = extra_context
        self.nodes = {}
        self._documents = documents

    def get_dependencies_for_node(self, node_name: str) -> Set[str]:
        """Get nodes that must complete before a node can run."""

        dependencies = set()

        # set[edge for edge in self.edges if edge.to_node == node_name]
        for edge in self.edges:
            if edge.to_node == node_name:
                dependencies.add(edge.from_node)

        return dependencies

    def add_node(self, node: "DAGNode"):
        # if self.nodes.get(node.name):
        #     raise ValueError(f"Node '{node.name}' already exists in DAG")
        node.dag = self
        self.nodes[node.name] = node

    def get_required_context_variables(self):
        node_names = self.nodes.keys()
        all_vars = []
        tmplts = list(
            itertools.chain(
                *[get_template_variables(i.template) for i in self.nodes.values() if i.template]
            )
        )
        return set(tmplts).difference(node_names)

    def __str__(self):
        return f"DAG: {self.name}"

    def __repr__(self):
        return f"DAG: {self.name}"

    @property
    def context(self) -> Dict[str, Any]:
        """Backward compatibility: return node outputs as dict"""
        return {k: v.output for k, v in self.nodes.items() if v.output is not None}


class DAGNode(ABC):
    dag: Optional["DAG"] = None
    name: str
    inputs: Optional[List[str]] = []
    model_name: Optional[str] = None
    output: Any = None

    template_name = Optional[str]
    document_paths: Optional[List[str]] = []

    def __init__(self, name, **kwargs):
        self.name = name
        self.inputs = kwargs.pop("inputs", [])

        self.template_name = kwargs.pop("template_name", None)
        self.template_text = kwargs.pop("template_text", None)

        for i in kwargs.keys():
            setattr(self, i, kwargs[i])

    def get_model(self):
        if self.model_name is None:
            return self.dag.get_model()
        return LLM(model_name=self.model_name)

    @abstractmethod
    def run(self, items: List[Any]) -> List[Any]:
        raise NotImplementedError

    @property
    def context(self) -> Dict[str, Any]:
        ctx = self.dag.extra_context.copy()

        if not self.inputs:
            self.inputs = ["documents"]

        for input_name in self.inputs:
            if input_name == "documents":
                ctx["documents"] = self.dag.load_documents()
            elif "." in input_name:
                # handle dotted access like "batch.codes"
                node_name, key = input_name.split(".", 1)
                if node_name in self.dag.nodes:
                    node = self.dag.nodes[node_name]
                    if hasattr(node, f"{key}_flat"):
                        ctx[input_name] = getattr(node, f"{key}_flat")
                    elif hasattr(node.output, key):
                        ctx[input_name] = getattr(node.output, key)
                    elif isinstance(node.output, dict) and key in node.output:
                        ctx[input_name] = node.output[key]
            else:
                if input_name in self.dag.nodes:
                    node_output = self.dag.nodes[input_name].output
                    if node_output is not None:
                        ctx[input_name] = node_output

        return ctx

    def get_template_name(self) -> str:
        if not self.template_name:
            return f"{self.name}.md"
        else:
            return self.template_name

    @property
    def template(self) -> str:
        if self.template_text is not None:
            return self.template_text
        try:
            try:
                current_directory = os.path.dirname(inspect.getfile(self.__class__))
            except OSError:
                current_directory = os.getcwd() + "/soak/pipelines/yaml/"

            with open(os.path.join(current_directory, self.get_template_name())) as f:
                return f.read()
        except FileNotFoundError:
            return None


class Split(DAGNode):
    near: Optional[List[str]] = ["\n\n", "\n", ".", " "]
    chunk_size: Optional[int] = None
    min_split: int = 500
    split_unit: Literal["chars", "tokens"] = "chars"
    encoding_name: str = "cl100k_base"
    token_encoder: Optional[tiktoken.Encoding] = None

    @property
    def template(self):
        return None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if len(self.inputs) > 1:
            raise ValueError("Split node can only have one input")

        self.chunk_size = kwargs.get("chunk_size", 10000)
        self.min_split = kwargs.get("min_split", 200)
        self.split_unit = kwargs.get("split_unit", "chars")

        logger.info(
            f"Splitting {self.inputs[0]} into chunks of size {self.chunk_size} and min_split {self.min_split}"
        )

        assert self.chunk_size > self.min_split, "chunk_size must be greater than min_split"

        if self.split_unit not in ["chars", "tokens"]:
            raise ValueError("split_unit must be 'chars' or 'tokens'")

        if self.split_unit == "tokens":
            self.token_encoder = kwargs.get("token_encoder") or tiktoken.get_encoding(
                self.encoding_name
            )

    def run(self) -> List[str]:
        logger.info(f"Running {self.name} ({self.__class__.__name__})")
        input_docs = self.context[self.inputs[0]]
        chunk_size = self.chunk_size or self.dag.chunk_size
        result = list(itertools.chain.from_iterable(map(self.split_document, input_docs)))
        logger.info(f"CREATED {len(result)} chunks")
        self.output = result
        return result

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

    def get_items(self) -> List[Dict[str, Any]]:

        # resolve futures at this point, so it's lazy
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
    return chatter_dag(multipart_prompt=rt, context=context, model=model, **kwargs).response


class Map(ItemsNode):
    # fn must accepts **kwargs, return iterable
    task = staticmethod(default_map_task)
    default_template = "{{input}} <prompt>: [[output]]"

    def run(self) -> List[Any]:
        logger.info(f"Running {self.name} ({self.__class__.__name__})")

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
    fn: Callable[..., Iterable[Any]] = None
    default_template = "{{input}} <prompt>: [[output]]"

    def run(self):
        logger.info(f"Running {self.name} ({self.__class__.__name__})")
        items = self.get_items()

        if not isinstance(items, str):
            assert len(items) == 1, "Transform nodes must have exactly one input item"

        if self.fn:
            res = self.fn(items)
        else:
            rt = render_strict_template(self.template, {**self.context, **items[0]})
            res = chatter_dag(multipart_prompt=rt, model=self.get_model()).response
            print(f"RESPONSE TYPE: {type(res)}")
        self.output = res
        return res


class Reduce(ItemsNode):
    fn: Optional[Callable] = None
    default_template = "{{input}}"

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
        logger.info(f"Running {self.name} ({self.__class__.__name__})")
        items = items or self.get_items()

        # if items is a BatchList, run on each batch
        if isinstance(items, BatchList):
            res = [self.run(items=i) for i in items.batches]
            self.output = res
            return res
        else:
            if self.fn:
                res = [self.fn(i) for i in items]
            elif self.template:
                # handle both dictionaries and strings
                rendered = []
                for item in items:
                    if isinstance(item, dict):
                        context = {**item}
                    else:
                        # item is a string, wrap it for template processing
                        context = {"input": item}
                    rendered.append(render_strict_template(self.template, context))
                res = "\n".join(rendered)
            else:
                res = items

            self.output = res
            return res


@dataclass
class BatchList(object):
    batches: List[Any]

    def __iter__(self):
        return iter(self.batches)


class Batch(ItemsNode):
    batch_fn: Optional[Callable] = None
    batch_size: int = 10
    default_template = None

    def run(self) -> List[List[Any]]:
        logger.info(f"Running {self.name} ({self.__class__.__name__})")
        batches_ = self.default_batch(self.get_items(), self.batch_size)
        batches = BatchList(batches=batches_)
        self.output = batches
        return batches

    def default_batch(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Batch items into lists of size batch_size."""
        return list(itertools.batched(items, batch_size))


class QualitativeAnalysisPipeline(DAG):

    def result(self):
        thms_node = self.nodes.get("themes")
        thms = getattr(thms_node.output, "themes", []) if thms_node and thms_node.output else []

        cds_node = self.nodes.get("codes")
        cds = getattr(cds_node.output, "codes", []) if cds_node and cds_node.output else []

        return QualitativeAnalysis(
            themes=thms,
            codes=cds,
            details={k: v.output for k, v in self.nodes.items() if v.output is not None},
            config={
                "extra_context": self.extra_context,
            },
        )


def pipeline_from_yaml(yaml_str: str) -> QualitativeAnalysisPipeline:
    config = yaml.safe_load(yaml_str)
    return pipeline_from_spec(config)


def pipeline_from_spec(config: OrderedDict) -> QualitativeAnalysisPipeline:

    name = config.get("name", "pipeline")
    extra_context = config.get("extra_context", {})
    document_paths = config.get("document_paths", [])
    documents = config.get("documents", [])

    dag = QualitativeAnalysisPipeline(name, extra_context=extra_context)
    dag.document_paths = document_paths
    dag._documents = documents

    node_constructors = {
        "split": Split,
        "map": Map,
        "reduce": Reduce,
        "transform": Transform,
        "batch": Batch,
    }

    for k, v in config.get("settings", {}).items():
        logger.debug(f"Setting {k} to {v}")
        setattr(dag, k, v)

    for node_def in config["steps"]:
        for kind, spec in node_def.items():
            node = node_constructors[kind](**spec)
            dag.add_node(node)

    return dag

