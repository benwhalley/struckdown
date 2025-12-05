"""Segment execution logic for struckdown."""

import logging
from collections import OrderedDict
from typing import List

from .jinja_utils import extract_jinja_variables
from .results import ChatterResult

logger = logging.getLogger(__name__)


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
