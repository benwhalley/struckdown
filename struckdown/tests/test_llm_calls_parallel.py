"""
Tests for parallel LLM calling functionality.
This file tests the parallel processing and dependency resolution of chatter.
"""

import asyncio
import unittest
from collections import OrderedDict
from unittest.mock import AsyncMock, Mock, patch

from struckdown import (ChatterResult, SegmentDependencyGraph, chatter,
                        chatter_async)


class ParallelLLMCallsTestCase(unittest.TestCase):
    """Tests focusing on parallel execution behavior, not implementation details"""

    def test_dependency_graph_creation(self):
        """Test that the dependency graph correctly identifies dependencies"""

        # Create segments with known dependencies
        segment1 = OrderedDict([("var1", Mock(text="First variable"))])
        segment2 = OrderedDict([("var2", Mock(text="Second using {{var1}}"))])
        segment3 = OrderedDict(
            [("var3", Mock(text="Third using {{var1}} and {{var2}}"))]
        )

        segments = [segment1, segment2, segment3]

        # Create dependency graph
        graph = SegmentDependencyGraph(segments)

        # Verify dependencies are detected correctly
        self.assertEqual(graph.dependency_graph["segment_0"], set())  # No dependencies
        self.assertEqual(
            graph.dependency_graph["segment_1"], {"segment_0"}
        )  # Depends on segment_0
        self.assertEqual(
            graph.dependency_graph["segment_2"], {"segment_0", "segment_1"}
        )  # Depends on both

        # Verify execution plan
        execution_plan = graph.get_execution_plan()
        expected_plan = [["segment_0"], ["segment_1"], ["segment_2"]]
        self.assertEqual(execution_plan, expected_plan)

    def test_parallel_independent_segments_execution_plan(self):
        """Test that independent segments can execute in parallel (plan has them in same batch)"""

        # Create independent segments
        segment1 = OrderedDict([("season", Mock(text="Favorite season"))])
        segment2 = OrderedDict([("city", Mock(text="Favorite city"))])

        segments = [segment1, segment2]

        # Verify dependency graph puts them in same batch
        graph = SegmentDependencyGraph(segments)
        execution_plan = graph.get_execution_plan()

        # Both should be in first batch (can execute in parallel)
        self.assertEqual(len(execution_plan), 1)
        self.assertEqual(set(execution_plan[0]), {"segment_0", "segment_1"})

    def test_dependent_segments_sequential_execution_plan(self):
        """Test that dependent segments must execute sequentially"""

        # Create dependent segments
        segment1 = OrderedDict([("season", Mock(text="Favorite season"))])
        segment2 = OrderedDict([("city", Mock(text="Favorite city"))])
        segment3 = OrderedDict([("poem", Mock(text="Describe {{city}} in {{season}}"))])

        segments = [segment1, segment2, segment3]

        # Verify dependency graph
        graph = SegmentDependencyGraph(segments)
        execution_plan = graph.get_execution_plan()

        # First two can run in parallel, third must wait for both
        self.assertEqual(len(execution_plan), 2)
        self.assertEqual(set(execution_plan[0]), {"segment_0", "segment_1"})
        self.assertEqual(set(execution_plan[1]), {"segment_2"})

    def test_mixed_parallel_and_sequential_dependencies(self):
        """Test complex dependency graph with both parallel and sequential execution"""

        # Create a diamond-shaped dependency:
        # a and b can run in parallel
        # c depends on a
        # d depends on both b and c
        segment_a = OrderedDict([("a", Mock(text="Task A"))])
        segment_b = OrderedDict([("b", Mock(text="Task B"))])
        segment_c = OrderedDict([("c", Mock(text="Task C using {{a}}"))])
        segment_d = OrderedDict([("d", Mock(text="Task D using {{b}} and {{c}}"))])

        segments = [segment_a, segment_b, segment_c, segment_d]

        graph = SegmentDependencyGraph(segments)
        execution_plan = graph.get_execution_plan()

        # Expected execution:
        # Batch 1: [a, b] (independent)
        # Batch 2: [c] (depends on a)
        # Batch 3: [d] (depends on b and c)
        self.assertEqual(len(execution_plan), 3)
        self.assertEqual(set(execution_plan[0]), {"segment_0", "segment_1"})
        self.assertEqual(set(execution_plan[1]), {"segment_2"})
        self.assertEqual(set(execution_plan[2]), {"segment_3"})


if __name__ == "__main__":
    unittest.main()
