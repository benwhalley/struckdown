"""
Simple unit tests for chatter functionality that don't require database setup.
"""

import unittest
from collections import OrderedDict
from unittest.mock import Mock

from struckdown import ChatterResult, SegmentDependencyGraph


class ChatterSimpleTestCase(unittest.TestCase):
    def test_chatter_result_properties(self):
        """Test ChatterResult helper properties work correctly"""
        result = ChatterResult()
        result["first"] = "first_value"
        result["second"] = "second_value"
        result["last"] = "last_value"

        # Test .response property returns the last item
        self.assertEqual(result.response, "last_value")

        # Test .outputs property returns a Box object
        outputs = result.outputs
        self.assertEqual(outputs.first, "first_value")
        self.assertEqual(outputs.second, "second_value")
        self.assertEqual(outputs.last, "last_value")

        # Test that it maintains OrderedDict behavior
        keys = list(result.keys())
        self.assertEqual(keys, ["first", "second", "last"])

    def test_chatter_result_ordering(self):
        """Test that ChatterResult preserves insertion order"""
        result = ChatterResult()

        # Add items in specific order
        result["zebra"] = "z"
        result["apple"] = "a"
        result["banana"] = "b"

        # Verify order is preserved
        keys = list(result.keys())
        self.assertEqual(keys, ["zebra", "apple", "banana"])

        # Verify response is last inserted
        self.assertEqual(result.response, "b")

    def test_segment_dependency_graph_no_dependencies(self):
        """Test dependency analysis for independent segments"""

        # Create mock segments with no dependencies
        segment1 = OrderedDict([("var1", Mock(text="First segment"))])
        segment2 = OrderedDict([("var2", Mock(text="Second segment"))])
        segment3 = OrderedDict([("var3", Mock(text="Third segment"))])

        segments = [segment1, segment2, segment3]

        graph = SegmentDependencyGraph(segments)
        execution_plan = graph.get_execution_plan()

        # All segments should be executable in parallel (single batch)
        self.assertEqual(len(execution_plan), 1)
        self.assertEqual(len(execution_plan[0]), 3)
        self.assertIn("segment_0", execution_plan[0])
        self.assertIn("segment_1", execution_plan[0])
        self.assertIn("segment_2", execution_plan[0])

    def test_segment_dependency_graph_with_dependencies(self):
        """Test dependency analysis for dependent segments"""

        # Create mock segments with dependencies
        segment1 = OrderedDict([("fruit", Mock(text="What fruit?"))])
        segment2 = OrderedDict([("color", Mock(text="What color?"))])
        segment3 = OrderedDict(
            [("story", Mock(text="Tell story about {{fruit}} and {{color}}"))]
        )

        segments = [segment1, segment2, segment3]

        graph = SegmentDependencyGraph(segments)
        execution_plan = graph.get_execution_plan()

        # Should have 2 batches: [fruit, color] then [story]
        self.assertEqual(len(execution_plan), 2)

        # First batch should have fruit and color (independent)
        self.assertEqual(len(execution_plan[0]), 2)
        self.assertIn("segment_0", execution_plan[0])
        self.assertIn("segment_1", execution_plan[0])

        # Second batch should have story (depends on fruit and color)
        self.assertEqual(len(execution_plan[1]), 1)
        self.assertIn("segment_2", execution_plan[1])

    def test_segment_dependency_graph_sequential_dependencies(self):
        """Test dependency analysis for sequential dependencies"""

        # Create mock segments with sequential dependencies
        segment1 = OrderedDict([("first", Mock(text="First task"))])
        segment2 = OrderedDict([("second", Mock(text="Second task using {{first}}"))])
        segment3 = OrderedDict([("third", Mock(text="Third task using {{second}}"))])

        segments = [segment1, segment2, segment3]

        graph = SegmentDependencyGraph(segments)
        execution_plan = graph.get_execution_plan()

        # Should have 3 batches (all sequential)
        self.assertEqual(len(execution_plan), 3)

        # Each batch should have exactly one segment
        for i, batch in enumerate(execution_plan):
            self.assertEqual(len(batch), 1)
            self.assertIn(f"segment_{i}", batch)

    def test_segment_dependency_graph_mixed_dependencies(self):
        """Test dependency analysis for mixed parallel and sequential dependencies"""

        # Create mock segments with mixed dependencies
        segment1 = OrderedDict([("a", Mock(text="Task A"))])
        segment2 = OrderedDict([("b", Mock(text="Task B"))])
        segment3 = OrderedDict([("c", Mock(text="Task C using {{a}}"))])
        segment4 = OrderedDict([("d", Mock(text="Task D using {{a}} and {{b}}"))])
        segment5 = OrderedDict([("e", Mock(text="Task E using {{c}} and {{d}}"))])

        segments = [segment1, segment2, segment3, segment4, segment5]

        graph = SegmentDependencyGraph(segments)
        execution_plan = graph.get_execution_plan()

        # Expected execution plan:
        # Batch 1: [a, b] (independent)
        # Batch 2: [c, d] (both depend on a and/or b)
        # Batch 3: [e] (depends on c and d)
        self.assertEqual(len(execution_plan), 3)

        # Verify batch contents
        batch1 = set(execution_plan[0])
        batch2 = set(execution_plan[1])
        batch3 = set(execution_plan[2])

        self.assertEqual(batch1, {"segment_0", "segment_1"})
        self.assertEqual(batch2, {"segment_2", "segment_3"})
        self.assertEqual(batch3, {"segment_4"})

    def test_segment_dependency_graph_complex_template_vars(self):
        """Test dependency analysis handles various template variable formats"""

        # Create segments with different template variable styles
        segment1 = OrderedDict(
            [("var1", Mock(text="First {{ var1 }}"))]
        )  # Self-reference (should be ignored)
        segment2 = OrderedDict(
            [("var2", Mock(text="Second {{var1}} and {{  var1  }}"))]
        )  # Multiple refs with spaces
        segment3 = OrderedDict(
            [("var3", Mock(text="Third {{var2}} and {{var1}}"))]
        )  # Multiple dependencies

        segments = [segment1, segment2, segment3]

        graph = SegmentDependencyGraph(segments)
        execution_plan = graph.get_execution_plan()

        # Should have 3 batches due to sequential dependencies
        self.assertEqual(len(execution_plan), 3)

        # Verify the order
        self.assertEqual(execution_plan[0], ["segment_0"])
        self.assertEqual(execution_plan[1], ["segment_1"])
        self.assertEqual(execution_plan[2], ["segment_2"])


if __name__ == "__main__":
    unittest.main()
