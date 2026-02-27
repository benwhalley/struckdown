"""
Tests for JSON schema filtering via schema_class() factory method.

Verifies that fields marked with exclude_from_completion=True are properly
excluded from schemas sent to LLMs by generating a simplified schema class.
"""

import unittest
from typing import List, Optional

from pydantic import Field

from struckdown.return_type_models import LLMConfig, ResponseModel, _schema_class_cache


def PostProcessedField(**kwargs):
    """Test helper - creates Field with exclude_from_completion."""
    kwargs.setdefault("json_schema_extra", {})
    kwargs["json_schema_extra"]["exclude_from_completion"] = True
    return Field(**kwargs)


class SimpleModel(ResponseModel):
    """Simple model with one excluded field."""

    name: str = Field(..., description="The name")
    internal_data: Optional[str] = Field(
        default=None, json_schema_extra={"exclude_from_completion": True}
    )


class NestedChild(ResponseModel):
    """Child model used in nested structures."""

    value: str = Field(..., description="A value")
    cached_result: Optional[str] = PostProcessedField(default=None)


class NestedParent(ResponseModel):
    """Parent model containing nested models."""

    title: str = Field(..., description="Title")
    children: List[NestedChild] = Field(default_factory=list)


class DeeplyNested(ResponseModel):
    """Model with multiple levels of nesting."""

    label: str = Field(..., description="Label")
    parents: List[NestedParent] = Field(default_factory=list)


class SchemaClassTestCase(unittest.TestCase):
    """Test the schema_class() factory method."""

    def setUp(self):
        # Clear cache before each test to ensure fresh schema classes
        _schema_class_cache.clear()

    def test_schema_class_excludes_llm_config(self):
        """schema_class() should not include llm_config field."""
        schema_cls = SimpleModel.schema_class()
        self.assertNotIn("llm_config", schema_cls.model_fields)

    def test_schema_class_excludes_marked_fields(self):
        """schema_class() should exclude fields with exclude_from_completion."""
        schema_cls = SimpleModel.schema_class()
        self.assertIn("name", schema_cls.model_fields)
        self.assertNotIn("internal_data", schema_cls.model_fields)

    def test_schema_class_is_cached(self):
        """schema_class() should return cached class on subsequent calls."""
        cls1 = SimpleModel.schema_class()
        cls2 = SimpleModel.schema_class()
        self.assertIs(cls1, cls2)

    def test_schema_class_transforms_nested_types(self):
        """Nested ResponseModel types should be transformed to their schema classes."""
        schema_cls = NestedParent.schema_class()

        # The children field should use NestedChildSchema, not NestedChild
        children_annotation = schema_cls.__annotations__["children"]
        # Get the inner type from List[...]
        from typing import get_args

        inner_type = get_args(children_annotation)[0]

        # The inner type should be a schema class (not the original NestedChild)
        self.assertNotEqual(inner_type, NestedChild)
        self.assertTrue(inner_type.__name__.endswith("Schema"))


class SchemaFilteringTestCase(unittest.TestCase):
    """Test that exclude_from_completion fields are filtered from JSON schema."""

    def setUp(self):
        _schema_class_cache.clear()

    def test_llm_config_excluded_from_simple_model(self):
        """LLMConfig should never appear in schema sent to LLM."""
        schema = SimpleModel.model_json_schema()

        # llm_config should not be in properties
        self.assertNotIn("llm_config", schema.get("properties", {}))

        # LLMConfig should not be in $defs
        defs = schema.get("$defs", {})
        self.assertNotIn("LLMConfig", defs)

    def test_excluded_field_filtered_from_properties(self):
        """Fields with exclude_from_completion should not appear in properties."""
        schema = SimpleModel.model_json_schema()

        self.assertIn("name", schema["properties"])
        self.assertNotIn("internal_data", schema["properties"])

    def test_nested_model_excluded_fields(self):
        """Excluded fields in nested models should be filtered from $defs."""
        schema = NestedParent.model_json_schema()
        defs = schema.get("$defs", {})

        # NestedChildSchema should be in $defs (it's referenced)
        nested_child_def = None
        for name, defn in defs.items():
            if "value" in defn.get("properties", {}):
                nested_child_def = defn
                break

        self.assertIsNotNone(nested_child_def)

        # But cached_result should be filtered from its properties
        self.assertIn("value", nested_child_def["properties"])
        self.assertNotIn("cached_result", nested_child_def["properties"])

        # LLMConfig should not be in $defs
        self.assertNotIn("LLMConfig", defs)

    def test_deeply_nested_filtering(self):
        """Filtering should work at all nesting levels."""
        schema = DeeplyNested.model_json_schema()
        defs = schema.get("$defs", {})

        # Find the NestedChild schema definition
        for name, defn in defs.items():
            props = defn.get("properties", {})
            if "cached_result" in props:
                self.fail(f"cached_result found in {name}")
            if "llm_config" in props:
                self.fail(f"llm_config found in {name}")

        # LLMConfig should never appear
        self.assertNotIn("LLMConfig", defs)

    def test_only_referenced_defs_remain(self):
        """$defs should only contain types that are actually referenced."""
        schema = SimpleModel.model_json_schema()

        # SimpleModel has no nested types, so $defs should be empty or not exist
        defs = schema.get("$defs", {})
        self.assertEqual(defs, {})

    def test_required_field_updated_when_excluded(self):
        """Required array should not include excluded fields."""
        schema = SimpleModel.model_json_schema()

        required = schema.get("required", [])
        self.assertIn("name", required)
        self.assertNotIn("internal_data", required)
        self.assertNotIn("llm_config", required)


class RealWorldModelsTestCase(unittest.TestCase):
    """Test with real-world model patterns from soak."""

    def setUp(self):
        _schema_class_cache.clear()

    def test_code_like_model(self):
        """Test model with Quote-like nested types and post-processed fields."""

        class Quote(ResponseModel):
            text: str = Field(..., description="Quote text")
            source: str = Field(default="", description="Source ID")

        class Code(ResponseModel):
            slug: str = Field(..., description="Code slug")
            name: str = Field(..., description="Code name")
            quotes: List[Quote] = Field(default_factory=list)
            # Post-processed field - should be excluded
            resolved_quotes: Optional[List[dict]] = PostProcessedField(default=None)

        schema = Code.model_json_schema()
        defs = schema.get("$defs", {})

        # Quote schema should be in $defs (it's referenced)
        quote_def = None
        for name, defn in defs.items():
            if "text" in defn.get("properties", {}):
                quote_def = defn
                break
        self.assertIsNotNone(quote_def)

        # LLMConfig should not be in $defs
        self.assertNotIn("LLMConfig", defs)

        # resolved_quotes should not be in properties
        self.assertNotIn("resolved_quotes", schema["properties"])

        # Quote's llm_config should also be filtered
        self.assertNotIn("llm_config", quote_def.get("properties", {}))

    def test_theme_like_model(self):
        """Test model with code references and multiple post-processed fields."""

        class Theme(ResponseModel):
            name: str = Field(..., description="Theme name")
            description: str = Field(..., description="Theme description")
            code_slugs: List[str] = Field(default_factory=list)
            # Multiple post-processed fields
            resolved_code_refs: Optional[List[dict]] = PostProcessedField(default=None)
            label: Optional[str] = PostProcessedField(default=None)

        schema = Theme.model_json_schema()

        # Post-processed fields should be excluded
        props = schema["properties"]
        self.assertIn("name", props)
        self.assertIn("description", props)
        self.assertIn("code_slugs", props)
        self.assertNotIn("resolved_code_refs", props)
        self.assertNotIn("label", props)
        self.assertNotIn("llm_config", props)

        # LLMConfig should not be in $defs
        defs = schema.get("$defs", {})
        self.assertNotIn("LLMConfig", defs)

    def test_wrapper_model_with_list(self):
        """Test wrapper model containing list of nested ResponseModels."""

        class Item(ResponseModel):
            value: str = Field(..., description="Item value")
            internal: Optional[str] = PostProcessedField(default=None)

        class ItemList(ResponseModel):
            items: List[Item] = Field(default_factory=list)

        schema = ItemList.model_json_schema()
        defs = schema.get("$defs", {})

        # Find Item schema in $defs
        item_def = None
        for name, defn in defs.items():
            if "value" in defn.get("properties", {}):
                item_def = defn
                break
        self.assertIsNotNone(item_def)

        # Item's internal field should be filtered
        item_props = item_def.get("properties", {})
        self.assertIn("value", item_props)
        self.assertNotIn("internal", item_props)
        self.assertNotIn("llm_config", item_props)

        # LLMConfig should not be in $defs
        self.assertNotIn("LLMConfig", defs)


class EdgeCasesTestCase(unittest.TestCase):
    """Test edge cases in schema filtering."""

    def setUp(self):
        _schema_class_cache.clear()

    def test_model_with_no_excluded_fields(self):
        """Model with no excluded fields should work normally."""

        class PlainModel(ResponseModel):
            field1: str = Field(..., description="Field 1")
            field2: int = Field(..., description="Field 2")

        schema = PlainModel.model_json_schema()

        self.assertIn("field1", schema["properties"])
        self.assertIn("field2", schema["properties"])
        self.assertNotIn("llm_config", schema.get("properties", {}))

        # $defs should be empty (no nested types)
        defs = schema.get("$defs", {})
        self.assertEqual(defs, {})

    def test_model_with_all_optional_excluded(self):
        """Model where all excluded fields are optional."""

        class OptionalExcluded(ResponseModel):
            visible: str = Field(..., description="Visible field")
            hidden1: Optional[str] = PostProcessedField(default=None)
            hidden2: Optional[int] = PostProcessedField(default=None)

        schema = OptionalExcluded.model_json_schema()

        self.assertIn("visible", schema["properties"])
        self.assertNotIn("hidden1", schema["properties"])
        self.assertNotIn("hidden2", schema["properties"])

    def test_recursive_nested_types(self):
        """Test that nested types are properly transformed."""

        class Leaf(ResponseModel):
            data: str = Field(..., description="Leaf data")

        class Branch(ResponseModel):
            leaves: List[Leaf] = Field(default_factory=list)

        class Tree(ResponseModel):
            branches: List[Branch] = Field(default_factory=list)

        schema = Tree.model_json_schema()
        defs = schema.get("$defs", {})

        # Both Branch and Leaf schemas should be in $defs
        # (they may have Schema suffix due to transformation)
        self.assertGreaterEqual(len(defs), 2)

        # LLMConfig should not be present
        self.assertNotIn("LLMConfig", defs)


class LLMConfigSpecificTestCase(unittest.TestCase):
    """Tests specifically for LLMConfig exclusion."""

    def setUp(self):
        _schema_class_cache.clear()

    def test_llm_config_never_in_defs(self):
        """LLMConfig should never appear in $defs, regardless of nesting."""

        class Model1(ResponseModel):
            x: str = Field(..., description="X")

        class Model2(ResponseModel):
            m1: Model1 = Field(..., description="Nested model")

        class Model3(ResponseModel):
            m2: Model2 = Field(..., description="Doubly nested")

        for Model in [Model1, Model2, Model3]:
            _schema_class_cache.clear()
            schema = Model.model_json_schema()
            defs = schema.get("$defs", {})
            self.assertNotIn(
                "LLMConfig", defs, f"LLMConfig found in {Model.__name__}.$defs"
            )

    def test_llm_config_field_never_in_properties(self):
        """llm_config field should never appear in properties."""

        class AnyModel(ResponseModel):
            field: str = Field(..., description="A field")

        schema = AnyModel.model_json_schema()

        # Check top-level
        self.assertNotIn("llm_config", schema.get("properties", {}))

        # Check all $defs
        for def_name, def_schema in schema.get("$defs", {}).items():
            self.assertNotIn(
                "llm_config",
                def_schema.get("properties", {}),
                f"llm_config found in {def_name}.properties",
            )


if __name__ == "__main__":
    unittest.main()
