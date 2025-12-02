"""Tests for YAML type loader."""

import tempfile
from pathlib import Path

import pytest

from struckdown.type_loader import YAMLTypeLoader, load_yaml_types


class TestYAMLTypeLoader:
    """Test the YAML type loader."""

    def test_load_simple_type(self):
        """Load a simple type definition."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: TestType
fields:
  response: str
""")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            name = loader.load_yaml_file(path)
            assert name == "TestType"

            models = loader.build_all_models()
            assert "TestType" in models

            model = models["TestType"]
            assert "response" in model.model_fields
        finally:
            path.unlink()

    def test_load_type_with_constraints(self):
        """Load a type with field constraints."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: ConstrainedType
fields:
  age:
    type: int
    description: Age in years
    ge: 0
    le: 150
  name:
    type: str
    min_length: 1
    max_length: 100
""")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            loader.load_yaml_file(path)
            models = loader.build_all_models()

            model = models["ConstrainedType"]
            assert "age" in model.model_fields
            assert "name" in model.model_fields

            age_field = model.model_fields["age"]
            assert age_field.metadata  # has constraints
        finally:
            path.unlink()

    def test_load_type_with_optional_field(self):
        """Load a type with optional field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: OptionalType
fields:
  required_field: str
  optional_field:
    type: str
    optional: true
""")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            loader.load_yaml_file(path)
            models = loader.build_all_models()

            model = models["OptionalType"]
            optional_field = model.model_fields["optional_field"]
            assert optional_field.default is None
        finally:
            path.unlink()

    def test_load_type_with_choices(self):
        """Load a type with enumerated choices."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: ChoiceType
fields:
  color:
    type: str
    choices: [red, green, blue]
""")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            loader.load_yaml_file(path)
            models = loader.build_all_models()

            model = models["ChoiceType"]
            assert "color" in model.model_fields
        finally:
            path.unlink()

    def test_load_type_with_list_field(self):
        """Load a type with list field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: ListType
fields:
  items:
    type: list[str]
    description: List of items
""")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            loader.load_yaml_file(path)
            models = loader.build_all_models()

            model = models["ListType"]
            items_field = model.model_fields["items"]
            # check it's a list type
            assert "list" in str(items_field.annotation).lower()
        finally:
            path.unlink()

    def test_load_type_with_llm_config(self):
        """Load a type with LLM config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: ConfiguredType
llm_config:
  temperature: 0.9
fields:
  response: str
""")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            loader.load_yaml_file(path)
            models = loader.build_all_models()

            model = models["ConfiguredType"]
            assert model.llm_config.temperature == 0.9
        finally:
            path.unlink()

    def test_load_directory(self):
        """Load all types from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # create two type files
            (tmppath / "type1.yaml").write_text("""
name: Type1
fields:
  field1: str
""")
            (tmppath / "type2.yaml").write_text("""
name: Type2
fields:
  field2: int
""")

            loader = YAMLTypeLoader()
            loaded = loader.load_directory(tmppath)

            assert "Type1" in loaded
            assert "Type2" in loaded

            models = loader.build_all_models()
            assert "Type1" in models
            assert "Type2" in models

    def test_inheritance(self):
        """Test type inheritance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            (tmppath / "base.yaml").write_text("""
name: BaseType
fields:
  base_field: str
""")
            (tmppath / "child.yaml").write_text("""
name: ChildType
extends: BaseType
fields:
  child_field: int
""")

            loader = YAMLTypeLoader()
            loader.load_directory(tmppath)
            models = loader.build_all_models()

            child = models["ChildType"]
            # child should have both fields
            assert "base_field" in child.model_fields or hasattr(child, "base_field")
            assert "child_field" in child.model_fields

    def test_nested_type_reference(self):
        """Test referencing another YAML-defined type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            (tmppath / "inner.yaml").write_text("""
name: Inner
fields:
  value: str
""")
            (tmppath / "outer.yaml").write_text("""
name: Outer
fields:
  inner:
    type: Inner
""")

            loader = YAMLTypeLoader()
            loader.load_directory(tmppath)
            models = loader.build_all_models()

            outer = models["Outer"]
            inner_field = outer.model_fields["inner"]
            # the type should be Inner
            assert "Inner" in str(inner_field.annotation)

    def test_circular_dependency_error(self):
        """Test that circular dependencies are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            (tmppath / "a.yaml").write_text("""
name: TypeA
extends: TypeB
fields:
  field_a: str
""")
            (tmppath / "b.yaml").write_text("""
name: TypeB
extends: TypeA
fields:
  field_b: str
""")

            loader = YAMLTypeLoader()
            loader.load_directory(tmppath)

            with pytest.raises(ValueError, match="Circular dependency"):
                loader.build_all_models()

    def test_missing_file(self):
        """Test handling of missing file."""
        loader = YAMLTypeLoader()
        result = loader.load_yaml_file(Path("/nonexistent/file.yaml"))
        assert result is None

    def test_invalid_yaml(self):
        """Test handling of invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            result = loader.load_yaml_file(path)
            assert result is None
        finally:
            path.unlink()

    def test_missing_name_field(self):
        """Test handling of YAML without name field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
fields:
  response: str
""")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            result = loader.load_yaml_file(path)
            assert result is None
        finally:
            path.unlink()

    def test_clean_str_repr(self):
        """Test that __str__ and __repr__ exclude llm_config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: CleanReprType
description: A type for testing clean repr
llm_config:
  temperature: 0.7
fields:
  name: str
  age: int
  powers:
    type: list[str]
""")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            loader.load_yaml_file(path)
            models = loader.build_all_models()

            model = models["CleanReprType"]
            instance = model(name="Test Hero", age=25, powers=["Flight", "Strength"])

            str_repr = str(instance)
            repr_repr = repr(instance)

            # llm_config should not appear in the string representation
            assert "llm_config" not in str_repr
            assert "llm_config" not in repr_repr

            # but the actual fields should appear
            assert "name=" in str_repr
            assert "Test Hero" in str_repr
            assert "age=" in str_repr
            assert "25" in str_repr
            assert "powers=" in str_repr
            assert "Flight" in str_repr

            # class name should be present
            assert "CleanReprType" in str_repr
            assert "CleanReprType" in repr_repr
        finally:
            path.unlink()


    def test_constraints_preserved_in_optional(self):
        """Test that field constraints are preserved when model is made optional."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: ConstrainedOptional
fields:
  age:
    type: int
    ge: 18
    le: 100
  name:
    type: str
    min_length: 2
    max_length: 50
""")
            f.flush()
            path = Path(f.name)

        try:
            loader = YAMLTypeLoader()
            loader.load_yaml_file(path)
            models = loader.build_all_models()

            base_model = models["ConstrainedOptional"]

            # verify base model has constraints in schema
            base_schema = base_model.model_json_schema()
            assert base_schema["properties"]["age"]["minimum"] == 18
            assert base_schema["properties"]["age"]["maximum"] == 100

            # make it optional (simulates [[type:x]] without !)
            optional_model = loader._make_optional(base_model, "ConstrainedOptional")

            # verify schema includes constraints after making optional
            schema = optional_model.model_json_schema()
            # optional fields use anyOf with the constrained type and null
            age_schema = schema["properties"]["age"]
            assert "anyOf" in age_schema
            # find the non-null type in anyOf
            int_schema = next(s for s in age_schema["anyOf"] if s.get("type") == "integer")
            assert int_schema.get("minimum") == 18
            assert int_schema.get("maximum") == 100

            # check string constraints too
            name_schema = schema["properties"]["name"]
            str_schema = next(s for s in name_schema["anyOf"] if s.get("type") == "string")
            assert str_schema.get("minLength") == 2
            assert str_schema.get("maxLength") == 50
        finally:
            path.unlink()


class TestLoadYamlTypes:
    """Test the load_yaml_types convenience function."""

    def test_load_single_file(self):
        """Load types from a single file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: SingleType
fields:
  response: str
""")
            f.flush()
            path = Path(f.name)

        try:
            # reset the global loader
            import struckdown.type_loader as tl
            tl._loader = None

            registered = load_yaml_types([path])
            assert "SingleType" in registered
        finally:
            path.unlink()
            tl._loader = None
