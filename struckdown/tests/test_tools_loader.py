"""Tests for Python tools loader."""

import tempfile
from pathlib import Path

import pytest

from struckdown.actions import Actions
from struckdown.tools_loader import load_tools_file, load_tools_directory


class TestToolsLoader:
    """Test the tools loader."""

    def setup_method(self):
        """Clear the actions registry before each test."""
        # save original registry
        self._original_registry = Actions._registry.copy()

    def teardown_method(self):
        """Restore the actions registry after each test."""
        Actions._registry = self._original_registry

    def test_load_simple_tool(self):
        """Load a simple tool from a Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from struckdown import Actions

@Actions.register('test_simple_tool')
def simple_tool(context: dict, arg1: str) -> str:
    return f"Hello, {arg1}!"
""")
            f.flush()
            path = Path(f.name)

        try:
            registered = load_tools_file(path)
            assert "test_simple_tool" in registered
            assert Actions.is_registered("test_simple_tool")
        finally:
            path.unlink()

    def test_load_multiple_tools(self):
        """Load multiple tools from a single file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from struckdown import Actions

@Actions.register('test_tool_a')
def tool_a(context: dict) -> str:
    return "A"

@Actions.register('test_tool_b')
def tool_b(context: dict) -> str:
    return "B"
""")
            f.flush()
            path = Path(f.name)

        try:
            registered = load_tools_file(path)
            assert "test_tool_a" in registered
            assert "test_tool_b" in registered
        finally:
            path.unlink()

    def test_load_tool_with_options(self):
        """Load a tool with registration options."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from struckdown import Actions

@Actions.register('test_options_tool', on_error='return_empty', default_save=False)
def options_tool(context: dict, query: str) -> str:
    return f"Query: {query}"
""")
            f.flush()
            path = Path(f.name)

        try:
            registered = load_tools_file(path)
            assert "test_options_tool" in registered
            assert Actions.get_default_save("test_options_tool") is False
        finally:
            path.unlink()

    def test_load_directory(self):
        """Load all tools from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            (tmppath / "tool1.py").write_text("""
from struckdown import Actions

@Actions.register('test_dir_tool1')
def tool1(context: dict) -> str:
    return "1"
""")
            (tmppath / "tool2.py").write_text("""
from struckdown import Actions

@Actions.register('test_dir_tool2')
def tool2(context: dict) -> str:
    return "2"
""")

            registered = load_tools_directory(tmppath)
            assert "test_dir_tool1" in registered
            assert "test_dir_tool2" in registered

    def test_skip_dunder_files(self):
        """__init__.py and similar files should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            (tmppath / "__init__.py").write_text("""
# this should be skipped
from struckdown import Actions

@Actions.register('test_init_tool')
def init_tool(context: dict) -> str:
    return "init"
""")
            (tmppath / "actual.py").write_text("""
from struckdown import Actions

@Actions.register('test_actual_tool')
def actual_tool(context: dict) -> str:
    return "actual"
""")

            registered = load_tools_directory(tmppath)
            assert "test_init_tool" not in registered
            assert "test_actual_tool" in registered

    def test_missing_file(self):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError):
            load_tools_file(Path("/nonexistent/file.py"))

    def test_non_python_file(self):
        """Non-Python files should be skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not python")
            f.flush()
            path = Path(f.name)

        try:
            registered = load_tools_file(path)
            assert registered == []
        finally:
            path.unlink()

    def test_import_error_handling(self):
        """Test handling of import errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
import nonexistent_module_that_does_not_exist
""")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception):  # could be ModuleNotFoundError or ImportError
                load_tools_file(path)
        finally:
            path.unlink()

    def test_file_without_decorators(self):
        """File without @Actions.register decorators returns empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def regular_function():
    return "not a tool"
""")
            f.flush()
            path = Path(f.name)

        try:
            registered = load_tools_file(path)
            assert registered == []
        finally:
            path.unlink()

    def test_empty_directory(self):
        """Empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            registered = load_tools_directory(tmppath)
            assert registered == []

    def test_nonexistent_directory(self):
        """Nonexistent directory returns empty list."""
        registered = load_tools_directory(Path("/nonexistent/directory"))
        assert registered == []
