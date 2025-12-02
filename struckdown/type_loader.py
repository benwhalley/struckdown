"""YAML type loader for custom Pydantic response models.

Allows users to define response types in YAML files that get converted
to Pydantic models at runtime.
"""

import hashlib
import logging
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union, get_args, get_origin

import yaml
from pydantic import Field, create_model

from .validation import ParsedOptions, parse_options
from .response_types import ResponseTypes
from .return_type_models import LLMConfig, ResponseModel

logger = logging.getLogger(__name__)


def _make_clean_repr(self) -> str:
    """Clean __repr__ that excludes llm_config and internal fields."""
    cls_name = self.__class__.__name__
    # get all fields except llm_config and private fields
    fields = []
    for name, value in self.__dict__.items():
        if name.startswith("_") or name == "llm_config":
            continue
        fields.append(f"{name}={value!r}")
    return f"{cls_name}({', '.join(fields)})"


def _make_clean_str(self) -> str:
    """Clean __str__ that shows field values in a readable format."""
    cls_name = self.__class__.__name__
    # get all fields except llm_config and private fields
    fields = []
    for name, value in self.__dict__.items():
        if name.startswith("_") or name == "llm_config":
            continue
        # format value nicely
        if isinstance(value, str):
            fields.append(f"{name}={value!r}")
        elif isinstance(value, list):
            fields.append(f"{name}={value}")
        else:
            fields.append(f"{name}={value}")
    return f"{cls_name}({', '.join(fields)})"


def _add_clean_repr_methods(model: type) -> type:
    """Add clean __str__ and __repr__ methods to a dynamically created model."""
    model.__repr__ = _make_clean_repr
    model.__str__ = _make_clean_str
    return model


# primitive type mappings
PRIMITIVE_TYPES: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "bool": bool,
    "boolean": bool,
    "date": date,
    "datetime": datetime,
    "time": time,
    "timedelta": timedelta,
    "duration": timedelta,
}


class YAMLTypeLoader:
    """Loads and registers YAML-defined response types."""

    def __init__(self):
        # registry of loaded models: name -> model class
        self._models: dict[str, type] = {}
        # raw YAML definitions before resolution
        self._definitions: dict[str, dict] = {}
        # track which models are currently being built (cycle detection)
        self._building: set[str] = set()
        # track which types this loader has already registered
        self._registered: set[str] = set()

    def load_yaml_file(self, path: Path) -> str | None:
        """Load a single YAML type definition file.

        Args:
            path: Path to YAML file

        Returns:
            Type name if loaded successfully, None otherwise
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            if not data or "name" not in data:
                logger.warning(f"YAML file {path} missing 'name' field, skipping")
                return None

            type_name = data["name"]

            # invalidate cached model if definition is being reloaded
            # (important for long-running processes like Django/Jupyter)
            if type_name in self._models:
                del self._models[type_name]
            self._registered.discard(type_name)

            self._definitions[type_name] = data
            self._definitions[type_name]["_source_path"] = str(path)

            logger.debug(f"Loaded YAML definition for '{type_name}' from {path}")
            return type_name

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def load_directory(self, directory: Path) -> list[str]:
        """Load all YAML files from a directory.

        Args:
            directory: Directory to scan for .yaml/.yml files

        Returns:
            List of loaded type names
        """
        loaded = []
        if not directory.is_dir():
            logger.warning(f"Not a directory: {directory}")
            return loaded

        for yaml_file in sorted(directory.glob("*.yaml")):
            name = self.load_yaml_file(yaml_file)
            if name:
                loaded.append(name)

        for yml_file in sorted(directory.glob("*.yml")):
            name = self.load_yaml_file(yml_file)
            if name:
                loaded.append(name)

        return loaded

    def build_all_models(self) -> dict[str, type]:
        """Build Pydantic models for all loaded definitions.

        Handles dependencies and inheritance automatically.

        Returns:
            Dict mapping type names to model classes
        """
        for type_name in self._definitions:
            if type_name not in self._models:
                self._build_model(type_name)

        return self._models

    def _build_model(self, type_name: str) -> type:
        """Build a single model, resolving dependencies first.

        Args:
            type_name: Name of the type to build

        Returns:
            Pydantic model class
        """
        # already built?
        if type_name in self._models:
            return self._models[type_name]

        # cycle detection
        if type_name in self._building:
            raise ValueError(
                f"Circular dependency detected while building '{type_name}'. "
                f"Currently building: {self._building}"
            )

        if type_name not in self._definitions:
            raise ValueError(f"Unknown type '{type_name}' - not loaded from YAML")

        self._building.add(type_name)

        try:
            defn = self._definitions[type_name]

            # handle inheritance first
            base_class = ResponseModel
            if "extends" in defn:
                parent_name = defn["extends"]
                if parent_name not in self._models:
                    self._build_model(parent_name)
                base_class = self._models[parent_name]

            # build field definitions and track explicitly required fields
            field_defs = {}
            explicitly_required = set()
            fields_data = defn.get("fields", {})

            for field_name, field_spec in fields_data.items():
                field_type, field_info = self._parse_field(field_name, field_spec)
                field_defs[field_name] = (field_type, field_info)
                # track if field was explicitly marked required: true
                if isinstance(field_spec, dict) and field_spec.get("required") is True:
                    explicitly_required.add(field_name)

            # create the model
            model = create_model(
                type_name,
                __base__=base_class,
                __module__="struckdown.yaml_types",
                **field_defs,
            )

            # add clean __str__ and __repr__ methods
            _add_clean_repr_methods(model)

            # store explicitly required fields for _make_optional to respect
            model._explicitly_required_fields = explicitly_required

            # store hash of YAML definition for cache invalidation
            # (any change to YAML will change this hash)
            defn_str = str(sorted(defn.items()))
            model._yaml_definition_hash = hashlib.md5(defn_str.encode()).hexdigest()

            # set model description if provided
            if "description" in defn:
                model.__doc__ = defn["description"]

            # set LLM config if provided
            if "llm_config" in defn:
                llm_config_data = defn["llm_config"]
                model.llm_config = LLMConfig(**llm_config_data)

            # store option mappings for parametric types
            if "options" in defn:
                model._yaml_options = defn["options"]

            self._models[type_name] = model
            logger.debug(f"Built model '{type_name}' with fields: {list(field_defs.keys())}")

            return model

        finally:
            self._building.discard(type_name)

    def _parse_field(self, field_name: str, field_spec: Any) -> tuple[type, Any]:
        """Parse a field specification into type and Field() info.

        Args:
            field_name: Name of the field
            field_spec: Either a type string or a dict with type/constraints

        Returns:
            (python_type, Field(...))
        """
        # simple form: just a type string
        if isinstance(field_spec, str):
            python_type = self._resolve_type(field_spec)
            return python_type, Field(...)

        # expanded form: dict with type and constraints
        if not isinstance(field_spec, dict):
            raise ValueError(
                f"Field '{field_name}' must be a string or dict, got {type(field_spec)}"
            )

        type_str = field_spec.get("type", "str")
        python_type = self._resolve_type(type_str)

        # handle optional/required - explicit required=false is same as optional=true
        is_optional = field_spec.get("optional", False)
        if "required" in field_spec:
            is_optional = not field_spec["required"]
        if is_optional:
            python_type = Optional[python_type]

        # build Field() kwargs
        field_kwargs: dict[str, Any] = {}

        # description
        if "description" in field_spec:
            field_kwargs["description"] = field_spec["description"]

        # default value
        if "default" in field_spec:
            field_kwargs["default"] = field_spec["default"]
        elif is_optional:
            field_kwargs["default"] = None

        # numeric constraints
        for constraint in ["ge", "gt", "le", "lt"]:
            if constraint in field_spec:
                field_kwargs[constraint] = field_spec[constraint]

        # string constraints
        if "min_length" in field_spec:
            field_kwargs["min_length"] = field_spec["min_length"]
        if "max_length" in field_spec:
            field_kwargs["max_length"] = field_spec["max_length"]
        if "pattern" in field_spec:
            field_kwargs["pattern"] = field_spec["pattern"]

        # choices -> Literal type
        if "choices" in field_spec:
            choices = tuple(field_spec["choices"])
            python_type = Literal[choices]  # type: ignore
            if is_optional:
                python_type = Optional[python_type]

        # determine if required
        if "default" not in field_kwargs and not is_optional:
            return python_type, Field(..., **field_kwargs)
        else:
            return python_type, Field(**field_kwargs)

    def _resolve_type(self, type_str: str) -> type:
        """Resolve a type string to a Python type.

        Handles: primitives, list[T], optional[T], references to other YAML types.
        """
        type_str = type_str.strip()

        # list types: list[str], list[Superhero]
        list_match = re.match(r"list\[(.+)\]", type_str, re.IGNORECASE)
        if list_match:
            inner_type = self._resolve_type(list_match.group(1))
            return list[inner_type]

        # optional types: optional[str]
        opt_match = re.match(r"optional\[(.+)\]", type_str, re.IGNORECASE)
        if opt_match:
            inner_type = self._resolve_type(opt_match.group(1))
            return Optional[inner_type]

        # primitive types
        if type_str.lower() in PRIMITIVE_TYPES:
            return PRIMITIVE_TYPES[type_str.lower()]

        # reference to another YAML-defined type
        if type_str in self._definitions:
            if type_str not in self._models:
                self._build_model(type_str)
            return self._models[type_str]

        # check if it's already registered in ResponseTypes
        existing = ResponseTypes.get(type_str)
        if existing is not None:
            return existing

        # fallback: treat as string
        logger.warning(f"Unknown type '{type_str}', treating as str")
        return str

    def register_all(self) -> list[str]:
        """Build all models and register them with ResponseTypes.

        Returns:
            List of registered type names
        """
        self.build_all_models()
        registered = []

        # determine struckdown package path for checking built-in types
        struckdown_path = Path(__file__).parent

        for type_name, model in self._models.items():
            # skip if already registered by this loader
            if type_name in self._registered:
                continue

            defn = self._definitions.get(type_name, {})
            source_path = defn.get("_source_path", "")

            # check if this type is from the struckdown package (built-in)
            is_builtin_yaml = False
            if source_path:
                source_path_obj = Path(source_path).resolve()
                is_builtin_yaml = struckdown_path in source_path_obj.parents

            # only warn if a non-builtin type overrides an existing type
            if ResponseTypes.is_registered(type_name) and not is_builtin_yaml:
                logger.warning(
                    f"YAML type '{type_name}' from {source_path} overrides built-in type"
                )

            # create factory function for the type
            factory = self._create_type_factory(type_name, model, defn)

            # register with ResponseTypes
            ResponseTypes._registry[type_name] = factory
            self._registered.add(type_name)
            registered.append(type_name)
            logger.debug(f"Registered YAML type: {type_name}")

        return registered

    def _create_type_factory(
        self, type_name: str, base_model: type, defn: dict
    ) -> callable:
        """Create a factory function for a YAML-defined type.

        The factory handles options and quantifiers like built-in types.
        """
        option_mappings = defn.get("options", {})
        # top-level required setting - makes type required by default
        default_required = defn.get("required", False)

        def yaml_type_factory(
            options: list[str] | None = None,
            quantifier: tuple | None = None,
            required_prefix: bool = False,
        ):
            opts = parse_options(options)
            is_required = default_required or opts.required or required_prefix

            # apply option mappings to create constrained model
            if option_mappings and options:
                model = self._apply_option_mappings(
                    base_model, type_name, opts, option_mappings
                )
            else:
                model = base_model

            # handle quantifiers (multiple responses)
            if quantifier:
                return self._wrap_in_list(model, type_name, quantifier, is_required)

            # handle required/optional
            if not is_required:
                return self._make_optional(model, type_name)

            return model

        return yaml_type_factory

    def _apply_option_mappings(
        self,
        base_model: type,
        type_name: str,
        opts: ParsedOptions,
        mappings: dict,
    ) -> type:
        """Apply option mappings to create a constrained model.

        Example mapping: {"min": "response.ge", "max": "response.le"}
        """
        # collect field modifications
        field_mods: dict[str, dict] = {}

        for opt_name, field_path in mappings.items():
            opt_value = opts.kwargs.get(opt_name)
            if opt_value is None:
                continue

            # parse field path: "response.ge" -> field="response", constraint="ge"
            parts = field_path.split(".")
            if len(parts) != 2:
                logger.warning(f"Invalid option mapping: {opt_name}={field_path}")
                continue

            field_name, constraint = parts
            if field_name not in field_mods:
                field_mods[field_name] = {}
            field_mods[field_name][constraint] = opt_value

        if not field_mods:
            return base_model

        # rebuild model with new constraints
        new_field_defs = {}
        for field_name, field_info in base_model.model_fields.items():
            # rebuild annotation with metadata (constraints like Ge, Le, etc.)
            base_annotation = field_info.annotation
            if field_info.metadata:
                base_annotation = Annotated[(base_annotation,) + tuple(field_info.metadata)]

            field_kwargs = {
                "description": field_info.description,
                "default": field_info.default if field_info.default is not None else ...,
            }

            if field_name in field_mods:
                # apply new constraints (these are passed to Field())
                field_kwargs.update(field_mods[field_name])

            new_field_defs[field_name] = (base_annotation, Field(**field_kwargs))

        capitalized_name = type_name[0].upper() + type_name[1:] if type_name else type_name

        constrained_model = create_model(
            f"{capitalized_name}Constrained",
            __base__=ResponseModel,
            __module__="struckdown.yaml_types",
            **new_field_defs,
        )
        return _add_clean_repr_methods(constrained_model)

    def _wrap_in_list(
        self,
        model: type,
        type_name: str,
        quantifier: tuple,
        is_required: bool,
    ) -> type:
        """Wrap a model in a list type for quantifiers."""
        min_items, max_items = quantifier

        field_kwargs = {}
        if min_items is not None:
            field_kwargs["min_length"] = min_items
        if max_items is not None:
            field_kwargs["max_length"] = max_items

        # build description
        if min_items == max_items:
            desc = f"Exactly {min_items}"
        elif max_items is None:
            desc = f"At least {min_items}" if min_items > 0 else "Any number of"
        elif min_items == 0:
            desc = f"Up to {max_items}"
        else:
            desc = f"Between {min_items} and {max_items}"

        description = f"{desc} {type_name} items."
        capitalized_name = type_name[0].upper() + type_name[1:] if type_name else type_name

        list_model = create_model(
            f"Multi{capitalized_name}Response",
            __base__=ResponseModel,
            __module__="struckdown.yaml_types",
            response=(list[model], Field(..., description=description, **field_kwargs)),
        )

        # add clean repr methods
        _add_clean_repr_methods(list_model)

        # copy LLM config
        if hasattr(model, "llm_config"):
            list_model.llm_config = model.llm_config

        return list_model

    def _make_optional(self, model: type, type_name: str) -> type:
        """Make response fields optional, except those explicitly marked required."""
        new_field_defs = {}

        # get fields that must stay required
        explicitly_required = getattr(model, "_explicitly_required_fields", set())

        for field_name, field_info in model.model_fields.items():
            # rebuild annotation with metadata (constraints like Ge, Le, etc.)
            base_annotation = field_info.annotation
            if field_info.metadata:
                # wrap annotation with its constraints
                base_annotation = Annotated[(base_annotation,) + tuple(field_info.metadata)]

            field_kwargs = {"description": field_info.description}

            if field_name in explicitly_required:
                # keep field as-is (required)
                new_field_defs[field_name] = (
                    base_annotation,
                    Field(
                        default=... if field_info.is_required() else field_info.default,
                        **field_kwargs,
                    ),
                )
            else:
                # make field optional with None default
                optional_type = Optional[base_annotation]
                new_field_defs[field_name] = (
                    optional_type,
                    Field(
                        default=None,
                        **field_kwargs,
                    ),
                )

        capitalized_name = type_name[0].upper() + type_name[1:] if type_name else type_name

        optional_model = create_model(
            f"Optional{capitalized_name}Response",
            __base__=ResponseModel,
            __module__="struckdown.yaml_types",
            **new_field_defs,
        )

        # add clean repr methods
        _add_clean_repr_methods(optional_model)

        if hasattr(model, "llm_config"):
            optional_model.llm_config = model.llm_config

        return optional_model


# global loader instance
_loader: YAMLTypeLoader | None = None


def get_loader() -> YAMLTypeLoader:
    """Get the global YAML type loader instance."""
    global _loader
    if _loader is None:
        _loader = YAMLTypeLoader()
    return _loader


def load_yaml_types(paths: list[Path]) -> list[str]:
    """Load YAML types from files or directories.

    Args:
        paths: List of YAML files or directories to load

    Returns:
        List of registered type names
    """
    loader = get_loader()

    for path in paths:
        if path.is_dir():
            loader.load_directory(path)
        elif path.is_file():
            loader.load_yaml_file(path)
        else:
            logger.warning(f"Path not found: {path}")

    return loader.register_all()


def discover_yaml_types(
    template_path: Path | None = None,
    cwd: Path | None = None,
) -> list[str]:
    """Auto-discover and load YAML types from conventional locations.

    Discovery order (highest priority first):
    1. types/ relative to template file
    2. types/ in current working directory
    3. Built-in types in struckdown/types/

    Args:
        template_path: Path to the .sd template file
        cwd: Current working directory (defaults to Path.cwd())

    Returns:
        List of registered type names
    """
    if cwd is None:
        cwd = Path.cwd()

    loader = get_loader()
    locations_searched = []

    # 1. types/ relative to template
    if template_path and template_path.parent.is_dir():
        types_dir = template_path.parent / "types"
        if types_dir.is_dir():
            loader.load_directory(types_dir)
            locations_searched.append(types_dir)

    # 2. types/ in cwd
    cwd_types = cwd / "types"
    if cwd_types.is_dir() and cwd_types not in locations_searched:
        loader.load_directory(cwd_types)
        locations_searched.append(cwd_types)

    # 3. built-in types in struckdown/types/
    builtin_types = Path(__file__).parent / "types"
    if builtin_types.is_dir():
        loader.load_directory(builtin_types)
        locations_searched.append(builtin_types)

    if locations_searched:
        logger.debug(f"Searched for YAML types in: {locations_searched}")

    return loader.register_all()
