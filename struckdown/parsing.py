from collections import OrderedDict, namedtuple
from importlib.resources import files
from pathlib import Path

from lark import Lark, Transformer
from pydantic import ValidationError

from .actions import Actions
from .response_types import ResponseTypes
from .return_type_models import LLMConfig

try:
    mindframe_grammar = (files(__package__) / "grammar.lark").read_text(
        encoding="utf-8"
    )
except Exception:
    # fallback to relative path from current file
    grammar_path = Path(__file__).parent / "grammar.lark"
    mindframe_grammar = grammar_path.read_text(encoding="utf-8")

MAX_LIST_LENGTH = 100
DEFAULT_RETURN_TYPE = "respond"


PromptPart = namedtuple(
    "PromptPart",
    [
        "key",
        "return_type",
        "options",
        "text",
        "system_message",  # Accumulated system message at this point
        "header_content",  # Accumulated header content at this point
        "quantifier",
        "action_type",
        "llm_kwargs",
        "required_prefix",
        "is_function",
        "has_explicit_var",
    ],
)


class MindframeTransformer(Transformer):
    def __init__(self):
        self.sections = []
        self.current_buf = []  # holds text/vars/etc
        self.current_parts = []  # list of (key, PromptPart)
        self.current_system = ""  # Accumulated system message
        self.current_header = ""  # Accumulated header content
        self.completion_counters = (
            {}
        )  # Track auto-variable generation per completion type
        self.action_counters = {}  # Track auto-variable generation per action name

    def _flush_section(self):
        if self.current_parts:
            self.sections.append(OrderedDict(self.current_parts))
        self.current_buf = []
        self.current_parts = []

    def markdown_text(self, items):
        text = str(items[0]).strip()
        self.current_buf.append({"type": "text", "text": text})
        return None

    def block_comment(self, items):
        # drop block comments entirely -- they are not included in the parsed output
        return None

    def placeholder(self, items):
        self.current_buf.append(
            {"type": "variable", "name": ".".join(map(str, items[0]))}
        )
        return None

    def templatetag(self, items):
        # todo: check for valid templatetags??
        self.current_buf.append(
            {
                "type": "templatetag",
                "text": str(items[0]),
            }  # already includes `{% ... %}`
        )
        return None

    def _validate_no_completions(self, content: str, block_type: str):
        """Validate that content doesn't contain completion syntax [[...]]

        Args:
            content: The content to validate
            block_type: Name of block type for error message (e.g., "¡SYSTEM", "¡HEADER")

        Raises:
            ValueError: If completion syntax is found
        """
        if '[[' in content:
            # Find the first occurrence for a helpful error message
            import re
            match = re.search(r'\[\[([^\]]+)\]\]', content)
            if match:
                completion_text = match.group(0)
                raise ValueError(
                    f"Completions are not allowed in {block_type} blocks.\n"
                    f"Found: {completion_text}\n\n"
                    f"{block_type} blocks can only contain:\n"
                    f"  - Static text\n"
                    f"  - Template variables: {{{{variable}}}}\n"
                    f"  - Template tags: {{% tag %}}\n\n"
                    f"To use completions, place them before the {block_type} block:\n"
                    f"  Example:\n"
                    f"    What language? [[pick:language|French,German]]\n"
                    f"    \n"
                    f"    {block_type}\n"
                    f"    Answer in {{{{language}}}}\n"
                    f"    /END"
                )

    def system_replace(self, items):
        """Handle ¡SYSTEM - replace current system message"""
        content = str(items[0]).strip() if items and items[0] else ""
        self._validate_no_completions(content, "¡SYSTEM")
        self.current_system = content
        return None

    def system_append(self, items):
        """Handle ¡SYSTEM+ - append to current system message"""
        content = str(items[0]).strip() if items and items[0] else ""
        self._validate_no_completions(content, "¡SYSTEM+")
        if self.current_system:
            self.current_system += "\n\n" + content
        else:
            self.current_system = content
        return None

    def header_replace(self, items):
        """Handle ¡HEADER - replace current header (empty wipes it)"""
        content = str(items[0]).strip() if items and items[0] else ""
        self._validate_no_completions(content, "¡HEADER")
        self.current_header = content
        return None

    def header_append(self, items):
        """Handle ¡HEADER+ - append to current header"""
        content = str(items[0]).strip() if items and items[0] else ""
        self._validate_no_completions(content, "¡HEADER+")
        if self.current_header:
            self.current_header += "\n\n" + content
        else:
            self.current_header = content
        return None

    def obliviate(self, items):
        self._flush_section()
        return None

    def single_completion(self, items):
        body = items[0]
        # Check if body dict contains is_function flag (set by typed_completion)
        is_function = body.get("is_function", False)
        self._record_completion(body, is_function=is_function)
        return None

    def _record_completion(self, body, is_function=False):
        prompt = self._collapse_prompt_text()

        # Parse options into plain options and LLM kwargs
        options_list = body.get("options", [])
        plain_options, llm_kwargs = self._parse_options(options_list)

        part = PromptPart(
            key=body["key"],
            return_type=body["return_type"],
            options=plain_options,  # Store only non-key=value options
            text=prompt,
            system_message=self.current_system,  # Capture accumulated system message
            header_content=self.current_header,  # Capture accumulated header content
            quantifier=body.get("quantifier", None),
            action_type=body.get("action_type"),
            llm_kwargs=llm_kwargs,  # Store parsed LLM parameters
            required_prefix=body.get("required_prefix", False),  # Store ! prefix flag
            is_function=is_function,  # Mark if this is a function call (no LLM)
            has_explicit_var=body.get(
                "has_explicit_var", True
            ),  # Default to True for LLM completions
        )
        self.current_parts.append((body["key"], part))
        self.current_buf = []  # reset prompt buffer
        # Don't auto-flush - let sections build up naturally within OBLIVIATE blocks

    def _parse_options(self, options_list):
        """Parse options list into separate plain options and key=value kwargs.

        Args:
            options_list: List of option strings that may contain key=value pairs

        Returns:
            tuple: (plain_options, kwargs_dict)
                plain_options: List of strings without '=' OR model-specific key=value like min=0
                kwargs_dict: Dict of LLM parameter key=value pairs (temperature, model)
        """
        # Get LLM parameter keys from LLMConfig schema
        LLM_PARAM_KEYS = set(LLMConfig.model_fields.keys())

        plain_options = []
        kwargs_dict = {}

        for opt in options_list:
            if "=" in opt:
                key, value = opt.split("=", 1)
                key = key.strip()
                value = value.strip()

                # If it's an LLM parameter, validate and add to kwargs_dict
                if key in LLM_PARAM_KEYS:
                    # Build a partial config to validate just this parameter
                    try:
                        # Create a minimal config with just this parameter to validate it
                        test_config = {key: value}
                        validated = LLMConfig.model_validate(test_config, strict=False)
                        # Use the validated (type-converted) value from pydantic
                        kwargs_dict[key] = getattr(validated, key)
                    except ValidationError as e:
                        # Extract user-friendly error message
                        error_msg = e.errors()[0]["msg"] if e.errors() else str(e)
                        raise ValueError(f"Invalid value for '{key}': {error_msg}")
                else:
                    # Model-specific options like min=0, max=100, required stay as plain options
                    plain_options.append(opt)
            else:
                plain_options.append(opt)

        return plain_options, kwargs_dict

    def _collapse_prompt_text(self):
        lines = []
        for item in self.current_buf:
            if item["type"] == "text":
                lines.append(item["text"])
            elif item["type"] == "variable":
                lines.append(f"{{{{{item['name']}}}}}")
            elif item["type"] == "templatetag":
                lines.append(item["text"])  # already in full {% ... %} form

        prompt_text = "\n".join(lines).strip()

        return prompt_text

    def _lookup_rt(self, key, options=None, quantifier=None, required_prefix=False):
        """Lookup return type, checking Actions registry first, then ResponseTypes.

        Args:
            key: Action type name (e.g., 'evidence', 'memory', 'pick', 'respond')
            options: Options from template (for Actions registry)
            quantifier: Quantifier from template (for Actions registry)
            required_prefix: Required prefix flag (for Actions registry)

        Returns:
            ResponseModel class or factory function

        Note:
            Actions registry is checked first to ensure custom actions take priority
            over built-in types. This enables disambiguation in [[...]] syntax.
        """
        # first check Actions registry (custom actions take priority)
        action_model = Actions.create_action_model(
            key, options, quantifier, required_prefix
        )
        if action_model is not None:
            return action_model

        # then check ResponseTypes registry (built-in types like bool, str, etc.)
        rt = ResponseTypes.get(key)
        if rt is not None:
            return rt

        # not found - return default response type
        return ResponseTypes.get("default")

    # All other methods: passthrough or reused from your original
    def typed_completion_with_var(self, items):
        """Handle [[type:var]] or [[!type:var]] or [[type+:var|options]]

        Grammar: BANG? CNAME quantifier? ":" CNAME ("|" option_list)?
        Possible items: [type, var, options] or [!, type, var, options] or [type, quantifier, var, options]
        """
        idx = 0
        required_prefix = False

        # Check if first item is "!" (required prefix)
        if items and str(items[0]) == "!":
            required_prefix = True
            idx = 1

        type_name = str(items[idx])
        idx += 1

        # Check if next item is a quantifier (tuple) or the variable name
        if idx < len(items) and isinstance(items[idx], tuple):
            quantifier = items[idx]
            idx += 1
        else:
            quantifier = None

        # Next item is the variable name
        var_name = str(items[idx])
        idx += 1

        # Last item (if present) is options
        options = items[idx] if idx < len(items) else []

        # Check if type_name is a registered action (for disambiguation)
        is_function = Actions.is_registered(type_name)

        return {
            "return_type": self._lookup_rt(
                type_name, options, quantifier, required_prefix
            ),
            "key": var_name,
            "options": options,
            "quantifier": quantifier,
            "action_type": type_name,
            "required_prefix": required_prefix,
            "is_function": is_function,
            "has_explicit_var": True,
        }

    def typed_completion_auto_var(self, items):
        """Handle [[type:]] or [[type+:]] - auto-generate variable

        Grammar: BANG? CNAME quantifier? ":" ("|" option_list)?
        """
        idx = 0
        required_prefix = False

        # Check if first item is "!" (required prefix)
        if items and str(items[0]) == "!":
            required_prefix = True
            idx = 1

        type_name = str(items[idx])
        idx += 1

        # Check if next item is a quantifier (tuple) or options (list)
        if idx < len(items) and isinstance(items[idx], tuple):
            quantifier = items[idx]
            idx += 1
        else:
            quantifier = None

        # Last item (if present) is options
        options = items[idx] if idx < len(items) else []

        # Auto-generate variable name
        if type_name not in self.completion_counters:
            self.completion_counters[type_name] = 0
        self.completion_counters[type_name] += 1
        var_name = f"_{type_name}_{self.completion_counters[type_name]:02d}"

        # Check if type_name is a registered action (for disambiguation)
        is_function = Actions.is_registered(type_name)

        return {
            "return_type": self._lookup_rt(
                type_name, options, quantifier, required_prefix
            ),
            "key": var_name,
            "options": options,
            "quantifier": quantifier,
            "action_type": type_name,
            "required_prefix": required_prefix,
            "is_function": is_function,
            "has_explicit_var": True,
        }

    def typed_completion_no_var(self, items):
        """Handle [[type]] or [[!type]] or [[type+|options]] - no variable specified

        Grammar: BANG? CNAME quantifier? ("|" option_list)?
        """
        idx = 0
        required_prefix = False

        # Check if first item is "!" (required prefix)
        if items and str(items[0]) == "!":
            required_prefix = True
            idx = 1

        type_name = str(items[idx])
        idx += 1

        # Check if next item is a quantifier (tuple) or options (list)
        if idx < len(items) and isinstance(items[idx], tuple):
            quantifier = items[idx]
            idx += 1
        else:
            quantifier = None

        # Last item (if present) is options
        options = items[idx] if idx < len(items) else []

        # Check if type_name is a registered action or response type
        is_function = Actions.is_registered(type_name)
        is_registered_type = is_function or ResponseTypes.get(type_name) is not None

        # If type_name matches a registered type, raise an error - user must be explicit
        if is_registered_type:
            raise ValueError(
                f"Ambiguous completion [[{type_name}]]: '{type_name}' is a registered completion type. "
                f"Please be explicit:\n"
                f"  - For a variable named '{type_name}': Not recommended, choose a different name\n"
                f"  - For type '{type_name}' with explicit variable: [[{type_name}:myvar]]\n"
                f"  - For type '{type_name}' with auto-generated variable: [[{type_name}:]]"
            )

        # It's an unregistered name like [[x]] - use as variable name
        var_name = type_name

        return {
            "return_type": self._lookup_rt(
                type_name, options, quantifier, required_prefix
            ),
            "key": var_name,
            "options": options,
            "quantifier": quantifier,
            "action_type": type_name,
            "required_prefix": required_prefix,
            "is_function": is_function,
            "has_explicit_var": False,  # No explicit var provided
        }

    def action_call_with_var(self, items):
        """Handle [[@action:myvar]] or [[@action:myvar|options]]

        items: [ACTION_PREFIX, action_name, var_name, options?]
        """
        # Skip ACTION_PREFIX token (items[0] is '@')
        action_name = str(items[1])
        var_name = str(items[2])
        options = items[3] if len(items) > 3 else []

        return {
            "return_type": self._lookup_rt(action_name, options, None, False),
            "key": var_name,
            "options": options,
            "quantifier": None,
            "action_type": action_name,
            "required_prefix": False,
            "is_function": True,
            "has_explicit_var": True,
        }

    def action_call_auto_var(self, items):
        """Handle [[@action:]] or [[@action:|options]] - auto-generate variable

        items: [ACTION_PREFIX, action_name, options?]
        """
        # Skip ACTION_PREFIX token (items[0] is '@')
        action_name = str(items[1])
        options = items[2] if len(items) > 2 else []

        # Auto-generate variable name
        if action_name not in self.action_counters:
            self.action_counters[action_name] = 0
        self.action_counters[action_name] += 1
        var_name = f"_{action_name}_{self.action_counters[action_name]:02d}"

        return {
            "return_type": self._lookup_rt(action_name, options, None, False),
            "key": var_name,
            "options": options,
            "quantifier": None,
            "action_type": action_name,
            "required_prefix": False,
            "is_function": True,
            "has_explicit_var": True,
        }

    def action_call_no_var(self, items):
        """Handle [[@action]] or [[@action|options]] - no variable specified

        items: [ACTION_PREFIX, action_name, options?]
        """
        # Skip ACTION_PREFIX token (items[0] is '@')
        action_name = str(items[1])
        options = items[2] if len(items) > 2 else []

        return {
            "return_type": self._lookup_rt(action_name, options, None, False),
            "key": action_name,
            "options": options,
            "quantifier": None,
            "action_type": action_name,
            "required_prefix": False,
            "is_function": True,
            "has_explicit_var": False,  # No explicit var - check default_save
        }

    def untyped_completion(self, items):
        return {
            "return_type": self._lookup_rt("default"),
            "key": str(items[0]),
            "options": items[1] if len(items) > 1 else [],
            "action_type": "default",
        }

    def option_list(self, items):
        return list(map(str, items))

    def option(self, item):
        return str(item[0]).strip()

    def var_path(self, items):
        return list(map(str, items))

    # Quantifier transformer methods
    def zero_or_more(self, items):
        return (0, None)

    def one_or_more(self, items):
        return (1, None)

    def zero_or_one(self, items):
        return (0, 1)

    def exact(self, items):
        n = int(str(items[0]))
        return (n, n)

    def at_least(self, items):
        n = int(str(items[0]))
        return (n, None)

    def between(self, items):
        min_n = int(str(items[0]))
        max_n = int(str(items[1]))
        return (min_n, max_n)

    def start(self, items):
        # end of parse: flush final section
        self._flush_section()
        return self.sections


def parser():
    return Lark(
        mindframe_grammar,
        parser="lalr",
        transformer=MindframeTransformer(),
    )


def _add_default_completion_if_needed(template: str) -> str:
    """
    Add [[response]] to final segment if it doesn't end with a completion placeholder.

    This allows the final segment to omit the completion placeholder while maintaining
    backward compatibility. Non-final segments are validated to ensure they have completions.

    Args:
        template: Raw template string

    Returns:
        Template with [[response]] appended to final segment if needed

    Raises:
        ValueError: If a non-final segment is missing a completion placeholder
    """
    SEGMENT_DELIMITER = "¡OBLIVIATE"

    # Helper to check if text ends with a completion placeholder
    def ends_with_completion(text: str) -> bool:
        stripped = text.rstrip()
        # Check for completion [[...]] (both LLM completions and actions use same syntax)
        if stripped.endswith("]]") and "[[" in stripped:
            return True
        return False

    # Check if we have multiple segments
    if SEGMENT_DELIMITER in template:
        segments = template.split(SEGMENT_DELIMITER)

        # Validate all non-final segments have completions
        for i, segment in enumerate(segments[:-1]):
            if not ends_with_completion(segment):
                raise ValueError(
                    f"Non-final segment {i+1} must end with a completion placeholder like [[variable]]. "
                    f"Only the final segment can omit the completion placeholder."
                )

        # Add default completion to final segment if needed
        final_segment = segments[-1]
        if not ends_with_completion(final_segment):
            segments[-1] = final_segment.rstrip() + "\n\n[[response]]"

        return SEGMENT_DELIMITER.join(segments)
    else:
        # Single segment - add completion if needed
        if not ends_with_completion(template):
            return template.rstrip() + "\n\n[[response]]"
        return template


def parse_syntax(syntax):
    """
    parse_syntax("hello {{cruel}} [[world]]")

    extract_all_placeholders("hello {{cruel}} [[world]]")
    """
    preprocessed = _add_default_completion_if_needed(syntax)
    return parser().parse(preprocessed.strip())


def _format_parse_error(error_str, template_text):
    """Convert technical Lark parsing errors into user-friendly messages"""
    import re

    # extract line and column if available
    line_match = re.search(r"line (\d+)", error_str)
    col_match = re.search(r"column (\d+)", error_str)

    line_num = int(line_match.group(1)) if line_match else None
    col_num = int(col_match.group(1)) if col_match else None

    # show the problematic line with context
    context = ""
    if line_num:
        lines = template_text.split("\n")
        if line_num <= len(lines):
            problematic_line = lines[line_num - 1]
            context = f"\n\nLine {line_num}: {problematic_line}"
            if col_num:
                pointer = " " * (len(f"Line {line_num}: ") + col_num - 1) + "^"
                context += f"\n{pointer}"

    # detect common error patterns and provide helpful messages

    # check for completion not at the very end
    if "[[" in template_text and template_text.strip().endswith("]]"):
        # find the last completion
        import re

        completion_matches = list(re.finditer(r"\[\[.*?\]\]", template_text))
        if completion_matches:
            last_completion = completion_matches[-1]
            text_after_completion = template_text[last_completion.end() :].strip()
            if text_after_completion:
                return f"Content found after final completion.{context}\n\nHint: Completions must be at the very end of the template. Move any content before the final [[completion]]."

    if "Unexpected token" in error_str and ("}}" in error_str or "CNAME" in error_str):
        # check if template text has unmatched opening braces
        open_count = template_text.count("{{")
        close_count = template_text.count("}}")
        if open_count > close_count:
            return f"Missing closing braces '}}' in template variable.{context}\n\nHint: Make sure all {{{{ variables }}}} have matching opening and closing braces."

    if "Unexpected token" in error_str and "[[" in template_text:
        return f"Syntax error in completion block [[ ]].{context}\n\nHint: Check that completion blocks are properly formatted like [[type:variable]] or [[variable]]."

    if "Unexpected token" in error_str and "{%" in template_text:
        return f"Syntax error in template tag.{context}\n\nHint: Template tags should be formatted like {{% tag %}}."

    # fallback for unmatched patterns
    return f"Template syntax error.{context}\n\nThe template contains invalid syntax. Common issues:\n- Template must end with a completion like [[variable]]\n- Missing closing braces in {{{{ variables }}}}\n- Malformed completion blocks [[ ]]\n- Invalid template tags {{% %}}"


def extract_all_placeholders(template_text):
    # TODO: do this from the parser and return in the proper order
    return extract_placeholders(template_text) + extract_template_tags(template_text)


def extract_placeholders(template_text):
    """Extract {{placeholder}} variable names that need user input (excluding LLM-generated ones)"""
    try:
        sections = parse_syntax(template_text)

        # collect all completion keys (LLM-generated variables)
        completion_keys = set()
        for section in sections:
            for key, prompt_part in section.items():
                # this key is generated by LLM completion
                completion_keys.add(key)

        # find ALL {{placeholder}} variables in the entire template text
        # this ensures we don't miss placeholders that appear between or after completions
        import re

        all_placeholders = []
        matches = re.findall(r"\{\{([^}]+)\}\}", template_text)
        # add to list while preserving order and avoiding duplicates
        for match in matches:
            if match not in all_placeholders:
                all_placeholders.append(match)

        # return only placeholders that are NOT generated by completions, preserving order
        user_input_placeholders = [
            p for p in all_placeholders if p not in completion_keys
        ]
        return user_input_placeholders

    except Exception as e:
        # convert technical parsing errors to user-friendly messages
        friendly_error = _format_parse_error(str(e), template_text)
        raise Exception(f"Template parsing failed: {friendly_error}")


def extract_template_tags(template_text):
    """Extract template tag names that need user input for simulation"""
    try:
        parse_syntax(template_text)
        # TODO FIX THIS TO USE THE PARSER SECTIONS
        # find ALL {% tag %} template tags in the entire template text
        # this ensures we don't miss template tags that appear between or after completions
        import re

        matches = re.findall(r"\{%\s*(\w+)(?:\s.*?)?\s*%\}", template_text)
        # add to list while preserving order and avoiding duplicates
        m = [m for m in matches]
        if len(set(m)) < len(m):
            m = [f"{j}_{i}" for i, j in enumerate(m)]
        return m

    except Exception as e:
        # convert technical parsing errors to user-friendly messages
        friendly_error = _format_parse_error(str(e), template_text)
        raise Exception(f"Template parsing failed: {friendly_error}")


def replace_template_tags_with_placeholders(template_text):
    """Replace template tags with user-provided content for simulation"""

    tag_replacements = extract_template_tags(template_text)
    try:
        # parse the template first to ensure it's valid
        parser().parse(template_text.strip())

        # TODO: DOES THIS NEED TO USE THE PARSER ?
        import re

        # find all template tags and replace them with the provided replacements
        def replace_tag(match):
            match.group(0)  # the full {% ... %} tag
            tag_name = match.group(1)  # just the tag name

            if tag_name in tag_replacements:
                return tag_replacements[tag_name]
            else:
                # if no replacement provided, replace with a safe placeholder to avoid Django processing
                return f"[Template tag: {tag_name} - no simulation provided]"

        # replace simple tags like {% turns %} but not complex ones like {% for ... %}
        result = re.sub(r"\{%\s*(\w+)(?:\s[^%]*)?\s*%\}", replace_tag, template_text)
        return result

    except Exception as e:
        raise e
        # convert technical parsing errors to user-friendly messages
        friendly_error = _format_parse_error(str(e), template_text)
        raise Exception(f"Template parsing failed: {friendly_error}")


"""
example_input = open('docs_/maximal_syntax_example.md').read()
sections = parse_syntax(example_input)  # ← list of OrderedDicts
list(sections[0].items())[0][1].text
sections
"""


def serialise_sections(sections):
    """Convert parsed output into JSON-serialisable structure"""
    serialised = []
    for section in sections:
        section_list = []
        for key in section:  # keep OrderedDict key order
            part = section[key]
            section_list.append(
                {
                    "key": part.key,
                    "return_type": part.return_type.__name__,
                    "options": part.options,
                    "text": part.text,
                }
            )
        serialised.append(section_list)
    return serialised
