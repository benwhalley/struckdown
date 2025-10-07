from collections import OrderedDict, namedtuple
from importlib.resources import files
from pathlib import Path

from lark import Lark, Transformer

from .return_type_models import ACTION_LOOKUP

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
    ["key", "return_type", "options", "text", "shared_header", "quantifier"],
)


class MindframeTransformer(Transformer):
    def __init__(self, action_lookup=ACTION_LOOKUP):
        self.action_lookup = action_lookup
        self.sections = []
        self.current_buf = []  # holds text/vars/etc
        self.current_parts = []  # list of (key, PromptPart)
        self.shared_header = ""  # holds shared header text

    def _flush_section(self):
        if self.current_parts:
            self.sections.append(OrderedDict(self.current_parts))
        self.current_buf = []
        self.current_parts = []

    def markdown_text(self, items):
        text = str(items[0]).strip()
        self.current_buf.append({"type": "text", "text": text})
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

    def begin_delimiter(self, items):
        # capture everything in current_buf as shared_header (including variables and template tags)
        header_parts = []
        for item in self.current_buf:
            if item["type"] == "text":
                header_parts.append(item["text"])
            elif item["type"] == "variable":
                header_parts.append(f"{{{{{item['name']}}}}}")
            elif item["type"] == "templatetag":
                header_parts.append(item["text"])

        if header_parts:
            self.shared_header = "\n".join(header_parts).strip()

        # clear buffer since this was header content
        self.current_buf = []
        return None

    def obliviate(self, items):
        self._flush_section()
        return None

    def single_completion(self, items):
        body = items[0]
        self._record_completion(body)
        return None

    def list_completion(self, items):
        repeat = items[0]
        body = items[1]
        body["repeat"] = repeat
        self._record_completion(body)
        return None

    def _record_completion(self, body):
        prompt = self._collapse_prompt_text()
        part = PromptPart(
            key=body["key"],
            return_type=body["return_type"],
            options=body.get("options", []),
            text=prompt,
            shared_header=self.shared_header,  # Store shared_header separately
            quantifier=body.get("quantifier", None),
        )
        self.current_parts.append((body["key"], part))
        self.current_buf = []  # reset prompt buffer
        # Don't auto-flush - let sections build up naturally within OBLIVIATE blocks

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

    def _lookup_rt(self, key):
        return self.action_lookup[key]

    # All other methods: passthrough or reused from your original
    def typed_completion(self, items):
        # items can be: [type, key, options] or [type, quantifier, key, options]
        type_name = str(items[0])

        # Check if second item is a quantifier (tuple) or key (string)
        if len(items) > 1 and isinstance(items[1], tuple):
            # Has quantifier: [type, quantifier, key, ...]
            quantifier = items[1]
            key = str(items[2])
            options = items[3] if len(items) > 3 else []
        else:
            # No quantifier: [type, key, ...]
            quantifier = None
            key = str(items[1]) if len(items) > 1 else "response"
            options = items[2] if len(items) > 2 else []

        return {
            "return_type": self._lookup_rt(type_name),
            "key": key,
            "options": options,
            "quantifier": quantifier,
        }

    def untyped_completion(self, items):
        return {
            "return_type": self._lookup_rt("default"),
            "key": str(items[0]),
            "options": items[1] if len(items) > 1 else [],
        }

    def option_list(self, items):
        return list(map(str, items))

    def option(self, item):
        return str(item[0])

    def var_path(self, items):
        return list(map(str, items))

    def wildcard(self, _):
        return (1, MAX_LIST_LENGTH)

    def fixed(self, items):
        return int(str(items[0]))

    def ranged(self, items):
        return (int(str(items[0])), int(str(items[1])))

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


def parser(action_lookup=ACTION_LOOKUP):
    return Lark(
        mindframe_grammar,
        parser="lalr",
        transformer=MindframeTransformer(action_lookup=action_lookup),
    )


def parse_syntax(syntax):
    """
    parse_syntax("hewllo {{cruel}} [[world]]")

    extract_all_placeholders("hewllo {{cruel}} [[world]]")
    """
    return parser().parse(syntax.strip())


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

    # check if template doesn't end with a completion
    template_stripped = template_text.strip()
    if not template_stripped.endswith("]]"):
        return f"Template must end with a completion.{context}\n\nHint: All Mindframe templates must end with a completion block like [[variable]] or [[type:variable]]."

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
