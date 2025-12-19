import re
import warnings
from collections import OrderedDict, namedtuple
from importlib.resources import files
from pathlib import Path

from lark import Lark, Transformer
from pydantic import ValidationError

from .actions import Actions
from .jinja_utils import extract_jinja_variables
from .response_types import ResponseTypes
from .return_type_models import LLMConfig
from typing import NamedTuple, Optional, Any


class OptionValue(NamedTuple):
    """
    Structured representation of parsed option values
    """
    
    key: Optional[str]
    value: Any
    is_variable_ref: bool

    def __str__(self):
        if self.key:
            return f"{self.key}={self.value}"
        return str(self.value)

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

# Regex pattern for finding [[...]] slot placeholders
SLOT_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")

# Mini-grammar for parsing just the inner content of [[...]] slots
# This is a subset of the main grammar, used for extracting slot keys
_slot_body_grammar = """
start: completion_body

completion_body: _AT CNAME _COLON CNAME (_PIPE option_list)? -> action_call_with_var
    | _AT CNAME _COLON (_PIPE option_list)? -> action_call_auto_var
    | _AT CNAME (_PIPE option_list)? -> action_call_no_var
    | BANG? CNAME quantifier? _COLON CNAME (_PIPE option_list)? -> typed_completion_with_var
    | BANG? CNAME quantifier? _COLON (_PIPE option_list)? -> typed_completion_auto_var
    | BANG? CNAME quantifier? (_PIPE option_list)? -> typed_completion_no_var

BANG: "!"
_PIPE: "|"
_AT: "@"
_COLON: ":"
_WHITESPACE: /\\s+/

quantifier: "*"                           -> zero_or_more
          | "+"                           -> one_or_more
          | "?"                           -> zero_or_one
          | "{" NUMBER "}"                -> exact
          | "{" NUMBER "," "}"            -> at_least
          | "{" NUMBER "," NUMBER "}"     -> between

option_list: option ("," option)*
option: CNAME "=" option_value -> keyed_option
      | positional_value -> positional_option

// keyed option values: unquoted = variable reference
option_value: STRING -> literal_string
            | SIGNED_FLOAT -> literal_float
            | NUMBER -> literal_number
            | CNAME -> variable_ref

// positional values: unquoted = literal string (for pick choices etc)
?positional_value: STRING -> literal_string
                 | SIGNED_FLOAT -> literal_float
                 | NUMBER -> literal_number
                 | EMOJI -> literal_emoji
                 | CNAME -> literal_cname

// Emoji terminal - covers common emoji ranges (emoticons, symbols, dingbats)
EMOJI: /[\\U0001F300-\\U0001FAFF\\U00002600-\\U000027BF]+/

%import common.CNAME
%import common.INT -> NUMBER
%import common.SIGNED_FLOAT
%import common.ESCAPED_STRING -> STRING
%ignore _WHITESPACE
"""

_cached_slot_parser = None


def _get_slot_body_parser():
    """Get cached mini-parser for slot body content."""
    global _cached_slot_parser
    if _cached_slot_parser is None:
        _cached_slot_parser = Lark(_slot_body_grammar, parser="lalr")
    return _cached_slot_parser


class SlotKeyTransformer(Transformer):
    """Extracts slot info from a slot body parse tree.

    Returns a dict with 'key', 'type', 'is_action', 'options', and 'auto' fields.
    """

    def _options_for_action(self, options):
        """Convert positional options to variable references for action calls.

        Only converts unquoted identifiers (valid variable names) to variable refs.
        Quoted strings containing spaces or special chars remain as literals.
        """
        result = []
        for opt in options:
            if opt.key is None and not opt.is_variable_ref and isinstance(opt.value, str):
                # Only convert to variable ref if it looks like a valid identifier
                # (no spaces, starts with letter/underscore, contains only alphanumeric/_)
                val = opt.value
                if val and val.replace("_", "").isalnum() and (val[0].isalpha() or val[0] == "_"):
                    result.append(OptionValue(key=None, value=val, is_variable_ref=True))
                else:
                    # Keep as literal (was a quoted string with spaces/special chars)
                    result.append(opt)
            else:
                result.append(opt)
        return result

    def action_call_with_var(self, items):
        action_name = str(items[0])
        var_name = str(items[1])
        options = self._options_for_action(items[2] if len(items) > 2 else [])
        return {
            "key": var_name,
            "type": action_name,
            "is_action": True,
            "options": options,
            "auto": False,
        }

    def action_call_auto_var(self, items):
        action_name = str(items[0])
        options = self._options_for_action(items[1] if len(items) > 1 else [])
        return {
            "key": None,
            "type": action_name,
            "is_action": True,
            "options": options,
            "auto": True,
        }

    def action_call_no_var(self, items):
        action_name = str(items[0])
        options = self._options_for_action(items[1] if len(items) > 1 else [])
        return {
            "key": action_name,
            "type": action_name,
            "is_action": True,
            "options": options,
            "auto": False,
        }

    def typed_completion_with_var(self, items):
        idx = 0
        if items and str(items[0]) == "!":
            idx = 1
        type_name = str(items[idx])
        idx += 1
        # skip quantifier if present
        if idx < len(items) and isinstance(items[idx], tuple):
            idx += 1
        var_name = str(items[idx])
        idx += 1
        options = items[idx] if idx < len(items) else []
        return {
            "key": var_name,
            "type": type_name,
            "is_action": False,
            "options": options,
            "auto": False,
        }

    def typed_completion_auto_var(self, items):
        idx = 0
        if items and str(items[0]) == "!":
            idx = 1
        type_name = str(items[idx])
        idx += 1
        # skip quantifier if present
        if idx < len(items) and isinstance(items[idx], tuple):
            idx += 1
        options = items[idx] if idx < len(items) else []
        return {
            "key": None,
            "type": type_name,
            "is_action": False,
            "options": options,
            "auto": True,
        }

    def typed_completion_no_var(self, items):
        idx = 0
        if items and str(items[0]) == "!":
            idx = 1
        type_name = str(items[idx])
        idx += 1
        # skip quantifier if present
        if idx < len(items) and isinstance(items[idx], tuple):
            idx += 1
        options = items[idx] if idx < len(items) else []
        return {
            "key": type_name,
            "type": type_name,
            "is_action": False,
            "options": options,
            "auto": False,
        }

    # quantifier handlers return tuples
    def zero_or_more(self, items):
        return (0, None)

    def one_or_more(self, items):
        return (1, None)

    def zero_or_one(self, items):
        return (0, 1)

    def exact(self, items):
        return (int(str(items[0])), int(str(items[0])))

    def at_least(self, items):
        return (int(str(items[0])), None)

    def between(self, items):
        return (int(str(items[0])), int(str(items[1])))

    def option_list(self, items):
        return list(items)

    def keyed_option(self, items):
        key = str(items[0])
        value, is_var = items[1]
        return OptionValue(key=key, value=value, is_variable_ref=is_var)

    def positional_option(self, items):
        value, is_var = items[0]
        return OptionValue(key=None, value=value, is_variable_ref=is_var)

    def literal_string(self, items):
        # strip quotes from STRING token
        s = str(items[0])
        return (s[1:-1] if s.startswith(('"', "'")) else s, False)

    def literal_float(self, items):
        return (float(str(items[0])), False)

    def literal_number(self, items):
        return (int(str(items[0])), False)

    def variable_ref(self, items):
        return (str(items[0]), True)

    def literal_cname(self, items):
        # unquoted positional value - treat as literal string, not variable ref
        return (str(items[0]), False)

    def literal_emoji(self, items):
        # emoji positional value - treat as literal string
        return (str(items[0]), False)

    def start(self, items):
        return items[0]


def parse_slot_body(inner_text: str) -> dict:
    """Parse slot inner content using Lark.

    Args:
        inner_text: Content between [[ and ]], e.g. "type:var|opts"

    Returns:
        Dict with 'key', 'type', 'is_action', 'options', 'auto' fields.
    """
    parser = _get_slot_body_parser()
    tree = parser.parse(inner_text.strip())
    return SlotKeyTransformer().transform(tree)


def extract_slot_key(inner_text: str) -> str:
    """Extract just the variable key from slot inner content.

    Args:
        inner_text: Content between [[ and ]]

    Returns:
        Variable key string, or the type name if no explicit key
    """
    result = parse_slot_body(inner_text)
    return result["key"] if result["key"] else result["type"]


def extract_option_variable_refs(options: list) -> set:
    """Extract variable reference names from a list of OptionValue.

    Args:
        options: List of OptionValue namedtuples from parsed slot

    Returns:
        Set of variable names that are referenced in the options
    """
    return {opt.value for opt in (options or []) if opt.is_variable_ref}


def extract_slot_variable_refs(text: str) -> set:
    """Extract all variable references from action parameters in template text.

    Parses each slot in the text and collects variable references from options.
    E.g., [[@evidence|query=topics,n=10]] → {"topics"}

    Args:
        text: Template string potentially containing [[@ action calls

    Returns:
        Set of variable names referenced in action parameters
    """
    variables = set()
    for match in SLOT_PATTERN.finditer(text):
        try:
            parsed = parse_slot_body(match.group(1))
            variables |= extract_option_variable_refs(parsed.get("options", []))
        except Exception:
            continue
    return variables


def find_slots_with_positions(text: str) -> list:
    """Find all [[...]] slots in text with positions.

    Args:
        text: Template text to search

    Returns:
        List of (key, start_pos, end_pos, inner_text) tuples
    """
    return [
        (extract_slot_key(m.group(1)), m.start(), m.end(), m.group(1))
        for m in SLOT_PATTERN.finditer(text)
    ]


class NamedSegment(OrderedDict):
    """OrderedDict subclass that adds optional segment name metadata.

    Maintains backward compatibility by behaving exactly like OrderedDict
    while allowing segments to optionally store a name for debugging/visualization.
    """

    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._segment_name = name

    @property
    def segment_name(self):
        """Get the segment name, or None if unnamed."""
        return self._segment_name


PromptPart = namedtuple(
    "PromptPart",
    [
        "key",
        "return_type",
        "options",
        "text",
        "system_message",  # Computed effective system message (globals + locals)
        "header_message",  # Computed effective header (globals + locals), sent as user role
        "quantifier",
        "action_type",
        "llm_kwargs",
        "required_prefix",
        "is_function",
        "has_explicit_var",
        "line_number",  # Line number in source file where completion is defined
        "block",  # If True, all subsequent segments depend on this completion's segment
        "is_break",  # If True, this is a break tag (early termination)
        "break_message",  # Optional message for break tag
        "together_group",  # If set, slots with same group ID are evaluated simultaneously
    ],
)


def get_completion_type(prompt_part: PromptPart) -> str:
    """Get completion type name for display/categorization.

    Returns the registered type name, automatically supporting new
    types as they're registered in ResponseTypes or Actions.

    Args:
        prompt_part: PromptPart to get type for

    Returns:
        Type name string (e.g., 'think', 'pick', 'respond', 'action')
    """
    # custom actions (registered via @Actions.register)
    # Note: break is now an action [[@break|msg]], not a special tag
    if prompt_part.is_function:
        return "action"

    # check if action_type is a registered ResponseType
    # (handles [[think:var]], [[pick:var]], [[custom_type:var]], etc.)
    if prompt_part.action_type and ResponseTypes.is_registered(prompt_part.action_type):
        return prompt_part.action_type

    # default type (usually 'respond')
    return "respond"


class MindframeTransformer(Transformer):
    """Lark transformer that converts parse tree into list of segments.

    Each transformer method corresponds to a grammar rule. Methods receive `items`
    containing child node results (after their transformers run). Underscore-prefixed
    terminals (e.g., _OPEN_BRACKET) are automatically filtered by Lark.

    The transformer accumulates state as it processes the tree:
    - current_buf: text/variables between completions
    - current_parts: completions in current segment
    - sections: completed segments (each is a NamedSegment of PromptParts)

    Final output: list of NamedSegment, each containing (key, PromptPart) pairs.
    """

    def __init__(self):
        self.sections = []
        self.current_buf = (
            []
        )  # text/variables/templatetags accumulated before next completion
        self.current_parts = []  # list of (key, PromptPart) for current segment

        # two-list system prompt model
        self.globals = []  # global system prompts (persist across checkpoints)
        self.locals = []  # local system prompts (cleared after each checkpoint)

        # two-list header model (same semantics, but sent as user role after system)
        self.header_globals = []  # global headers (persist across checkpoints)
        self.header_locals = []  # local headers (cleared after each checkpoint)

        self.checkpoint_counter = 0  # counter for auto-naming checkpoints

        self.completion_counters = (
            {}
        )  # track auto-variable generation per completion type
        self.action_counters = {}  # track auto-variable generation per action name
        self.pending_segment_name = None  # name to apply to next segment

        # track completions for validating local system tags
        self.completions_in_current_checkpoint = []

        # together group tracking (for parallel slot execution)
        self.together_group_stack = []  # stack of active group IDs
        self.together_group_counter = 0  # counter for generating unique group IDs

    def _flush_section(self):
        """Flush current section and clear locals"""
        if self.current_parts:
            # Use pending segment name, then clear it
            self.sections.append(
                NamedSegment(self.pending_segment_name, self.current_parts)
            )
            self.pending_segment_name = None
        self.current_buf = []
        self.current_parts = []
        self.locals = []
        self.header_locals = []
        self.completions_in_current_checkpoint = []

    def _extract_line_number(self, items):
        """Extract line number from first Token in items.

        Args:
            items: List of items from transformer, may contain Tokens

        Returns:
            Line number (int) or 0 if no Token found
        """
        for item in items:
            if hasattr(item, "line"):
                return item.line
        return 0

    # -------------------------------------------------------------------------
    # Basic content handlers - accumulate text/variables into current_buf
    # -------------------------------------------------------------------------

    def markdown_text(self, items):
        """Handle plain text content between special syntax.

        Grammar: markdown: /.../ -> markdown_text
        Items: [Token] containing the matched text
        """
        text = str(items[0]).strip()
        self.current_buf.append({"type": "text", "text": text})
        return None

    def block_comment(self, items):
        """Handle <!-- HTML comments --> - dropped from output.

        Grammar: block_comment: BLOCK_COMMENT_START BLOCK_COMMENT_CONTENT? BLOCK_COMMENT_END
        Items: [start_token, optional_content, end_token]
        """
        return None

    def placeholder(self, items):
        """Handle {{variable}} or {{variable.path}} template variables.

        Grammar: placeholder: "{{" var_path "}}"
        Items: [var_path_list] where var_path_list is ["var"] or ["var", "path"]
        """
        self.current_buf.append(
            {"type": "variable", "name": ".".join(map(str, items[0]))}
        )
        return None

    def templatetag(self, items):
        """Handle Jinja2 {% ... %} template tags (passed through unchanged).

        Grammar: templatetag: /{%.*?%}/s -> templatetag
        Items: [Token] containing the full tag like "{% if x %}"
        """
        self.current_buf.append(
            {
                "type": "templatetag",
                "text": str(items[0]),
            }
        )
        return None

    # -------------------------------------------------------------------------
    # Checkpoint handlers - flush current segment and start new one
    # -------------------------------------------------------------------------

    def checkpoint_closed(self, items):
        """Handle <checkpoint>name</checkpoint> - named checkpoint.

        Grammar: checkpoint_tag: CHECKPOINT_CLOSED -> checkpoint_closed
        Items: [Token] containing full tag like "<checkpoint>My Section</checkpoint>"
        """
        # items[0] is the full tag like "<checkpoint>Name</checkpoint>"
        tag = str(items[0]) if items else ""
        # Extract content between <checkpoint...> and </checkpoint>
        match = re.search(r"<checkpoint\s*>(.*?)</checkpoint>", tag, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Strip HTML comments from content
            content = re.sub(r"<!--(.|\n)*?-->", "", content).strip()
            # Normalize whitespace (multiline names become single line)
            name = " ".join(content.split()) if content else None
        else:
            name = None
        self._flush_checkpoint(name)
        return None

    def checkpoint_auto(self, items):
        """Handle <checkpoint> - auto-named checkpoint.

        Grammar: checkpoint_tag: CHECKPOINT_OPEN -> checkpoint_auto
        Items: [Token] containing just "<checkpoint>"
        """
        self.checkpoint_counter += 1
        name = f"Checkpoint {self.checkpoint_counter}"
        self._flush_checkpoint(name)
        return None

    def obliviate_closed(self, items):
        """Handle <obliviate>name</obliviate> - named checkpoint (alias).

        Grammar: obliviate_tag: OBLIVIATE_CLOSED -> obliviate_closed
        Items: [Token] containing full tag
        """
        tag = str(items[0]) if items else ""
        match = re.search(r"<obliviate\s*>(.*?)</obliviate>", tag, re.DOTALL)
        if match:
            content = match.group(1).strip()
            content = re.sub(r"<!--(.|\n)*?-->", "", content).strip()
            name = " ".join(content.split()) if content else None
        else:
            name = None
        self._flush_checkpoint(name)
        return None

    def obliviate_auto(self, items):
        """Handle <obliviate> - auto-named checkpoint (alias).

        Grammar: obliviate_tag: OBLIVIATE_OPEN -> obliviate_auto
        Items: [Token] containing just "<obliviate>"
        """
        self.checkpoint_counter += 1
        name = f"Checkpoint {self.checkpoint_counter}"
        self._flush_checkpoint(name)
        return None

    def _flush_checkpoint(self, name):
        """Flush current section and set up next checkpoint

        Special case: If this is the first checkpoint and there are no completions yet,
        just set the pending name without flushing (avoids creating an empty segment).
        """
        # Check for unused local system tags before flushing
        self._warn_unused_local_system_tags()

        if not self.current_parts and not self.sections:
            # First checkpoint before any completions -- just set the name (no-op case)
            self.pending_segment_name = name
        else:
            # Normal case: flush current segment, then set pending name for next segment
            self._flush_section()
            self.pending_segment_name = name

    def _warn_unused_local_system_tags(self):
        """Warn if local system tags were set but no completions followed"""
        if self.locals and not self.completions_in_current_checkpoint:
            warnings.warn(
                "Local system tag was set but no completions followed in this checkpoint. "
                "Local system prompts only apply to completions within the same checkpoint."
            )

    # -------------------------------------------------------------------------
    # System message handlers - modify globals/locals lists
    # -------------------------------------------------------------------------

    def system_tag(self, items):
        """Handle <system [modifiers]>content</system> - set system prompts.

        Grammar: system_tag: SYSTEM_OPEN SYSTEM_CONTENT? SYSTEM_CLOSE
        Items: [SYSTEM_OPEN token, optional SYSTEM_CONTENT, SYSTEM_CLOSE]

        Modifiers parsed from opening tag:
        - scope: 'local' (cleared after checkpoint) or 'global' (default, persists)
        - action: 'replace' (clear list) or 'append' (default, add to list)
        """
        scope = "global"
        action = "append"

        # First item is SYSTEM_OPEN (e.g., "<system local replace>")
        opening_tag = str(items[0]) if items else "<system>"

        # Parse modifiers from opening tag
        if "local" in opening_tag:
            scope = "local"
        elif "global" in opening_tag:
            scope = "global"

        if "replace" in opening_tag:
            action = "replace"
        elif "append" in opening_tag:
            action = "append"

        # Second item (if present and not SYSTEM_CLOSE) is content
        # Items are: [SYSTEM_OPEN, (optional SYSTEM_CONTENT), SYSTEM_CLOSE]
        content = ""
        if len(items) > 2:  # Has content between open and close
            content = str(items[1]).strip()
            # Strip HTML comments from content
            content = re.sub(r"<!--(.|\n)*?-->", "", content).strip()

        # Apply the system tag
        if action == "replace":
            if scope == "global":
                self.globals = [content] if content else []
            else:  # local
                self.locals = [content] if content else []
        else:  # append
            if scope == "global":
                if content:  # Only append non-empty content
                    self.globals.append(content)
            else:  # local
                if content:
                    self.locals.append(content)

        return None

    def header_tag(self, items):
        """Handle <header [modifiers]>content</header> - set header messages.

        Grammar: header_tag: HEADER_OPEN HEADER_CONTENT? HEADER_CLOSE
        Items: [HEADER_OPEN token, optional HEADER_CONTENT, HEADER_CLOSE]

        Same semantics as system_tag, but content is sent as user role message
        (appears after system message in the message list).

        Modifiers parsed from opening tag:
        - scope: 'local' (cleared after checkpoint) or 'global' (default, persists)
        - action: 'replace' (clear list) or 'append' (default, add to list)
        """
        scope = "global"
        action = "append"

        # First item is HEADER_OPEN (e.g., "<header local replace>")
        opening_tag = str(items[0]) if items else "<header>"

        # Parse modifiers from opening tag
        if "local" in opening_tag:
            scope = "local"
        elif "global" in opening_tag:
            scope = "global"

        if "replace" in opening_tag:
            action = "replace"
        elif "append" in opening_tag:
            action = "append"

        # Second item (if present and not HEADER_CLOSE) is content
        content = ""
        if len(items) > 2:  # Has content between open and close
            content = str(items[1]).strip()
            # Strip HTML comments from content
            content = re.sub(r"<!--(.|\n)*?-->", "", content).strip()

        # Apply the header tag
        if action == "replace":
            if scope == "global":
                self.header_globals = [content] if content else []
            else:  # local
                self.header_locals = [content] if content else []
        else:  # append
            if scope == "global":
                if content:  # Only append non-empty content
                    self.header_globals.append(content)
            else:  # local
                if content:
                    self.header_locals.append(content)

        return None

    def user_tag(self, items):
        """Handle <user>content</user> - inline user message.

        Grammar: user_tag: USER_OPEN USER_CONTENT? USER_CLOSE
        Items: [USER_OPEN token, optional USER_CONTENT, USER_CLOSE]

        Content is stored in current_buf with role marker for later processing
        by the segment processor.
        """
        content = ""
        if len(items) > 2:  # Has content between open and close
            content = str(items[1]).strip()
            # Strip HTML comments from content
            content = re.sub(r"<!--(.|\n)*?-->", "", content).strip()

        if content:
            self.current_buf.append(
                {
                    "type": "role_content",
                    "role": "user",
                    "text": content,
                }
            )
        return None

    def assistant_tag(self, items):
        """Handle <assistant>content</assistant> - inline assistant message.

        Grammar: assistant_tag: ASSISTANT_OPEN ASSISTANT_CONTENT? ASSISTANT_CLOSE
        Items: [ASSISTANT_OPEN token, optional ASSISTANT_CONTENT, ASSISTANT_CLOSE]

        Content is stored in current_buf with role marker for later processing
        by the segment processor.
        """
        content = ""
        if len(items) > 2:  # Has content between open and close
            content = str(items[1]).strip()
            # Strip HTML comments from content
            content = re.sub(r"<!--(.|\n)*?-->", "", content).strip()

        if content:
            self.current_buf.append(
                {
                    "type": "role_content",
                    "role": "assistant",
                    "text": content,
                }
            )
        return None

    def include_tag(self, items):
        """Handle <include src="path"/> - file inclusion (fallback).

        Grammar: include_tag: "<include" _WHITESPACE "src=" INCLUDE_SRC "/>"
        Items: [INCLUDE_SRC token like '"path/to/file.sd"']

        Note: Includes are normally resolved by resolve_includes() BEFORE parsing.
        This handler only runs if an include wasn't resolved (shouldn't happen).
        """
        # items[0] is the INCLUDE_SRC token like '"path/to/file.sd"'
        src_token = str(items[0]) if items else '""'
        # Remove surrounding quotes
        path = src_token.strip('"')

        # Record as text - the actual resolution happens in resolve_includes()
        # which runs BEFORE parsing. If we get here, it means the include
        # wasn't resolved (which shouldn't happen in normal flow).
        self.current_buf.append(
            {"type": "text", "text": f"<!-- UNRESOLVED INCLUDE: {path} -->"}
        )
        return None

    # -------------------------------------------------------------------------
    # Together tag handlers - parallel slot groups
    # -------------------------------------------------------------------------

    def together_open_tag(self, items):
        """Handle <together> - start a parallel group.

        Slots inside <together>...</together> are evaluated simultaneously
        and don't see each other's results (context isolation).
        """
        self.together_group_counter += 1
        group_id = f"together_{self.together_group_counter}"
        self.together_group_stack.append(group_id)
        return None

    def together_close_tag(self, items):
        """Handle </together> - end current parallel group with validation.

        Validates that no slot inside the group references another slot
        from the same group (which would be undefined at execution time).
        """
        if not self.together_group_stack:
            raise ValueError("</together> without matching <together>")

        group_id = self.together_group_stack.pop()

        # Validate: check for invalid variable references within the group
        # Get all slots in this group (from current_parts)
        group_slot_keys = {
            key for key, part in self.current_parts if part.together_group == group_id
        }

        # Check if any slot's prompt text references another slot in the same group
        for key, part in self.current_parts:
            if part.together_group == group_id:
                referenced_vars = extract_jinja_variables(part.text)
                invalid_refs = referenced_vars & group_slot_keys
                if invalid_refs:
                    raise ValueError(
                        f"Invalid reference in <together> block: {invalid_refs} "
                        f"cannot be used because slots in a <together> block "
                        f"are evaluated simultaneously and don't see each other's results."
                    )

        return None

    def _get_current_together_group(self):
        """Get current parallel group ID, or None if not in a group."""
        return self.together_group_stack[-1] if self.together_group_stack else None

    def _compute_effective_system(self):
        """Compute effective system prompt = globals + locals."""
        all_parts = self.globals + self.locals
        return "\n\n".join(all_parts) if all_parts else ""

    def _compute_effective_header(self):
        """Compute effective header = header_globals + header_locals."""
        all_parts = self.header_globals + self.header_locals
        return "\n\n".join(all_parts) if all_parts else ""

    # -------------------------------------------------------------------------
    # Completion handlers - create PromptPart and add to current segment
    # -------------------------------------------------------------------------

    def single_completion(self, items):
        """Handle [[...]] - wrapper that receives completion_body result.

        Grammar: single_completion: _OPEN_BRACKET completion_body _CLOSE_BRACKET
        Items: [body_dict] - the dict returned by a completion_body handler

        This is the entry point for all completions. The inner completion_body
        handlers (typed_completion_*, action_call_*) return a dict with completion
        details, which this method passes to _record_completion().
        """
        body = items[0] if items else None
        if body is None or not isinstance(body, dict):
            raise ValueError(f"No completion body found in items: {items}")
        is_function = body.get("is_function", False)
        self._record_completion(body, is_function=is_function)
        return None

    def _record_completion(self, body, is_function=False):
        prompt = self._collapse_prompt_text()

        # Check for "block" flag before parsing options
        # Options are now OptionValue namedtuples with (key, value, is_variable_ref)
        options_list = body.get("options", [])
        block = False
        filtered_options = []
        for opt in options_list:
            if opt.key == "block":
                # block=true or block=false
                block = str(opt.value).lower() in ("true", "1", "yes")
            elif opt.key is None and opt.value == "block":
                # positional "block" option
                block = True
            else:
                filtered_options.append(opt)

        # Parse remaining options into plain options and LLM kwargs
        plain_options, llm_kwargs = self._parse_options(filtered_options)

        part = PromptPart(
            key=body["key"],
            return_type=body["return_type"],
            options=plain_options,
            text=prompt,
            system_message=self._compute_effective_system(),
            header_message=self._compute_effective_header(),
            quantifier=body.get("quantifier", None),
            action_type=body.get("action_type"),
            llm_kwargs=llm_kwargs,
            required_prefix=body.get("required_prefix", False),
            is_function=is_function,
            has_explicit_var=body.get("has_explicit_var", True),
            line_number=body.get("line_number", 0),
            block=block,
            is_break=False,
            break_message=None,
            together_group=self._get_current_together_group(),
        )
        self.current_parts.append((body["key"], part))
        self.current_buf = []
        self.completions_in_current_checkpoint.append(body["key"])

    def _parse_options(self, options_list):
        """Parse options list into separate plain options and LLM kwargs.

        Args:
            options_list: List of OptionValue namedtuples

        Returns:
            tuple: (plain_options, kwargs_dict)
                plain_options: List of OptionValue not matching LLM params
                kwargs_dict: Dict of LLM parameter key=value pairs (temperature, model)
        """
        LLM_PARAM_KEYS = set(LLMConfig.model_fields.keys())

        plain_options = []
        kwargs_dict = {}

        for opt in options_list:
            if opt.key in LLM_PARAM_KEYS:
                try:
                    test_config = {opt.key: opt.value}
                    validated = LLMConfig.model_validate(test_config, strict=False)
                    kwargs_dict[opt.key] = getattr(validated, opt.key)
                except ValidationError as e:
                    error_msg = e.errors()[0]["msg"] if e.errors() else str(e)
                    raise ValueError(f"Invalid value for '{opt.key}': {error_msg}")
            else:
                # non-LLM options stay as-is
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
                lines.append(item["text"])
            elif item["type"] == "role_content":
                # Preserve role markers in special format for segment processor
                role = item["role"]
                text = item["text"]
                lines.append(f"<{role}>{text}</{role}>")

        prompt_text = "\n".join(lines).strip()
        return prompt_text

    def _lookup_rt(self, key, options=None, quantifier=None, required_prefix=False):
        """Lookup return type, checking Actions registry first, then ResponseTypes.

        If the registered type is a factory function, calls it with options
        to get the actual Pydantic model class.
        """
        from types import FunctionType

        action_model = Actions.create_action_model(
            key, options, quantifier, required_prefix
        )
        if action_model is not None:
            return action_model

        rt = ResponseTypes.get(key)
        if rt is not None:
            # If it's a factory function, call it with options to get the model
            if isinstance(rt, FunctionType):
                return rt(options, quantifier, required_prefix)
            return rt

        # Fall back to default
        default_rt = ResponseTypes.get("default")
        if isinstance(default_rt, FunctionType):
            return default_rt(options, quantifier, required_prefix)
        return default_rt

    def typed_completion_with_var(self, items):
        """Handle [[type:var]] - typed completion with explicit variable name.

        Grammar: BANG? CNAME quantifier? _COLON CNAME (_PIPE option_list)?
        Items: [optional BANG, type_name, optional quantifier, var_name, optional options]

        Examples: [[bool:is_valid]], [[!int:count]], [[pick{3}:choices|a,b,c]]
        Returns: dict with completion details for single_completion()
        """
        line_number = self._extract_line_number(items)

        idx = 0
        required_prefix = False

        if items and str(items[0]) == "!":
            required_prefix = True
            idx = 1

        type_name = str(items[idx])
        idx += 1

        if idx < len(items) and isinstance(items[idx], tuple):
            quantifier = items[idx]
            idx += 1
        else:
            quantifier = None

        var_name = str(items[idx])
        idx += 1

        options = items[idx] if idx < len(items) else []

        is_function = Actions.is_registered(type_name)

        # Validate that the explicit type is registered
        if not is_function and not ResponseTypes.is_registered(type_name):
            raise ValueError(
                f"Unknown type '{type_name}' in [[{type_name}:{var_name}]] at line {line_number}. "
                f"Available types: {sorted(ResponseTypes.list_registered())}"
            )

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
            "line_number": line_number,
        }

    def typed_completion_auto_var(self, items):
        """Handle [[type:]] - typed completion with auto-generated variable name.

        Grammar: BANG? CNAME quantifier? _COLON (_PIPE option_list)?
        Items: [optional BANG, type_name, optional quantifier, optional options]

        Examples: [[bool:]], [[int:]], [[pick:|a,b,c]]
        Variable name generated as _{type}_{counter:02d}, e.g., _bool_01
        """
        line_number = self._extract_line_number(items)

        idx = 0
        required_prefix = False

        if items and str(items[0]) == "!":
            required_prefix = True
            idx = 1

        type_name = str(items[idx])
        idx += 1

        if idx < len(items) and isinstance(items[idx], tuple):
            quantifier = items[idx]
            idx += 1
        else:
            quantifier = None

        options = items[idx] if idx < len(items) else []

        if type_name not in self.completion_counters:
            self.completion_counters[type_name] = 0
        self.completion_counters[type_name] += 1
        var_name = f"_{type_name}_{self.completion_counters[type_name]:02d}"

        is_function = Actions.is_registered(type_name)

        # Validate that the explicit type is registered
        if not is_function and not ResponseTypes.is_registered(type_name):
            raise ValueError(
                f"Unknown type '{type_name}' in [[{type_name}:]] at line {line_number}. "
                f"Available types: {sorted(ResponseTypes.list_registered())}"
            )

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
            "line_number": line_number,
        }

    def typed_completion_no_var(self, items):
        """Handle [[name]] - simple completion, variable name = type name.

        Grammar: BANG? CNAME quantifier? (_PIPE option_list)?
        Items: [optional BANG, name, optional quantifier, optional options]

        Examples: [[response]], [[summary]], [[!answer]]
        Raises ValueError if name matches a registered type (ambiguous).
        """

        def parse_prefix(items: list) -> tuple[bool, list]:
            """Extract optional '!' prefix, returning (required_prefix, remaining)."""
            match items:
                case ["!", *rest]:
                    return (True, rest)
                case [first, *rest] if str(first) == "!":
                    return (True, rest)
                case _:
                    return (False, list(items))

        def parse_name_and_quantifier(items: list) -> tuple[str, tuple | None, list]:
            """Extract name and optional quantifier, returning (name, quantifier, remaining)."""
            match items:
                case [name, (_, _) as quant, *rest]:
                    return (str(name), quant, rest)
                case [name, *rest]:
                    return (str(name), None, rest)
                case _:
                    return ("", None, [])

        def validate_name(name: str) -> bool:
            """Check if name conflicts with registered types. Returns is_function flag."""
            is_function = Actions.is_registered(name)
            is_registered_type = is_function or ResponseTypes.get(name) is not None
            if is_registered_type:
                raise ValueError(
                    f"Ambiguous completion [[{name}]]: '{name}' is a registered completion type. "
                    f"Please be explicit:\n"
                    f"  - For a variable named '{name}': Not recommended, choose a different name\n"
                    f"  - For type '{name}' with explicit variable: [[{name}:myvar]]\n"
                    f"  - For type '{name}' with auto-generated variable: [[{name}:]]"
                )
            return is_function

        line_number = self._extract_line_number(items)
        required_prefix, after_prefix = parse_prefix(items)
        name, quantifier, after_name = parse_name_and_quantifier(after_prefix)
        is_function = validate_name(name)
        options = after_name[0] if after_name else []

        return {
            "return_type": self._lookup_rt(name, options, quantifier, required_prefix),
            "key": name,
            "options": options,
            "quantifier": quantifier,
            "action_type": name,
            "required_prefix": required_prefix,
            "is_function": is_function,
            "has_explicit_var": False,
            "line_number": line_number,
        }

    # -------------------------------------------------------------------------
    # Action call handlers - function calls with @ prefix
    # -------------------------------------------------------------------------

    def _options_for_action(self, options):
        """Convert positional options to variable references for action calls.

        In action calls like [[@evidence|topics]], positional options should be
        variable references (looked up from context), not literal strings.
        This differs from pick/extract slots where positional options are choices.

        Only converts unquoted identifiers (valid variable names) to variable refs.
        Quoted strings containing spaces or special chars remain as literals.
        """
        result = []
        for opt in options:
            if opt.key is None and not opt.is_variable_ref and isinstance(opt.value, str):
                # Only convert to variable ref if it looks like a valid identifier
                val = opt.value
                if val and val.replace("_", "").isalnum() and (val[0].isalpha() or val[0] == "_"):
                    result.append(OptionValue(key=None, value=val, is_variable_ref=True))
                else:
                    # Keep as literal (was a quoted string with spaces/special chars)
                    result.append(opt)
            else:
                result.append(opt)
        return result

    def action_call_with_var(self, items):
        """Handle [[@action:var]] - action call with explicit variable name.

        Grammar: _AT CNAME _COLON CNAME (_PIPE option_list)?
        Items: [action_name, var_name, optional options]

        Examples: [[@uppercase:result]], [[@search:hits|query=foo]]
        """
        line_number = self._extract_line_number(items)

        action_name = str(items[0])
        var_name = str(items[1])
        options = self._options_for_action(items[2] if len(items) > 2 else [])

        if not Actions.is_registered(action_name):
            warnings.warn(
                f"Unknown action '@{action_name}' at line {line_number}. "
                f"Treating as no-op. Available actions: {Actions.list_registered() or ['(none)']}"
            )
            action_name = "noop"

        return {
            "return_type": self._lookup_rt(action_name, options, None, False),
            "key": var_name,
            "options": options,
            "quantifier": None,
            "action_type": action_name,
            "required_prefix": False,
            "is_function": True,
            "has_explicit_var": True,
            "line_number": line_number,
        }

    def action_call_auto_var(self, items):
        """Handle [[@action:]] - action call with auto-generated variable name.

        Grammar: _AT CNAME _COLON (_PIPE option_list)?
        Items: [action_name, optional options]

        Examples: [[@uppercase:]], [[@search:|query=foo]]
        Variable name generated as _{action}_{counter:02d}
        """
        line_number = self._extract_line_number(items)

        action_name = str(items[0])
        options = self._options_for_action(items[1] if len(items) > 1 else [])

        if not Actions.is_registered(action_name):
            warnings.warn(
                f"Unknown action '@{action_name}' at line {line_number}. "
                f"Treating as no-op. Available actions: {Actions.list_registered() or ['(none)']}"
            )
            action_name = "noop"

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
            "line_number": line_number,
        }

    def action_call_no_var(self, items):
        """Handle [[@action]] - action call, variable name = action name.

        Grammar: _AT CNAME (_PIPE option_list)?
        Items: [action_name, optional options]

        Examples: [[@uppercase]], [[@search|query=foo]]
        """
        line_number = self._extract_line_number(items)

        action_name = str(items[0])
        options = self._options_for_action(items[1] if len(items) > 1 else [])

        original_action_name = action_name
        if not Actions.is_registered(action_name):
            warnings.warn(
                f"Unknown action '@{action_name}' at line {line_number}. "
                f"Treating as no-op. Available actions: {Actions.list_registered() or ['(none)']}"
            )
            action_name = "noop"

        return {
            "return_type": self._lookup_rt(action_name, options, None, False),
            "key": original_action_name,  # keep original name as variable key
            "options": options,
            "quantifier": None,
            "action_type": action_name,
            "required_prefix": False,
            "is_function": True,
            "has_explicit_var": False,
            "line_number": line_number,
        }

    # -------------------------------------------------------------------------
    # Helper methods for parsing sub-elements
    # -------------------------------------------------------------------------

    def option_list(self, items):
        """Handle option_list: option ("," option)* -> list of OptionValue."""
        return list(items)

    def keyed_option(self, items):
        """Handle keyed_option: CNAME "=" option_value -> OptionValue with key."""
        key = str(items[0])
        value, is_var = items[1]
        return OptionValue(key=key, value=value, is_variable_ref=is_var)

    def positional_option(self, items):
        """Handle positional_option: option_value -> OptionValue without key."""
        value, is_var = items[0]
        return OptionValue(key=None, value=value, is_variable_ref=is_var)

    def literal_string(self, items):
        """Handle literal_string: STRING -> (value, False)."""
        s = str(items[0])
        return (s[1:-1] if s.startswith(('"', "'")) else s, False)

    def literal_float(self, items):
        """Handle literal_float: SIGNED_FLOAT -> (value, False)."""
        return (float(str(items[0])), False)

    def literal_number(self, items):
        """Handle literal_number: NUMBER -> (value, False)."""
        return (int(str(items[0])), False)

    def variable_ref(self, items):
        """Handle variable_ref: CNAME -> (varname, True)."""
        return (str(items[0]), True)

    def literal_cname(self, items):
        """Handle literal_cname: CNAME in positional context -> (value, False)."""
        return (str(items[0]), False)

    def literal_emoji(self, items):
        """Handle literal_emoji: EMOJI in positional context -> (value, False)."""
        return (str(items[0]), False)

    def var_path(self, items):
        """Handle var_path: CNAME ("." CNAME)* -> list of path components."""
        return list(map(str, items))

    # -------------------------------------------------------------------------
    # Quantifier handlers - return (min, max) tuples
    # -------------------------------------------------------------------------

    def zero_or_more(self, items):
        """Handle * quantifier -> (0, None) meaning 0 to unlimited."""
        return (0, None)

    def one_or_more(self, items):
        """Handle + quantifier -> (1, None) meaning 1 to unlimited."""
        return (1, None)

    def zero_or_one(self, items):
        """Handle ? quantifier -> (0, 1) meaning optional."""
        return (0, 1)

    def exact(self, items):
        """Handle {n} quantifier -> (n, n) meaning exactly n."""
        n = int(str(items[0]))
        return (n, n)

    def at_least(self, items):
        """Handle {n,} quantifier -> (n, None) meaning n or more."""
        n = int(str(items[0]))
        return (n, None)

    def between(self, items):
        """Handle {min,max} quantifier -> (min, max) meaning between min and max."""
        min_n = int(str(items[0]))
        max_n = int(str(items[1]))
        return (min_n, max_n)

    # -------------------------------------------------------------------------
    # Top-level handler
    # -------------------------------------------------------------------------

    def start(self, items):
        """Handle start rule - finalize parsing and return all segments.

        Grammar: start: (block_comment | checkpoint_tag | ... )*
        Items: [None, None, ...] - child handlers return None after accumulating state

        Returns: list of NamedSegment, each containing (key, PromptPart) pairs
        """
        self._warn_unused_local_system_tags()
        self._flush_section()
        return self.sections


# Module-level cached Lark parser (compiled once, reused for all parses)
_cached_lark = None


def _get_cached_lark():
    """Get or create the cached Lark parser.

    The Lark grammar compilation is expensive (~13ms). By caching the compiled
    parser at module level, subsequent parses only need ~1ms.
    """
    global _cached_lark
    if _cached_lark is None:
        _cached_lark = Lark(
            mindframe_grammar,
            parser="lalr",
            propagate_positions=True,
            # No transformer - we'll apply it separately after parsing
        )
    return _cached_lark


def parser_with_state(initial_globals=None):
    """Create parser with state tracking for system message globals.

    Args:
        initial_globals: Optional list of global system message strings to start with.
            Used when parsing checkpoint segments to maintain globals across segments.

    Returns:
        Tuple of (CachedParserWrapper, MindframeTransformer instance).
        The wrapper's parse() method applies the transformer automatically.
        The transformer can be inspected after parsing to get the final globals state.
    """
    transformer = MindframeTransformer()
    if initial_globals is not None:
        transformer.globals = list(initial_globals)  # Copy to avoid mutation

    lark = _get_cached_lark()

    # Return a wrapper that combines cached parser with fresh transformer
    class CachedParserWrapper:
        def parse(self, text):
            from lark.exceptions import VisitError

            tree = lark.parse(text)
            try:
                return transformer.transform(tree)
            except VisitError as e:
                # Unwrap VisitError to preserve original exception type
                if e.orig_exc:
                    raise e.orig_exc from None
                raise

    return CachedParserWrapper(), transformer


def parser():
    """Create parser (backward compatible wrapper)."""
    lark, _ = parser_with_state()
    return lark


def resolve_includes(
    template_text: str, base_path: Path = None, search_paths: list = None
) -> str:
    """Resolve <include src="path"/> tags at compile time.

    Recursively inlines included file contents. This happens BEFORE Jinja2 rendering,
    allowing included files to contain Jinja2 syntax that will be evaluated later.

    Args:
        template_text: Raw template text with potential <include> tags
        base_path: Base path for resolving relative includes (usually template directory)
        search_paths: Additional search paths for includes

    Returns:
        Template text with all <include> tags replaced by file contents
    """
    if search_paths is None:
        search_paths = []

    if base_path and base_path not in search_paths:
        search_paths = [base_path] + search_paths

    # Pattern to match <include src="path"/> or <include src='path'/>
    include_pattern = re.compile(r'<include\s+src=["\']([^"\']+)["\']\s*/>')

    def resolve_single_include(match):
        rel_path = match.group(1)

        # Try each search path
        for search_path in search_paths:
            full_path = search_path / rel_path
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                # Recursively resolve includes in the included file
                return resolve_includes(content, full_path.parent, search_paths)

        # Raise error if include file not found
        searched = (
            list(dict.fromkeys(str(p.resolve()) for p in search_paths))
            if search_paths
            else ["."]
        )
        raise FileNotFoundError(
            f"Include file not found: {rel_path}\nSearched in:\n  "
            + "\n  ".join(searched)
        )

    return include_pattern.sub(resolve_single_include, template_text)


def split_by_checkpoint(template_text: str) -> list:
    """Split template into segments by <checkpoint> tags.

    This is a compile-time operation that splits BEFORE Jinja2 rendering.
    Each segment will be Jinja2-rendered separately at execution time.

    Args:
        template_text: Template text with <checkpoint> tags

    Returns:
        List of (segment_text, checkpoint_name) tuples
    """
    # Pattern for checkpoint tags - self-closing, opening-only, and with content
    # <checkpoint/> or <checkpoint> or <checkpoint>Name</checkpoint>
    checkpoint_pattern = re.compile(
        r"<checkpoint\s*>([^<]*?)</checkpoint>|<checkpoint\s*/>|<checkpoint\s*>|"
        r"<obliviate\s*>([^<]*?)</obliviate>|<obliviate\s*/>|<obliviate\s*>",
        re.DOTALL,
    )

    segments = []
    last_end = 0
    checkpoint_counter = 0

    for match in checkpoint_pattern.finditer(template_text):
        # Get text before this checkpoint
        segment_text = template_text[last_end : match.start()]

        # Determine checkpoint name
        if match.group(1):  # <checkpoint>Name</checkpoint>
            name = match.group(1).strip()
        elif match.group(2):  # <obliviate>Name</obliviate>
            name = match.group(2).strip()
        else:  # Auto-named
            checkpoint_counter += 1
            name = f"Checkpoint {checkpoint_counter}"

        if segment_text.strip():
            segments.append((segment_text, name))

        last_end = match.end()

    # Add final segment after last checkpoint
    final_segment = template_text[last_end:]
    if final_segment.strip():
        segments.append((final_segment, None))

    # If no checkpoints found, return single segment
    if not segments:
        segments.append((template_text, None))

    return segments


def _add_default_completion_if_needed(template: str) -> str:
    """Add [[response]] to final segment if it doesn't end with a completion placeholder.

    This allows the final segment to omit the completion placeholder while maintaining
    backward compatibility. Non-final segments are validated to ensure they have completions.

    Args:
        template: Raw template string

    Returns:
        Template with [[response]] appended to final segment if needed

    Raises:
        ValueError: If a non-final segment is missing a completion placeholder
    """

    # Helper to check if text ends with a completion placeholder
    def ends_with_completion(text: str) -> bool:
        stripped = text.rstrip()
        if stripped.endswith("]]") and "[[" in stripped:
            return True
        return False

    # Check if we have multiple segments (using new checkpoint syntax)
    # Match any checkpoint tag: <checkpoint...>, <obliviate...>
    checkpoint_pattern = (
        r"<(?:checkpoint|obliviate)(?:\s[^>]*)?>.*?(?:</(?:checkpoint|obliviate)>|\n|$)"
    )

    if re.search(r"<(?:checkpoint|obliviate)", template):
        # Split on checkpoint tags while preserving them
        segments = re.split(
            r"(<(?:checkpoint|obliviate)(?:\s[^>]*)?>.*?(?:</(?:checkpoint|obliviate)>|(?=\n)|(?=$)))",
            template,
        )

        # Filter out empty segments and reconstruct
        non_empty_segments = []
        current_segment = ""

        for part in segments:
            if re.match(r"<(?:checkpoint|obliviate)", part):
                # This is a checkpoint tag
                if current_segment.strip():
                    non_empty_segments.append(current_segment)
                non_empty_segments.append(part)
                current_segment = ""
            else:
                current_segment += part

        if current_segment.strip():
            non_empty_segments.append(current_segment)

        # Find content segments (not checkpoint tags)
        content_segments = [
            s
            for s in non_empty_segments
            if not re.match(r"<(?:checkpoint|obliviate)", s)
        ]

        # Note: Non-final segments without completions are now allowed.
        # They can contain just <system> tags or other context setup.
        # The segment will be processed but no LLM call made.

        # Add default completion to final content segment if needed
        if content_segments and not ends_with_completion(content_segments[-1]):
            # Find the last content segment in the original list and append
            for i in range(len(non_empty_segments) - 1, -1, -1):
                if not re.match(r"<(?:checkpoint|obliviate)", non_empty_segments[i]):
                    non_empty_segments[i] = (
                        non_empty_segments[i].rstrip() + "\n\n[[response]]"
                    )
                    break

            return "".join(non_empty_segments)

        return template
    else:
        # Single segment - add completion if needed
        if not ends_with_completion(template):
            return template.rstrip() + "\n\n[[response]]"
        return template


def parse_syntax(syntax):
    """Parse struckdown syntax into sections"""
    preprocessed = _add_default_completion_if_needed(syntax)
    return parser().parse(preprocessed.strip())


def get_slot_names(syntax):
    """Return a set of all slot names defined in the struckdown syntax."""
    parsed = parse_syntax(syntax)
    return {key for segment in parsed for key in segment.keys()}


def _format_parse_error(error_str, template_text):
    """Convert technical Lark parsing errors into user-friendly messages"""
    line_match = re.search(r"line (\d+)", error_str)
    col_match = re.search(r"column (\d+)", error_str)

    line_num = int(line_match.group(1)) if line_match else None
    col_num = int(col_match.group(1)) if col_match else None

    context = ""
    if line_num:
        lines = template_text.split("\n")
        if line_num <= len(lines):
            problematic_line = lines[line_num - 1]
            context = f"\n\nLine {line_num}: {problematic_line}"
            if col_num:
                pointer = " " * (len(f"Line {line_num}: ") + col_num - 1) + "^"
                context += f"\n{pointer}"

    if "[[" in template_text and template_text.strip().endswith("]]"):
        completion_matches = list(re.finditer(r"\[\[.*?\]\]", template_text))
        if completion_matches:
            last_completion = completion_matches[-1]
            text_after_completion = template_text[last_completion.end() :].strip()
            if text_after_completion:
                return f"Content found after final completion.{context}\n\nHint: Completions must be at the very end of the template."

    if "Unexpected token" in error_str and ("}}" in error_str or "CNAME" in error_str):
        open_count = template_text.count("{{")
        close_count = template_text.count("}}")
        if open_count > close_count:
            return f"Missing closing braces '}}' in template variable.{context}"

    if "Unexpected token" in error_str and "[[" in template_text:
        return f"Syntax error in completion block [[ ]].{context}"

    if "Unexpected token" in error_str and "{%" in template_text:
        return f"Syntax error in template tag.{context}"

    return f"Template syntax error.{context}"


def extract_all_placeholders(template_text):
    return extract_placeholders(template_text) + extract_template_tags(template_text)


def extract_placeholders(template_text):
    """Extract {{placeholder}} variable names that need user input (excluding LLM-generated ones)"""
    try:
        sections = parse_syntax(template_text)

        completion_keys = set()
        for section in sections:
            for key, prompt_part in section.items():
                completion_keys.add(key)

        all_placeholders = []
        matches = re.findall(r"\{\{([^}]+)\}\}", template_text)
        for match in matches:
            if match not in all_placeholders:
                all_placeholders.append(match)

        user_input_placeholders = [
            p for p in all_placeholders if p not in completion_keys
        ]
        return user_input_placeholders

    except Exception as e:
        friendly_error = _format_parse_error(str(e), template_text)
        raise Exception(f"Template parsing failed: {friendly_error}")


def extract_template_tags(template_text):
    """Extract template tag names that need user input for simulation"""
    try:
        parse_syntax(template_text)
        matches = re.findall(r"\{%\s*(\w+)(?:\s.*?)?\s*%\}", template_text)
        m = [m for m in matches]
        if len(set(m)) < len(m):
            m = [f"{j}_{i}" for i, j in enumerate(m)]
        return m

    except Exception as e:
        friendly_error = _format_parse_error(str(e), template_text)
        raise Exception(f"Template parsing failed: {friendly_error}")


def replace_template_tags_with_placeholders(template_text):
    """Replace template tags with user-provided content for simulation"""
    tag_replacements = extract_template_tags(template_text)
    try:
        parser().parse(template_text.strip())

        def replace_tag(match):
            match.group(0)
            tag_name = match.group(1)

            if tag_name in tag_replacements:
                return tag_replacements[tag_name]
            else:
                return f"[Template tag: {tag_name} - no simulation provided]"

        result = re.sub(r"\{%\s*(\w+)(?:\s[^%]*)?\s*%\}", replace_tag, template_text)
        return result

    except Exception as e:
        raise e


def serialise_sections(sections):
    """Convert parsed output into JSON-serialisable structure"""
    serialised = []
    for section in sections:
        section_list = []
        for key in section:
            part = section[key]
            section_list.append(
                {
                    "key": part.key,
                    "return_type": (
                        part.return_type.__name__ if part.return_type else None
                    ),
                    "options": part.options,
                    "text": part.text,
                }
            )
        serialised.append(section_list)
    return serialised
