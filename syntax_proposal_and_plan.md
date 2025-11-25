# Struckdown XML-Style Syntax Proposal & Implementation Plan

**Status**: ✅ Approved - Ready for implementation
**Date**: 2025-11-23 (Updated with user decisions)
**Breaking Changes**: Yes - New major version required (v2.0.0)

---

## Executive Summary

Migrate Struckdown from custom `¡KEYWORD` syntax to XML/HTML-style tags for better editor support, familiarity, and tooling. Major changes:

- `¡OBLIVIATE` / `¡SEGMENT` → `<checkpoint>` (or `<obliviate>` as synonym)
- `¡SYSTEM` / `¡SYSTEM+` → `<system>` with modifiers (global/local scoping)
- `¡HEADER` / `¡HEADER+` → **REMOVED** (no replacement)
- Keep `[[...]]` completions and `{{...}}` variables unchanged
- New two-list system prompt model: globals[] + locals[]
- **Clean break**: All old `¡` syntax removed from parser

---

## 1. Proposed Syntax

### 1.1 Checkpoint Tags

Replace segment delimiters with checkpoint tags (three styles):

```xml
<!-- Style 1: Auto-named (alone on line) -->
<checkpoint>
<!-- Creates checkpoint with auto-generated name: checkpoint_1, checkpoint_2, etc. -->

<!-- Style 2: Line-based naming (name on same line) -->
<checkpoint> Analysis Phase
<!-- Creates checkpoint named "Analysis Phase" -->

<!-- Style 3: Closed tag (multiline names) -->
<checkpoint>
Long checkpoint name
that spans multiple lines
</checkpoint>

<!-- Synonym: obliviate (works with all three styles) -->
<obliviate>
<obliviate> Section Name
<obliviate>Name</obliviate>
```

**Semantic meaning**: Creates a memory boundary. Everything before the checkpoint is "forgotten" except for extracted variables. Variables carry forward, but the prompt text/context does not.

**Auto-naming**: Style 1 generates sequential names: `Checkpoint 1`, `Checkpoint 2`, etc.

**HTML comments**: All `<!-- comments -->` are stripped from the final LLM prompt (not sent to model).

### 1.2 System Tags (Two-List Model)

System prompts use a **two-list model**:
- **globals[]** - Persists across all checkpoints (until replaced)
- **locals[]** - Cleared after each checkpoint

**Default**: No system prompt (both lists start empty)

**Tag syntax**: `<system [scope] [action]>content</system>`
- **scope**: `local` or `global` (default: `global`)
- **action**: `append` or `replace` (default: `append`)

**All combinations**:
```xml
<!-- Shorthand: append global (most common) -->
<system>
You are a helpful assistant
</system>

<!-- Explicit: append global (same as above) -->
<system global append>
You are a helpful assistant
</system>

<!-- Append local: adds to current checkpoint only -->
<system local>
For this checkpoint, act as a critic
</system>

<!-- Explicit: append local (same as above) -->
<system local append>
Additional context for this checkpoint
</system>

<!-- Replace global: wipes globals[], sets new content -->
<system global replace>
You are now a completely different persona
</system>

<!-- Replace local: wipes locals[], sets new content -->
<system local replace>
Ignore prior local instructions, use this instead
</system>
```

**Breakdown by frequency**:

| Tag | Meaning | Use Case |
|-----|---------|----------|
| `<system>` | Append global | Set initial system prompt or add global instruction |
| `<system local>` | Append local | Add instruction for current checkpoint only |
| `<system global>` | Append global | Explicit global (same as `<system>`) |
| `<system global replace>` | Replace global | Change persona/role entirely |
| `<system local replace>` | Replace local | Override local instructions for checkpoint |
| `<system global append>` | Append global | Explicit append (same as `<system>`) |
| `<system local append>` | Append local | Explicit local append (same as `<system local>`) |

**Key behaviors**:
- Default is **no system prompt** (empty globals and locals)
- `<system>` appends to globals[] (not replace)
- locals[] cleared after each checkpoint
- Effective prompt = `"\n\n".join(globals) + "\n\n" + "\n\n".join(locals)`
- HTML comments inside system tags are stripped before processing

### 1.3 System Tag Ordering

When multiple system tags appear before the first completion:

```xml
<system>First</system>              <!-- globals = ["First"] -->
<system local>Second</system>       <!-- locals = ["Second"] -->
<system>Third</system>              <!-- globals = ["First", "Third"] -->

Question: [[q]]
<!-- q sees: "First\n\nThird\n\nSecond" -->
<!-- Order: all globals joined, then all locals joined -->
```

**Processing order**:
1. All tags processed in document order
2. globals[] and locals[] accumulate separately
3. Effective prompt = `globals[0] + "\n\n" + globals[1] + ... + "\n\n" + locals[0] + "\n\n" + locals[1] + ...`

### 1.4 Complete Example

```xml
<!-- HTML comments are stripped from LLM prompt -->

<system>
<!-- Set initial system prompt -->
You are a kind and helpful therapist.
</system>
<!-- globals = ["You are a kind and helpful therapist."], locals = [] -->

Tell me a joke [[joke]]

<!-- Auto-named checkpoint (checkpoint_1) -->
<checkpoint>

<!-- Context wiped, locals[] cleared, only {{joke}} variable available -->
<!-- globals = ["You are a kind..."], locals = [] -->

<system local>
<!-- Add local instruction for this checkpoint only -->
You never tell jokes, only riddles.
</system>
<!-- globals = ["You are a kind..."], locals = ["You never tell jokes..."] -->

Tell me a joke [[notajoke]]
<!-- notajoke sees: "You are a kind and helpful therapist.\n\nYou never tell jokes, only riddles." -->

<!-- Named checkpoint (line-based) -->
<checkpoint> Truth vs Fiction

<!-- locals[] cleared -->
<!-- globals = ["You are a kind..."], locals = [] -->

<system global replace>
<!-- Replace global: change persona entirely -->
You never tell the truth - it's a fun game.
</system>
<!-- globals = ["You never tell the truth..."], locals = [] -->

What is the height of Mt. Everest? [[height]]
<!-- height sees: "You never tell the truth - it's a fun game." -->

<system local>
<!-- Add local exception for this question -->
Except if the question is about a mountain - then tell the truth.
</system>
<!-- globals = ["You never tell the truth..."], locals = ["Except if..."] -->

What is the height of Mt. Everest? [[height_real]]
<!-- height_real sees: "You never tell the truth - it's a fun game.\n\nExcept if the question is about a mountain - then tell the truth." -->

<!-- Named checkpoint (closed tag, multiline) -->
<checkpoint>
Final Summary
Section
</checkpoint>

<!-- locals[] cleared again -->
<!-- globals = ["You never tell the truth..."], locals = [] -->

Summarize our conversation: [[summary]]
<!-- summary sees: "You never tell the truth - it's a fun game." -->
```

---

## 2. ✅ Resolved Design Decisions

All critical design questions have been resolved by the user. This section documents the final decisions.

### 2.1 Header Blocks → REMOVED

**Decision**: Headers are **removed entirely**. No replacement.

**Rationale**: Simplifies the system. System prompts can handle what headers were used for.

**Impact**:
- Remove all header-related code from parsing.py
- Remove header tests from test_system_messages.py
- Update docs to not mention headers
- No migration path - users must manually integrate header content into system prompts

---

### 2.2 System Prompt Semantics → Two-List Model

**Decision**: Two separate lists that combine at render time:
- **globals[]** - Persists across checkpoints, wiped only by `<system global replace>`
- **locals[]** - Cleared after each checkpoint

**Default**: Both lists start empty (no system prompt by default)

**Tag syntax**: `<system [scope] [action]>content</system>`
- **scope**: `local` or `global` (default: `global`)
- **action**: `append` or `replace` (default: `append`)

**Common operations**:
- `<system>` = `<system global append>` (append to globals)
- `<system local>` = `<system local append>` (append to locals)
- `<system global>` = `<system global append>` (explicit append to globals)
- `<system global replace>` - Wipes globals[], sets new content
- `<system local replace>` - Wipes locals[], sets new content

**Effective prompt**: `"\n\n".join(globals) + "\n\n" + "\n\n".join(locals)`

**HTML comments**: Stripped from system tag content before processing

---

### 2.3 Multiple System Tags → Process in Order

**Decision**: Tags processed in document order, accumulate into respective lists.

**Example**:
```xml
<system>First</system>                  <!-- globals = ["First"], locals = [] -->
<system append local>Second</system>    <!-- globals = ["First"], locals = ["Second"] -->
<system append global>Third</system>    <!-- globals = ["First", "Third"], locals = ["Second"] -->

Question: [[q]]
<!-- q sees: "First\n\nThird\n\nSecond" -->
```

**Conflict handling**: Error if both `append` and `replace` modifiers specified together (e.g., `<system append replace>`).

---

### 2.4 System Tag Placement → Anywhere (applies to subsequent completions)

**Decision**: System tags can appear anywhere in a checkpoint. They apply to all **subsequent** completions in the same checkpoint.

```xml
<checkpoint />

<system>You are helpful</system>
Question 1: [[q1]]  <!-- q1 sees "You are helpful" -->

<system append local>Additional context</system>
Question 2: [[q2]]  <!-- q2 sees "You are helpful\n\nAdditional context" -->

<checkpoint />
<!-- q2's local context cleared here -->
```

**Validation rule**: If a local system tag (`<system append local>` or `<system replace local>`) appears after a completion AND there are no more completions in the same checkpoint, raise an error (unused local system prompt).

```xml
Question: [[q]]
<system append local>This won't be used</system>  <!-- ❌ ERROR: No subsequent completions -->
<checkpoint />
```

---

### 2.5 Empty Checkpoints → Name First Segment

**Decision**: First checkpoint before any completions just names the first segment (doesn't create empty segment).

```xml
<checkpoint>intro</checkpoint>  <!-- Names the first segment -->
<system>You are helpful</system>
Question: [[q]]
<!-- Segment is named "intro" -->
```

---

### 2.6 Checkpoint Syntax → Three Styles

**Decision**: Support three checkpoint syntaxes for flexibility:

**Style 1 - Auto-named** (alone on line):
```xml
<checkpoint>
```
Auto-generates: `checkpoint_1`, `checkpoint_2`, etc.

**Style 2 - Line-based naming** (name on same line):
```xml
<checkpoint> Section Name
```
Creates checkpoint named "Section Name"

**Style 3 - Closed tag** (multiline names):
```xml
<checkpoint>
Long name
spanning lines
</checkpoint>
```
Creates checkpoint with full multiline name (whitespace normalized)

**Synonym**: `<obliviate>` works with all three styles

**Grammar precedence**: Tries closed tag first, then line-based, then auto-named

---

### 2.7 Empty System Tag → Append Empty String

**Decision**: `<system></system>` = `<system global append>` with empty content (appends nothing, effectively a no-op).

To wipe system prompt, use: `<system global replace></system>`

---

### 2.8 Unnamed Checkpoints → Auto-generate Names

**Decision**: Auto-generate sequential names using format: `Checkpoint N`, where N is the ordinal position of that checkpoint in the document (counting ALL checkpoints, both named and unnamed).

**Example**:
```xml
<checkpoint>Introduction</checkpoint>  <!-- Named checkpoint (position 1) -->
<checkpoint>                           <!-- Auto-named: "Checkpoint 2" (position 2) -->
<checkpoint>Methodology</checkpoint>   <!-- Named checkpoint (position 3) -->
<checkpoint>                           <!-- Auto-named: "Checkpoint 4" (position 4) -->
```

**Rationale**: Ordinal position provides clear sequential tracking. Numbering all checkpoints (not just unnamed ones) is simpler and more predictable.

---

### 2.9 Migration of Old Syntax → Clean Break

**Decision**: Remove ALL old `¡` syntax from parser entirely:
- `¡SYSTEM ... /END` → removed
- `¡HEADER ... /END` → removed
- `¡OBLIVIATE` → removed
- `¡SEGMENT` → removed

**Rationale**: This is a new library. Clean break allows simpler implementation without legacy baggage. No migration path needed.

---

### 2.10 Break Tags → Early Termination Signal

**NEW FEATURE**: Add `<break>` tags as replacement for `[[!end]]` completion.

**Syntax**:
```xml
<!-- Break without message -->
<break />
<break/>

<!-- Break with message -->
<break>
optional message explaining why we're breaking
</break>
```

**Behavior**:
- Acts as early termination signal (like `[[!end]]` in current syntax)
- Optional message captured and added to returned context
- Message stored as: `{'break_message': 'optional message'}`
- If no message provided, `break_message` is `None` or empty string

**Example**:
```xml
<system>You are helpful</system>

Analyze this text: {{input_text}}

Is it spam? [[bool:is_spam]]

<break>Detected spam, skipping further analysis</break>

<!-- Everything after <break> is not processed -->
Rate quality: [[number:quality]]
```

If `is_spam` is true and break is hit, the result includes:
```python
{
    'is_spam': True,
    'break_message': 'Detected spam, skipping further analysis'
    # 'quality' is not present - never reached
}
```

**Break behavior - CONFIRMED**:
- **Unconditional**: Always processed when reached (not conditional or LLM-generated)
- **Placement**: Can appear anywhere (enforcement not required, but logically should be after at least one completion)
- **Context preservation**: Does NOT wipe context (unlike checkpoint) - just terminates early and returns accumulated results

**Implementation Impact**:
- Add to grammar: `break_tag` similar to `checkpoint_tag`
- Add to transformer: detect breaks and signal early termination
- Execution layer must handle break signal and stop processing
- Return `break_message` in final context dictionary

---

## 3. ✅ All Design Questions Resolved

All design decisions have been finalized. See Section 2 for complete details.

**Key confirmations**:
1. ✅ System tags can appear anywhere - apply to subsequent completions
2. ✅ Auto-checkpoint naming: `checkpoint_1`, `checkpoint_2`, ...
3. ✅ Break tags are unconditional (always processed when reached)
4. ✅ Break tags can appear anywhere (no enforcement needed)
5. ✅ Break preserves context (doesn't wipe like checkpoint)

---

## 4. Architecture: Single-Stage Lark Parsing

### 4.1 Parsing Strategy

**Decision**: Use **single-stage Lark grammar** (same approach as current implementation).

**Why single-stage works**:
1. ✅ Clear delimiters - each syntax has distinct markers (`<checkpoint>`, `[[...]]`, `{{...}}`)
2. ✅ Rule precedence - Lark tries rules in order, structure tags matched before markdown
3. ✅ Negative lookahead - markdown terminal excludes our syntax, allows `<div>` etc.
4. ✅ Simpler architecture - one parser, one transformer, one pass
5. ✅ Proven approach - existing grammar already works this way

**Single grammar handles everything**:
- Structure tags: `<checkpoint>`, `<obliviate>`, `<system>`, `<break>`, `¡OBLIVIATE`
- Completions: `[[type:var|opts]]`
- Variables: `{{variable}}`
- Template tags: `{% include %}`
- Markdown: everything else (via negative lookahead)

### 4.2 Why Single-Stage Lark?

**Advantages**:
1. ✅ Single pass - faster, simpler
2. ✅ Declarative, maintainable grammar
3. ✅ Built-in line number tracking (`propagate_positions=True`)
4. ✅ HTML-safe - `<div>` in markdown is ignored via negative lookahead
5. ✅ Good error messages from Lark
6. ✅ Matches existing architecture (easier migration)

**Trade-offs**:
- Users must escape/backtick our tags when discussing them: `` `<system>` `` or `&lt;system&gt;`
- **Acceptable**: Technical tool, reasonable requirement
- Keeps Jinja2 `{% include %}` for file includes (no XML `<include>` tag)
- **Rationale**: Jinja2 includes already work perfectly, provide power features (conditionals, dynamic paths)

### 4.3 Updated Grammar (Single-Stage)

```lark
// grammar.lark - Single-stage parsing of all syntax

?start: (block_comment | checkpoint_tag | obliviate_tag | system_tag | break_tag | single_completion | placeholder | templatetag | markdown)* completion?

// HTML comments (dropped from LLM prompt)
block_comment: BLOCK_COMMENT_START BLOCK_COMMENT_CONTENT? BLOCK_COMMENT_END
BLOCK_COMMENT_CONTENT: /(.|\\n)+?(?=-->)/
BLOCK_COMMENT_START: "<!--"
BLOCK_COMMENT_END: "-->"

// Three checkpoint styles (precedence order matters!)
checkpoint_tag: "<checkpoint" WS? ">" CHECKPOINT_CONTENT "</checkpoint>"  -> checkpoint_closed
              | "<checkpoint>" LINE_CONTENT NEWLINE                       -> checkpoint_line
              | "<checkpoint>" WS? NEWLINE                                -> checkpoint_auto

// Obliviate synonym (same three styles)
obliviate_tag: "<obliviate" WS? ">" OBLIVIATE_CONTENT "</obliviate>"  -> obliviate_closed
             | "<obliviate>" LINE_CONTENT NEWLINE                      -> obliviate_line
             | "<obliviate>" WS? NEWLINE                               -> obliviate_auto

// System tags: <system [scope] [action]>content</system>
system_tag: "<system" system_mods? ">" SYSTEM_CONTENT "</system>"

system_mods: WS+ system_scope (WS+ system_action)?
           | WS+ system_action (WS+ system_scope)?
           | WS+ system_scope
           | WS+ system_action

system_scope: "local" | "global"
system_action: "append" | "replace"

// Break tags
break_tag: "<break" WS? ">" BREAK_CONTENT "</break>"
         | "<break" WS? "/>"

// Checkpoint/system/break content terminals
CHECKPOINT_CONTENT: /(.|\n)*?(?=<\/checkpoint>)/
OBLIVIATE_CONTENT: /(.|\n)*?(?=<\/obliviate>)/
LINE_CONTENT: /[^\n]+/  // Captures rest of line
SYSTEM_CONTENT: /(.|\n)*?(?=<\/system>)/
BREAK_CONTENT: /(.|\n)*?(?=<\/break>)/

// Markdown: everything else (with negative lookahead to exclude our syntax)
// Matches anything that's NOT our special tags, completions, or variables
markdown: /(?:(?!<(?:checkpoint|obliviate|system|break|!--)|{{|\[\[|{%)(?:.|\n))+/s  -> markdown_text

// Existing completion/placeholder/templatetag rules unchanged
placeholder: "{{" WS? var_path WS? "}}"
var_path: CNAME ("." CNAME)*

templatetag: /{%\s.*?%}/s  -> templatetag

completion: single_completion
single_completion: "[[" WS? completion_body WS? "]]"

completion_body: ACTION_PREFIX CNAME ":" CNAME ("|" option_list)? -> action_call_with_var
    | ACTION_PREFIX CNAME ":" ("|" option_list)? -> action_call_auto_var
    | ACTION_PREFIX CNAME ("|" option_list)? -> action_call_no_var
    | BANG? CNAME quantifier? ":" CNAME ("|" option_list)? -> typed_completion_with_var
    | BANG? CNAME quantifier? ":" ("|" option_list)? -> typed_completion_auto_var
    | BANG? CNAME quantifier? ("|" option_list)? -> typed_completion_no_var

ACTION_PREFIX: "@"
BANG: "!"

quantifier: "*"                           -> zero_or_more
          | "+"                           -> one_or_more
          | "?"                           -> zero_or_one
          | "{" NUMBER "}"                -> exact
          | "{" NUMBER "," "}"            -> at_least
          | "{" NUMBER "," NUMBER "}"     -> between

option_list: option ("," option)*
option: STRING | /[^,\]\|]+/

%import common.CNAME
%import common.INT -> NUMBER
%import common.ESCAPED_STRING -> STRING
%import common.WS
%import common.NEWLINE
%ignore WS
```

**Key feature**: `markdown` terminal uses negative lookahead to ensure:
- `<div>`, `<span>`, etc. → matched as markdown ✅
- `<checkpoint>`, `<system>`, etc. → matched as structure tags ✅
- `[[...]]`, `{{...}}` → matched as completions/variables ✅

### 4.4 Implementation Flow

**Complete processing pipeline:**

```
1. Jinja2 pre-processing (existing - unchanged)
   - Process {% include %} directives
   - Expand {{template_variables}} from Python context
   ↓
2. Single-stage Lark parsing (simplified from existing)
   - Parse all syntax in one pass: structure tags + completions + variables + markdown
   - Extract system prompt modifiers
   - Build sections with PromptParts
   ↓
3. Execution (unchanged)
   - Execute completions with LLM
```

```python
from lark import Lark, Transformer

# Single parser for all syntax
parser = Lark(
    grammar,
    start='start',
    propagate_positions=True  # Track line numbers
)

def parse_syntax(template):
    """Single-pass Lark parsing

    Note: Jinja2 preprocessing ({% include %}, {{vars}})
    happens BEFORE this function is called.
    """
    # Parse all syntax in one pass
    tree = parser.parse(template)

    # Transform to sections
    transformer = MindframeTransformer()
    sections = transformer.transform(tree)

    return sections

class StructureTransformer(Transformer):
    def __init__(self, content_parser):
        self.content_parser = content_parser
        self.sections = []
        self.globals = []        # System prompt globals
        self.locals = []         # System prompt locals
        self.checkpoint_counter = 0

    def checkpoint_named(self, items):
        name = str(items[0]).strip()
        self._flush_checkpoint(name)

    def checkpoint_unnamed(self, items):
        self.checkpoint_counter += 1
        name = f"checkpoint_{self.checkpoint_counter}"
        self._flush_checkpoint(name)

    def checkpoint_malformed(self, items):
        raise ValueError(
            "Malformed checkpoint tag. Use <checkpoint /> or "
            "<checkpoint>name</checkpoint>"
        )

    def system_tag(self, items):
        # Parse modifiers and content
        modifiers = self._parse_system_modifiers(items)
        content = str(items[-1])  # SYSTEM_CONTENT terminal

        if modifiers['replace']:
            if modifiers['global']:
                self.globals = [content]
            else:  # local
                self.locals = [content]
        else:  # append
            if modifiers['global']:
                self.globals.append(content)
            else:  # local
                self.locals.append(content)

    def text_block(self, items):
        # Parse content with content_parser
        text = str(items[0])
        content_tree = self.content_parser.parse(text)
        # Extract completions, variables from content_tree
        self._process_content(content_tree)

    def _flush_checkpoint(self, name):
        # Flush current section
        # Clear locals
        self.locals = []
        # Set pending segment name
        ...
```

### 4.5 Transformer Implementation (Pseudo-code)

```python
class MindframeTransformer(Transformer):
    """Single transformer for all syntax (extends existing)"""

    def __init__(self):
        self.sections = []
        self.current_parts = []
        self.current_buf = []

        # System prompt state (NEW - two-list model)
        self.globals = []      # Global system prompts
        self.locals = []       # Local system prompts
        self.checkpoint_counter = 0

    # NEW: Structure tag handlers
    def block_comment(self, items):
        """Drop HTML comments"""
        return None

    def checkpoint_closed(self, items):
        """Handle <checkpoint>name</checkpoint>"""
        content = str(items[0]).strip()
        content = re.sub(r'<!--(.|\n)*?-->', '', content)  # Strip comments
        name = ' '.join(content.split())
        self._flush_checkpoint(name)

    def checkpoint_line(self, items):
        """Handle <checkpoint> Name"""
        name = str(items[0]).strip()
        self._flush_checkpoint(name)

    def checkpoint_auto(self, items):
        """Handle <checkpoint> (auto-named)"""
        self.checkpoint_counter += 1
        name = f"Checkpoint {self.checkpoint_counter}"
        self._flush_checkpoint(name)

    def obliviate_auto(self, items):
        """Handle <obliviate> (synonym for checkpoint)"""
        return self.checkpoint_auto(items)

    def system_tag(self, items):
        """Handle <system [scope] [action]>content</system>"""
        scope, action = 'global', 'append'  # Defaults

        # Parse modifiers
        for item in items[:-1]:
            if hasattr(item, 'data'):
                if item.data == 'system_scope':
                    scope = str(item.children[0])
                elif item.data == 'system_action':
                    action = str(item.children[0])

        content = str(items[-1]).strip()
        content = re.sub(r'<!--(.|\n)*?-->', '', content).strip()

        # Apply
        if action == 'replace':
            (self.globals if scope == 'global' else self.locals) = [content]
        else:  # append
            (self.globals if scope == 'global' else self.locals).append(content)

    # EXISTING: Completion/variable handlers (update system prompt)
    def typed_completion_with_var(self, items):
        """Handle [[type:var]] - updated to use two-list system"""
        # ... build PromptPart as before
        part.system_message = self._compute_effective_system()  # NEW
        self.current_parts.append((key, part))

    def _compute_effective_system(self):
        """NEW: Compute effective system = globals + locals"""
        all_parts = self.globals + self.locals
        return "\n\n".join(all_parts) if all_parts else ""

    def _flush_checkpoint(self, name):
        """NEW: Flush section, clear locals"""
        if self.current_parts:
            self.sections.append(NamedSegment(name, self.current_parts))
            self.current_parts = []
        self.locals = []  # Clear locals only
```

### 4.6 Error Handling Examples

**User writes malformed tag**:
```xml
<checkpoint>  <!-- Missing closing -->
```
→ Lark error: "Expected </checkpoint> or />"

**User discusses tags**:
```xml
Use <system> tags to configure...
```
→ Lark error: "Expected closing </system>"
→ **Solution**: Use backticks `` `<system>` `` or escape `&lt;system&gt;`

**HTML in markdown**:
```xml
<system>You are helpful</system>
Explain <div> tags: [[answer]]
```
→ ✅ Works! `<div>` matched as markdown

---

## 5. Implementation Impact Analysis

### 5.1 Core Files Affected

#### **grammar.lark** (MEDIUM-HIGH IMPACT - Replace old rules with new)

**Changes needed**:
1. ✅ Remove old rules: `obliviate`, `segment_delimiter`, `system_block`, `system_append_block`, `header_block`, `header_append_block`
2. ✅ Remove old terminals: `OBLIVIATE_LINE`, `SEGMENT_LINE`, `SYSTEM_CONTENT`, `HEADER_CONTENT`
3. ✅ Add new rules: `checkpoint_tag`, `obliviate_tag`, `system_tag`, `break_tag`, `block_comment`
4. ✅ Add new terminals: `CHECKPOINT_CONTENT`, `OBLIVIATE_CONTENT`, `LINE_CONTENT`, `SYSTEM_CONTENT`, `BREAK_CONTENT`
5. ✅ Update `start` rule to include new tags
6. ✅ Update `markdown` terminal to exclude new syntax via negative lookahead
7. ✅ Keep: `placeholder`, `completion`, `templatetag` (unchanged)

**Result**: Single grammar handling all syntax in one pass (similar to existing architecture).

---

#### **parsing.py** (HIGH IMPACT - Extend existing transformer)

**Extend MindframeTransformer** (single transformer approach):

**Add new methods to existing MindframeTransformer**:

```python
class MindframeTransformer(Transformer):
    """Extends existing transformer with new tag handlers"""

    def __init__(self):
        super().__init__()
        # Existing state
        self.sections = []
        self.current_parts = []
        self.current_buf = []

        # NEW: System prompt state (two-list model)
        self.globals = []      # Global system prompts
        self.locals = []       # Local system prompts
        self.checkpoint_counter = 0

    # NEW: Structure tag handlers
    def block_comment(self, items):
        """Drop HTML comments from output"""
        return None

    def checkpoint_closed(self, items):
        """Handle <checkpoint>name</checkpoint>"""
        content = str(items[0]).strip()
        content = re.sub(r'<!--(.|\n)*?-->', '', content)
        name = ' '.join(content.split())
        self._flush_checkpoint(name)

    def checkpoint_line(self, items):
        """Handle <checkpoint> Name"""
        name = str(items[0]).strip()
        self._flush_checkpoint(name)

    def checkpoint_auto(self, items):
        """Handle <checkpoint> (auto-named)"""
        self.checkpoint_counter += 1
        name = f"Checkpoint {self.checkpoint_counter}"
        self._flush_checkpoint(name)

    def obliviate_auto(self, items):
        """Handle <obliviate> (synonym for checkpoint)"""
        return self.checkpoint_auto(items)

    def system_tag(self, items):
        """Handle <system [scope] [action]>content</system>"""
        scope, action = 'global', 'append'  # Defaults

        # Parse modifiers
        for item in items[:-1]:
            if hasattr(item, 'data'):
                if item.data == 'system_scope':
                    scope = str(item.children[0])
                elif item.data == 'system_action':
                    action = str(item.children[0])

        content = str(items[-1]).strip()
        content = re.sub(r'<!--(.|\n)*?-->', '', content).strip()

        # Apply
        if action == 'replace':
            (self.globals if scope == 'global' else self.locals) = [content]
        else:  # append
            (self.globals if scope == 'global' else self.locals).append(content)

    def break_tag(self, items):
        """Handle <break> or <break>message</break>"""
        message = str(items[0]).strip() if items else None
        self._signal_break(message)

    # UPDATED: Existing completion handlers compute system prompt
    def typed_completion_with_var(self, items):
        """Handle [[type:var]] - updated to use two-list system"""
        # ... existing logic to build PromptPart
        part.system_message = self._compute_effective_system()  # NEW
        self.current_parts.append((key, part))

    # NEW: Helper methods
    def _compute_effective_system(self):
        """Compute effective system = globals + locals"""
        all_parts = self.globals + self.locals
        return "\n\n".join(all_parts) if all_parts else ""

    def _flush_checkpoint(self, name):
        """Flush section, clear locals"""
        if self.current_parts:
            self.sections.append(NamedSegment(name, self.current_parts))
            self.current_parts = []
        self.locals = []  # Clear locals only

    def _signal_break(self, message):
        """Signal early termination"""
        self.break_message = message
        # ... signal termination
```

**Key changes**:
- ✅ Remove ALL old `¡` syntax handling methods (`obliviate`, `system_block`, `header_block`)
- ✅ Remove `current_system`, `current_header` state (replaced by globals/locals)
- ✅ Add new XML-style tag handler methods (checkpoint_*, obliviate_*, system_tag, break_tag, block_comment)
- ✅ Update completion methods to call `_compute_effective_system()`
- ✅ Single-pass architecture (extends existing pattern)

**Estimated effort**: ~300 lines refactored, ~150 lines new

---

#### **__init__.py** (LOW IMPACT)

**Changes required**:
- Usage of `system_message` field remains unchanged (computed at parse time)
- Remove `header_content` field from PromptPart (headers removed)
- No changes needed to execution/chatter logic

**Impact**: Minimal - downstream code already uses computed `system_message` string.

---

#### **visualize.py** (MEDIUM IMPACT)

**Changes required**:
- Update graph generation to show new syntax in node labels
- Update section name display to use checkpoint names
- May need to visualize system prompt scoping (base vs global vs local)

**Risks**:
- Visualizations may be less clear without updates
- Graph complexity may increase with scoping info

---

### 3.2 Preprocessing & Helper Functions

#### **_add_default_completion_if_needed()** (HIGH IMPACT)

**Current logic**:
```python
SEGMENT_DELIMITER = "¡OBLIVIATE"
if SEGMENT_DELIMITER in template:
    segments = template.split(SEGMENT_DELIMITER)
    # validate non-final segments have completions
```

**New logic needed**:
```python
# Can't simply split on "<checkpoint" because:
# 1. Multiple checkpoint syntaxes (<checkpoint />, <checkpoint>name</checkpoint>)
# 2. Need to preserve checkpoint names
# 3. Need more robust parsing

# Option A: Use regex to find checkpoints
import re
checkpoint_pattern = r'<checkpoint(?:\s*/>|>.*?(?:</checkpoint>|(?=<checkpoint|$)))'
segments = re.split(checkpoint_pattern, template)

# Option B: Do this after initial parse
# Let grammar handle checkpoint detection, validate in transformer
```

**Recommendation**: Move this validation into the transformer since we already have parsed structure.

**TODO**: Refactor validation logic to work with parsed structure, not string splitting.

---

#### **Error messages** (MEDIUM IMPACT)

**Changes required**:
- Update all error messages referencing `¡OBLIVIATE` → `<checkpoint>`
- Update all error messages referencing `¡SYSTEM` → `<system>`
- Add new errors for invalid modifier combinations
- Update line number reporting for new syntax

---

### 3.3 Testing Impact

#### **Tests to Update** (HIGH IMPACT)

All existing tests use old syntax. Must update:

1. **test_system_messages.py** - Complete rewrite
   - 18 test cases covering system/header behaviors
   - All use `¡SYSTEM`, `¡HEADER`, `¡OBLIVIATE`
   - Need to add tests for new global/local scoping

2. **test_chatter_simple.py** - Update examples
3. **test_temporal_types.py** - Update if using system blocks
4. **test_optional_completion.py** - Update if using system blocks
5. **test_cache_integration.py** - Update if using system blocks
6. **test_auto_escaping.py** - Update if using system blocks

#### **Tests to Create** (HIGH IMPACT)

New test coverage needed:

1. **test_checkpoint_syntax.py**:
   - Named checkpoints: `<checkpoint>name</checkpoint>`
   - Self-closing: `<checkpoint />`
   - Malformed: `<checkpoint>`
   - Empty content: `<checkpoint></checkpoint>`
   - Special characters in names

2. **test_system_scoping.py**:
   - Local append (default)
   - Global append persistence
   - Local replace (checkpoint-scoped override)
   - Global replace (base replacement)
   - Multiple tags in same checkpoint
   - Interaction between scopes

3. **test_modifier_combinations.py**:
   - Valid: `append`, `append global`, `replace`, `replace global`
   - Invalid: `append replace`, `append replace global`
   - Multiple appends accumulating
   - Replace overwriting prior local

4. **test_edge_cases.py**:
   - System tags after completions (should error)
   - Checkpoints with no completions
   - Empty system tags
   - Malformed closing tags

**Estimated effort**: 40+ new test cases to achieve same coverage as current.

---

### 3.4 Documentation Impact

#### **Files to Update**:

1. **README.md** - All examples use old syntax
   - QuickStart section
   - Basic Syntax section
   - System Messages section
   - Memory Boundaries section

2. **docs/QUICKSTART.md** - Step-by-step tutorial
3. **docs/CLI_USAGE.md** - Command reference
4. **docs/TUTORIAL_RAG.md** - RAG examples
5. **docs/CUSTOM_ACTIONS.md** - Action examples

6. **examples/*.sd** - 14+ example files:
   - 01_basic_completion.sd
   - 02_simple_chain.sd
   - 03_template_variables.sd
   - 04_return_types.sd
   - 05_shared_header.sd
   - 06_list_completions.sd
   - 07_complex_workflow.sd
   - 08_template_tags.sd
   - 09_temporal_extraction.sd
   - 10_number_extraction.sd
   - 11_temporal_extraction.sd
   - new-syntax.sd (already exists!)
   - xmlstyle.sd (already exists!)
   - includes/*.sd

#### **New Documentation**:


1. **SYSTEM_SCOPING.md** - Deep dive on system prompt scoping
   - Base vs global vs local
   - Use cases for each
   - Best practices




### 3.5 VSCode Extension Impact

#### **syntaxes/struckdown.tmLanguage.json** (HIGH IMPACT)

**Add new patterns** to main patterns array:
```json
{
  "include": "#struckdown-xml-tags"
}
```

**Add to repository**:
```json
"struckdown-xml-tags": {
  "patterns": [
    {
      "comment": "Checkpoint tags",
      "patterns": [
        {
          "name": "meta.tag.checkpoint.struckdown",
          "match": "(<)(checkpoint)(\\s*/>)",
          "captures": {
            "1": {"name": "punctuation.definition.tag.begin.struckdown"},
            "2": {"name": "entity.name.tag.checkpoint.struckdown"},
            "3": {"name": "punctuation.definition.tag.end.struckdown"}
          }
        },
        {
          "name": "meta.tag.checkpoint.struckdown",
          "begin": "(<)(checkpoint)(>)",
          "end": "(</)(checkpoint)(>)|(?=<checkpoint)",
          "beginCaptures": {
            "1": {"name": "punctuation.definition.tag.begin.struckdown"},
            "2": {"name": "entity.name.tag.checkpoint.struckdown"},
            "3": {"name": "punctuation.definition.tag.end.struckdown"}
          },
          "endCaptures": {
            "1": {"name": "punctuation.definition.tag.begin.struckdown"},
            "2": {"name": "entity.name.tag.checkpoint.struckdown"},
            "3": {"name": "punctuation.definition.tag.end.struckdown"}
          },
          "contentName": "entity.name.checkpoint.name.struckdown"
        },
        {
          "comment": "Malformed checkpoint (no closing)",
          "name": "meta.tag.checkpoint.malformed.struckdown",
          "match": "(<)(checkpoint)(>)",
          "captures": {
            "1": {"name": "punctuation.definition.tag.begin.struckdown"},
            "2": {"name": "entity.name.tag.checkpoint.struckdown"},
            "3": {"name": "punctuation.definition.tag.end.struckdown"}
          }
        }
      ]
    },
    {
      "comment": "System tags with modifiers",
      "name": "meta.tag.system.struckdown",
      "begin": "(<)(system)(?:\\s+(append|replace))?(?:\\s+(global))?(>)",
      "end": "(</)(system)(>)",
      "beginCaptures": {
        "1": {"name": "punctuation.definition.tag.begin.struckdown"},
        "2": {"name": "entity.name.tag.system.struckdown"},
        "3": {"name": "storage.modifier.system.struckdown"},
        "4": {"name": "storage.modifier.scope.struckdown"},
        "5": {"name": "punctuation.definition.tag.end.struckdown"}
      },
      "endCaptures": {
        "1": {"name": "punctuation.definition.tag.begin.struckdown"},
        "2": {"name": "entity.name.tag.system.struckdown"},
        "3": {"name": "punctuation.definition.tag.end.struckdown"}
      },
      "patterns": [
        {"include": "#struckdown-template-variables"},
        {"include": "#struckdown-django-tags"}
      ]
    }
  ]
}
```

**Keep old patterns** (optional for transition period):
- Keep `struckdown-keywords` for `¡SYSTEM`, `¡OBLIVIATE` etc.

#### **themes/struckdown-dark.json** (MEDIUM IMPACT)

**Add token colors**:
```json
{
  "scope": ["entity.name.tag.checkpoint.struckdown", "entity.name.tag.system.struckdown"],
  "settings": {
    "foreground": "#E06C75",
    "fontStyle": "bold"
  }
},
{
  "scope": "entity.name.checkpoint.name.struckdown",
  "settings": {
    "foreground": "#C678DD",
    "fontStyle": "italic"
  }
},
{
  "scope": ["storage.modifier.system.struckdown", "storage.modifier.scope.struckdown"],
  "settings": {
    "foreground": "#61AFEF",
    "fontStyle": "italic"
  }
},
{
  "scope": ["punctuation.definition.tag.begin.struckdown", "punctuation.definition.tag.end.struckdown"],
  "settings": {
    "foreground": "#ABB2BF"
  }
}
```

#### **themes/struckdown-light.json** (MEDIUM IMPACT)

Similar colors, adjusted for light background.

---


## 4. ✅ All Design Questions Resolved

All design questions have been answered and finalized:

1. **Header handling**: ✅ **REMOVED** - Headers completely removed, no replacement
2. **Local replace semantics**: ✅ **Independent lists** - `global replace` only affects `globals[]`, `local replace` only affects `locals[]`. Each list is independent.
3. **Modifier conflicts**: ✅ **Error** - Raise error if both `append` and `replace` specified in same tag
4. **Malformed tag parsing**: ✅ **Line-based** - `<checkpoint>` looks for content on same line, then stops
5. **Old syntax migration**: ✅ **Clean break** - ALL `¡` syntax removed from parser (no backwards compatibility)
6. **Empty tags**: ✅ **No-op** - `<system></system>` appends empty string to global (no effect)
7. **Validation strictness**: ✅ **Error on unused local** - If local system tag appears with no subsequent completions in section, raise error
8. **Unnamed checkpoints**: ✅ **Auto-name by position** - Format: "Checkpoint N" where N is ordinal position in document (counting ALL checkpoints, named + unnamed)
9. **Tag order validation**: ✅ **Any order** - System tags can appear anywhere, apply to subsequent completions

---

## 5. Implementation Phases

### Phase 1: Grammar Updates (Week 1)
- Update grammar.lark with new XML-style rules (single-stage)
- Remove ALL old `¡` syntax rules
- Add negative lookahead for markdown terminal
- Unit tests for grammar parsing

### Phase 2: Transformer Extension (Week 1-2)
- Extend MindframeTransformer with new tag handlers
- Implement two-list system prompt model (globals/locals)
- Implement `_compute_effective_system()`
- Update completion handlers to use new system prompt logic
- Unit tests for scoping

### Phase 3: Testing (Week 2)
- Update existing tests to use new syntax
- Create new test suite (checkpoint syntax, system scoping, break tags)
- Integration tests
- Edge case testing

### Phase 4: VSCode Extension (Week 2)
- Update tmLanguage grammar for XML-style tags
- Update color themes
- Test highlighting

### Phase 5: Documentation (Week 2-3)
- Update all docs with new syntax examples
- Update all example `.sd` files
- Update README with new syntax

---

## 6. Risks & Mitigation

### High Risk

1. **Scoping logic bugs**: Mitigate with comprehensive unit tests, state machine diagram

### Medium Risk

2. **Grammar conflicts with markdown**: Reduced - single-stage proven approach (existing pattern)
3. **VSCode extension complexity**: Mitigate with incremental updates, test in real editor
4. **User confusion on scoping**: Mitigate with clear docs, examples, error messages

### Low Risk

5. **Performance impact**: Single-stage is faster than two-stage
6. **Visualization updates lag**: Acceptable - visualize can iterate separately

---

## 7. Success Criteria

1. All tests pass with new syntax
2. VSCode highlighting works correctly for all tags
3. Documentation updated and clear
4. No performance regression (parse time remains fast)
5. Clear error messages for common mistakes
6. Two-list system prompt model working correctly

---

## Appendix A: Syntax Comparison Table

| Feature | Old Syntax | New Syntax (v2.0) |
|---------|-----------|------------|
| Segment delimiter | `¡OBLIVIATE` or `¡SEGMENT` | `<checkpoint>` or `<obliviate>` |
| Named segment | `¡OBLIVIATE segment_name` | `<checkpoint>segment_name</checkpoint>` or `<checkpoint> segment_name` |
| Unnamed segment | _(not available)_ | `<checkpoint>` (auto-named by position) |
| System (replace, global) | `¡SYSTEM\nContent\n/END` | `<system>Content</system>` ⭐ |
| System (append, global) | `¡SYSTEM+\nContent\n/END` | `<system append>Content</system>` ⭐ |
| System (replace, local) | _(not available)_ | `<system replace local>Content</system>` ✨ |
| System (append, local) | _(not available)_ | `<system append local>Content</system>` ✨ |
| Header (replace) | `¡HEADER\nContent\n/END` | ~~REMOVED~~ |
| Header (append) | `¡HEADER+\nContent\n/END` | ~~REMOVED~~ |
| Early termination | `[[!end]]` | `<break>message</break>` or `<break />` ✨ |
| Completions | `[[variable]]` | `[[variable]]` _(unchanged)_ |
| Variables | `{{variable}}` | `{{variable}}` _(unchanged)_ |
| Comments | `<!-- comment -->` | `<!-- comment -->` _(stripped from LLM prompt)_ ✨ |
| Template tags | `{% tag %}` | `{% tag %}` _(unchanged)_ |
| File includes | `{% include 'file.sd' %}` | `{% include 'file.sd' %}` _(unchanged)_ |

**Legend**:
- ⭐ = Shorthand defaults to global scope
- ✨ = New feature in v2.0

---

## Appendix B: State Machine Diagram (Two-List Model)

```
System Prompt State Machine:

┌──────────────────────────────────────────────────────────┐
│ Global State (persists across checkpoints)              │
│                                                          │
│  globals: List[str]  = []                                │
│                                                          │
│  Operations:                                             │
│    <system replace global> → globals = [new_content]     │
│    <system append global>  → globals.append(new_content) │
│                                                          │
│  Shorthands (all default to global):                     │
│    <system>         = append global                      │
│    <system append>  = append global                      │
│    <system replace> = replace global                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Local State (cleared each checkpoint)                   │
│                                                          │
│  locals: List[str]  = []                                 │
│                                                          │
│  Operations:                                             │
│    <system replace local> → locals = [new_content]       │
│    <system append local>  → locals.append(new_content)   │
│                                                          │
│  Cleared after each <checkpoint> or <obliviate>          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Effective System Prompt (computed per completion)       │
│                                                          │
│  effective = "\n\n".join(globals) + "\n\n".join(locals)  │
│                                                          │
│  Example:                                                │
│    globals = ["A", "B"]                                  │
│    locals = ["C", "D"]                                   │
│    effective = "A\n\nB\n\nC\n\nD"                         │
└──────────────────────────────────────────────────────────┘

Checkpoint Flow:
═══════════════

Checkpoint 1:
  <system>A</system>                 → globals = ["A"], locals = []
  <system append local>B</system>    → globals = ["A"], locals = ["B"]
  [[q1]]                             → sees "A\n\nB"

<checkpoint />                       → locals cleared!

Checkpoint 2:
  <system append>C</system>          → globals = ["A", "C"], locals = []
  [[q2]]                             → sees "A\n\nC"

<checkpoint />                       → locals cleared!

Checkpoint 3:
  <system replace>D</system>         → globals = ["D"], locals = []
  [[q3]]                             → sees "D"
```

---

## Summary & Next Steps

### ✅ Fully Approved & Ready for Implementation

**All design decisions finalized**:
- **Syntax**: XML-style tags (`<checkpoint>`, `<system>`, `<break>`)
- **System prompts**: Two-list model (globals[] + locals[])
- **Headers**: Removed entirely
- **Checkpoints**: Three styles (closed tag, line-based, auto-named)
- **Auto-naming**: "Checkpoint N" where N is ordinal position in document (all checkpoints counted)
- **Synonyms**: `<obliviate>` works same as `<checkpoint>`
- **Migration**: Clean break - ALL `¡` syntax removed from parser
- **Break tags**: Unconditional early termination, preserves context
- **System tag placement**: Anywhere (applies to subsequent completions)
- **Validation**: Error on conflicting modifiers, error on unused local system tags

### 📋 Implementation Plan

**Ready to proceed with**:

1. ✅ **Phase 1**: Grammar updates (grammar.lark) - single-stage parsing, remove ALL `¡` syntax
2. ✅ **Phase 2**: Extend MindframeTransformer (parsing.py) - new tag handlers + two-list system
3. ✅ **Phase 3**: Break tag functionality + validation logic
4. ✅ **Phase 4**: VSCode extension syntax highlighting
5. ✅ **Phase 5**: Test suite updates (40+ test cases) + documentation

**Estimated timeline**: 2 weeks for full implementation (clean break, no migration complexity)

### 🚀 Next Actions

1. Create GitHub issue/milestone for v2.0 release
2. Begin Phase 1: Grammar & parsing implementation
3. Set up feature branch: `feature/xml-syntax-v2`

---

**END OF DOCUMENT**
