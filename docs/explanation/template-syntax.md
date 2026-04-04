---
layout: default
title: Template Syntax
parent: Explanation
nav_order: 3
---

# Template Syntax

Struckdown templates combine Jinja2 templating with special syntax for structured LLM interactions.

## Completion Slots

Slots define where the LLM should produce output. The basic syntax is:

```
[[variable]]                    # Basic text completion
[[type:variable]]               # Typed completion
[[type:variable|options]]       # With constraints
```

### Available Types

| Type | Use for | Example |
|------|---------|---------|
| `extract` | Verbatim text extraction | `[[extract:quote]]` |
| `respond` | Natural response (default) | `[[respond:answer]]` or `[[answer]]` |
| `think` | Internal reasoning | `[[think:analysis]]` |
| `speak` | Conversational dialogue | `[[speak:greeting]]` |
| `bool` | True/False decisions | `[[bool:is_urgent]]` |
| `int` | Whole numbers | `[[int:count]]` |
| `number` | Int or float | `[[number:price\|min=0]]` |
| `pick` | Choose from options | `[[pick:category\|sales,support,billing]]` |
| `date` | Date extraction | `[[date:deadline]]` |
| `datetime` | Date with time | `[[datetime:appointment]]` |
| `time` | Time only | `[[time:start_time]]` |
| `json` | Structured JSON output | `[[json:metadata]]` |
| `record` | JSON object | `[[record:person]]` |

### Examples

```python
from struckdown import chatter

# Basic completion
result = chatter("What is the capital of France? [[answer]]")
print(result["answer"])  # "Paris"

# Typed boolean
result = chatter("Is the sky blue? [[bool:is_blue]]")
print(result["is_blue"])  # True

# Pick from options
result = chatter("Classify: 'I love it!' [[pick:sentiment|positive,negative,neutral]]")
print(result["sentiment"])  # "positive"

# Number with constraints
result = chatter("Rate 1-10: 'Great product' [[int:score|min=1,max=10]]")
print(result["score"])  # 8
```

## Quantifiers (Lists)

Extract multiple items using quantifiers:

```
[[type*:var]]           # Zero or more items
[[type+:var]]           # One or more items (at least one)
[[type?:var]]           # Zero or one item (optional)
[[type{3}:var]]         # Exactly 3 items
[[type{2,5}:var]]       # Between 2 and 5 items
[[type{3,}:var]]        # At least 3 items
```

### Examples

```bash
# Exactly 3 fruits
sd chat "Name 3 fruits: [[pick{3}:fruits|apple,banana,orange,grape]]"

# One or more (at least one required)
sd chat "List the main points: [[extract+:points]]"

# Zero or more (can be empty)
sd chat "Any warnings? [[extract*:warnings]]"
```

## Required Fields

Mark slots as required using `!` prefix or explicit option:

```
[[!type:var]]           # ! prefix = required
[[type:var|required]]   # Explicit required option
```

Required slots must have a valid response -- the LLM cannot skip them.

## Constraints

Add validation constraints to slots:

```
[[number:score|min=0,max=100]]              # Numeric range
[[number:price|min=0,max=1000,required]]    # Required with constraints
[[int:count|min=1,max=10]]                  # Integer range
[[extract:code|pattern="\\d{3}-\\d{4}"]]    # Regex pattern
```

### Pattern Matching

Constrain text extraction with regex:

```bash
# Module code: 4 letters followed by digits
sd chat 'Module code: [[extract:code|pattern="\w{4}\d+"]]'

# UK postcode
sd chat 'Postcode: [[extract:postcode|pattern="[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}"]]'
```

## Template Variables

Reference extracted values or input data:

```
{{variable}}            # Reference extracted variable
{{variable.field}}      # Access nested JSON field
{{input}}               # Reference input data (batch processing)
```

### Example

```
Extract the name: [[extract:name]]

<checkpoint>

Hello {{name}}, tell me about yourself: [[response]]
```

## System Messages

Control LLM behaviour with system messages:

```
<system>You are an expert analyst.</system>          # Global (persists across checkpoints)
<system local>Focus on accuracy.</system>            # Local (cleared at checkpoint)
<system replace>New global instructions.</system>    # Replace previous system
```

### Example

```
<system>
You are an experienced data analyst.
Be precise and factual.
If information is missing, say "Not found" rather than guessing.
</system>

Analyse this data: {{input}}

[[analysis]]
```

## Checkpoints (Memory Boundaries)

Use `<checkpoint>` to create memory boundaries and save tokens:

```
First, read this document carefully:
{{document}}

Extract the key points: [[extract+:key_points]]

<checkpoint>

# After checkpoint, only {{key_points}} is available
# Previous messages are cleared (saves tokens)

Based on these points: {{key_points}}

Provide recommendations: [[recommendations]]
```

**Critical**: Variables from before a checkpoint must be included as `{{variable}}` to remain visible in subsequent sections.

## Parallelisation

Run multiple completions in parallel with isolated contexts:

```
<together>
[[analysis_a]]
[[analysis_b]]
[[analysis_c]]
</together>
# All three run in parallel
```

## Jinja2 Templating

Full Jinja2 syntax is supported:

### Conditionals

```jinja
{% if include_examples %}
Here are some examples:
- Example 1
- Example 2
{% endif %}

Analyse: {{content}}
[[analysis]]
```

### Loops

```jinja
Review these items:
{% for item in items %}
- {{item.name}}: {{item.description}}
{% endfor %}

[[review]]
```

### Filters

```jinja
{{text | upper}}
{{items | join(", ")}}
{{content | truncate(100)}}
```

## File Includes

Include other template files:

```jinja
{% include 'system-prompt.sd' %}

User: {{question}}

[[answer]]
```

Include paths are resolved relative to the template file, then common locations like `templates/` and `~/.struckdown/includes/`.

## Comments

Comments are removed before processing:

```jinja
{# Jinja2 comment - not sent to LLM #}

<!-- HTML comment - also removed -->

Actual prompt content here.
[[response]]
```

## Built-in Actions

Actions perform operations without LLM calls:

```
[[@set:varname|"literal value"]]           # Set variable without LLM
[[@set:copy|other_variable]]               # Copy variable
[[@fetch:content|url="https://..."]]       # Fetch URL content
[[@search:results|query="topic",n=5]]      # Web search
[[@timestamp:now]]                         # Current timestamp
[[@timestamp:now|format="%Y-%m-%d"]]       # Formatted timestamp
[[@break|reason="Done"]]                   # Early termination
```

## Role Messages

Simulate conversation turns:

```
<header>Context that appears before each segment</header>
<user>Simulated user message</user>
<assistant>Simulated assistant response</assistant>
```

## Model/Temperature Overrides

Override LLM settings per-slot:

```
[[think:reasoning|temperature=0.3]]
[[pick:choice|red,blue|model=gpt-4]]
[[extract:data|model=gpt-4,temperature=0.0]]
[[think:deep|thinking=high,temperature=0.3]]
[[respond:summary|thinking=off]]
```

Supported per-slot parameters: `temperature`, `thinking`, `model`, `max_tokens`, `seed`. See [Model Overrides](../how-to/model-overrides.md) for details.

## Streaming

Free-text slots (`respond`, `speak`, `think`, `extract`, `poem`) stream token-by-token when using the async API or CLI. Constrained slots (`pick`, `bool`, `int`, etc.) complete atomically. No template syntax changes are required -- streaming is handled automatically.

## Custom Pydantic Types

Use custom Pydantic models for complex structured output:

```python
from pydantic import BaseModel
from struckdown import chatter

class Person(BaseModel):
    name: str
    age: int
    occupation: str

result = chatter("""
Extract person info from: {{text}}
[[Person:person]]
""", context={
    "text": "John is a 30-year-old engineer",
    "Person": Person
})

person = result["person"]  # Person(name="John", age=30, occupation="engineer")
```
