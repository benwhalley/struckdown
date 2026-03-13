# Struckdown Prompt Writer

Create well-engineered Struckdown prompts for structured LLM interactions.

## Usage

```
/struckdown <intent description>
/struckdown <intent> with slots: name, score, comment
```

**Examples:**
```
/struckdown extract contact details from business cards
/struckdown analyse customer feedback sentiment with slots: sentiment, urgency, topic
/struckdown generate quiz questions about a topic
/struckdown summarise clinical notes into structured format
```

## What this skill does

1. Gathers detailed requirements through conversation
2. Writes an optimised Struckdown prompt using all available syntax features
3. Saves the prompt as a `.sd` file in the current directory
4. Tests the prompt with real or simulated data
5. Iterates based on output quality
6. Suggests CLI commands for batch processing

## Output

- **Prompt file**: `<descriptive-name>.sd` in current working directory
- **Test results**: Shown interactively during development

---

## Instructions for Claude

When this skill is invoked, follow this workflow:

### Phase 1: Requirements Gathering

**Ask upfront questions to work autonomously.** Use AskUserQuestion to clarify:

1. **Purpose**: What is the prompt for? (extraction, generation, classification, analysis, etc.)

2. **Slot names**: What variables should be extracted/generated? Get specific names if not provided.

3. **Data source**:
   - Will this process files? What format? (txt, csv, xlsx, json)
   - Is there sample data in the current directory?
   - Or should we generate simulated test data?

4. **Constraints**:
   - Do any slots need type constraints? (bool, int, number, pick, date, etc.)
   - Are there minimum/maximum values needed?
   - Should any slots be required (must have a response)?
   - Should any slots return lists? How many items?

5. **Persona/Context**:
   - Does the LLM need a specific role/expertise?
   - Any domain-specific instructions?

6. **Testing permission**: Ask: "May I run test iterations to refine the prompt? This uses API calls but ensures quality output."

### Phase 2: Scan for Data

Before writing the prompt, check for potential data sources:

```bash
ls -la *.txt *.csv *.xlsx *.json *.md 2>/dev/null | head -20
```

If CSV/Excel files exist that might be relevant, mention them and suggest using `--head 3` for testing.

### Phase 3: Write the Prompt

Create the `.sd` file using these Struckdown features appropriately:

---

## STRUCKDOWN SYNTAX REFERENCE

### Completion Slots

```
[[variable]]                    # Basic text completion
[[type:variable]]               # Typed completion
[[type:variable|options]]       # With constraints
```

**Available types:**

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

### Quantifiers (for lists)

```
[[type*:var]]           # Zero or more items
[[type+:var]]           # One or more items (at least one)
[[type?:var]]           # Zero or one item (optional)
[[type{3}:var]]         # Exactly 3 items
[[type{2,5}:var]]       # Between 2 and 5 items
[[type{3,}:var]]        # At least 3 items
```

### Required Fields

```
[[!type:var]]           # ! prefix = required
[[type:var|required]]   # Explicit required option
```

Use the exclamation prefix or "required" option when the slot MUST have a valid response.

### Constraints

```
[[number:score|min=0,max=100]]              # Numeric range
[[number:price|min=0,max=1000,required]]    # Required with constraints
[[int:count|min=1,max=10]]                  # Integer range
[[extract:code|pattern="\\d{3}-\\d{4}"]]    # Regex pattern
```

### System Messages

```
<system>You are an expert analyst.</system>          # Global (persists across checkpoints)
<system local>Focus on accuracy for this section.</system>  # Local (cleared at checkpoint)
<system replace>New global instructions.</system>   # Replace previous system
```

### Checkpoints (Memory Boundaries)

```
First section...
[[summary]]

<checkpoint>

# After checkpoint, only {{summary}} is available
# All previous messages are cleared (saves tokens)
Based on {{summary}}, analyse further: [[analysis]]
```

**Critical**: Variables from before a checkpoint MUST be included as `{{variable}}` to remain visible.

### Template Variables

```
{{variable}}            # Reference extracted variable
{{variable.field}}      # Access nested JSON field
{{input}}               # Reference input data (for batch processing)
```

### Jinja2 Features

```
{% if condition %}...{% endif %}     # Conditional sections
{% for item in list %}...{% endfor %}  # Loops
{% set var = value %}                # Variable assignment
{# Comment - not sent to LLM #}      # Comments
```

### HTML Comments (Dropped)

```
<!-- This comment is NOT sent to the LLM -->
```

### Built-in Actions

```
[[@set:varname|"literal value"]]           # Set variable without LLM
[[@set:copy|other_variable]]               # Copy variable
[[@fetch:content|url="https://..."]]       # Fetch URL content
[[@search:results|query="topic",n=5]]      # Web search
[[@timestamp:now]]                         # Current timestamp
[[@timestamp:now|format="%Y-%m-%d"]]       # Formatted timestamp
[[@break|reason="Done"]]                   # Early termination
```

### Parallelisation

```
<together>
[[analysis_a]]
[[analysis_b]]
[[analysis_c]]
</together>
# All three run in parallel with isolated contexts
```

### Headers and Role Messages

```
<header>Context that appears before each segment</header>
<user>Simulated user message</user>
<assistant>Simulated assistant response</assistant>
```

---

## PROMPT ENGINEERING STRATEGIES

### 1. Use System Messages Wisely

Give the LLM a clear persona and constraints:

```
<system>
You are an experienced data analyst.
Be precise and factual.
If information is missing, say "Not found" rather than guessing.
</system>
```

### 2. Chain of Thought with Two Slots

Use a thinking slot before the final answer:

```
Analyse this data: {{input}}

First, identify the key patterns: [[think:analysis]]

<checkpoint>

Based on your analysis: {{analysis}}

Provide your final recommendation: [[recommendation]]
```

### 3. Use Checkpoints for Token Efficiency

Split long prompts into segments. Only carry forward what's needed:

```
# Section 1: Extract
Document: {{document}}
Extract all names mentioned: [[extract+:names]]
Extract all dates mentioned: [[date*:dates]]

<checkpoint>

# Section 2: Analyse (document text no longer in context)
Names found: {{names}}
Dates found: {{dates}}

Identify relationships between people and dates: [[relationships]]
```

### 4. Constrain Outputs Appropriately

- Use `pick` for classification: `[[pick:sentiment|positive,negative,neutral]]`
- Use `bool` for yes/no: `[[bool:is_complete]]`
- Use `int` or `number` with min/max for scores: `[[int:confidence|min=1,max=5]]`
- Use `required` when you MUST have a value: `[[!pick:category|a,b,c]]`

### 5. Lists vs Single Values

- `[[item]]` - single response
- `[[item{3}]]` - exactly 3 items
- `[[item{1,5}]]` - between 1 and 5 items
- `[[item+]]` - one or more (at least one)
- `[[item*]]` - zero or more (can be empty)

### 6. Minimise Checkpoint Context

Only include variables that the next section actually needs:

```
# BAD - carries forward everything
<checkpoint>
{{full_document}} {{analysis}} {{metadata}} {{timestamps}}

# GOOD - only what's needed
<checkpoint>
Key findings: {{key_findings}}
```

---

## TESTING WORKFLOW

### For Single Prompts

```bash
uv run sd chat -p prompt.sd --show-context
```

### For Batch Processing

```bash
# Test with first 3 rows of CSV
uv run sd batch data.csv -p prompt.sd --head 3 -o test_output.json

# Process all files
uv run sd batch *.txt -p prompt.sd -o results.csv
```

### Inspect and Iterate

1. Run the prompt with test data
2. Check the output for:
   - Are slot values correct/sensible?
   - Are required fields populated?
   - Are constraints being respected?
   - Is the output format correct?
3. If issues found, revise the prompt:
   - Clarify instructions in system message
   - Add constraints to slots
   - Use `required` for must-have fields
   - Add examples in the prompt text

---

## CUSTOM PYDANTIC TYPES

If the user needs complex structured output beyond basic types, Struckdown supports custom Pydantic models.

**Before implementing custom types, confirm with the user:**

1. Show them the proposed model structure
2. Explain how it will constrain the output
3. Get approval before proceeding

Custom types are defined in YAML and require additional setup.

---

## SKILL WORKFLOW

1. **Gather requirements** - Ask all clarifying questions upfront
2. **Check for data** - Look for CSV/Excel/text files in current directory
3. **Write initial prompt** - Create `.sd` file with appropriate syntax
4. **Test the prompt** - Run with sample data (with permission)
5. **Review output** - Show user the test results
6. **Iterate if needed** - Refine based on output quality
7. **Provide batch command** - Suggest CLI command for full processing

---

## EXAMPLE: Complete Prompt Creation

**User request**: "Extract contact info from business cards"

**Questions to ask**:
- What fields? (name, email, phone, company, title, address?)
- Any required? (name always required?)
- Multiple phone numbers possible?
- Should we classify contact type?

**Resulting prompt** (`business_card_extractor.sd`):

```
<system>
You are a data extraction specialist.
Extract contact information exactly as written.
If a field is not present, leave it empty.
</system>

Business card text:
{{input}}

Extract the following information:

Name: [[!extract:name]]
Job title: [[extract:title]]
Company: [[extract:company]]
Email: [[extract:email]]
Phone numbers: [[extract*:phone]]
Address: [[extract:address]]
Website: [[extract:website]]

Contact type: [[pick:contact_type|business,personal,unknown]]
```

**Test command**:
```bash
uv run sd batch cards/*.txt -p business_card_extractor.sd --head 3 -o contacts.json
```

---

## IMPORTANT REMINDERS

1. **Always save as `.sd` file** - Never just show the prompt, save it
2. **Ask questions first** - Gather all requirements before writing
3. **Test with permission** - Always ask before running API calls
4. **Show test output** - Let user see results before finalising
5. **Iterate** - If output is surprising, revise the prompt
6. **Suggest batch commands** - Help user run the prompt at scale
7. **Preserve shape when editing** - If updating an existing prompt, keep overall structure and slot names unless user requests changes

## ERROR HANDLING

- If slot extraction fails: Check constraints aren't too strict
- If LLM ignores instructions: Strengthen system message
- If output format wrong: Use typed slots instead of free text
- If missing values: Add "required" option or exclamation prefix
- If too verbose: Use `extract` type instead of `respond`
