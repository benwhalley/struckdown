---
layout: default
title: CLI
parent: Reference
nav_order: 1
---

# Struckdown CLI Usage Guide

Complete reference for the `sd` command-line interface.

## Table of Contents

- [Installation](#installation)
- [Commands](#commands)
  - [sd chat](#sd-chat)
  - [sd batch](#sd-batch)
- [Global Options](#global-options)
- [Batch Processing Options](#batch-processing-options)
- [Progress Bars](#progress-bars)
- [Output Streams](#output-streams)
- [Examples](#examples)

---

## Installation

```bash
# Using uv (recommended)
uv tool install git+https://github.com/benwhalley/struckdown/

# Or install in current environment
uv pip install git+https://github.com/benwhalley/struckdown/
```

**Environment Setup:**
```bash
export LLM_API_KEY="your-api-key"          # e.g., from openai.com
export LLM_API_BASE="https://api.openai.com/v1"
export DEFAULT_LLM="litellm/gpt-4.1-mini"
```

---

## Commands

### `sd chat`

Run a single prompt interactively. Best for testing prompts and quick experiments.

**Syntax:**
```bash
sd chat [OPTIONS] PROMPT...
```

**Options:**
- `--model-name TEXT` - Override default LLM model
- `--show-context` - Print the resolved prompt context after execution
- `--verbose` - Enable debug logging

**Examples:**
```bash
# Simple completion
sd chat "Tell me a joke: [[joke]]"

# Multiple slots
sd chat "Name a color: [[pick:color|red,blue,green]] Describe it: [[description]]"

# With context display
sd chat "Pick a number: [[int:number]]" --show-context
```

---

### `sd batch`

Process multiple inputs in batch mode. Supports files, globs, stdin, and chaining.

**Syntax:**
```bash
sd batch [OPTIONS] [INPUTS...] [PROMPT]
```

**Arguments:**
- `INPUTS` - Input files or glob patterns (e.g., `*.txt`, `data/*.json`)
- `PROMPT` - Inline prompt with extraction slots

**Input Sources:**
1. **Files:** `sd batch file1.txt file2.txt "[[summary]]"`
2. **Globs:** `sd batch inputs/*.txt "[[summary]]"`
3. **Stdin:** `cat data.txt | sd batch "[[summary]]"`
4. **JSON stdin:** `echo '{"name":"Alice"}' | sd batch "Hello {{name}} [[greeting]]"`

**Output Formats:**
- **JSON** (`.json`) - Default, structured data
- **CSV** (`.csv`) - Flattened tabular format
- **Excel** (`.xlsx`) - Spreadsheet format
- **Markdown** (`.md`, `.txt`) - Markdown tables
- **stdout** - JSON to stdout if no `-o` specified

---

## Global Options


### `--help`
Show help for any command.

```bash
sd --help
sd chat --help
sd batch --help
```

**Output:** stdout
**Exit code:** 0

---

## Batch Processing Options

### `-o` / `--output PATH`
Output file path. Format auto-detected from extension.

```bash
sd batch *.txt "[[name]]" -o results.json   # JSON output
sd batch *.txt "[[name]]" -o results.csv    # CSV output
sd batch *.txt "[[name]]" -o results.xlsx   # Excel output
sd batch *.txt "[[name]]" -o results.md     # Markdown table
```

**Default:** Outputs JSON to stdout if omitted.

---

### `-p` / `--prompt PATH`
Load prompt from file instead of inline.

```bash
# prompt.sd contains: "Extract name: [[name]]"
sd batch *.txt -p prompt.sd -o results.json
```

**Cannot be combined with inline prompt.**

---

### `-k` / `--keep-inputs`
Include input fields in output (filename, content, basename).

```bash
sd batch *.txt "[[summary]]" -k -o results.json
```

**Output includes:**
```json
{
  "filename": "input.txt",
  "input": "original text...",
  "content": "original text...",
  "basename": "input",
  "summary": "extracted summary"
}
```

**Default:** Only extracted fields + filename for traceability.

---

### `-q` / `--quiet`
Suppress progress output to stderr.

```bash
sd batch *.txt "[[name]]" -o results.json --quiet
```

**Behavior:**
- ✗ No progress bar
- ✗ No verbose logs (unless `--verbose` also specified)
- ✓ Errors still shown on stderr

**Use case:** Scripting, cron jobs, CI/CD pipelines.

---

### `--verbose`
Enable debug logging to stderr.

```bash
sd batch *.txt "[[name]]" --verbose 2> debug.log
```

**Output includes:**
- Prompts sent to LLM
- LLM responses
- Cache hits/misses
- Per-item processing status
- Stack traces on errors

**Destination:** stderr

---

### `--model-name TEXT`
Override default LLM model for this run.

```bash
sd batch *.txt "[[summary]]" --model-name "litellm/gpt-4"
```

**Default:** `$DEFAULT_LLM` environment variable.

---

## Progress Bars

`sd batch` shows a progress bar by default during processing:

```
Processing ⠋ ━━━━━━━━━━━━━━━━━━━━━━ 47/100 47% 0:00:23
```

**Display includes:**
- Spinner animation
- Progress bar
- Items completed / total
- Percentage
- Estimated time remaining (ETA)

### Automatic Behavior

**Shown when:**
- stderr is a TTY (interactive terminal)
- NOT in `--quiet` mode

**Hidden when:**
- stderr is redirected: `sd batch ... 2> errors.log`
- stdout is piped: `sd batch ... | jq .`
- `--quiet` flag is used

### Manual Control

```bash
# Force show (default in terminal)
sd batch *.txt "[[name]]" -o out.json

# Force hide
sd batch *.txt "[[name]]" -o out.json --quiet

# Redirect to ignore progress
sd batch *.txt "[[name]]" -o out.json 2>/dev/null
```

---

## Output Streams

Struckdown follows CLI best practices for output streams:

### stdout (File Descriptor 1)
**Primary output** - machine-readable results, intended for piping/capture.

**Contains:**
- Extracted results (JSON by default)
- `--version` output
- `--help` output
- Success messages from output formatters (via logger.info, but these go to stderr)

**Example:**
```bash
sd batch *.txt "[[name]]" > results.json   # Only results captured
```

---

### stderr (File Descriptor 2)
**Diagnostics** - human-readable status, errors, warnings.

**Contains:**
- Progress bars
- Error messages
- Warnings
- Verbose debug logs
- Logger output (info, warning, error levels)

**Example:**
```bash
sd batch *.txt "[[name]]" 2> errors.log    # Only diagnostics captured
```

---

### Exit Codes

| Code | Meaning | Examples |
|------|---------|----------|
| 0 | Success | Command completed without errors |
| 1 | Error | LLM failure, file not found, processing error |
| 2 | Usage error | Invalid arguments, missing required options |

**Example:**
```bash
sd batch --invalid-flag
# Exit code: 2 (usage error)

sd batch missing-file.txt "[[x]]"
# Exit code: 1 (file not found error)

sd batch *.txt "[[x]]" -o out.json
# Exit code: 0 (success)
```

---

## Examples

### Basic Batch Processing

**Extract names from text files:**
```bash
sd batch documents/*.txt "Extract person's name: [[name]]" -o names.json
```

**Output:**
```json
[
  {"filename": "doc1.txt", "name": "Alice"},
  {"filename": "doc2.txt", "name": "Bob"}
]
```

---

### Piping and Chaining

**Chain multiple extraction steps:**
```bash
sd batch *.txt "Purpose: [[purpose]] Name: [[name]]" | \
  sd batch "{{name}}: {{purpose}}. Amazon equivalent? [[product]]" -k
```

**Process and filter with jq:**
```bash
sd batch *.txt "Price: [[number:price]]" | jq '.[] | select(.price > 100)'
```

---

### Quiet Mode for Scripts

**Silent execution, log errors:**
```bash
#!/bin/bash
sd batch data/*.txt "[[summary]]" -o results.json --quiet 2> errors.log

if [ $? -eq 0 ]; then
  echo "Processing complete: results.json"
else
  echo "Errors occurred, check errors.log"
  exit 1
fi
```

---

### Verbose Debugging

**Capture full debug output:**
```bash
sd batch *.txt "[[name]]" --verbose 2> debug.log
```

**Debug log contains:**
- Prompts sent to LLM
- Raw LLM responses
- Cache operations
- Template rendering details
- Full stack traces on errors

---

### Reading from Stdin

**Plain text:**
```bash
echo "Hello world" | sd batch "Translate to Spanish: [[translation]]"
```

**JSON input:**
```bash
echo '[{"name":"Alice"},{"name":"Bob"}]' | \
  sd batch "Hello {{name}}! [[greeting]]"
```

**From file:**
```bash
cat data.txt | sd batch "Summarize: [[summary]]" -o summary.json
```

---

### Multiple Output Formats

**CSV for spreadsheet import:**
```bash
sd batch *.txt "Name: [[name]] Age: [[int:age]]" -o people.csv
```

**Excel for reports:**
```bash
sd batch *.txt "Product: [[product]] Price: [[number:price]]" -o report.xlsx
```

**Markdown for documentation:**
```bash
sd batch *.txt "Feature: [[feature]] Status: [[pick:status|done,wip,todo]]" -o status.md
```

---

### Error Handling

**Separate successful results and errors:**
```bash
sd batch *.txt "[[summary]]" -o results.json 2> errors.log

# Check exit code
if [ $? -ne 0 ]; then
  echo "Some items failed, check errors.log"
fi
```

**Verbose mode for debugging failures:**
```bash
sd batch failing-input.txt "[[x]]" --verbose
```

---

## Tips and Best Practices

### Performance
- **Use caching:** Repeated prompts use cached results (see `$STRUCKDOWN_CACHE`)
- **Batch processing:** Process multiple files in one command, not a loop
- **Progress monitoring:** Leave progress bars on in long-running jobs

### Composability
- **Pipe results:** `sd batch ... | sd batch ...` for multi-stage extraction
- **Use jq:** Post-process JSON output with `jq` for filtering/transformation
- **Redirect smartly:** Capture results and errors separately

### Debugging
- **Start with chat:** Test prompts with `sd chat` before batch processing
- **Use --verbose:** Enable verbose mode to see exactly what's happening
- **Check exit codes:** Use `$?` in scripts to detect failures

### Scripting
- **Use --quiet:** Suppress progress in cron jobs and CI/CD
- **Check exit codes:** Always check `$?` after `sd batch`
- **Log errors:** Redirect stderr to a log file for auditing

---

## Related Documentation

- [Getting Started](../tutorials/getting-started.md) - Quick start guide
- [Template Syntax](../explanation/template-syntax.md) - Struckdown template syntax
- [Model Overrides](../how-to/model-overrides.md) - Per-slot LLM configuration
- [Number Extraction](../how-to/number-extraction.md) - Numeric validation

---

## Migration from `chatter`

The legacy `chatter` CLI has been removed in v0.1.6.

**Old command:**
```bash
chatter run "Tell me a joke: [[joke]]"
```

**New command:**
```bash
sd chat "Tell me a joke: [[joke]]"
```

**Functionality is identical.** All flags and options work the same way.
