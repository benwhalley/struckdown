---
layout: default
title: CLI
parent: Reference
nav_order: 1
---

# Struckdown CLI Usage Guide

Reference for the `sd` command-line interface. Run `sd <command> --help` for
the option list of the version you have installed.

## Table of Contents

- [Installation](#installation)
- [Commands](#commands)
  - [sd chat](#sd-chat)
  - [sd batch](#sd-batch)
  - [Other commands](#other-commands)
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
export DEFAULT_LLM="gpt-4.1-mini"
```

If `DEFAULT_LLM` is not set, the library falls back to `gpt-4.1-mini`. Model
names use the `provider:model` form (`openai:gpt-4o`,
`anthropic:claude-sonnet-4-20250514`, `ollama:llama3`). A bare name with no
prefix defaults to OpenAI. When `LLM_API_BASE` is set, every request goes
through that proxy and the provider prefix is stripped.

`sd test` checks that these settings work; see [Other commands](#other-commands).

---

## Commands

| Command | Purpose |
|---------|---------|
| `sd chat` | Run one prompt, or an interactive conversation |
| `sd batch` | Run one prompt over many inputs |
| `sd explain` | Show a prompt's structure and execution plan |
| `sd graph` | Print a section dependency graph as Mermaid text |
| `sd preview` | Render a `.sd` file with syntax highlighting |
| `sd flat` | Resolve `{% raw %}{% include %}{% endraw %}` directives and print the result |
| `sd edit` | Open the browser-based playground on local files |
| `sd serve` | Run the playground as a hosted service |
| `sd test` | Check LLM and embedding connections |
| `sd setup` | Install the Claude Code skill and the VS Code extension |
| `sd install-skill` | Install the Claude Code skill only |
| `sd install-vscode` | Install the VS Code extension only |

### `sd chat`

Run a single prompt. Useful for testing prompts and quick experiments.

**Syntax:**
```bash
sd chat [OPTIONS] [PROMPT]...
```

The prompt can be given inline, read from a file with `-p`, or piped on stdin
(`cat prompt.sd | sd chat`).

**Options:**

| Option | Description |
|--------|-------------|
| `-p`, `--prompt-file PATH` | Read the prompt from a file |
| `-s`, `--source TEXT` | File or URL whose content is available as `{% raw %}{{source}}{% endraw %}` |
| `--raw` | When `--source` is a URL, fetch raw HTML instead of extracted markdown |
| `-c`, `--context KEY=VALUE` | Set a context variable; repeatable |
| `-d`, `--data TEXT` | Load a JSON file as context. `file.json` gives `{% raw %}{{data.key}}{% endraw %}`; `key=file.json` gives `{% raw %}{{data.key.subkey}}{% endraw %}` |
| `--model-name TEXT` | Model as `provider:model`. Default: `$DEFAULT_LLM` |
| `--seed INT` | Random seed, where the model supports one |
| `--thinking TEXT` | Reasoning mode: `on`, `off`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `--show-context` / `--no-show-context` | Print the resolved context after the run (default: off) |
| `-v`, `--verbose` | Counted: `-v` shows messages, `-vv` adds detail and info logs, `-vvv` debug logs |
| `-I`, `--include-path PATH` | Extra directory to search for includes; repeatable |
| `--type PATH` | YAML type definition file or directory; repeatable |
| `--tools PATH` | Python tools file or directory; repeatable |
| `--history PATH` | Conversation history file for the `@history` action (one line per turn, alternating roles) |
| `-i`, `--interactive` | Keep the conversation going after the first response |
| `-o`, `--output PATH` | Write slot outputs as a JSON object to a file |
| `--debug-api` | Log full API requests (messages and tool schema) as JSON |
| `--dump PATH` | Save each API call as JSON in this directory, one file per slot (`{slot_name}.json`) |
| `--no-stream` | Disable streaming output, for models that struggle with structured streaming |
| `--strict-params` | Raise an error on unsupported LLM parameters instead of warning |

**Examples:**
```bash
# Simple completion
sd chat "Tell me a joke: [[joke]]"

# Multiple slots
sd chat "Name a colour: [[pick:colour|red,blue,green]] Describe it: [[description]]"

# With context display
sd chat "Pick a number: [[int:number]]" --show-context

# Prompt file, source document and context variables
sd chat -p prompt.sd -s input.txt -c topic=sun -o result.json

# Different model
sd chat "joke [[joke]]" --model-name anthropic:claude-sonnet-4-20250514
```

---

### `sd batch`

Run one prompt over many inputs, concurrently. Inputs come from files, globs,
URLs or stdin.

**Syntax:**
```bash
sd batch [OPTIONS] [PROMPT]
```

**Arguments:**
- `PROMPT` - Inline prompt with slots. Omit it when using `-p`.

`sd batch` takes one positional argument. Input files go through `-i`, which
can be repeated. Passing a file as a second positional argument is a usage
error (exit code 2).

**Input Sources:**
1. **Files:** `sd batch -i file1.txt -i file2.txt "[[summary]]"`
2. **Globs:** `sd batch -i 'inputs/*.txt' "[[summary]]"` (quote the pattern; `**` is recursive)
3. **Stdin:** `cat data.txt | sd batch "[[summary]]"`
4. **JSON stdin:** `echo '{"name":"Alice"}' | sd batch "Hello {% raw %}{{name}}{% endraw %} [[greeting]]"`

How each input becomes one or more items:

| Input | Items | Fields |
|-------|-------|--------|
| Text file or document (`.txt`, `.pdf`, `.docx`, ...) | One | `input`, `content`, `source` (all the text), `filename`, `basename` |
| `.json` | One per object (a list gives several) | The object's keys, plus `filename` if absent |
| `.csv`, `.xlsx` | One per row | The column names |
| URL | Fetched and read as a document | As for text |
| Plain text on stdin | One | `input`, `content`, `filename` (`<stdin>`) |
| JSON on stdin | One per object | The object's keys |

If the prompt references no template variables at all, `{% raw %}{{input}}{% endraw %}` is
prepended to it, so `sd batch -i '*.txt' "[[summary]]"` sees each file's text.

**Output Formats:**
- **JSON** (`.json`) - Structured data
- **CSV** (`.csv`) - Flattened tabular format
- **Excel** (`.xlsx`) - Spreadsheet format
- **Markdown** (`.md`, `.txt`) - Markdown tables
- **stdout** - JSON to stdout if no `-o` is given

The full option list is under [Batch Processing Options](#batch-processing-options).

---

### Other commands

#### `sd explain`

```bash
sd explain PROMPT_FILE [-o OUTPUT]
```

Parses a prompt and shows the external inputs it needs, its sections and
completion slots, the dependencies of each completion, source line numbers,
and any parsing errors. With `-o plan.html` it writes HTML; any other
extension gets plain text.

`sd check` is a hidden, deprecated alias for `sd explain`.

#### `sd graph`

```bash
sd graph PROMPT_FILE [-o diagram.mmd]
```

Prints Mermaid diagram text showing section nodes with their slot names, the
dependencies between sections, and external inputs. It does not render an
image; for a rendered view use `sd explain -o plan.html`.

#### `sd preview`

```bash
sd preview [PROMPT_FILE] [-o out.html] [-r] [-f]
```

Renders a `.sd` file with syntax highlighting and opens it in the browser,
with includes resolved.

| Option | Description |
|--------|-------------|
| `-o`, `--output PATH` | Save HTML to a file instead of opening the browser |
| `-r`, `--raw` | Do not resolve includes |
| `-f`, `--fragment` | Print an HTML fragment (no page wrapper) to stdout; reads stdin if no file is given |

#### `sd flat`

```bash
sd flat PROMPT_FILE [-o flattened.sd]
```

Prints the template with every `{% raw %}{% include %}{% endraw %}` expanded, to stdout or
to the `-o` file.

#### `sd edit`

```bash
sd edit [PATH]
```

Starts a local web server with a browser-based editor for writing and
testing prompts. `PATH` is a file to edit or a directory to open as a
workspace (default: the current directory).

| Option | Description |
|--------|-------------|
| `-p`, `--port INT` | Port (default: first free port from 9000) |
| `--no-browser` | Do not open a browser |
| `-I`, `--include PATH` | Extra include path for actions and types |
| `-r`, `--reload` | Restart the server when files change |
| `-m`, `--models TEXT` | Comma-separated list of models offered in the selector |

#### `sd serve`

```bash
sd serve [OPTIONS]
```

Runs the playground without local file access, for hosting as a web
service. Users supply their own API keys in the settings panel unless a
server-side key is given.

| Option | Description |
|--------|-------------|
| `-p`, `--port INT` | Port (default: `PORT` env var, else 8000) |
| `-h`, `--host TEXT` | Host to bind to (default `0.0.0.0`) |
| `--api-key TEXT` | Server-side API key |
| `-m`, `--models TEXT` | Comma-separated allowed models; falls back to `STRUCKDOWN_ALLOWED_MODELS` |

For production the help suggests gunicorn:
`gunicorn -w 4 -b 0.0.0.0:8000 "struckdown.playground:create_app(remote_mode=True)"`.

#### `sd test`

```bash
sd test [-v]
```

Checks that `LLM_API_KEY` and `LLM_API_BASE` are set, then makes a simple
completion call and an embedding call. If credentials are missing it asks for
them and saves them to a `.env` file. `-v` shows detailed output.

#### `sd setup`

```bash
sd setup [-f] [--skip-skill] [--skip-vscode]
```

Installs the Claude Code skill (the `/struckdown` command, into
`~/.claude/commands/`) and the VS Code extension for `.sd`/`.soak` syntax
highlighting. `-f` overwrites existing installations.

#### `sd install-skill`

```bash
sd install-skill [-f]
```

Copies `struckdown.md` to `~/.claude/commands/` so Claude Code can run it as
`/struckdown`. `-f` overwrites an existing file.

#### `sd install-vscode`

```bash
sd install-vscode [-f]
```

Copies the extension to `~/.vscode/extensions/`. `-f` overwrites an existing
installation.

---

## Global Options

| Option | Description |
|--------|-------------|
| `-V`, `--version` | Show the version and exit |
| `--install-completion` | Install shell completion |
| `--show-completion` | Print the completion script |
| `--help` | Show help for any command |

```bash
sd --help
sd chat --help
sd batch --help
```

**Output:** stdout
**Exit code:** 0

Note that in `sd batch`, `-h` means `--head`, not help; use `--help`.

---

## Batch Processing Options

Summary:

| Option | Description |
|--------|-------------|
| `-i`, `--input TEXT` | Input file, glob or URL; repeatable |
| `-p`, `--prompt PATH` | Read the prompt from a file |
| `-o`, `--output PATH` | Output file, format from extension; repeatable |
| `-k`, `--keep-inputs` | Include input fields in the output |
| `--template PATH` | Jinja2 template applied to non-JSON outputs |
| `-m`, `--model TEXT` | Model name (default: `$DEFAULT_LLM`) |
| `--seed INT` | Random seed, where the model supports one |
| `-j`, `--concurrency INT` | Maximum concurrent API requests (default 20) |
| `-v`, `--verbose` | Debug logging |
| `--debug-api` | Log full API requests and error details |
| `-q`, `--quiet` | Hide the progress bar |
| `-I`, `--include-path PATH` | Extra directory to search for includes; repeatable |
| `-c`, `--compare TEXT` | Compare an input column with a completion; repeatable |
| `--statsonly` | Print only the comparison statistics, as JSON |
| `-h`, `--head INT` | Process only the first N inputs |
| `-e`, `--classification-errors [INT]` | Show misclassified examples |
| `--min-n-compare INT` | Minimum ground-truth count for a category to enter aggregate metrics (default 1) |
| `-t`, `--type PATH` | YAML type definition file or directory; repeatable |
| `--tools PATH` | Python tools file or directory; repeatable |
| `-M`, `--map TEXT` | Map input columns to template variable names; repeatable |

`types/` and `actions/` directories next to the prompt file (or in the
current directory) are also discovered automatically. A `templates/`
directory in the current directory is always on the include path.

### `-i` / `--input TEXT`
Input file, glob pattern or URL. Repeat it for several sources.

```bash
sd batch -i '*.txt' -i 'extra/*.pdf' "[[summary]]"
sd batch -i data.csv "Classify: {% raw %}{{text}}{% endraw %} [[pick:label|yes,no]]"
```

A pattern that matches nothing logs a warning (`No files matched pattern`)
and is skipped. If no `-i` value matches any file or URL, the command prints
`Error: No input files or URLs found` and exits with code 1.

---

### `-o` / `--output PATH`
Output file path. Format is detected from the extension. Repeat `-o` to write
several files from one run.

```bash
sd batch -i '*.txt' "[[name]]" -o results.json   # JSON output
sd batch -i '*.txt' "[[name]]" -o results.csv    # CSV output
sd batch -i '*.txt' "[[name]]" -o results.xlsx   # Excel output
sd batch -i '*.txt' "[[name]]" -o results.md     # Markdown table
sd batch -i '*.txt' "[[name]]" -o results.json -o results.csv
```

**Default:** Outputs JSON to stdout if omitted.

---

### `--template PATH`
A Jinja2 template applied to every non-JSON output. JSON outputs are always
written as standard JSON. Requires at least one non-JSON `-o`.

```bash
sd batch -i '*.txt' -p prompt.sd -o results.json -o report.html --template report.j2
```

The command's own `--help` text shows this example with `-t`, but `-t` is
the short form of `--type`; use `--template`.

---

### `-p` / `--prompt PATH`
Load the prompt from a file instead of inline.

```bash
# prompt.sd contains: "Extract name: [[name]]"
sd batch -i '*.txt' -p prompt.sd -o results.json
```

**Cannot be combined with an inline prompt.** Includes are resolved relative
to the prompt file.

---

### `-k` / `--keep-inputs`
Include input fields in the output.

```bash
sd batch -i '*.txt' "[[summary]]" -k -o results.json
```

**Output for a text file includes:**
```json
{
  "filename": "input.txt",
  "input": "original text...",
  "content": "original text...",
  "source": "original text...",
  "basename": "input",
  "summary": "extracted summary"
}
```

If a slot has the same name as an input column, the input is stored as
`name.data` and the completion as `name.predicted`.

**Default:** Only the slot outputs plus `filename`, for traceability.

Every row also carries `_error`, `_error_class` and `_error_text` columns;
see [Exit Codes](#exit-codes).

---

### `-m` / `--model TEXT`
Model for this run, as `provider:model` or a bare name.

```bash
sd batch -i '*.txt' "[[summary]]" -m openai:gpt-4o
```

**Default:** `$DEFAULT_LLM`.

---

### `-j` / `--concurrency INT`
Maximum number of API requests in flight at once. Default 20.

```bash
sd batch -i '*.txt' "[[summary]]" -j 5
```

---

### `--seed INT`
Random seed passed to the API, for models that support it.

---

### `-h` / `--head INT`
Process only the first N inputs. After the run, the cost summary adds an
estimate for the full input set, scaled by input size for text files or by
item count for tabular data.

```bash
sd batch -i data.csv -p prompt.sd -h 10
```

---

### `-M` / `--map TEXT`
Copy an input column to a template variable name. `target=source` makes
`{% raw %}{{target}}{% endraw %}` hold the `source` column; a bare `name` is a same-name copy.

```bash
sd batch -i data.csv -p prompt.sd -M input=content -o results.json
```

---

### `-I` / `--include-path PATH`, `-t` / `--type PATH`, `--tools PATH`
Extra directories for includes, extra YAML type definitions, and extra
Python actions. All three can be repeated.

```bash
sd batch -i '*.txt' -p prompt.sd -I ./includes -t ./types --tools ./actions
```

---

### Comparing against ground truth: `-c`, `--statsonly`, `-e`, `--min-n-compare`

When the input already holds a correct answer (a labelled CSV, for example),
`-c` compares it with a completion and prints agreement statistics.

- `-c label` compares the `label` column with the `label` slot.
- `-c label=predicted` compares the `label` column with the `predicted` slot.
- `-c all` pairs every slot with the input column of the same name.

Statistics go to stdout when there is no `-o`, otherwise to stderr.

```bash
sd batch -i labelled.csv -p classify.sd -k -c label -o results.csv
```

`--statsonly` prints only the statistics, as JSON, and writes no results. It
requires at least one `-c`.

`-e` shows examples of misclassifications: `-e` alone shows all of them,
`-e 3` shows up to three per error type.

`--min-n-compare N` excludes categories with fewer than N ground-truth cases
from the macro and weighted F1 figures (default 1).

---

### `-q` / `--quiet`
Hide the progress bar.

```bash
sd batch -i '*.txt' "[[name]]" -o results.json --quiet
```

**Behaviour:**
- No progress bar
- The cost summary and the list of failed items are still printed to stderr
- `--verbose` output is still printed if `--verbose` is also given

**Use case:** Scripting, cron jobs, CI pipelines.

---

### `-v` / `--verbose`
Enable debug logging to stderr. Unlike `sd chat`, this is a plain flag, not
counted.

```bash
sd batch -i '*.txt' "[[name]]" --verbose 2> debug.log
```

**Output includes:**
- Debug-level log messages from struckdown
- The types and actions loaded or discovered
- One line per processed item
- A traceback for each failed item

**Destination:** stderr

---

### `--debug-api`
Log full API requests and error details.

---

## Progress Bars

`sd batch` shows a progress bar by default during processing:

```
⠋ 47/100 completions ━━━━━━━━━━━━━━━━━━━━━━ 47% 0:00:23
```

The total is an estimate: number of inputs times number of slots in the
prompt.

**Display includes:**
- Spinner animation
- Completions so far / estimated total
- Progress bar
- Percentage
- Estimated time remaining

### Automatic Behaviour

**Shown when** stderr is a terminal and `--quiet` is not set.

**Hidden when:**
- stderr is redirected: `sd batch ... 2> errors.log`
- `--quiet` is used

Piping stdout (`sd batch ... | jq .`) does not hide it, because the bar is
drawn on stderr.

### Manual Control

```bash
# Shown (default in a terminal)
sd batch -i '*.txt' "[[name]]" -o out.json

# Hidden
sd batch -i '*.txt' "[[name]]" -o out.json --quiet

# Discard all diagnostics
sd batch -i '*.txt' "[[name]]" -o out.json 2>/dev/null
```

---

## Output Streams

### stdout (File Descriptor 1)
Machine-readable results, for piping or capture.

**Contains:**
- Results (JSON) when no `-o` is given
- Comparison statistics when `-c` is used without `-o`
- `--version` and `--help` output

**Example:**
```bash
sd batch -i '*.txt' "[[name]]" > results.json   # Only results captured
```

---

### stderr (File Descriptor 2)
Human-readable status, errors and warnings.

**Contains:**
- Progress bars
- The cost summary (always printed at the end of a batch)
- The list of failed items
- Error messages and warnings
- Verbose debug logs

**Example:**
```bash
sd batch -i '*.txt' "[[name]]" 2> errors.log    # Only diagnostics captured
```

---

### Exit Codes

| Code | Meaning | Examples |
|------|---------|----------|
| 0 | Run finished | All items processed, including runs where some items failed |
| 1 | Error before or instead of processing | No input found, prompt file missing, both inline and `-p` prompt given, `--statsonly` without `-c` |
| 2 | Usage error | Unknown option, too many positional arguments |

A failure on one item does not stop the batch or change the exit code. The
item gets an output row with `_error` set to `true` and the exception in
`_error_class` and `_error_text`, and stderr ends with
`Completed with N error(s):` and one line per failure. Check the `_error`
column, not only the exit code.

**Example:**
```bash
sd batch --invalid-flag
# Exit code: 2 (usage error)

sd batch missing-file.txt "[[x]]"
# Exit code: 2 (two positional arguments)

sd batch -i missing-file.txt "[[x]]"
# Exit code: 1 ("No input files or URLs found")

sd batch -i '*.txt' "[[x]]" -o out.json
# Exit code: 0
```

---

## Examples

### Basic Batch Processing

**Extract names from text files:**
```bash
sd batch -i 'documents/*.txt' "Extract person's name: [[name]]" -o names.json
```

**Output (trimmed; each row also has the `_error` columns):**
```json
[
  {"filename": "documents/doc1.txt", "name": "Alice"},
  {"filename": "documents/doc2.txt", "name": "Bob"}
]
```

---

### Piping and Chaining

**Chain multiple extraction steps:**
```bash
sd batch -i '*.txt' "Purpose: [[purpose]] Name: [[name]]" | \
  sd batch "{% raw %}{{name}}: {{purpose}}{% endraw %}. Amazon equivalent? [[product]]" -k
```

**Process and filter with jq:**
```bash
sd batch -i '*.txt' "Price: [[number:price]]" | jq '.[] | select(.price > 100)'
```

---

### Quiet Mode for Scripts

**Silent execution, then check for failed items:**
```bash
#!/bin/bash
sd batch -i 'data/*.txt' "[[summary]]" -o results.json --quiet 2> errors.log || exit 1

if jq -e 'any(.[]; ._error)' results.json > /dev/null; then
  echo "Some items failed, check errors.log"
  exit 1
fi
```

---

### Trying a Prompt on a Sample

**Run the first ten rows and see the projected cost of the full run:**
```bash
sd batch -i data.csv -p prompt.sd -h 10 -o sample.csv
```

---

### Reading from Stdin

**Plain text:**
```bash
echo "Hello world" | sd batch "Translate to Spanish: [[translation]]"
```

**JSON input:**
```bash
echo '[{"name":"Alice"},{"name":"Bob"}]' | \
  sd batch "Hello {% raw %}{{name}}{% endraw %}! [[greeting]]"
```

**From file:**
```bash
cat data.txt | sd batch "Summarise: [[summary]]" -o summary.json
```

---

### Multiple Output Formats

**CSV for spreadsheet import:**
```bash
sd batch -i '*.txt' "Name: [[name]] Age: [[int:age]]" -o people.csv
```

**Excel for reports:**
```bash
sd batch -i '*.txt' "Product: [[product]] Price: [[number:price]]" -o report.xlsx
```

**Markdown for documentation:**
```bash
sd batch -i '*.txt' "Feature: [[feature]] Status: [[pick:status|done,wip,todo]]" -o status.md
```

---

### Debugging Failures

```bash
sd batch -i failing-input.txt "[[x]]" --verbose
sd batch -i failing-input.txt "[[x]]" --debug-api
```

---

## Tips

### Performance
- **Caching:** repeated calls with the same prompt use cached results (see `$STRUCKDOWN_CACHE`)
- **One command, many files:** `sd batch` runs items concurrently (`-j`); a shell loop does not
- **Sample first:** `-h 10` shows the cost of a small run and an estimate for the whole set

### Composability
- **Pipe results:** `sd batch ... | sd batch ...` for multi-stage extraction
- **Use jq:** post-process JSON output with `jq`
- **Redirect:** capture results and diagnostics separately

### Debugging
- **Start with chat:** test prompts with `sd chat` before batch processing
- **Inspect structure:** `sd explain prompt.sd` shows inputs, slots and dependencies
- **Check `_error`:** failed items do not change the exit code

---

## Related Documentation

- [Getting Started](../tutorials/getting-started.md) - Quick start guide
- [Template Syntax](../explanation/template-syntax.md) - Struckdown template syntax
- [Model Overrides](../how-to/model-overrides.md) - Per-slot LLM configuration
- [Number Extraction](../how-to/number-extraction.md) - Numeric validation

---

## Migration from `chatter`

The legacy `chatter` CLI was removed in v0.1.6.

**Old command:**
```bash
chatter run "Tell me a joke: [[joke]]"
```

**New command:**
```bash
sd chat "Tell me a joke: [[joke]]"
```
