# Struckdown Playground Implementation Plan

## Executive Summary

This document outlines the implementation of `sd edit`, a web-based playground for editing and testing struckdown prompts. The tool provides:

- Real-time prompt editing with syntax highlighting and live parse error display
- Input specification via individual fields or batch xlsx upload
- Live output rendering with incremental batch results
- Works both locally (`sd edit myfile.sd`) and as a remote hosted service

---

## Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Backend | Flask | Lightweight, quick startup, suitable for CLI and remote hosting |
| CSS Framework | Bootstrap 5.3 | Familiar, comprehensive component library |
| Partial Updates | HTMX 2.x | Server-rendered HTML, no build step |
| Syntax Editor | CodeMirror 6 | Modern, extensible syntax highlighting |
| Table Display | DataTables | Feature-rich table with sorting, filtering, export |
| Icons | Bootstrap Icons | Consistent with Bootstrap |
| Streaming | Server-Sent Events (SSE) | For incremental batch row updates |

---

## Code Organisation

```
struckdown/
├── playground/
│   ├── __init__.py           # Package init, exports create_app, find_available_port
│   ├── core.py               # Framework-agnostic logic
│   │   ├── extract_required_inputs()
│   │   ├── extract_slot_names()
│   │   ├── validate_syntax()
│   │   ├── run_single()
│   │   ├── run_batch_streaming()    # Generator yielding rows as completed
│   │   ├── encode_state() / decode_state()
│   │   └── load_xlsx_preview()
│   │
│   ├── flask_app.py          # Flask application factory
│   │   ├── create_app(prompt_file=None, include_paths=None, remote_mode=False)
│   │   ├── Routes:
│   │   │   ├── GET  /                       # Main editor page
│   │   │   ├── GET  /e/<encoded_state>      # Load from URL-encoded state (remote)
│   │   │   ├── POST /api/save               # Save syntax to file (local only)
│   │   │   ├── POST /api/run                # Execute single and return outputs
│   │   │   ├── POST /api/run-batch          # Start batch, return task_id
│   │   │   ├── GET  /api/batch-stream/<id>  # SSE stream of batch results
│   │   │   ├── POST /api/analyse            # Analyse template, return inputs/slots/errors
│   │   │   ├── POST /api/upload             # Upload xlsx, return file_id + preview
│   │   │   └── GET  /api/download/<id>      # Download completed batch as xlsx
│   │
│   ├── templates/
│   │   ├── base.html                 # Bootstrap, HTMX, DataTables CDN links
│   │   ├── editor.html               # Main editor page
│   │   └── partials/
│   │       ├── outputs_single.html   # Single-mode output cards
│   │       ├── outputs_batch.html    # Batch-mode DataTable
│   │       ├── inputs_panel.html     # Dynamic inputs form (collapsible)
│   │       ├── settings_panel.html   # Model, mode selection (collapsible)
│   │       └── error_banner.html     # Parse error display
│   │
│   └── static/
│       ├── js/
│       │   ├── editor.js             # CodeMirror setup, debounced analysis
│       │   ├── batch-stream.js       # SSE handler for incremental updates
│       │   └── panels.js             # Collapse/expand panel handling
│       └── css/
│           ├── playground.css        # Layout, panel styles
│           └── codemirror-struckdown.css  # Syntax highlighting colours
```

---

## UI Design

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STRUCKDOWN PLAYGROUND          [myfile.sd]           [⚙ Settings] [📥 Inputs] │
├─────────────────────────────────────────────────────────────────────────────┤
│  ⚠ Parse error at line 5: unexpected ']'                            [dismiss] │  ← Error banner (hidden when valid)
├───────────────────────────────────┬─────────────────────────────────────────┤
│                                   │                                         │
│  EDITOR                           │  OUTPUTS                                │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────────────────┐│
│  │ Tell me a joke about        │  │  │ ☐ joke                              ││
│  │ {{topic}}                   │  │  │   Why did the chicken cross...      ││
│  │                             │  │  ├─────────────────────────────────────┤│
│  │ [[joke]]                    │  │  │ ☐ rating                            ││
│  │                             │  │  │   7/10                              ││
│  │ Rate it: [[!number:rating]] │  │  └─────────────────────────────────────┘│
│  │                             │  │                                         │
│  └─────────────────────────────┘  │                                         │
│                                   │                                         │
│  [💾 Save & Run]                  │  Cost: $0.003 | 245 tokens              │
│                                   │                                         │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

### Settings Panel (Collapsed by Default)

Bootstrap collapse component, toggled by header button:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ▼ SETTINGS                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Model name                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ openai/gpt-4o                                                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  LLM API Key (remote mode only, stored in browser)                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ sk-...                                                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Mode:  ○ Single input   ● Batch file (xlsx/csv)                           │
│                                                                             │
│  ── Batch File ──                                                           │
│  [📁 Upload xlsx/csv]                                                       │
│  └─ data.xlsx (15 rows) ✓                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Inputs Panel (Collapsed by Default)

Separate from settings; shows dynamic fields based on syntax:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ▼ INPUTS                                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  topic                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ programming                                                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  author                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Shakespeare                                                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  (These fields are derived from {{variables}} in your prompt)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Batch Mode Output (DataTable)

When batch mode selected, outputs panel shows DataTable with both inputs and outputs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BATCH RESULTS (12/15 complete)                          [⬇ Download xlsx]  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Show columns: [✓] topic  [✓] author  [✓] joke  [✓] rating                  │
├──────────┬────────────┬─────────────────────────────┬───────────────────────┤
│  topic ▼ │ author     │ joke                        │ rating                │
├──────────┼────────────┼─────────────────────────────┼───────────────────────┤
│  cats    │ Shakespeare│ Why do cats make terrible...│ 6/10                  │
│  dogs    │ Hemingway  │ What do you call a dog...   │ 8/10                  │
│  birds   │ Austen     │ ⏳ Processing...            │                       │
│  fish    │ Dickens    │ (pending)                   │                       │
│  ...     │ ...        │ ...                         │ ...                   │
└──────────┴────────────┴─────────────────────────────┴───────────────────────┘
```

- Input columns shown by default (needed to identify rows)
- Output columns shown by default
- All columns toggleable via checkboxes
- Rows update incrementally via SSE as they complete
- Download button exports completed results as xlsx

---

## Data Flow

### Single Mode Execution

```
User edits prompt
    ↓
[Save & Run] clicked
    ↓
POST /api/save { syntax: "..." }     →  Write to file (local mode only)
    ↓
POST /api/run {
    syntax: "...",
    inputs: { topic: "programming" },
    model: "openai/gpt-4o",
    api_key: "sk-..." (remote mode only)
}
    ↓
Server: core.run_single(syntax, inputs, model, credentials)
    ↓
Returns: {
    outputs: { joke: "...", rating: 7 },
    cost: { input_tokens: 100, output_tokens: 50, cost: 0.003 },
    error: null
}
    ↓
HTMX swaps #outputs-panel with rendered partials/outputs_single.html
```

### Batch Mode Execution (Incremental)

```
User uploads xlsx
    ↓
POST /api/upload (multipart form)
    ↓
Server stores file in temp dir, returns:
    { file_id: "abc123", rows: 15, columns: ["topic", "author"], preview: [...] }
    ↓
[Save & Run] clicked
    ↓
POST /api/run-batch {
    syntax: "...",
    file_id: "abc123",
    model: "openai/gpt-4o",
    api_key: "sk-..."
}
    ↓
Server creates batch task, returns: { task_id: "xyz789" }
    ↓
Client opens SSE connection: GET /api/batch-stream/xyz789
    ↓
Server processes rows concurrently, yields events:
    event: row
    data: { index: 0, inputs: {...}, outputs: {...}, status: "complete" }

    event: row
    data: { index: 1, inputs: {...}, outputs: {...}, status: "complete" }

    event: progress
    data: { completed: 2, total: 15 }

    ... (continues until all rows done)

    event: done
    data: { download_id: "xyz789", cost: {...} }
    ↓
Client JS updates DataTable rows incrementally as events arrive
```

### Real-time Syntax Analysis

```
User types in editor
    ↓ (debounced 500ms)
POST /api/analyse { syntax: "..." }
    ↓
Server: core.validate_syntax(syntax) + core.extract_required_inputs(syntax)
    ↓
Returns: {
    valid: false,
    error: { line: 5, column: 12, message: "unexpected ']'" },
    inputs_required: ["topic", "author"],
    slots_defined: ["joke", "rating"]
}
    ↓
If error: Show error banner above editor
If inputs changed: HTMX fetches partials/inputs_panel.html (preserving values)
```

---

## Local vs Remote Mode

| Aspect | Local (`sd edit`) | Remote (hosted service) |
|--------|-------------------|-------------------------|
| File editing | Reads/writes local .sd file | No file access |
| State persistence | File on disk | URL-encoded state |
| API credentials | From environment | User provides in UI, stored in localStorage |
| Custom actions | Loaded from `-I` paths and cwd | Only built-in safe actions |
| Batch file storage | Temp directory | Temp directory with TTL cleanup |
| URL structure | `http://localhost:9000/` | `https://playground.example.com/e/{encoded}` |

### URL-Encoded State (Remote Mode)

```python
import base64
import json
import zlib

def encode_state(syntax: str, model: str, inputs: dict) -> str:
    """Encode editor state to URL-safe string."""
    state = {"s": syntax, "m": model, "i": inputs}
    json_bytes = json.dumps(state, separators=(',', ':')).encode('utf-8')
    compressed = zlib.compress(json_bytes, level=9)
    return base64.urlsafe_b64encode(compressed).decode('ascii')

def decode_state(encoded: str) -> dict:
    """Decode URL state back to components."""
    compressed = base64.urlsafe_b64decode(encoded)
    json_bytes = zlib.decompress(compressed)
    return json.loads(json_bytes)
```

When user saves in remote mode:
1. Encode current state (syntax, model, inputs)
2. Update browser URL to `/e/{encoded}`
3. User can copy/share URL to restore exact state

---

## Server Architecture

### Flask App Configuration

```python
def create_app(
    prompt_file: Path = None,      # Local mode: file to edit
    include_paths: List[Path] = None,
    remote_mode: bool = False,     # True for hosted service
):
    app = Flask(__name__)

    # Store config in app
    app.config['PROMPT_FILE'] = prompt_file
    app.config['INCLUDE_PATHS'] = include_paths or []
    app.config['REMOTE_MODE'] = remote_mode
    app.config['UPLOAD_FOLDER'] = Path(tempfile.gettempdir()) / 'struckdown-uploads'

    # Load custom actions (same as sd chat)
    if not remote_mode:
        from struckdown.actions import discover_actions, load_actions
        from struckdown.type_loader import discover_yaml_types, load_yaml_types

        action_files = discover_actions(include_paths or [Path.cwd()])
        load_actions(action_files)

        type_files = discover_yaml_types(include_paths or [Path.cwd()])
        load_yaml_types(type_files)

    return app
```

### Batch Processing with SSE

```python
from flask import Response, stream_with_context
import anyio

@app.route('/api/batch-stream/<task_id>')
def batch_stream(task_id):
    """Server-Sent Events stream for batch progress."""

    def generate():
        # Get task from storage
        task = get_task(task_id)

        async def process():
            async for result in core.run_batch_streaming(
                syntax=task['syntax'],
                rows=task['rows'],
                model=task['model'],
                credentials=task.get('credentials'),
            ):
                yield f"event: row\ndata: {json.dumps(result)}\n\n"

            yield f"event: done\ndata: {json.dumps({'task_id': task_id})}\n\n"

        # Run async generator in sync context
        for event in anyio.from_thread.run_sync(process):
            yield event

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache'}
    )
```

### Port Selection (Local Mode)

```python
import socket

def find_available_port(start=9000, max_attempts=100):
    """Find an available port starting from 9000."""
    for port in range(start, start + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available port found in range 9000-9099")
```

---

## CLI Integration

```python
# In sd_cli.py

@app.command()
def edit(
    file: Optional[Path] = typer.Argument(
        None,
        help="Struckdown file to edit (default: untitled.sd in cwd)"
    ),
    port: Optional[int] = typer.Option(
        None, "-p", "--port",
        help="Port to run server on (default: auto-select from 9000+)"
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser",
        help="Don't open browser automatically"
    ),
    include: List[Path] = typer.Option(
        [], "-I", "--include",
        help="Additional include paths for actions and types"
    ),
):
    """Open interactive playground for editing struckdown prompts."""
    from struckdown.playground import create_app, find_available_port

    # Resolve file path
    if file is None:
        file = Path.cwd() / "untitled.sd"

    if not file.exists():
        file.write_text("# Your struckdown prompt\n\n[[response]]\n")
        console.print(f"Created {file}")

    # Find port
    if port is None:
        port = find_available_port()

    # Create app
    flask_app = create_app(
        prompt_file=file.resolve(),
        include_paths=[p.resolve() for p in include],
        remote_mode=False,
    )

    url = f"http://localhost:{port}"
    console.print(f"Playground: {url}")
    console.print(f"Editing: {file}")
    console.print("Press Ctrl+C to stop")

    if not no_browser:
        import webbrowser
        import threading
        # Delay browser open slightly to let server start
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    flask_app.run(host='localhost', port=port, debug=False, threaded=True)
```

---

## Core Logic Module

```python
# playground/core.py

from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
from struckdown import chatter_async, extract_jinja_variables, LLM, LLMCredentials
from struckdown.parsing import find_slots_with_positions, extract_slot_key, parser
from lark.exceptions import UnexpectedToken, UnexpectedCharacters


def extract_required_inputs(syntax: str) -> Dict[str, List[str]]:
    """
    Analyse template to determine required inputs.

    Returns dict with:
        inputs_required: {{vars}} not filled by [[slots]]
        slots_defined: [[slots]] that will be created
    """
    jinja_vars = extract_jinja_variables(syntax)
    slots = find_slots_with_positions(syntax)
    slot_names = {extract_slot_key(s[0]) for s in slots}

    # Inputs are vars not defined by slots
    inputs_required = sorted(jinja_vars - slot_names)

    return {
        "inputs_required": inputs_required,
        "slots_defined": sorted(slot_names)
    }


def validate_syntax(syntax: str) -> Dict:
    """
    Validate struckdown syntax.

    Returns dict with:
        valid: bool
        error: { line, column, message } or None
    """
    try:
        # Attempt to parse
        p = parser()
        p.parse(syntax)
        return {"valid": True, "error": None}
    except (UnexpectedToken, UnexpectedCharacters) as e:
        return {
            "valid": False,
            "error": {
                "line": e.line,
                "column": e.column,
                "message": str(e)
            }
        }
    except Exception as e:
        return {
            "valid": False,
            "error": {
                "line": 1,
                "column": 1,
                "message": str(e)
            }
        }


async def run_single(
    syntax: str,
    inputs: Dict[str, Any],
    model_name: str = None,
    credentials: LLMCredentials = None,
    include_paths: List[Path] = None,
) -> Dict:
    """Execute template with given inputs."""
    try:
        model = LLM(model=model_name) if model_name else LLM()
        result = await chatter_async(
            syntax,
            model=model,
            credentials=credentials,
            context=inputs,
            include_paths=include_paths,
        )

        return {
            "outputs": result.outputs,
            "cost": result.cost.model_dump() if result.cost else None,
            "error": None
        }
    except Exception as e:
        return {
            "outputs": {},
            "cost": None,
            "error": str(e)
        }


async def run_batch_streaming(
    syntax: str,
    rows: List[Dict[str, Any]],
    model_name: str = None,
    credentials: LLMCredentials = None,
    include_paths: List[Path] = None,
    max_concurrent: int = 10,
) -> AsyncGenerator[Dict, None]:
    """
    Execute template for each row, yielding results as they complete.

    Yields dicts with:
        index: row index
        inputs: original input values
        outputs: slot values
        status: "complete" | "error"
        error: error message if status is "error"
    """
    import anyio

    model = LLM(model=model_name) if model_name else LLM()
    semaphore = anyio.Semaphore(max_concurrent)

    async def process_row(index: int, row: Dict) -> Dict:
        async with semaphore:
            try:
                result = await chatter_async(
                    syntax,
                    model=model,
                    credentials=credentials,
                    context=row,
                    include_paths=include_paths,
                )
                return {
                    "index": index,
                    "inputs": row,
                    "outputs": result.outputs,
                    "status": "complete",
                    "error": None
                }
            except Exception as e:
                return {
                    "index": index,
                    "inputs": row,
                    "outputs": {},
                    "status": "error",
                    "error": str(e)
                }

    # Process all rows concurrently, yield as they complete
    async with anyio.create_task_group() as tg:
        results_queue = []

        for i, row in enumerate(rows):
            tg.start_soon(lambda idx=i, r=row: results_queue.append(process_row(idx, r)))

        # Note: This is simplified; real implementation would use
        # anyio memory channels for proper streaming
        for result in results_queue:
            yield await result
```

---

## Security (Remote Mode)

1. **Action Whitelist:** Only allow safe built-in actions
   - Allowed: `@set`, `@break`, `@timestamp`
   - Blocked: `@fetch`, `@search` (could be enabled with rate limiting)

2. **Rate Limiting:** Flask-Limiter
   - 10 requests/minute for `/api/run`
   - 5 requests/minute for `/api/run-batch`

3. **API Key Handling:**
   - Never logged server-side
   - Stored only in browser localStorage
   - Sent per-request, not stored in session

4. **Input Sanitisation:**
   - Syntax length limit (e.g., 50KB)
   - Row count limit for batch (e.g., 100 rows)

5. **Execution Timeout:** 60 seconds per LLM call

6. **File Upload:**
   - Max 5MB xlsx files
   - Temp files deleted after 1 hour

---

## Test Plan

### Unit Tests (pytest)

```python
# tests/test_playground_core.py

def test_extract_required_inputs_basic():
    """Variables not filled by slots are required inputs."""
    syntax = "Tell me about {{topic}}\n[[response]]"
    result = extract_required_inputs(syntax)
    assert result["inputs_required"] == ["topic"]
    assert result["slots_defined"] == ["response"]

def test_extract_required_inputs_slot_fills_var():
    """Variables filled by earlier slots are not required."""
    syntax = "[[topic]]\nExpand on {{topic}}\n[[expansion]]"
    result = extract_required_inputs(syntax)
    assert result["inputs_required"] == []
    assert "topic" in result["slots_defined"]

def test_validate_syntax_valid():
    """Valid syntax returns no error."""
    result = validate_syntax("Hello [[greeting]]")
    assert result["valid"] is True
    assert result["error"] is None

def test_validate_syntax_invalid():
    """Invalid syntax returns error with line info."""
    result = validate_syntax("Hello [[greeting")  # Missing ]]
    assert result["valid"] is False
    assert result["error"]["line"] >= 1

def test_encode_decode_state_roundtrip():
    """State encoding/decoding preserves data."""
    state = {"s": "[[test]]", "m": "gpt-4", "i": {"x": "y"}}
    encoded = encode_state(**state)
    decoded = decode_state(encoded)
    assert decoded == state
```

### Integration Tests (Flask test client)

```python
# tests/test_playground_flask.py

import pytest
from struckdown.playground import create_app

@pytest.fixture
def client(tmp_path):
    prompt_file = tmp_path / "test.sd"
    prompt_file.write_text("[[greeting]]")
    app = create_app(prompt_file=prompt_file, remote_mode=False)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page_loads(client):
    """Home page renders editor."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'CodeMirror' in response.data

def test_save_updates_file(client, tmp_path):
    """Save endpoint writes to file."""
    response = client.post('/api/save', json={'syntax': 'New [[slot]]'})
    assert response.status_code == 200
    assert (tmp_path / "test.sd").read_text() == 'New [[slot]]'

def test_analyse_returns_inputs(client):
    """Analyse endpoint extracts required inputs."""
    response = client.post('/api/analyse', json={
        'syntax': '{{name}} says [[greeting]]'
    })
    data = response.get_json()
    assert data['inputs_required'] == ['name']
    assert data['slots_defined'] == ['greeting']

def test_analyse_returns_parse_error(client):
    """Analyse endpoint returns parse errors."""
    response = client.post('/api/analyse', json={
        'syntax': '[[broken'
    })
    data = response.get_json()
    assert data['valid'] is False
    assert 'error' in data
```

### E2E Tests (Playwright)

```python
# tests/test_playground_e2e.py

import pytest

@pytest.mark.playwright
def test_editor_loads_file_content(page, playground_server):
    """Editor shows content from file."""
    page.goto(playground_server.url)
    editor = page.locator('.cm-content')
    assert '[[greeting]]' in editor.inner_text()

@pytest.mark.playwright
def test_parse_error_shown_in_banner(page, playground_server):
    """Parse errors appear in error banner."""
    page.goto(playground_server.url)
    page.locator('.cm-content').fill('[[broken')
    page.wait_for_timeout(600)  # Wait for debounce
    assert page.locator('.error-banner').is_visible()

@pytest.mark.playwright
def test_inputs_panel_updates_dynamically(page, playground_server):
    """Adding {{var}} creates input field."""
    page.goto(playground_server.url)
    page.locator('.cm-content').fill('{{topic}} [[response]]')
    page.wait_for_timeout(600)
    page.click('[data-bs-target="#inputs-panel"]')  # Open inputs panel
    assert page.locator('input[name="topic"]').is_visible()

@pytest.mark.playwright
def test_save_and_run_shows_output(page, playground_server, mock_llm):
    """Save & Run displays outputs."""
    page.goto(playground_server.url)
    page.click('button:has-text("Save & Run")')
    page.wait_for_selector('.output-card')
    assert page.locator('.output-card').count() > 0

@pytest.mark.playwright
def test_batch_upload_shows_table(page, playground_server, sample_xlsx):
    """Uploading xlsx shows DataTable."""
    page.goto(playground_server.url)
    page.click('[data-bs-target="#settings-panel"]')
    page.locator('input[type="file"]').set_input_files(sample_xlsx)
    assert page.locator('.dataTables_wrapper').is_visible()

@pytest.mark.playwright
def test_batch_rows_update_incrementally(page, playground_server, sample_xlsx, mock_llm):
    """Batch rows appear as they complete."""
    page.goto(playground_server.url)
    page.click('[data-bs-target="#settings-panel"]')
    page.locator('input[type="file"]').set_input_files(sample_xlsx)
    page.click('button:has-text("Save & Run")')
    # First row should appear before all complete
    page.wait_for_selector('td:has-text("complete")', timeout=5000)
```

---

## Implementation Tasks

1. **Package Structure**
   - Create `struckdown/playground/__init__.py`
   - Create `struckdown/playground/core.py` with extraction/validation functions
   - Create `struckdown/playground/flask_app.py` with routes

2. **Templates & Static**
   - Create `templates/base.html` with Bootstrap, HTMX, DataTables
   - Create `templates/editor.html` main layout
   - Create partial templates for outputs, inputs, settings
   - Create `static/js/editor.js` for CodeMirror setup
   - Create `static/js/batch-stream.js` for SSE handling
   - Create `static/css/playground.css`
   - Create `static/css/codemirror-struckdown.css`

3. **API Endpoints**
   - `GET /` - render editor page
   - `GET /e/<encoded>` - render from URL state
   - `POST /api/save` - save syntax to file
   - `POST /api/analyse` - validate and extract inputs
   - `POST /api/run` - execute single
   - `POST /api/upload` - upload xlsx
   - `POST /api/run-batch` - start batch task
   - `GET /api/batch-stream/<id>` - SSE stream
   - `GET /api/download/<id>` - download xlsx

4. **CLI Command**
   - Add `edit` command to `sd_cli.py`
   - Port selection logic
   - Browser auto-open

5. **CodeMirror Integration**
   - Struckdown syntax highlighting mode
   - `[[slot]]` highlighting (distinct colour)
   - `{{var}}` highlighting (distinct colour)
   - `[[@action]]` highlighting

6. **DataTables Integration**
   - Column visibility toggles
   - Input/output column distinction
   - Export to xlsx button
   - Incremental row updates via SSE

7. **Error Handling**
   - Parse error banner display
   - Execution error display in outputs
   - Network error handling

8. **Responsive Design**
   - Collapsible panels
   - Resizable columns (nice to have)
   - Mobile-friendly stacking

9. **Tests**
   - Unit tests for core functions
   - Flask integration tests
   - Playwright E2E tests

---

## Appendix: Existing Code Reference

### Key Functions to Reuse

From `struckdown/__init__.py`:
- `chatter_async()` - Main execution function
- `extract_jinja_variables()` - Get {{var}} references

From `struckdown/parsing.py`:
- `find_slots_with_positions()` - Get [[slot]] locations
- `extract_slot_key()` - Parse slot body for variable name
- `parser()` - Get Lark parser instance for validation

From `struckdown/sd_cli.py`:
- `batch_async()` - Batch processing logic (adapt for streaming)
- `_resolve_template_includes()` - Handle includes
- Action/type discovery patterns
