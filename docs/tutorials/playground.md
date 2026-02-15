---
layout: default
title: Playground
parent: Tutorials
nav_order: 3
---

# Struckdown Playground

The Struckdown Playground is a web-based editor for creating and testing struckdown prompts interactively. 

It provides real-time syntax validation, dynamic input fields, batch processing, and instant output rendering.


## Quick Start (Local Mode)

```bash
# Edit a new or existing file
sd edit myfile.sd

# Auto-creates untitled.sd if no file specified
sd edit

# Use a specific port
sd edit -p 8080

# Include custom actions/types from a directory
sd edit -I ./custom-actions myfile.sd

# Don't auto-open browser
sd edit --no-browser
```

The playground opens in your default browser at `http://localhost:9000` (or next available port).

## Features

### Editor
- **Syntax validation** -- Real-time parsing with error messages showing line/column
- **Debounced analysis** -- 500ms delay before re-analysing to avoid excessive requests
- **Keyboard shortcuts** -- Ctrl+S / Cmd+S triggers Save & Run

### Inputs Panel
- **Dynamic input fields** -- Automatically detects `{{variables}}` in your prompt
- **Single/Batch toggle** -- Switch between individual inputs and batch file upload
- **Persistent values** -- Input values saved to localStorage, restored on reload
- **Session isolation** -- Each browser tab gets a unique session ID in the URL hash

### Outputs Panel
- **Slot-based display** -- Each `[[slot]]` shown as a card in template order
- **Pin to top** -- Pin important outputs to keep them visible
- **Cost tracking** -- Shows token count and estimated cost
- **Copy to clipboard** -- One-click copy for any output

### Batch Mode
- **File upload** -- Upload xlsx or csv files
- **Incremental results** -- Rows complete and display as they finish (SSE streaming)
- **Download results** -- Export completed batch as xlsx

## CLI Options

```
sd edit [FILE] [OPTIONS]

Arguments:
  FILE    Struckdown file to edit (default: untitled.sd)

Options:
  -p, --port INTEGER       Port to run server on (default: auto 9000+)
  --no-browser             Don't open browser automatically
  -I, --include PATH       Additional include paths for actions and types
  --help                   Show help message
```

## Local vs Remote Mode

| Aspect | Local Mode | Remote Mode |
|--------|------------|-------------|
| File access | Reads/writes local .sd file | No file access |
| State | File on disk | URL-encoded state |
| API credentials | From environment (`LLM_API_KEY`) | User enters in UI (or `--api-key` flag) |
| Custom actions | Loaded from -I paths and cwd | Built-in actions only |
| URL | `http://localhost:9000/` | `https://your-domain.com/e/{encoded}` |

---

## Deploying Remote Mode

Remote mode allows hosting the playground as a public web service where users can create and share prompts without local file access.

**Important:** In remote mode, API keys are NOT read from environment variables by default. This ensures that server credentials are never accidentally exposed. Users must either:
1. Enter their own API key in the settings panel, or
2. The server operator provides a key via `--api-key`

### Quick Start (Development)

```bash
# Start remote mode server (users must provide their own API keys)
sd serve

# Provide a server-side API key (for internal deployments)
sd serve --api-key=$MY_API_KEY

# Use specific port
sd serve -p 9000

# Bind to localhost only (for reverse proxy)
sd serve -h 127.0.0.1 -p 8000
```

### Basic Deployment (Python)

```python
# server.py
import os
from struckdown.playground import create_app

app = create_app(
    prompt_file=None,       # No local file
    include_paths=[],       # No custom actions
    remote_mode=True,       # Enable remote mode
    # Optional: provide server-side API key
    # server_api_key=os.environ.get("LLM_API_KEY"),
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

### Production Deployment with Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:8000 "server:app"
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install struckdown
RUN pip install struckdown gunicorn

# Create server script
COPY server.py .

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "server:app"]
```

```yaml
# docker-compose.yml
version: "3.8"
services:
  playground:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
```

### URL Sharing (Remote Mode)

In remote mode, users can share their prompts via URL:

1. Create a prompt in the editor
2. The state is encoded in the URL path: `/e/{base64_compressed_state}`
3. Share the URL -- recipients see the exact same prompt and inputs

The encoding uses zlib compression + base64url, keeping URLs reasonably short.

---

## Security Considerations

### For Remote/Public Deployments

**1. Reverse Proxy with HTTPS**

Always deploy behind a reverse proxy (nginx, Caddy) with TLS:

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name playground.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support for batch streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

**2. Rate Limiting**

Add rate limiting to prevent abuse. Example with Flask-Limiter:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = create_app(remote_mode=True)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Stricter limits for execution endpoints
@limiter.limit("10 per minute")
@app.route("/api/run", methods=["POST"])
def run_limited():
    # The original route handles the logic
    pass
```

**3. Request Size Limits**

Configure maximum request sizes:

```python
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload
```

**4. Execution Timeouts**

The LLM calls should have timeouts. Configure via environment:

```bash
export LLM_TIMEOUT=60  # seconds
```

**5. Action Restrictions**

In remote mode, consider restricting which actions are available:

```python
# Disable potentially dangerous actions
from struckdown.actions import ACTION_LOOKUP

# Remove web fetch action for public deployments
if "fetch" in ACTION_LOOKUP:
    del ACTION_LOOKUP["fetch"]
if "search" in ACTION_LOOKUP:
    del ACTION_LOOKUP["search"]
```

**6. API Key Handling**

- API keys entered by users are stored only in browser localStorage
- Keys are sent per-request, never stored server-side
- Keys are never logged

**7. CORS (if needed)**

```python
from flask_cors import CORS

app = create_app(remote_mode=True)
CORS(app, origins=["https://your-frontend.com"])
```

### Recommended Production Stack

```
Internet
    │
    ▼
┌─────────────┐
│   Caddy/    │  ← TLS termination, rate limiting
│   nginx     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Gunicorn   │  ← Multiple workers
│  (4 workers)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Flask     │  ← Struckdown Playground
│   App       │
└─────────────┘
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for LLM provider (local mode only -- not read in remote mode) | (required in local mode) |
| `LLM_API_BASE` | Custom API base URL | Provider default |
| `DEFAULT_LLM` | Default model name | `openai/gpt-4o-mini` |
| `STRUCKDOWN_CACHE` | Cache directory (`0` to disable) | `~/.cache/struckdown` |

### Security Settings (Remote Mode)

| Variable | Description | Default |
|----------|-------------|---------|
| `STRUCKDOWN_RATE_LIMIT` | Rate limit for API endpoints (Flask-Limiter format) | `100/hour` |
| `STRUCKDOWN_MAX_SYNTAX_LENGTH` | Maximum template length in characters | `1000000` (1M) |
| `STRUCKDOWN_MAX_UPLOAD_SIZE` | Maximum upload file size in bytes | `5242880` (5MB) |
| `STRUCKDOWN_ZIP_MAX_SIZE` | Maximum uncompressed zip size in bytes | `52428800` (50MB) |
| `STRUCKDOWN_ZIP_MAX_FILES` | Maximum files in a zip archive | `500` |

**Action Restrictions:** In remote mode, actions like `@fetch` and `@search` are disabled by default for security (they could be used to probe internal networks). Safe actions like `@set`, `@break`, and `@timestamp` remain available.

---

## Troubleshooting

### Port already in use

```bash
# Specify a different port
sd edit -p 9001
```

### Custom actions not loading

Ensure the path exists and contains valid action files:

```bash
sd edit -I /path/to/actions myfile.sd
```

### Browser doesn't open

Use `--no-browser` and manually navigate:

```bash
sd edit --no-browser
# Then open http://localhost:9000 manually
```

### Session data not persisting

- Check that localStorage is enabled in your browser
- Each session has a unique ID in the URL hash (`#s=sd_xxxxxxxx`)
- Different tabs/URLs have separate sessions

---

## API Endpoints (for integrations)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main editor page |
| `/e/<encoded>` | GET | Load from URL-encoded state |
| `/api/analyse` | POST | Validate syntax, extract inputs/slots |
| `/api/run` | POST | Execute single prompt |
| `/api/save` | POST | Save to file (local mode only) |
| `/api/upload` | POST | Upload xlsx/csv for batch |
| `/api/run-batch` | POST | Start batch task |
| `/api/batch-stream/<id>` | GET | SSE stream of batch results |
| `/api/download/<id>` | GET | Download batch results as xlsx |
| `/api/encode-state` | POST | Encode state for URL sharing |

### Example: Analyse endpoint

```bash
curl -X POST http://localhost:9000/api/analyse \
  -H "Content-Type: application/json" \
  -d '{"syntax": "Tell me about {{topic}}\n\n[[response]]"}'
```

Response:

```json
{
  "valid": true,
  "error": null,
  "inputs_required": ["topic"],
  "slots_defined": ["response"]
}
```

### Example: Run endpoint

```bash
curl -X POST http://localhost:9000/api/run \
  -H "Content-Type: application/json" \
  -d '{
    "syntax": "Tell me a joke about {{topic}}\n\n[[joke]]",
    "inputs": {"topic": "programming"},
    "model": "openai/gpt-4o-mini"
  }'
```

Response:

```json
{
  "outputs": {"joke": "Why do programmers prefer dark mode? Because light attracts bugs!"},
  "cost": {"total_tokens": 45, "total_cost": 0.0001},
  "error": null
}
```
