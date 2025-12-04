"""Flask application for the Struckdown Playground.

Security considerations for remote mode:
- API keys are stored only in browser localStorage, never on server
- Uploaded files use SpooledTemporaryFile (in-memory up to threshold, auto-cleanup)
- Error messages are sanitised to prevent credential leakage
- Rate limiting is applied to execution endpoints
"""

import json
import os
import socket
import tempfile
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    RateLimitError,
    Timeout,
)

from . import core

# Security configuration from environment
STRUCKDOWN_RATE_LIMIT = os.environ.get("STRUCKDOWN_RATE_LIMIT", "100/hour")
STRUCKDOWN_MAX_SYNTAX_LENGTH = int(os.environ.get("STRUCKDOWN_MAX_SYNTAX_LENGTH", "1000000"))
STRUCKDOWN_MAX_UPLOAD_SIZE = int(os.environ.get("STRUCKDOWN_MAX_UPLOAD_SIZE", "5242880"))  # 5MB
STRUCKDOWN_ZIP_MAX_SIZE = int(os.environ.get("STRUCKDOWN_ZIP_MAX_SIZE", "52428800"))  # 50MB
STRUCKDOWN_ZIP_MAX_FILES = int(os.environ.get("STRUCKDOWN_ZIP_MAX_FILES", "500"))
# Threshold for spooling to disk (files smaller than this stay in memory)
STRUCKDOWN_SPOOL_THRESHOLD = int(os.environ.get("STRUCKDOWN_SPOOL_THRESHOLD", "1048576"))  # 1MB

# Allowed file extensions
ALLOWED_BATCH_EXTENSIONS = {".xlsx", ".csv", ".zip"}
# Binary file types that should be rejected for source upload
BINARY_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".bin", ".dat", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg", ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flv", ".ogg", ".woff", ".woff2", ".ttf", ".otf", ".eot"}


def safe_error_message(e: Exception, remote_mode: bool) -> str:
    """
    Return a safe error message for the given exception.

    In remote mode, returns user-friendly messages without internal details.
    In local mode, returns the full exception message for debugging.
    """
    if not remote_mode:
        return str(e)

    # Map specific exceptions to user-friendly messages
    if isinstance(e, AuthenticationError):
        return "Invalid API key. Please check your credentials."
    elif isinstance(e, RateLimitError):
        return "Rate limited by API provider. Please try again later."
    elif isinstance(e, ContextWindowExceededError):
        return "Prompt too long for this model. Try a shorter prompt or different model."
    elif isinstance(e, ContentPolicyViolationError):
        return "Content blocked by API provider's content policy."
    elif isinstance(e, (APIConnectionError, Timeout)):
        return "Could not connect to API. Please check your API base URL and try again."
    elif isinstance(e, ValueError) and "API key" in str(e).lower():
        return "API key and API base URL are required."
    else:
        # Generic message for unknown errors
        return "Execution failed. Please check your prompt and try again."


def sanitise_error_string(error_str: str, remote_mode: bool) -> str:
    """
    Sanitise an error string for display.

    In remote mode, maps known error patterns to user-friendly messages.
    In local mode, returns the original error string.
    """
    if not remote_mode or not error_str:
        return error_str

    error_lower = error_str.lower()

    # Pattern-based error mapping for common issues
    if "authenticationerror" in error_lower or "invalid api key" in error_lower:
        return "Invalid API key. Please check your credentials."
    elif "ratelimit" in error_lower:
        return "Rate limited by API provider. Please try again later."
    elif "context" in error_lower and "exceed" in error_lower:
        return "Prompt too long for this model. Try a shorter prompt or different model."
    elif "content" in error_lower and "policy" in error_lower:
        return "Content blocked by API provider's content policy."
    elif "connection" in error_lower or "timeout" in error_lower:
        return "Could not connect to API. Please check your API base URL and try again."
    elif "provider not provided" in error_lower or "pass model as" in error_lower:
        return "Invalid model name. Use format like 'openai/gpt-4o' or 'anthropic/claude-3-sonnet'."
    elif "badrequest" in error_lower:
        return "Invalid request. Please check your model name and settings."
    elif "api key" in error_lower and "required" in error_lower:
        return "API key and API base URL are required."
    elif "notfounderror" in error_lower or "model not found" in error_lower:
        return "Model not found. Please check the model name."
    else:
        # Generic message for unknown errors
        return "Execution failed. Please check your prompt and settings."


def find_available_port(start: int = 9000, max_attempts: int = 100) -> int:
    """Find an available port starting from 9000."""
    for port in range(start, start + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port found in range {start}-{start + max_attempts - 1}")


# In-memory storage for batch tasks and uploaded files
# These are request-scoped and cleaned up after use or on timeout
_batch_tasks: Dict[str, Dict] = {}
_uploaded_files: Dict[str, Dict] = {}  # Stores parsed data in memory (no disk files)
_source_files: Dict[str, Dict] = {}  # For file mode (content stored in memory)
_file_timestamps: Dict[str, float] = {}  # Track when files were uploaded for cleanup

# Cleanup interval and max age for uploaded files (seconds)
_CLEANUP_INTERVAL = 300  # 5 minutes
_FILE_MAX_AGE = 1800  # 30 minutes
_last_cleanup = time.time()


def _cleanup_old_files():
    """Remove uploaded files and tasks older than _FILE_MAX_AGE."""
    global _last_cleanup
    now = time.time()

    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return

    _last_cleanup = now
    cutoff = now - _FILE_MAX_AGE

    # Clean up old uploaded files
    expired_files = [fid for fid, ts in _file_timestamps.items() if ts < cutoff]
    for fid in expired_files:
        _uploaded_files.pop(fid, None)
        _source_files.pop(fid, None)
        _file_timestamps.pop(fid, None)

    # Clean up old batch tasks
    expired_tasks = [
        tid for tid, task in _batch_tasks.items()
        if task.get("created_at", 0) < cutoff and task.get("status") in ("complete", "error")
    ]
    for tid in expired_tasks:
        _batch_tasks.pop(tid, None)


def create_app(
    prompt_file: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    remote_mode: bool = False,
    server_api_key: Optional[str] = None,
) -> Flask:
    """
    Create Flask application for the playground.

    Args:
        prompt_file: Local mode - path to the .sd file being edited
        include_paths: Directories to search for includes/actions/types
        remote_mode: True for hosted service (no file access, URL-encoded state)
        server_api_key: Optional server-side API key for remote mode
                        (if not set, users must provide their own)
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    # Store config
    app.config["PROMPT_FILE"] = prompt_file
    app.config["INCLUDE_PATHS"] = include_paths or []
    app.config["REMOTE_MODE"] = remote_mode
    app.config["SERVER_API_KEY"] = server_api_key
    app.config["MAX_CONTENT_LENGTH"] = STRUCKDOWN_MAX_UPLOAD_SIZE

    # Rate limiting (only in remote mode)
    limiter = None
    if remote_mode:
        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=[],  # No default limit, apply only to specific endpoints
            storage_uri="memory://",
        )

    # Load custom actions/types in local mode
    if not remote_mode and include_paths:
        try:
            from struckdown.actions import discover_actions, load_actions
            from struckdown.type_loader import discover_yaml_types, load_yaml_types

            search_paths = include_paths + [Path.cwd()]
            action_files = discover_actions(search_paths)
            load_actions(action_files)

            type_files = discover_yaml_types(search_paths)
            load_yaml_types(type_files)
        except Exception:
            pass  # Ignore errors loading custom actions

    @app.route("/")
    def index():
        """Render the main editor page."""
        syntax = ""
        if prompt_file and prompt_file.exists():
            syntax = prompt_file.read_text()

        return render_template(
            "editor.html",
            syntax=syntax,
            filename=prompt_file.name if prompt_file else None,
            remote_mode=remote_mode,
            has_server_api_key=bool(server_api_key),
        )

    @app.route("/e/<encoded_state>")
    def load_from_state(encoded_state: str):
        """Load editor from URL-encoded state (remote mode)."""
        state = core.decode_state(encoded_state)
        return render_template(
            "editor.html",
            syntax=state["syntax"],
            model=state["model"],
            inputs=state["inputs"],
            filename=None,
            remote_mode=True,
            has_server_api_key=bool(server_api_key),
            encoded_state=encoded_state,
        )

    @app.route("/api/save", methods=["POST"])
    def save():
        """Save syntax to file (local mode only)."""
        if remote_mode:
            return jsonify({"error": "Cannot save in remote mode"}), 400

        data = request.get_json()
        syntax = data.get("syntax", "")

        if prompt_file:
            prompt_file.write_text(syntax)
            # Return the new modification time
            mtime = prompt_file.stat().st_mtime
            return jsonify({"success": True, "mtime": mtime})

        return jsonify({"error": "No file specified"}), 400

    @app.route("/api/file-status")
    def file_status():
        """Get file modification time and content (local mode only)."""
        if remote_mode:
            return jsonify({"error": "Not available in remote mode"}), 400

        if not prompt_file or not prompt_file.exists():
            return jsonify({"error": "No file"}), 400

        mtime = prompt_file.stat().st_mtime
        content = prompt_file.read_text()
        return jsonify({"mtime": mtime, "content": content})

    @app.route("/api/analyse", methods=["POST"])
    def analyse():
        """Analyse template syntax."""
        data = request.get_json()
        syntax = data.get("syntax", "")

        # Validate syntax
        validation = core.validate_syntax(syntax)

        # Extract inputs/slots
        extraction = core.extract_required_inputs(syntax)

        return jsonify({
            "valid": validation["valid"],
            "error": validation["error"],
            "inputs_required": extraction["inputs_required"],
            "slots_defined": extraction["slots_defined"],
        })

    @app.route("/api/run", methods=["POST"])
    def run():
        """Execute template with inputs."""
        import anyio
        from struckdown import LLMCredentials
        from struckdown.errors import StruckdownLLMError

        data = request.get_json()
        syntax = data.get("syntax", "")
        inputs = data.get("inputs", {})
        model_name = data.get("model")

        # Validate syntax length
        if len(syntax) > STRUCKDOWN_MAX_SYNTAX_LENGTH:
            return jsonify({
                "error": f"Syntax too long ({len(syntax)} chars, max {STRUCKDOWN_MAX_SYNTAX_LENGTH})",
                "outputs": {},
                "cost": None
            }), 400

        # Check for disallowed actions in remote mode
        if remote_mode:
            disallowed = core.check_disallowed_actions(syntax)
            if disallowed:
                return jsonify({
                    "error": f"Actions not allowed in remote mode: {', '.join(disallowed)}",
                    "outputs": {},
                    "cost": None
                }), 400

        # Determine credentials
        credentials = None
        if remote_mode:
            # In remote mode, use server key if available, otherwise require user key
            user_api_key = data.get("api_key")
            user_api_base = data.get("api_base")

            # Use server key if provided, otherwise require user credentials
            if server_api_key:
                credentials = LLMCredentials(api_key=server_api_key)
            elif user_api_key:
                # User must provide both api_key and api_base, or api_base from env
                api_base = user_api_base or os.environ.get("LLM_API_BASE")
                if not api_base:
                    return jsonify({
                        "error": "API base URL required. Set it in Settings or contact the administrator.",
                        "outputs": {},
                        "cost": None
                    }), 400
                credentials = LLMCredentials(api_key=user_api_key, base_url=api_base)
            else:
                return jsonify({
                    "error": "API key required. Enter your API key in Settings.",
                    "outputs": {},
                    "cost": None
                }), 400
        # In local mode, credentials=None will use environment variables

        async def execute():
            return await core.run_single(
                syntax=syntax,
                inputs=inputs,
                model_name=model_name,
                credentials=credentials,
                include_paths=app.config["INCLUDE_PATHS"],
            )

        try:
            result = anyio.run(execute)
            # Sanitise error message if present (core.run_single catches exceptions internally)
            if result.get("error"):
                result["error"] = sanitise_error_string(result["error"], remote_mode)
            return jsonify(result)
        except StruckdownLLMError as e:
            # Extract the underlying LLM exception for better error messages
            error_msg = safe_error_message(e.__cause__ or e, remote_mode)
            return jsonify({"error": error_msg, "outputs": {}, "cost": None})
        except Exception as e:
            error_msg = safe_error_message(e, remote_mode)
            return jsonify({"error": error_msg, "outputs": {}, "cost": None})

    # Apply rate limiting to run endpoint in remote mode
    if limiter:
        run = limiter.limit(STRUCKDOWN_RATE_LIMIT)(run)

    @app.route("/api/upload", methods=["POST"])
    def upload():
        """Upload xlsx/csv/zip file for batch processing.

        Files are processed in memory using SpooledTemporaryFile.
        Only the parsed data is retained; the original file is discarded.
        """
        # Trigger cleanup on each request
        _cleanup_old_files()

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        # Validate file extension
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_BATCH_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_BATCH_EXTENSIONS))
            return jsonify({"error": f"File type '{suffix}' not allowed. Allowed types: {allowed}"}), 400

        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Use SpooledTemporaryFile: stays in memory up to threshold, auto-deletes
        try:
            with tempfile.SpooledTemporaryFile(
                max_size=STRUCKDOWN_SPOOL_THRESHOLD,
                suffix=suffix
            ) as spooled:
                # Copy uploaded file to spooled temp
                file.save(spooled)
                spooled.seek(0)

                # For processing, we need a path -- write to a real temp file briefly
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(spooled.read())
                    tmp_path = Path(tmp.name)

                try:
                    # Parse the file (xlsx/csv/zip)
                    data = core.load_batch_file(tmp_path)
                finally:
                    # Immediately delete the temp file after parsing
                    tmp_path.unlink(missing_ok=True)

            # Store only parsed data in memory (no file path)
            _uploaded_files[file_id] = {
                "filename": file.filename,
                "data": data,
            }
            _file_timestamps[file_id] = time.time()

            return jsonify({
                "file_id": file_id,
                "filename": file.filename,
                "row_count": data["row_count"],
                "columns": data["columns"],
                "preview": data["rows"][:5],
            })
        except Exception as e:
            error_msg = safe_error_message(e, remote_mode)
            return jsonify({"error": error_msg}), 400

    @app.route("/api/upload-source", methods=["POST"])
    def upload_source():
        """Upload a single file for file mode (content becomes {{source}}).

        File content is read directly into memory; no disk storage.
        """
        # Trigger cleanup on each request
        _cleanup_old_files()

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        # Reject binary file types
        suffix = Path(file.filename).suffix.lower()
        if suffix in BINARY_EXTENSIONS:
            return jsonify({"error": f"Binary file type '{suffix}' not supported. Please upload a text-based file."}), 400

        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Read file content directly into memory
        try:
            # Read as bytes first, then decode
            file_bytes = file.read()
            file_size = len(file_bytes)
            content = file_bytes.decode("utf-8", errors="replace")

            # Store only content in memory (no disk path)
            _source_files[file_id] = {
                "filename": file.filename,
                "content": content,
                "size": file_size,
            }
            _file_timestamps[file_id] = time.time()

            return jsonify({
                "file_id": file_id,
                "filename": file.filename,
                "size": file_size,
            })
        except Exception as e:
            error_msg = safe_error_message(e, remote_mode)
            return jsonify({"error": error_msg}), 400

    @app.route("/api/run-file", methods=["POST"])
    def run_file():
        """Execute template with uploaded file as {{source}}."""
        import anyio
        from struckdown import LLMCredentials
        from struckdown.errors import StruckdownLLMError

        data = request.get_json()
        syntax = data.get("syntax", "")
        file_id = data.get("file_id")
        model_name = data.get("model")

        # Validate syntax length
        if len(syntax) > STRUCKDOWN_MAX_SYNTAX_LENGTH:
            return jsonify({
                "error": f"Syntax too long ({len(syntax)} chars, max {STRUCKDOWN_MAX_SYNTAX_LENGTH})",
                "outputs": {},
                "cost": None
            }), 400

        # Check for disallowed actions in remote mode
        if remote_mode:
            disallowed = core.check_disallowed_actions(syntax)
            if disallowed:
                return jsonify({
                    "error": f"Actions not allowed in remote mode: {', '.join(disallowed)}",
                    "outputs": {},
                    "cost": None
                }), 400

        if file_id not in _source_files:
            return jsonify({"error": "File not found or expired", "outputs": {}, "cost": None}), 400

        file_data = _source_files[file_id]

        # Determine credentials
        credentials = None
        if remote_mode:
            user_api_key = data.get("api_key")
            user_api_base = data.get("api_base")

            if server_api_key:
                credentials = LLMCredentials(api_key=server_api_key)
            elif user_api_key:
                api_base = user_api_base or os.environ.get("LLM_API_BASE")
                if not api_base:
                    return jsonify({
                        "error": "API base URL required. Set it in Settings or contact the administrator.",
                        "outputs": {},
                        "cost": None
                    }), 400
                credentials = LLMCredentials(api_key=user_api_key, base_url=api_base)
            else:
                return jsonify({
                    "error": "API key required. Enter your API key in Settings.",
                    "outputs": {},
                    "cost": None
                }), 400

        # Inject file content as {{source}}
        inputs = {"source": file_data["content"]}

        async def execute():
            return await core.run_single(
                syntax=syntax,
                inputs=inputs,
                model_name=model_name,
                credentials=credentials,
                include_paths=app.config["INCLUDE_PATHS"],
            )

        try:
            result = anyio.run(execute)
            # Sanitise error message if present
            if result.get("error"):
                result["error"] = sanitise_error_string(result["error"], remote_mode)
            return jsonify(result)
        except StruckdownLLMError as e:
            error_msg = safe_error_message(e.__cause__ or e, remote_mode)
            return jsonify({"error": error_msg, "outputs": {}, "cost": None})
        except Exception as e:
            error_msg = safe_error_message(e, remote_mode)
            return jsonify({"error": error_msg, "outputs": {}, "cost": None})

    # Apply rate limiting to run-file endpoint in remote mode
    if limiter:
        run_file = limiter.limit(STRUCKDOWN_RATE_LIMIT)(run_file)

    @app.route("/api/run-batch", methods=["POST"])
    def run_batch():
        """Start batch processing and return task ID."""
        from struckdown import LLMCredentials

        data = request.get_json()
        syntax = data.get("syntax", "")
        file_id = data.get("file_id")
        model_name = data.get("model")

        # Validate syntax length
        if len(syntax) > STRUCKDOWN_MAX_SYNTAX_LENGTH:
            return jsonify({"error": f"Syntax too long ({len(syntax)} chars, max {STRUCKDOWN_MAX_SYNTAX_LENGTH})"}), 400

        # Check for disallowed actions in remote mode
        if remote_mode:
            disallowed = core.check_disallowed_actions(syntax)
            if disallowed:
                return jsonify({"error": f"Actions not allowed in remote mode: {', '.join(disallowed)}"}), 400

        if file_id not in _uploaded_files:
            return jsonify({"error": "File not found or expired"}), 400

        # Determine credentials
        credentials = None
        if remote_mode:
            user_api_key = data.get("api_key")
            user_api_base = data.get("api_base")

            if server_api_key:
                credentials = LLMCredentials(api_key=server_api_key)
            elif user_api_key:
                api_base = user_api_base or os.environ.get("LLM_API_BASE")
                if not api_base:
                    return jsonify({
                        "error": "API base URL required. Set it in Settings or contact the administrator."
                    }), 400
                credentials = LLMCredentials(api_key=user_api_key, base_url=api_base)
            else:
                return jsonify({"error": "API key required. Enter your API key in Settings."}), 400
        # In local mode, credentials=None will use environment variables

        file_data = _uploaded_files[file_id]
        rows = file_data["data"]["rows"]

        # Create task with timestamp for cleanup
        task_id = str(uuid.uuid4())
        _batch_tasks[task_id] = {
            "status": "pending",
            "syntax": syntax,
            "rows": rows,
            "model": model_name,
            "results": [],
            "completed": 0,
            "created_at": time.time(),
            "total": len(rows),
            "columns": file_data["data"]["columns"],
        }

        # Start background processing
        def process_batch():
            task = _batch_tasks[task_id]
            task["status"] = "running"

            def on_row_complete(result):
                task["results"].append(result)
                task["completed"] += 1

            try:
                core.run_batch_sync(
                    syntax=syntax,
                    rows=rows,
                    model_name=model_name,
                    credentials=credentials,
                    include_paths=app.config["INCLUDE_PATHS"],
                    on_row_complete=on_row_complete,
                )
                task["status"] = "complete"
            except Exception as e:
                task["status"] = "error"
                task["error"] = safe_error_message(e, remote_mode)

        thread = threading.Thread(target=process_batch, daemon=True)
        thread.start()

        return jsonify({"task_id": task_id})

    # Apply rate limiting to run-batch endpoint in remote mode
    if limiter:
        run_batch = limiter.limit(STRUCKDOWN_RATE_LIMIT)(run_batch)

    @app.route("/api/batch-stream/<task_id>")
    def batch_stream(task_id: str):
        """SSE stream for batch progress."""
        if task_id not in _batch_tasks:
            return jsonify({"error": "Task not found"}), 404

        def generate():
            task = _batch_tasks[task_id]
            last_sent = 0

            while True:
                # Send any new results
                while last_sent < len(task["results"]):
                    result = task["results"][last_sent]
                    yield f"event: row\ndata: {json.dumps(result)}\n\n"
                    last_sent += 1

                # Send progress update
                yield f"event: progress\ndata: {json.dumps({'completed': task['completed'], 'total': task['total']})}\n\n"

                # Check if done
                if task["status"] in ("complete", "error"):
                    if task["status"] == "error":
                        yield f"event: error\ndata: {json.dumps({'error': task.get('error', 'Unknown error')})}\n\n"
                    else:
                        yield f"event: done\ndata: {json.dumps({'task_id': task_id})}\n\n"
                    break

                # Small delay before next poll
                import time
                time.sleep(0.1)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/download/<task_id>")
    def download(task_id: str):
        """Download completed batch results as xlsx."""
        import pandas as pd
        from io import BytesIO

        if task_id not in _batch_tasks:
            return jsonify({"error": "Task not found"}), 404

        task = _batch_tasks[task_id]
        if task["status"] != "complete":
            return jsonify({"error": "Task not complete"}), 400

        # Build dataframe from results
        rows_data = []
        for result in sorted(task["results"], key=lambda r: r["index"]):
            row = {**result["inputs"], **result["outputs"]}
            if result["error"]:
                row["_error"] = result["error"]
            rows_data.append(row)

        df = pd.DataFrame(rows_data)

        # Write to bytes
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        return Response(
            output.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=results_{task_id[:8]}.xlsx"},
        )

    @app.route("/api/encode-state", methods=["POST"])
    def encode_state_route():
        """Encode current state for URL sharing."""
        data = request.get_json()
        encoded = core.encode_state(
            syntax=data.get("syntax", ""),
            model=data.get("model", ""),
            inputs=data.get("inputs", {}),
        )
        return jsonify({"encoded": encoded})

    @app.route("/partials/inputs", methods=["POST"])
    def render_inputs_partial():
        """Render inputs panel with current fields."""
        data = request.get_json()
        inputs_required = data.get("inputs_required", [])
        current_values = data.get("current_values", {})

        return render_template(
            "partials/inputs_panel.html",
            inputs_required=inputs_required,
            current_values=current_values,
        )

    @app.route("/partials/outputs", methods=["POST"])
    def render_outputs_partial():
        """Render outputs panel with results."""
        data = request.get_json()
        outputs = data.get("outputs", {})
        error = data.get("error")
        cost = data.get("cost")
        slots_defined = data.get("slots_defined", [])

        return render_template(
            "partials/outputs_single.html",
            outputs=outputs,
            error=error,
            cost=cost,
            slots_defined=slots_defined,
        )

    return app
