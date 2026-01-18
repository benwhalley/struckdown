"""Flask application for the Struckdown Playground.

Security considerations for remote mode:
- API keys are stored only in browser localStorage, never on server
- Uploaded files use SpooledTemporaryFile (in-memory up to threshold, auto-cleanup)
- Error messages are sanitised to prevent credential leakage
- Rate limiting is applied to execution endpoints
"""

import json
import os
import random
import socket
import tempfile
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from flask import (Flask, Response, jsonify, render_template, request,
                   stream_with_context)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from litellm.exceptions import (APIConnectionError, AuthenticationError,
                                ContentPolicyViolationError,
                                ContextWindowExceededError, RateLimitError,
                                Timeout)
from werkzeug.utils import secure_filename

from . import core, prompt_cache

# Security configuration from environment
STRUCKDOWN_RATE_LIMIT = os.environ.get("STRUCKDOWN_RATE_LIMIT", "100/hour")
STRUCKDOWN_UPLOAD_RATE_LIMIT = os.environ.get(
    "STRUCKDOWN_UPLOAD_RATE_LIMIT", "100/minute"
)
STRUCKDOWN_PROMPT_RATE_LIMIT = os.environ.get(
    "STRUCKDOWN_PROMPT_RATE_LIMIT", "10/minute"
)
STRUCKDOWN_PROMPT_LOAD_RATE_LIMIT = os.environ.get(
    "STRUCKDOWN_PROMPT_LOAD_RATE_LIMIT", "6/minute"  # 1 every 10 seconds
)
STRUCKDOWN_MAX_SYNTAX_LENGTH = int(
    os.environ.get("STRUCKDOWN_MAX_SYNTAX_LENGTH", "1000000")
)
STRUCKDOWN_MAX_UPLOAD_SIZE = int(
    os.environ.get("STRUCKDOWN_MAX_UPLOAD_SIZE", "5242880")
)  # 5MB
STRUCKDOWN_ZIP_MAX_SIZE = int(
    os.environ.get("STRUCKDOWN_ZIP_MAX_SIZE", "52428800")
)  # 50MB
STRUCKDOWN_ZIP_MAX_FILES = int(os.environ.get("STRUCKDOWN_ZIP_MAX_FILES", "500"))
STRUCKDOWN_BATCH_TIMEOUT = int(
    os.environ.get("STRUCKDOWN_BATCH_TIMEOUT", "300")
)  # 5 minutes
# Threshold for spooling to disk (files smaller than this stay in memory)
STRUCKDOWN_SPOOL_THRESHOLD = int(
    os.environ.get("STRUCKDOWN_SPOOL_THRESHOLD", "1048576")
)  # 1MB

# Allowed file extensions
ALLOWED_BATCH_EXTENSIONS = {".xlsx", ".csv", ".zip"}
# Binary file types that should be rejected for source upload
BINARY_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".dat",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".svg",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".ogg",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
}


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
        return (
            "Prompt too long for this model. Try a shorter prompt or different model."
        )
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
        return (
            "Prompt too long for this model. Try a shorter prompt or different model."
        )
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


def get_json_safe() -> Optional[dict]:
    """Parse request JSON safely.

    Note: Python's json module doesn't support max_depth, so we just use standard parsing.
    For DoS protection we rely on MAX_CONTENT_LENGTH limiting request size.
    """
    data = request.get_data()
    if not data:
        return None
    return json.loads(data)


def find_available_port(start: int = 9000, max_attempts: int = 100) -> int:
    """Find an available port starting from 9000."""
    for port in range(start, start + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available port found in range {start}-{start + max_attempts - 1}"
    )


def get_random_placeholder() -> str:
    """Get a random placeholder prompt from bundled examples."""
    placeholders_dir = Path(__file__).parent.parent / "examples" / "placeholders"
    if not placeholders_dir.exists():
        return "# Your prompt here\n\n[[response]]\n"

    sd_files = list(placeholders_dir.glob("*.sd"))
    if not sd_files:
        return "# Your prompt here\n\n[[response]]\n"

    chosen = random.choice(sd_files)
    try:
        return chosen.read_text()
    except Exception:
        return "# Your prompt here\n\n[[response]]\n"


def _create_untitled_file(workspace_dir: Path, content: str) -> Optional[Path]:
    """Create an untitled.sd file in the workspace, adding numbers if needed."""
    # Try untitled.sd first
    candidate = workspace_dir / "untitled.sd"
    if not candidate.exists():
        try:
            candidate.write_text(content)
            return candidate
        except Exception:
            return None

    # Try untitled-1.sd, untitled-2.sd, etc.
    for i in range(1, 100):
        candidate = workspace_dir / f"untitled-{i}.sd"
        if not candidate.exists():
            try:
                candidate.write_text(content)
                return candidate
            except Exception:
                return None

    return None


# Cleanup interval for caches
_CLEANUP_INTERVAL = 300  # 5 minutes
_last_cleanup = time.time()


def _cleanup_old_files():
    """Clean up old batch tasks and trigger upload cache cleanup."""
    global _last_cleanup
    now = time.time()

    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return

    _last_cleanup = now

    # Import here to avoid circular imports
    from . import evidence_cache, task_cache, upload_cache

    # Clean up upload cache (handles expiry and size limits)
    upload_cache.cleanup_cache()

    # Clean up evidence cache
    evidence_cache.cleanup_cache()

    # Clean up task cache
    task_cache.cleanup_tasks()


def create_app(
    prompt_file: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    remote_mode: bool = False,
    server_api_key: Optional[str] = None,
    allowed_models: Optional[List[str]] = None,
) -> Flask:
    """
    Create Flask application for the playground.

    Args:
        prompt_file: Local mode - path to the initial .sd file to open
        workspace_dir: Local mode - directory containing .sd files (for multi-file support)
        include_paths: Directories to search for includes/actions/types
        remote_mode: True for hosted service (no file access, URL-encoded state)
        server_api_key: Optional server-side API key for remote mode
                        (if not set, users must provide their own)
        allowed_models: Optional list of allowed model names. If provided,
                        the UI shows a dropdown instead of free text input.
    """
    from . import task_cache, upload_cache

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    # Store config
    app.config["PROMPT_FILE"] = prompt_file
    app.config["WORKSPACE_DIR"] = workspace_dir
    app.config["INCLUDE_PATHS"] = include_paths or []
    app.config["REMOTE_MODE"] = remote_mode
    app.config["SERVER_API_KEY"] = server_api_key
    app.config["ALLOWED_MODELS"] = allowed_models
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
            from struckdown.type_loader import (discover_yaml_types,
                                                load_yaml_types)

            search_paths = include_paths + [Path.cwd()]
            action_files = discover_actions(search_paths)
            load_actions(action_files)

            type_files = discover_yaml_types(search_paths)
            load_yaml_types(type_files)
        except Exception:
            pass  # Ignore errors loading custom actions

    @app.before_request
    def check_csrf():
        """Validate CSRF token on all POST requests.

        Applied in both remote and local mode because:
        - Remote: prevents cross-site attacks using user's stored API key
        - Local: prevents malicious sites from targeting localhost
        """
        if request.method != "POST":
            return

        # Check for token in header
        token = request.headers.get("X-CSRF-Token")
        if not token or len(token) < 10:
            return jsonify({"error": "Missing or invalid CSRF token"}), 403

    @app.route("/")
    def index():
        """Render the main editor page."""
        syntax = ""
        current_file_path = ""
        filename = None

        if prompt_file and prompt_file.exists():
            syntax = prompt_file.read_text()
            filename = prompt_file.name
            # Get path relative to workspace
            if workspace_dir:
                try:
                    current_file_path = str(
                        prompt_file.resolve().relative_to(workspace_dir.resolve())
                    )
                except ValueError:
                    current_file_path = prompt_file.name
            else:
                current_file_path = prompt_file.name
        elif not remote_mode and workspace_dir:
            # No file specified but we have a workspace - create untitled.sd
            syntax = get_random_placeholder()
            untitled_path = _create_untitled_file(workspace_dir, syntax)
            if untitled_path:
                current_file_path = untitled_path.name
                filename = untitled_path.name
        else:
            # Remote mode - just use placeholder without file
            syntax = get_random_placeholder()

        # Get default model from environment
        default_model = os.environ.get("DEFAULT_LLM", "")

        return render_template(
            "editor.html",
            syntax=syntax,
            filename=filename,
            current_file_path=current_file_path,
            remote_mode=remote_mode,
            has_server_api_key=bool(server_api_key),
            allowed_models=allowed_models,
            model=default_model,
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
            allowed_models=allowed_models,
        )

    @app.route("/p/<prompt_id>")
    def load_from_prompt(prompt_id: str):
        """Load editor from stored prompt (remote mode)."""
        if not prompt_cache.validate_prompt_id(prompt_id):
            return "Invalid prompt ID", 400

        try:
            syntax = prompt_cache.get_prompt(prompt_id)
        except FileNotFoundError:
            return "Prompt not found", 404

        # Get default model from environment
        default_model = os.environ.get("DEFAULT_LLM", "")

        return render_template(
            "editor.html",
            syntax=syntax,
            model=default_model,
            inputs={},
            filename=None,
            remote_mode=True,
            has_server_api_key=bool(server_api_key),
            prompt_id=prompt_id,
            allowed_models=allowed_models,
        )

    # Apply rate limiting to prompt load endpoint (prevents hash enumeration)
    if limiter:
        load_from_prompt = limiter.limit(STRUCKDOWN_PROMPT_LOAD_RATE_LIMIT)(
            load_from_prompt
        )

    @app.route("/api/save", methods=["POST"])
    def save():
        """Save syntax to file (local mode only)."""
        if remote_mode:
            return jsonify({"error": "Cannot save in remote mode"}), 400

        data = get_json_safe()
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

    @app.route("/api/files")
    def list_files():
        """List all .sd files in the workspace directory."""
        if remote_mode:
            return jsonify({"error": "Not available in remote mode"}), 400

        workspace = app.config.get("WORKSPACE_DIR")
        if not workspace or not workspace.is_dir():
            return jsonify({"files": [], "workspace": None})

        files = []
        for sd_file in sorted(workspace.rglob("*.sd")):
            # Skip hidden files/directories
            if any(
                part.startswith(".") for part in sd_file.relative_to(workspace).parts
            ):
                continue
            rel_path = str(sd_file.relative_to(workspace))
            files.append(
                {
                    "path": rel_path,
                    "name": sd_file.name,
                    "mtime": sd_file.stat().st_mtime,
                }
            )

        return jsonify({"files": files, "workspace": str(workspace)})

    def _validate_workspace_path(filepath: str) -> Optional[Path]:
        """Validate and resolve a file path within the workspace. Returns None if invalid."""
        workspace = app.config.get("WORKSPACE_DIR")
        if not workspace:
            return None

        # Resolve the path and check it's within workspace
        try:
            file_path = (workspace / filepath).resolve()
            workspace_resolved = workspace.resolve()
            if not str(file_path).startswith(str(workspace_resolved)):
                return None  # Directory traversal attempt
            return file_path
        except (ValueError, OSError):
            return None

    @app.route("/api/files/<path:filepath>")
    def get_file(filepath: str):
        """Read content of a specific file in the workspace."""
        if remote_mode:
            return jsonify({"error": "Not available in remote mode"}), 400

        file_path = _validate_workspace_path(filepath)
        if not file_path:
            return jsonify({"error": "Invalid path"}), 400

        if not file_path.exists():
            return jsonify({"error": "File not found"}), 404

        return jsonify(
            {
                "content": file_path.read_text(),
                "mtime": file_path.stat().st_mtime,
                "path": filepath,
            }
        )

    @app.route("/api/files/<path:filepath>", methods=["POST"])
    def save_file(filepath: str):
        """Save content to a specific file in the workspace."""
        if remote_mode:
            return jsonify({"error": "Not available in remote mode"}), 400

        file_path = _validate_workspace_path(filepath)
        if not file_path:
            return jsonify({"error": "Invalid path"}), 400

        data = get_json_safe()
        syntax = data.get("syntax", "")

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(syntax)

        return jsonify({"success": True, "mtime": file_path.stat().st_mtime})

    @app.route("/api/files/new", methods=["POST"])
    def create_file():
        """Create a new .sd file in the workspace."""
        if remote_mode:
            return jsonify({"error": "Not available in remote mode"}), 400

        workspace = app.config.get("WORKSPACE_DIR")
        if not workspace:
            return jsonify({"error": "No workspace configured"}), 400

        data = get_json_safe()
        filename = data.get("filename", "").strip()

        if not filename:
            return jsonify({"error": "Filename required"}), 400

        # Ensure .sd extension
        if not filename.endswith(".sd"):
            filename += ".sd"

        # Validate the path
        file_path = _validate_workspace_path(filename)
        if not file_path:
            return jsonify({"error": "Invalid filename"}), 400

        if file_path.exists():
            return jsonify({"error": "File already exists"}), 400

        # Create with default content
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "# LLM Instructions\n\nUse markdown-style syntax (see the help tab for details) \n\n[[response]]\n"
        )

        return jsonify(
            {"success": True, "path": filename, "mtime": file_path.stat().st_mtime}
        )

    @app.route("/api/analyse", methods=["POST"])
    def analyse():
        """Analyse template syntax."""
        data = get_json_safe()
        syntax = data.get("syntax", "")

        # Validate syntax
        validation = core.validate_syntax(syntax)

        # Extract inputs/slots
        extraction = core.extract_required_inputs(syntax)

        # Check for @history usage
        uses_history = core.uses_history_action(syntax)

        return jsonify(
            {
                "valid": validation["valid"],
                "error": validation["error"],
                "inputs_required": extraction["inputs_required"],
                "slots_defined": extraction["slots_defined"],
                "uses_history": uses_history,
            }
        )

    @app.route("/api/run", methods=["POST"])
    def run():
        """Execute template with inputs."""
        import anyio

        from struckdown import LLMCredentials
        from struckdown.errors import StruckdownLLMError

        data = get_json_safe()
        syntax = data.get("syntax", "")
        inputs = data.get("inputs", {})
        model_name = data.get("model")
        strict_undefined = data.get("strict_undefined", False)
        session_id = data.get("session_id")  # For evidence loading

        # Validate syntax length
        if len(syntax) > STRUCKDOWN_MAX_SYNTAX_LENGTH:
            return (
                jsonify(
                    {
                        "error": f"Syntax too long ({len(syntax)} chars, max {STRUCKDOWN_MAX_SYNTAX_LENGTH})",
                        "outputs": {},
                        "cost": None,
                    }
                ),
                400,
            )

        # Check for disallowed actions in remote mode
        if remote_mode:
            disallowed = core.check_disallowed_actions(syntax)
            if disallowed:
                return (
                    jsonify(
                        {
                            "error": f"Actions not allowed in remote mode: {', '.join(disallowed)}",
                            "outputs": {},
                            "cost": None,
                        }
                    ),
                    400,
                )

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
                    return (
                        jsonify(
                            {
                                "error": "API base URL required. Set it in Settings or contact the administrator.",
                                "outputs": {},
                                "cost": None,
                            }
                        ),
                        400,
                    )
                credentials = LLMCredentials(api_key=user_api_key, base_url=api_base)
            else:
                return (
                    jsonify(
                        {
                            "error": "API key required. Enter your API key in Settings.",
                            "outputs": {},
                            "cost": None,
                        }
                    ),
                    400,
                )
        # In local mode, credentials=None will use environment variables

        async def execute():
            return await core.run_single(
                syntax=syntax,
                inputs=inputs,
                model_name=model_name,
                credentials=credentials,
                include_paths=app.config["INCLUDE_PATHS"],
                strict_undefined=strict_undefined,
                evidence_session_id=session_id,
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

    @app.route("/api/run-incremental", methods=["POST"])
    def run_incremental():
        """Execute template with inputs, streaming incremental results via SSE.

        Returns Server-Sent Events with event types:
        - slot_completed: individual slot result
        - checkpoint: checkpoint boundary reached
        - complete: final result
        - error: processing error
        """
        import asyncio

        from struckdown import LLMCredentials, chatter_incremental_async

        data = get_json_safe()
        syntax = data.get("syntax", "")
        inputs = data.get("inputs", {})
        model_name = data.get("model")
        strict_undefined = data.get("strict_undefined", False)
        session_id = data.get("session_id")  # For evidence loading

        # Inject evidence if session has uploaded evidence files
        if session_id:
            from . import evidence_cache

            evidence_chunks = evidence_cache.get_evidence_for_session(session_id)
            if evidence_chunks:
                inputs = {**inputs, "_evidence_store": evidence_chunks}

        # Validate syntax length
        if len(syntax) > STRUCKDOWN_MAX_SYNTAX_LENGTH:

            def error_gen():
                yield f'event: error\ndata: {json.dumps({"error": f"Syntax too long ({len(syntax)} chars, max {STRUCKDOWN_MAX_SYNTAX_LENGTH})"})}\n\n'

            return Response(
                stream_with_context(error_gen()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Check for disallowed actions in remote mode
        if remote_mode:
            disallowed = core.check_disallowed_actions(syntax)
            if disallowed:

                def error_gen():
                    yield f'event: error\ndata: {json.dumps({"error": f"Actions not allowed in remote mode: {", ".join(disallowed)}"})}\n\n'

                return Response(
                    stream_with_context(error_gen()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

        # Determine credentials (same as /api/run)
        credentials = None
        if remote_mode:
            user_api_key = data.get("api_key")
            user_api_base = data.get("api_base")

            if server_api_key:
                credentials = LLMCredentials(api_key=server_api_key)
            elif user_api_key:
                api_base = user_api_base or os.environ.get("LLM_API_BASE")
                if not api_base:

                    def error_gen():
                        yield f'event: error\ndata: {json.dumps({"error": "API base URL required."})}\n\n'

                    return Response(
                        stream_with_context(error_gen()),
                        mimetype="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        },
                    )
                credentials = LLMCredentials(api_key=user_api_key, base_url=api_base)
            else:

                def error_gen():
                    yield f'event: error\ndata: {json.dumps({"error": "API key required."})}\n\n'

                return Response(
                    stream_with_context(error_gen()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

        def generate():
            """Generator that streams incremental events."""
            from struckdown import LLM

            async def stream_events():
                """Async generator to collect events."""
                model = LLM(model_name=model_name) if model_name else LLM()
                try:
                    async for event in chatter_incremental_async(
                        syntax,
                        model=model,
                        credentials=credentials,
                        context=inputs,
                        strict_undefined=strict_undefined,
                    ):
                        yield event
                except Exception as e:
                    # Yield error event
                    from struckdown import ChatterResult, ProcessingError

                    yield ProcessingError(
                        segment_index=0,
                        slot_key=None,
                        error_message=safe_error_message(e, remote_mode),
                        partial_results=ChatterResult(),
                    )

            # Run the async generator synchronously
            loop = asyncio.new_event_loop()
            try:
                gen = stream_events()
                while True:
                    try:
                        event = loop.run_until_complete(gen.__anext__())
                        # Serialise and yield as SSE
                        event_data = event.model_dump_json()
                        yield f"event: {event.type}\ndata: {event_data}\n\n"
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Apply rate limiting to run-incremental endpoint in remote mode
    if limiter:
        run_incremental = limiter.limit(STRUCKDOWN_RATE_LIMIT)(run_incremental)

    @app.route("/api/run-chat", methods=["POST"])
    def run_chat():
        """Execute template for chat mode, with history messages injected.

        Similar to /api/run-incremental but accepts history_messages to inject
        into context for the [[@history]] action.

        Returns Server-Sent Events with event types:
        - slot_completed: individual slot result
        - checkpoint: checkpoint boundary reached
        - complete: final result with all slots
        - error: processing error
        """
        import asyncio

        from struckdown import LLMCredentials, chatter_incremental_async

        data = get_json_safe()
        syntax = data.get("syntax", "")
        inputs = data.get("inputs", {})
        model_name = data.get("model")
        history_messages = data.get("history_messages", [])
        strict_undefined = data.get("strict_undefined", False)
        session_id = data.get("session_id")  # For evidence loading

        # Inject evidence if session has uploaded evidence files
        if session_id:
            from . import evidence_cache

            evidence_chunks = evidence_cache.get_evidence_for_session(session_id)
            if evidence_chunks:
                inputs = {**inputs, "_evidence_store": evidence_chunks}

        # Validate syntax length
        if len(syntax) > STRUCKDOWN_MAX_SYNTAX_LENGTH:

            def error_gen():
                yield f'event: error\ndata: {json.dumps({"error_message": f"Syntax too long ({len(syntax)} chars, max {STRUCKDOWN_MAX_SYNTAX_LENGTH})"})}\n\n'

            return Response(
                stream_with_context(error_gen()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Check for disallowed actions in remote mode
        if remote_mode:
            disallowed = core.check_disallowed_actions(syntax)
            if disallowed:

                def error_gen():
                    yield f'event: error\ndata: {json.dumps({"error_message": f"Actions not allowed in remote mode: {", ".join(disallowed)}"})}\n\n'

                return Response(
                    stream_with_context(error_gen()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

        # Determine credentials (same as /api/run)
        credentials = None
        if remote_mode:
            user_api_key = data.get("api_key")
            user_api_base = data.get("api_base")

            if server_api_key:
                credentials = LLMCredentials(api_key=server_api_key)
            elif user_api_key:
                api_base = user_api_base or os.environ.get("LLM_API_BASE")
                if not api_base:

                    def error_gen():
                        yield f'event: error\ndata: {json.dumps({"error_message": "API base URL required."})}\n\n'

                    return Response(
                        stream_with_context(error_gen()),
                        mimetype="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        },
                    )
                credentials = LLMCredentials(api_key=user_api_key, base_url=api_base)
            else:

                def error_gen():
                    yield f'event: error\ndata: {json.dumps({"error_message": "API key required."})}\n\n'

                return Response(
                    stream_with_context(error_gen()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

        # Build context with inputs and history messages
        context = dict(inputs)
        if history_messages:
            context["_history_messages"] = history_messages

        def generate():
            """Generator that streams incremental events."""
            from struckdown import LLM

            async def stream_events():
                """Async generator to collect events."""
                model = LLM(model_name=model_name) if model_name else LLM()
                try:
                    async for event in chatter_incremental_async(
                        syntax,
                        model=model,
                        credentials=credentials,
                        context=context,
                        strict_undefined=strict_undefined,
                    ):
                        yield event
                except Exception as e:
                    # Yield error event
                    from struckdown import ChatterResult, ProcessingError

                    yield ProcessingError(
                        segment_index=0,
                        slot_key=None,
                        error_message=safe_error_message(e, remote_mode),
                        partial_results=ChatterResult(),
                    )

            # Run the async generator synchronously
            loop = asyncio.new_event_loop()
            try:
                gen = stream_events()
                while True:
                    try:
                        event = loop.run_until_complete(gen.__anext__())
                        # Serialise and yield as SSE
                        event_data = event.model_dump_json()
                        yield f"event: {event.type}\ndata: {event_data}\n\n"
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Apply rate limiting to run-chat endpoint in remote mode
    if limiter:
        run_chat = limiter.limit(STRUCKDOWN_RATE_LIMIT)(run_chat)

    @app.route("/api/upload", methods=["POST"])
    def upload():
        """Upload files for batch processing.

        Supports:
        - Single xlsx/csv/zip file: parsed as batch data
        - Multiple files: each file becomes a row with 'source' and 'filename' columns

        Files are processed in memory using SpooledTemporaryFile.
        Only the parsed data is retained; the original file is discarded.
        """
        # Trigger cleanup on each request
        _cleanup_old_files()

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        files = request.files.getlist("file")
        if not files or not files[0].filename:
            return jsonify({"error": "No file selected"}), 400

        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Check if this is a single structured file (xlsx/csv/zip)
        if len(files) == 1:
            file = files[0]
            suffix = Path(file.filename).suffix.lower()

            if suffix in ALLOWED_BATCH_EXTENSIONS:
                # Single xlsx/csv/zip: use existing parsing logic
                try:
                    with tempfile.SpooledTemporaryFile(
                        max_size=STRUCKDOWN_SPOOL_THRESHOLD, suffix=suffix
                    ) as spooled:
                        file.save(spooled)
                        spooled.seek(0)

                        with tempfile.NamedTemporaryFile(
                            suffix=suffix, delete=False
                        ) as tmp:
                            tmp.write(spooled.read())
                            tmp_path = Path(tmp.name)

                        try:
                            data = core.load_batch_file(tmp_path)
                        finally:
                            tmp_path.unlink(missing_ok=True)

                    upload_cache.store_upload(
                        file_id,
                        {
                            "type": "batch",
                            "filename": file.filename,
                            "data": data,
                        },
                    )

                    return jsonify(
                        {
                            "file_id": file_id,
                            "filename": file.filename,
                            "row_count": data["row_count"],
                            "columns": data["columns"],
                            "preview": data["rows"][:5],
                        }
                    )
                except Exception as e:
                    error_msg = safe_error_message(e, remote_mode)
                    return jsonify({"error": error_msg}), 400

        # Multiple files (or single non-xlsx/csv/zip): treat as individual source files
        # Each file becomes a row with 'source' and 'filename' columns (like zip handling)
        try:
            if len(files) > STRUCKDOWN_ZIP_MAX_FILES:
                return (
                    jsonify(
                        {
                            "error": f"Too many files. Maximum is {STRUCKDOWN_ZIP_MAX_FILES}."
                        }
                    ),
                    400,
                )

            rows = []
            total_size = 0
            skipped_binary = []

            for f in files:
                filename = secure_filename(f.filename) if f.filename else "unnamed"
                suffix = Path(filename).suffix.lower()

                # Skip binary files
                if suffix in BINARY_EXTENSIONS:
                    skipped_binary.append(filename)
                    continue

                # Read content
                content = f.read()
                total_size += len(content)

                if total_size > STRUCKDOWN_ZIP_MAX_SIZE:
                    return (
                        jsonify(
                            {
                                "error": f"Total file size exceeds {STRUCKDOWN_ZIP_MAX_SIZE // (1024*1024)}MB limit."
                            }
                        ),
                        400,
                    )

                # Decode as UTF-8 with replacement for invalid chars
                try:
                    text_content = content.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = content.decode("utf-8", errors="replace")

                rows.append({"source": text_content, "filename": filename})

            if not rows:
                if skipped_binary:
                    return (
                        jsonify(
                            {
                                "error": f"All files were binary and skipped: {', '.join(skipped_binary)}"
                            }
                        ),
                        400,
                    )
                return jsonify({"error": "No valid files to process"}), 400

            data = {
                "rows": rows,
                "columns": ["source", "filename"],
                "row_count": len(rows),
            }

            # Determine display name
            display_name = (
                f"{len(files)} files" if len(files) > 1 else files[0].filename
            )

            upload_cache.store_upload(
                file_id,
                {
                    "type": "batch",
                    "filename": display_name,
                    "data": data,
                },
            )

            response_data = {
                "file_id": file_id,
                "filename": display_name,
                "row_count": data["row_count"],
                "columns": data["columns"],
                "preview": data["rows"][:5],
            }

            if skipped_binary:
                response_data["warning"] = (
                    f"Skipped binary files: {', '.join(skipped_binary)}"
                )

            return jsonify(response_data)

        except Exception as e:
            error_msg = safe_error_message(e, remote_mode)
            return jsonify({"error": error_msg}), 400

    # Apply rate limiting to upload endpoint in remote mode
    if limiter:
        upload = limiter.limit(STRUCKDOWN_UPLOAD_RATE_LIMIT)(upload)

    @app.route("/api/upload-source", methods=["POST"])
    def upload_source():
        """Upload a single file for file mode (content becomes {{source}}).

        File content is stored in disk cache.
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
            return (
                jsonify(
                    {
                        "error": f"Binary file type '{suffix}' not supported. Please upload a text-based file."
                    }
                ),
                400,
            )

        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Read file content and store to disk cache
        try:
            # Read as bytes first, then decode
            file_bytes = file.read()
            file_size = len(file_bytes)
            content = file_bytes.decode("utf-8", errors="replace")

            # Store to disk cache
            upload_cache.store_upload(
                file_id,
                {
                    "type": "source",
                    "filename": file.filename,
                    "content": content,
                    "size": file_size,
                },
            )

            return jsonify(
                {
                    "file_id": file_id,
                    "filename": file.filename,
                    "size": file_size,
                }
            )
        except Exception as e:
            error_msg = safe_error_message(e, remote_mode)
            return jsonify({"error": error_msg}), 400

    # Apply rate limiting to upload-source endpoint in remote mode
    if limiter:
        upload_source = limiter.limit(STRUCKDOWN_UPLOAD_RATE_LIMIT)(upload_source)

    @app.route("/api/upload-evidence", methods=["POST"])
    def upload_evidence():
        """Upload evidence files for BM25 search via [[@evidence]] action.

        Accepts .txt and .md files. Files are chunked and stored per session.
        """
        from . import chunking, evidence_cache

        # Trigger cleanup
        _cleanup_old_files()

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        files = request.files.getlist("file")
        if not files or not files[0].filename:
            return jsonify({"error": "No file selected"}), 400

        session_id = request.form.get("session_id")
        if not session_id or not evidence_cache.validate_session_id(session_id):
            return jsonify({"error": "Invalid or missing session_id"}), 400

        results = []
        errors = []

        for file in files:
            filename = secure_filename(file.filename) if file.filename else "unnamed"
            suffix = Path(filename).suffix.lower()

            # Only accept .txt and .md files
            if suffix not in (".txt", ".md"):
                errors.append(f"{filename}: only .txt and .md files allowed")
                continue

            try:
                # Read and decode content
                content = file.read()
                text = content.decode("utf-8", errors="replace")

                # Chunk using sentence-based splitter
                chunks = chunking.chunk_text_sentences(text)

                # Generate file ID and store
                file_id = str(uuid.uuid4())
                evidence_cache.store_evidence(session_id, file_id, filename, chunks)

                results.append(
                    {
                        "file_id": file_id,
                        "filename": filename,
                        "chunk_count": len(chunks),
                    }
                )
            except Exception as e:
                errors.append(f"{filename}: {safe_error_message(e, remote_mode)}")

        if not results and errors:
            return jsonify({"error": "; ".join(errors)}), 400

        response = {"files": results, "session_id": session_id}
        if errors:
            response["warnings"] = errors

        return jsonify(response)

    # Apply rate limiting to upload-evidence endpoint in remote mode
    if limiter:
        upload_evidence = limiter.limit(STRUCKDOWN_UPLOAD_RATE_LIMIT)(upload_evidence)

    @app.route("/api/evidence")
    def list_evidence_files():
        """List all evidence files for a session."""
        from . import evidence_cache

        session_id = request.args.get("session_id")
        if not session_id or not evidence_cache.validate_session_id(session_id):
            return jsonify({"files": []})

        files = evidence_cache.list_evidence(session_id)
        return jsonify({"files": files, "session_id": session_id})

    @app.route("/api/evidence/<file_id>", methods=["DELETE"])
    def delete_evidence_file(file_id: str):
        """Delete an evidence file from a session."""
        from . import evidence_cache

        session_id = request.args.get("session_id")
        if not session_id or not evidence_cache.validate_session_id(session_id):
            return jsonify({"error": "Invalid session_id"}), 400

        if not evidence_cache.validate_file_id(file_id):
            return jsonify({"error": "Invalid file_id"}), 400

        deleted = evidence_cache.delete_evidence(session_id, file_id)
        return jsonify({"deleted": deleted, "file_id": file_id})

    @app.route("/api/run-file", methods=["POST"])
    def run_file():
        """Execute template with uploaded file as {{source}}."""
        import anyio

        from struckdown import LLMCredentials
        from struckdown.errors import StruckdownLLMError

        data = get_json_safe()
        syntax = data.get("syntax", "")
        file_id = data.get("file_id")
        model_name = data.get("model")
        strict_undefined = data.get("strict_undefined", False)

        # Validate syntax length
        if len(syntax) > STRUCKDOWN_MAX_SYNTAX_LENGTH:
            return (
                jsonify(
                    {
                        "error": f"Syntax too long ({len(syntax)} chars, max {STRUCKDOWN_MAX_SYNTAX_LENGTH})",
                        "outputs": {},
                        "cost": None,
                    }
                ),
                400,
            )

        # Check for disallowed actions in remote mode
        if remote_mode:
            disallowed = core.check_disallowed_actions(syntax)
            if disallowed:
                return (
                    jsonify(
                        {
                            "error": f"Actions not allowed in remote mode: {', '.join(disallowed)}",
                            "outputs": {},
                            "cost": None,
                        }
                    ),
                    400,
                )

        try:
            file_data = upload_cache.get_upload(file_id)
            if file_data.get("type") != "source":
                return (
                    jsonify(
                        {"error": "Invalid file type", "outputs": {}, "cost": None}
                    ),
                    400,
                )
        except (FileNotFoundError, ValueError):
            return (
                jsonify(
                    {"error": "File not found or expired", "outputs": {}, "cost": None}
                ),
                400,
            )

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
                    return (
                        jsonify(
                            {
                                "error": "API base URL required. Set it in Settings or contact the administrator.",
                                "outputs": {},
                                "cost": None,
                            }
                        ),
                        400,
                    )
                credentials = LLMCredentials(api_key=user_api_key, base_url=api_base)
            else:
                return (
                    jsonify(
                        {
                            "error": "API key required. Enter your API key in Settings.",
                            "outputs": {},
                            "cost": None,
                        }
                    ),
                    400,
                )

        # Inject file content as {{source}}
        inputs = {"source": file_data["content"]}

        async def execute():
            return await core.run_single(
                syntax=syntax,
                inputs=inputs,
                model_name=model_name,
                credentials=credentials,
                include_paths=app.config["INCLUDE_PATHS"],
                strict_undefined=strict_undefined,
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

        data = get_json_safe()
        syntax = data.get("syntax", "")
        file_id = data.get("file_id")
        model_name = data.get("model")
        strict_undefined = data.get("strict_undefined", False)

        # Validate syntax length
        if len(syntax) > STRUCKDOWN_MAX_SYNTAX_LENGTH:
            return (
                jsonify(
                    {
                        "error": f"Syntax too long ({len(syntax)} chars, max {STRUCKDOWN_MAX_SYNTAX_LENGTH})"
                    }
                ),
                400,
            )

        # Check for disallowed actions in remote mode
        if remote_mode:
            disallowed = core.check_disallowed_actions(syntax)
            if disallowed:
                return (
                    jsonify(
                        {
                            "error": f"Actions not allowed in remote mode: {', '.join(disallowed)}"
                        }
                    ),
                    400,
                )

        try:
            file_data = upload_cache.get_upload(file_id)
            if file_data.get("type") != "batch":
                return jsonify({"error": "Invalid file type"}), 400
        except (FileNotFoundError, ValueError):
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
                    return (
                        jsonify(
                            {
                                "error": "API base URL required. Set it in Settings or contact the administrator."
                            }
                        ),
                        400,
                    )
                credentials = LLMCredentials(api_key=user_api_key, base_url=api_base)
            else:
                return (
                    jsonify(
                        {"error": "API key required. Enter your API key in Settings."}
                    ),
                    400,
                )
        # In local mode, credentials=None will use environment variables

        rows = file_data["data"]["rows"]

        # Create task in file-based cache
        task_id = str(uuid.uuid4())
        task_cache.create_task(
            task_id,
            {
                "status": "pending",
                "total": len(rows),
                "completed": 0,
                "columns": file_data["data"]["columns"],
                "results": [],
                "events": [],
            },
        )

        # Start background processing with timeout
        def process_batch():
            import anyio

            task_cache.update_task(task_id, status="running", start_time=time.time())
            start_time = time.time()

            def on_row_complete(result):
                # Convert to row event format for SSE
                row_event = {
                    "index": result["row_index"],
                    "inputs": result["inputs"],
                    "outputs": result["outputs"],
                    "status": result["status"],
                    "error": result["error"],
                }
                task_cache.append_result(task_id, row_event)
                # Check timeout on each row completion
                elapsed = time.time() - start_time
                if elapsed > STRUCKDOWN_BATCH_TIMEOUT:
                    raise TimeoutError(
                        f"Batch timeout exceeded ({STRUCKDOWN_BATCH_TIMEOUT}s)"
                    )

            try:
                # Run with timeout using anyio
                async def run_with_timeout():
                    with anyio.fail_after(STRUCKDOWN_BATCH_TIMEOUT):
                        async for event in core.run_batch_streaming(
                            syntax=syntax,
                            rows=rows,
                            model_name=model_name,
                            credentials=credentials,
                            include_paths=app.config["INCLUDE_PATHS"],
                            on_row_complete=on_row_complete,
                            slot_level=True,
                            strict_undefined=strict_undefined,
                        ):
                            # Store slot events for SSE streaming
                            if event.get("type") == "slot":
                                task_cache.append_event(task_id, event)

                anyio.run(run_with_timeout)
                task_cache.update_task(task_id, status="complete")
            except TimeoutError:
                task_cache.update_task(
                    task_id,
                    status="error",
                    error=f"Batch processing timed out after {STRUCKDOWN_BATCH_TIMEOUT} seconds",
                )
            except Exception as e:
                task_cache.update_task(
                    task_id, status="error", error=safe_error_message(e, remote_mode)
                )

        thread = threading.Thread(target=process_batch, daemon=True)
        thread.start()

        return jsonify({"task_id": task_id})

    # Apply rate limiting to run-batch endpoint in remote mode
    if limiter:
        run_batch = limiter.limit(STRUCKDOWN_RATE_LIMIT)(run_batch)

    @app.route("/api/batch-stream/<task_id>")
    def batch_stream(task_id: str):
        """SSE stream for batch progress with slot-level updates."""
        task = task_cache.get_task(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404

        def generate():
            last_row_sent = 0
            last_event_sent = 0

            while True:
                # Get current task state from file
                task = task_cache.get_task(task_id)
                if not task:
                    yield f"event: error\ndata: {json.dumps({'error': 'Task not found'})}\n\n"
                    break

                # Send progress update FIRST so client can initialize table
                yield f"event: progress\ndata: {json.dumps({'completed': task.get('completed', 0), 'total': task.get('total', 0)})}\n\n"

                # Send any new slot events (for cell-level updates)
                events = task.get("events", [])
                while last_event_sent < len(events):
                    event = events[last_event_sent]
                    yield f"event: slot\ndata: {json.dumps(event)}\n\n"
                    last_event_sent += 1

                # Send any new row results (for row completion)
                results = task.get("results", [])
                while last_row_sent < len(results):
                    result = results[last_row_sent]
                    yield f"event: row\ndata: {json.dumps(result)}\n\n"
                    last_row_sent += 1

                # Check if done
                status = task.get("status")
                if status in ("complete", "error"):
                    if status == "error":
                        yield f"event: error\ndata: {json.dumps({'error': task.get('error', 'Unknown error')})}\n\n"
                    else:
                        yield f"event: done\ndata: {json.dumps({'task_id': task_id})}\n\n"
                    break

                # Small delay before next poll
                time.sleep(0.1)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/download/<task_id>")
    def download(task_id: str):
        """Download completed batch results as xlsx."""
        from io import BytesIO

        import pandas as pd

        task = task_cache.get_task(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404

        if task.get("status") != "complete":
            return jsonify({"error": "Task not complete"}), 400

        # Build dataframe from results
        rows_data = []
        for result in sorted(task.get("results", []), key=lambda r: r["index"]):
            row = {**result["inputs"], **result["outputs"]}
            if result.get("error"):
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
            headers={
                "Content-Disposition": f"attachment; filename=results_{task_id[:8]}.xlsx"
            },
        )

    @app.route("/api/encode-state", methods=["POST"])
    def encode_state_route():
        """Encode current state for URL sharing."""
        data = get_json_safe()
        encoded = core.encode_state(
            syntax=data.get("syntax", ""),
            model=data.get("model", ""),
            inputs=data.get("inputs", {}),
        )
        return jsonify({"encoded": encoded})

    @app.route("/api/save-prompt", methods=["POST"])
    def save_prompt():
        """Save prompt text and return content hash (remote mode only).

        Uses content-based hashing so same content always returns same hash.
        Only saves the prompt text -- never inputs or outputs.
        """
        if not remote_mode:
            return jsonify({"error": "Only available in remote mode"}), 400

        data = get_json_safe()
        syntax = data.get("syntax", "")

        if not syntax or not syntax.strip():
            return jsonify({"error": "Empty prompt"}), 400

        if len(syntax) > STRUCKDOWN_MAX_SYNTAX_LENGTH:
            return jsonify({"error": "Prompt too long"}), 400

        # Store and get content-based hash
        prompt_id = prompt_cache.store_prompt(syntax)

        return jsonify({"prompt_id": prompt_id})

    # Apply rate limiting to save-prompt endpoint in remote mode
    if limiter:
        save_prompt = limiter.limit(STRUCKDOWN_PROMPT_RATE_LIMIT)(save_prompt)

    @app.route("/partials/inputs", methods=["POST"])
    def render_inputs_partial():
        """Render inputs panel with current fields."""
        data = get_json_safe()
        inputs_required = data.get("inputs_required", [])
        current_values = data.get("current_values", {})
        uses_history = data.get("uses_history", False)

        return render_template(
            "partials/inputs_panel.html",
            inputs_required=inputs_required,
            current_values=current_values,
            uses_history=uses_history,
        )

    @app.route("/partials/outputs", methods=["POST"])
    def render_outputs_partial():
        """Render outputs panel with results."""
        data = get_json_safe()
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
