"""Core logic for the Struckdown Playground.

Framework-agnostic functions for template analysis, validation, and execution.
"""

import base64
import json
import os
import re
import zlib
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import anyio
import pandas as pd
from lark.exceptions import UnexpectedCharacters, UnexpectedToken

from struckdown import (
    LLM,
    LLMCredentials,
    chatter_async,
    extract_jinja_variables,
)
from struckdown.parsing import extract_slot_key, find_slots_with_positions, parser

# Security configuration from environment
STRUCKDOWN_ZIP_MAX_SIZE = int(os.environ.get("STRUCKDOWN_ZIP_MAX_SIZE", "52428800"))  # 50MB
STRUCKDOWN_ZIP_MAX_FILES = int(os.environ.get("STRUCKDOWN_ZIP_MAX_FILES", "500"))
STRUCKDOWN_STATE_MAX_SIZE = int(os.environ.get("STRUCKDOWN_STATE_MAX_SIZE", "1048576"))  # 1MB max decompressed state


def extract_required_inputs(syntax: str) -> Dict[str, List[str]]:
    """
    Analyse template to determine required inputs.

    Returns dict with:
        inputs_required: {{vars}} not filled by [[slots]]
        slots_defined: [[slots]] that will be created
    """
    # Get all {{var}} references
    jinja_vars = extract_jinja_variables(syntax)

    # Get all [[slot]] definitions in order they appear
    slots = find_slots_with_positions(syntax)
    slot_names_set = {s[0] for s in slots}  # s[0] is the extracted key

    # Preserve order of first appearance for slots
    seen = set()
    slots_ordered = []
    for s in slots:
        if s[0] not in seen:
            seen.add(s[0])
            slots_ordered.append(s[0])

    # Inputs are vars not defined by slots
    inputs_required = sorted(jinja_vars - slot_names_set)

    return {
        "inputs_required": inputs_required,
        "slots_defined": slots_ordered,
    }


def check_disallowed_actions(syntax: str) -> List[str]:
    """
    Check for actions that are not allowed in remote mode.

    Returns list of disallowed action names found in the syntax.
    """
    from struckdown.actions import Actions

    # Find all action calls: [[@action_name:...]] or [[@action_name|...]]
    action_pattern = re.compile(r'\[\[@(\w+)[:\|]')
    found_actions = set(action_pattern.findall(syntax))

    # Check which ones are disallowed in remote mode
    disallowed = []
    for action_name in found_actions:
        if Actions.is_registered(action_name) and not Actions.is_allowed_remote(action_name):
            disallowed.append(action_name)

    return sorted(disallowed)


def uses_history_action(syntax: str) -> bool:
    """Check if the syntax uses the [[@history]] action."""
    pattern = re.compile(r'\[\[@history(?:[:\|]|]])')
    return bool(pattern.search(syntax))


def validate_syntax(syntax: str) -> Dict:
    """
    Validate struckdown syntax.

    Returns dict with:
        valid: bool
        error: { line, column, message } or None
    """
    if not syntax or not syntax.strip():
        return {"valid": True, "error": None}

    try:
        # Attempt to parse
        p = parser()
        p.parse(syntax)
        return {"valid": True, "error": None}
    except (UnexpectedToken, UnexpectedCharacters) as e:
        return {
            "valid": False,
            "error": {
                "line": getattr(e, "line", 1),
                "column": getattr(e, "column", 1),
                "message": str(e),
            },
        }
    except Exception as e:
        return {
            "valid": False,
            "error": {
                "line": 1,
                "column": 1,
                "message": str(e),
            },
        }


def encode_state(syntax: str, model: str = "", inputs: dict = None) -> str:
    """Encode editor state to URL-safe string."""
    state = {"s": syntax, "m": model or "", "i": inputs or {}}
    json_bytes = json.dumps(state, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(json_bytes, level=9)
    return base64.urlsafe_b64encode(compressed).decode("ascii")


def decode_state(encoded: str) -> dict:
    """Decode URL state back to components.

    Uses max_length parameter to prevent decompression bombs.
    """
    try:
        compressed = base64.urlsafe_b64decode(encoded)

        # Use max_length to limit decompressed size and prevent decompression bombs
        decompressor = zlib.decompressobj()
        json_bytes = decompressor.decompress(compressed, max_length=STRUCKDOWN_STATE_MAX_SIZE)

        # Check if there's unconsumed data (meaning we hit the limit)
        if decompressor.unconsumed_tail:
            raise ValueError("Decompressed state exceeds size limit")

        state = json.loads(json_bytes)
        return {
            "syntax": state.get("s", ""),
            "model": state.get("m", ""),
            "inputs": state.get("i", {}),
        }
    except Exception:
        return {"syntax": "", "model": "", "inputs": {}}


def load_xlsx_data(file_path: Path) -> Dict:
    """
    Load xlsx/csv file and return data for batch processing.

    Returns dict with:
        rows: list of dicts (one per row)
        columns: list of column names
        row_count: number of rows
    """
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Convert to list of dicts, handling NaN values
    rows = df.fillna("").to_dict(orient="records")

    return {
        "rows": rows,
        "columns": list(df.columns),
        "row_count": len(rows),
    }


def load_zip_data(file_path: Path) -> Dict:
    """
    Load zip archive and return data for batch processing.

    Each file in the archive becomes a row with:
        - source: file content (text)
        - filename: original filename

    Returns dict with:
        rows: list of dicts (one per file)
        columns: list of column names
        row_count: number of files

    Raises:
        ValueError: If zip exceeds security limits (file count, total size)
    """
    import zipfile

    path = Path(file_path)
    rows = []

    with zipfile.ZipFile(path, "r") as zf:
        # Security: check total uncompressed size and file count
        file_infos = [info for info in zf.infolist()
                      if not info.is_dir()
                      and not info.filename.startswith("__")
                      and not info.filename.startswith(".")]

        if len(file_infos) > STRUCKDOWN_ZIP_MAX_FILES:
            raise ValueError(
                f"Zip contains too many files ({len(file_infos)}, max {STRUCKDOWN_ZIP_MAX_FILES})"
            )

        total_uncompressed = sum(info.file_size for info in file_infos)
        if total_uncompressed > STRUCKDOWN_ZIP_MAX_SIZE:
            raise ValueError(
                f"Zip uncompressed size too large ({total_uncompressed} bytes, max {STRUCKDOWN_ZIP_MAX_SIZE})"
            )

        for info in file_infos:
            try:
                content = zf.read(info.filename).decode("utf-8", errors="replace")
                # Sanitise filename (strip path components for security)
                safe_filename = Path(info.filename).name
                rows.append({
                    "source": content,
                    "filename": safe_filename,
                })
            except Exception:
                # Skip files that can't be read as text
                continue

    return {
        "rows": rows,
        "columns": ["source", "filename"],
        "row_count": len(rows),
    }


def load_batch_file(file_path: Path) -> Dict:
    """
    Load batch file (xlsx, csv, or zip) and return data for processing.

    Returns dict with:
        rows: list of dicts (one per row/file)
        columns: list of column names
        row_count: number of rows/files
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".zip":
        return load_zip_data(path)
    else:
        return load_xlsx_data(path)


async def run_single(
    syntax: str,
    inputs: Dict[str, Any],
    model_name: str = None,
    credentials: LLMCredentials = None,
    include_paths: List[Path] = None,
) -> Dict:
    """
    Execute template with given inputs.

    Returns dict with:
        outputs: dict of slot_name -> value
        cost: cost info dict or None
        error: error message string or None
    """
    try:
        model = LLM(model_name=model_name) if model_name else LLM()
        result = await chatter_async(
            syntax,
            model=model,
            credentials=credentials,
            context=inputs,
            include_paths=include_paths,
        )

        # Build cost info from ChatterResult properties
        cost_info = None
        try:
            cost_info = {
                "total_cost": result.total_cost,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
            }
        except Exception:
            pass

        return {
            "outputs": result.outputs,
            "cost": cost_info,
            "error": None,
        }
    except Exception as e:
        return {
            "outputs": {},
            "cost": None,
            "error": str(e),
        }


async def run_batch_streaming(
    syntax: str,
    rows: List[Dict[str, Any]],
    model_name: str = None,
    credentials: LLMCredentials = None,
    include_paths: List[Path] = None,
    max_concurrent: int = 10,
    on_row_complete=None,
    slot_level: bool = False,
) -> AsyncGenerator[Dict, None]:
    """
    Execute template for each row, yielding results as they complete.

    When slot_level=False (default), yields row-level events:
        type: "row"
        index: row index
        inputs: original input values
        outputs: slot values
        status: "complete" | "error"
        error: error message if status is "error"

    When slot_level=True, yields slot-level events as well:
        type: "slot"
        row_index: row index
        slot_key: slot name
        value: slot output value
        elapsed_ms: time for this slot
        was_cached: whether result was cached

        type: "row_complete"
        row_index: row index
        outputs: all slot values for this row
        status: "complete" | "error"
        error: error message if status is "error"

    Args:
        syntax: The struckdown template
        rows: List of input dicts (one per row)
        model_name: LLM model name
        credentials: API credentials
        include_paths: Paths for includes
        max_concurrent: Max concurrent executions
        on_row_complete: Optional callback called with result dict
        slot_level: If True, yield slot-level events for incremental updates
    """
    from struckdown import chatter_incremental_async

    model = LLM(model_name=model_name) if model_name else LLM()
    semaphore = anyio.Semaphore(max_concurrent)

    # Use a larger buffer for slot-level events
    buffer_size = len(rows) * 20 if slot_level else len(rows)
    send_channel, receive_channel = anyio.create_memory_object_stream(max_buffer_size=buffer_size)

    async def process_row(index: int, row: Dict):
        async with semaphore:
            if slot_level:
                # Use incremental mode for slot-level events
                outputs = {}
                try:
                    async for event in chatter_incremental_async(
                        syntax,
                        model=model,
                        credentials=credentials,
                        context=row,
                        include_paths=include_paths,
                    ):
                        if event.type == "slot_completed":
                            outputs[event.slot_key] = event.result.output
                            await send_channel.send({
                                "type": "slot",
                                "row_index": index,
                                "slot_key": event.slot_key,
                                "value": event.result.output,
                                "elapsed_ms": event.elapsed_ms,
                                "was_cached": event.was_cached,
                            })
                        elif event.type == "complete":
                            row_result = {
                                "type": "row_complete",
                                "row_index": index,
                                "inputs": row,
                                "outputs": outputs,
                                "status": "complete",
                                "error": None,
                            }
                            await send_channel.send(row_result)
                            if on_row_complete:
                                on_row_complete(row_result)
                        elif event.type == "error":
                            row_result = {
                                "type": "row_complete",
                                "row_index": index,
                                "inputs": row,
                                "outputs": outputs,
                                "status": "error",
                                "error": event.error_message,
                            }
                            await send_channel.send(row_result)
                            if on_row_complete:
                                on_row_complete(row_result)
                except Exception as e:
                    row_result = {
                        "type": "row_complete",
                        "row_index": index,
                        "inputs": row,
                        "outputs": outputs,
                        "status": "error",
                        "error": str(e),
                    }
                    await send_channel.send(row_result)
                    if on_row_complete:
                        on_row_complete(row_result)
            else:
                # Original row-level mode
                try:
                    result = await chatter_async(
                        syntax,
                        model=model,
                        credentials=credentials,
                        context=row,
                        include_paths=include_paths,
                    )
                    row_result = {
                        "type": "row",
                        "index": index,
                        "inputs": row,
                        "outputs": result.outputs,
                        "status": "complete",
                        "error": None,
                    }
                except Exception as e:
                    row_result = {
                        "type": "row",
                        "index": index,
                        "inputs": row,
                        "outputs": {},
                        "status": "error",
                        "error": str(e),
                    }

                await send_channel.send(row_result)
                if on_row_complete:
                    on_row_complete(row_result)

    async def producer():
        async with anyio.create_task_group() as tg:
            for i, row in enumerate(rows):
                tg.start_soon(process_row, i, row)
        await send_channel.aclose()

    # Start producer in background
    async with anyio.create_task_group() as tg:
        tg.start_soon(producer)

        # Yield results as they arrive
        async with receive_channel:
            async for result in receive_channel:
                yield result


def run_batch_sync(
    syntax: str,
    rows: List[Dict[str, Any]],
    model_name: str = None,
    credentials: LLMCredentials = None,
    include_paths: List[Path] = None,
    max_concurrent: int = 10,
    on_row_complete=None,
) -> List[Dict]:
    """
    Synchronous wrapper for batch processing.

    Returns list of result dicts in completion order.
    """
    results = []

    async def collect_results():
        async for result in run_batch_streaming(
            syntax=syntax,
            rows=rows,
            model_name=model_name,
            credentials=credentials,
            include_paths=include_paths,
            max_concurrent=max_concurrent,
            on_row_complete=on_row_complete,
        ):
            results.append(result)

    anyio.run(collect_results)
    return results
