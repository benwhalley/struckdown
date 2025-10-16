import json
import logging
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import typer
from decouple import config as env_config

from . import ACTION_LOOKUP, LLM, LLMCredentials, chatter
from .output_formatters import write_output

from jinja2 import Environment, meta

app = typer.Typer(help="struckdown: structured conversations with language models")

logger = logging.getLogger(__name__)


def auto_prepend_input(prompt: str) -> str:
    """
    Auto-prepend {{input}} to the prompt if no template variables are referenced.

    Logic:
    - If {{input}} or {{content}} is already referenced: do nothing
    - If other placeholders exist (e.g., {{name}}, {{price}}): do nothing
    - If NO placeholders exist at all: prepend {{input}}

    Uses Jinja2's meta.find_undeclared_variables to parse the template
    and detect variable references.
    """
    env = Environment()
    try:
        # Parse the template to find all variable references
        ast = env.parse(prompt)
        variables = meta.find_undeclared_variables(ast)

        # If any template variables exist, don't prepend
        if variables:
            return prompt
        else:
            # Auto-prepend {{input}} if no variables are referenced
            logger.debug("Auto-prepending {{input}} to prompt")
            return f"{{{{input}}}}\n\n{prompt}"
    except Exception as e:
        # If parsing fails, return prompt unchanged
        logger.warning(f"Failed to parse template for auto-prepend: {e}")
        return prompt


@app.command()
def chat(
    prompt: List[str] = typer.Argument(
        ..., help="Prompt with slots, e.g. tell a joke [[joke]]"
    ),
    model_name: Optional[str] = typer.Option(
        env_config("DEFAULT_LLM", default=None, cast=str),
        help="LLM model name (overrides DEFAULT_LLM env var)",
    ),
    show_context: bool = typer.Option(False, help="Print the resolved prompt context"),
    verbose: bool = typer.Option(False, help="Print the full ChatterResult object"),
):
    """
    Run a single chatter prompt (interactive mode).

    Example:
        sd chat "tell a joke [[joke]]"
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("struckdown").setLevel(logging.DEBUG)

    prompt_str = " ".join(prompt)
    credentials = LLMCredentials()
    model = LLM(model_name=model_name)

    result = chatter(
        multipart_prompt=prompt_str,
        model=model,
        credentials=credentials,
        action_lookup=ACTION_LOOKUP,
    )

    for k, v in result.results.items():
        typer.echo(f"{k}: {v.output}")

    if show_context:
        typer.echo("\nFinal context:")
        typer.echo(result.outputs)


@app.command()
def batch(
    inputs: List[str] = typer.Argument(
        None,
        help="Input files or glob patterns (e.g., inputs/*.txt). If omitted, reads from stdin."
    ),
    prompt: Optional[str] = typer.Argument(
        None,
        help="Prompt with slots, e.g. 'extract name [[name]]'"
    ),
    prompt_file: Optional[Path] = typer.Option(
        None,
        "-p",
        "--prompt",
        help="Path to file containing the prompt"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file (format inferred from extension: .json, .csv, .xlsx, .md, .txt)"
    ),
    keep_inputs: bool = typer.Option(
        False,
        "-k",
        "--keep-inputs",
        help="Include input fields (input, content, filename, basename) in output"
    ),
    model_name: Optional[str] = typer.Option(
        env_config("DEFAULT_LLM", default=None, cast=str),
        help="LLM model name (overrides DEFAULT_LLM env var)",
    ),
    verbose: bool = typer.Option(False, help="Enable debug logging"),
):
    """
    Process multiple inputs in batch mode.

    Examples:
        sd batch inputs/*.txt "extract [[name]]" -o results.json
        sd batch data.json "welcome for {name} [[msg]]"
        cat file.txt | sd batch "extract [[name]]"
        sd batch inputs/*.txt -p prompt.sd -o results.csv
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("struckdown").setLevel(logging.DEBUG)

    # Validate prompt arguments
    if prompt_file and prompt:
        typer.echo("Error: Cannot specify both inline prompt and --prompt file", err=True)
        raise typer.Exit(1)

    if not prompt_file and not prompt:
        typer.echo("Error: Must specify either prompt or --prompt file", err=True)
        raise typer.Exit(1)

    # Load prompt from file if specified
    if prompt_file:
        if not prompt_file.exists():
            typer.echo(f"Error: Prompt file not found: {prompt_file}", err=True)
            raise typer.Exit(1)
        prompt = prompt_file.read_text(encoding="utf-8")

    # Auto-prepend {{input}} if not referenced in prompt
    prompt = auto_prepend_input(prompt)

    # Determine input source
    input_data = []

    if inputs:
        # Process file arguments (may include globs)
        file_paths = []
        for pattern in inputs:
            # Check if it's a glob pattern or regular file
            matches = glob(pattern, recursive=True)
            if matches:
                file_paths.extend(matches)
            elif Path(pattern).exists():
                file_paths.append(pattern)
            else:
                logger.warning(f"No files matched pattern: {pattern}")

        if not file_paths:
            typer.echo("Error: No input files found", err=True)
            raise typer.Exit(1)

        # Process each file
        for file_path in file_paths:
            path = Path(file_path)
            try:
                input_data.extend(_read_input_file(path))
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                if verbose:
                    raise

    elif not sys.stdin.isatty():
        # Read from stdin
        stdin_content = sys.stdin.read()
        if not stdin_content.strip():
            typer.echo("Error: stdin is empty", err=True)
            raise typer.Exit(1)

        try:
            # Try to parse as JSON
            data = json.loads(stdin_content)
            if isinstance(data, list):
                input_data = data
            elif isinstance(data, dict):
                input_data = [data]
            else:
                typer.echo(f"Error: stdin JSON must be dict or list, got {type(data)}", err=True)
                raise typer.Exit(1)
        except json.JSONDecodeError:
            # Treat as plain text
            input_data = [{"input": stdin_content, "content": stdin_content, "filename": "<stdin>"}]

    else:
        typer.echo("Error: No input provided (specify files or pipe to stdin)", err=True)
        raise typer.Exit(1)

    # Process each input
    credentials = LLMCredentials()
    model = LLM(model_name=model_name)
    results = []
    errors = []

    for i, input_item in enumerate(input_data):
        try:
            # Execute chatter with the input context
            result = chatter(
                multipart_prompt=prompt,
                model=model,
                credentials=credentials,
                context=input_item,
                action_lookup=ACTION_LOOKUP,
            )

            # Merge input data with extracted results
            if keep_inputs:
                output_item = input_item.copy()
            else:
                # Start with empty dict, only include extracted results
                output_item = {}
                # Keep filename for traceability
                if 'filename' in input_item:
                    output_item['filename'] = input_item['filename']

            for key, segment_result in result.results.items():
                output_item[key] = segment_result.output

            results.append(output_item)

            if verbose:
                typer.echo(f"Processed item {i+1}/{len(input_data)}: {output_item.get('filename', f'item_{i}')}", err=True)

        except Exception as e:
            error_msg = f"Error processing item {i+1}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            if verbose:
                import traceback
                typer.echo(traceback.format_exc(), err=True)

    # Report errors if any
    if errors:
        typer.echo(f"\nCompleted with {len(errors)} error(s):", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)

    # Write output
    if results:
        write_output(results, output)
    else:
        typer.echo("Error: No results produced", err=True)
        raise typer.Exit(1)


def _read_input_file(path: Path) -> List[dict]:
    """
    Read an input file and return a list of input items.

    For text files: returns [{"input": "...", "content": "...", "filename": "..."}]
    For JSON files:
        - If dict: returns [dict]
        - If list: returns list
    """
    extension = path.suffix.lower()

    if extension == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Ensure each item has filename
            for item in data:
                if isinstance(item, dict) and "filename" not in item:
                    item["filename"] = str(path)
            return data
        elif isinstance(data, dict):
            if "filename" not in data:
                data["filename"] = str(path)
            return [data]
        else:
            raise ValueError(f"JSON file must contain dict or list, got {type(data)}")

    else:
        # Treat as text file
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return [{
            "input": content,
            "content": content,
            "filename": str(path),
            "basename": path.stem
        }]


if __name__ == "__main__":
    app()
