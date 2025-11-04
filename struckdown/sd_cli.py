import json
import logging
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import anyio
import typer
from decouple import config as env_config
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.console import Console

from . import ACTION_LOOKUP, LLM, LLMCredentials, chatter, chatter_async, __version__
from .output_formatters import write_output, render_template

from jinja2 import Environment, meta

app = typer.Typer(help="struckdown: structured conversations with language models")

logger = logging.getLogger(__name__)

# Concurrency settings
MAX_CONCURRENCY = env_config("SD_MAX_CONCURRENCY", default=20, cast=int)
semaphore = anyio.Semaphore(MAX_CONCURRENCY)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    )
):
    """struckdown: structured conversations with language models"""
    pass


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
    prompt: Optional[List[str]] = typer.Argument(
        None, help="Prompt with slots, e.g. tell a joke [[joke]]"
    ),
    prompt_file: Optional[Path] = typer.Option(
        None,
        "-p",
        "--prompt-file",
        help="Path to file containing the prompt"
    ),
    model_name: Optional[str] = typer.Option(
        env_config("DEFAULT_LLM", default=None, cast=str),
        help="LLM model name (overrides DEFAULT_LLM env var)",
    ),
    show_context: bool = typer.Option(False, help="Print the resolved prompt context"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Print the full ChatterResult object"),
):
    """
    Run a single chatter prompt (interactive mode).

    Examples:
        sd chat "tell a joke [[joke]]"
        cat prompt.sd | sd chat
        sd chat -p prompt.sd
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("struckdown").setLevel(logging.DEBUG)

    # Determine prompt source
    prompt_str = None

    if prompt_file:
        # Read from prompt file
        if not prompt_file.exists():
            typer.echo(f"Error: Prompt file not found: {prompt_file}", err=True)
            raise typer.Exit(1)
        prompt_str = prompt_file.read_text(encoding="utf-8")
    elif prompt:
        # Use positional arguments
        prompt_str = " ".join(prompt)
    elif not sys.stdin.isatty():
        # Read from stdin
        stdin_content = sys.stdin.read()
        if not stdin_content.strip():
            typer.echo("Error: stdin is empty", err=True)
            raise typer.Exit(1)
        prompt_str = stdin_content.strip()
    else:
        typer.echo("Error: No prompt provided. Use a positional argument, -p/--prompt-file, or pipe to stdin.", err=True)
        raise typer.Exit(1)
    credentials = LLMCredentials()
    model = LLM(model_name=model_name)

    result = chatter(
        multipart_prompt=prompt_str,
        model=model,
        credentials=credentials,
    )

    for k, v in result.results.items():
        typer.echo(f"{k}: {v.output}")

    if show_context:
        typer.echo("\nFinal context:")
        typer.echo(result.outputs)


async def batch_async(
    prompt: str,
    input_data: List[dict],
    output: Optional[List[Path]],
    keep_inputs: bool,
    template: Optional[Path],
    model_name: Optional[str],
    max_concurrent: int,
    verbose: bool,
    quiet: bool,
):
    """
    Async implementation of batch processing with concurrent execution.
    """
    # Process each input
    credentials = LLMCredentials()
    model = LLM(model_name=model_name)
    results = [None] * len(input_data)
    errors = []

    # Determine if we should show progress bar
    show_progress = not quiet and sys.stderr.isatty()

    # Create console for progress output (always to stderr)
    console = Console(stderr=True)

    # Create semaphore for concurrency control
    sem = anyio.Semaphore(max_concurrent)

    # Process with or without progress bar
    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing", total=len(input_data))

            async with anyio.create_task_group() as tg:
                for idx, input_item in enumerate(input_data):

                    async def run_and_store(index=idx, item=input_item, progress_bar=progress, progress_task=task):
                        async with sem:
                            try:
                                # Execute chatter_async with the input context
                                result = await chatter_async(
                                    multipart_prompt=prompt,
                                    model=model,
                                    credentials=credentials,
                                    context=item,
                                )

                                # Merge input data with extracted results
                                if keep_inputs:
                                    output_item = item.copy()
                                else:
                                    # Start with empty dict, only include extracted results
                                    output_item = {}
                                    # Keep filename for traceability
                                    if 'filename' in item:
                                        output_item['filename'] = item['filename']

                                for key, segment_result in result.results.items():
                                    output_item[key] = segment_result.output

                                results[index] = output_item

                                if verbose:
                                    console.print(f"Processed item {index+1}/{len(input_data)}: {output_item.get('filename', f'item_{index}')}")

                            except Exception as e:
                                error_msg = f"Error processing item {index+1}: {e}"
                                logger.error(error_msg)
                                errors.append(error_msg)
                                if verbose:
                                    import traceback
                                    console.print(traceback.format_exc())

                            finally:
                                # Update progress bar on completion
                                if progress_bar is not None:
                                    progress_bar.update(progress_task, advance=1)

                    tg.start_soon(run_and_store)
    else:
        # No progress bar, just process concurrently
        async with anyio.create_task_group() as tg:
            for idx, input_item in enumerate(input_data):

                async def run_and_store(index=idx, item=input_item):
                    async with sem:
                        try:
                            # Execute chatter_async with the input context
                            result = await chatter_async(
                                multipart_prompt=prompt,
                                model=model,
                                credentials=credentials,
                                context=item,
                            )

                            # Merge input data with extracted results
                            if keep_inputs:
                                output_item = item.copy()
                            else:
                                # Start with empty dict, only include extracted results
                                output_item = {}
                                # Keep filename for traceability
                                if 'filename' in item:
                                    output_item['filename'] = item['filename']

                            for key, segment_result in result.results.items():
                                output_item[key] = segment_result.output

                            results[index] = output_item

                            if verbose:
                                console.print(f"Processed item {index+1}/{len(input_data)}: {output_item.get('filename', f'item_{index}')}")

                        except Exception as e:
                            error_msg = f"Error processing item {index+1}: {e}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                            if verbose:
                                import traceback
                                console.print(traceback.format_exc())

                tg.start_soon(run_and_store)

    # Report errors if any
    if errors:
        typer.echo(f"\nCompleted with {len(errors)} error(s):", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)

    # Write output(s)
    if not results or all(r is None for r in results):
        typer.echo("Error: No results produced", err=True)
        raise typer.Exit(1)

    # Filter out None results from errors
    results = [r for r in results if r is not None]

    if output:
        # Write to multiple outputs
        for output_path in output:
            # Check if this is a JSON file
            is_json = str(output_path).lower().endswith('.json')

            if is_json or not template:
                # Use format auto-detection for JSON or when no template specified
                write_output(results, output_path)
            else:
                # Use template rendering for non-JSON outputs when template is specified
                render_template(results, output_path, template)
    else:
        # No outputs specified, write to stdout
        write_output(results, None)


@app.command()
def batch(
    prompt: Optional[str] = typer.Argument(
        None,
        help="Prompt with slots, e.g. 'extract name [[name]]'. Omit if using --prompt/-p flag."
    ),
    input_files: Optional[List[str]] = typer.Option(
        None,
        "-i",
        "--input",
        help="Input files or glob patterns (e.g., -i '*.txt' -i 'data/*.json'). Can be specified multiple times."
    ),
    prompt_file: Optional[Path] = typer.Option(
        None,
        "-p",
        "--prompt",
        help="Path to file containing the prompt"
    ),
    output: Optional[List[Path]] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file (format inferred from extension: .json, .csv, .xlsx, .md, .txt). Can be specified multiple times."
    ),
    keep_inputs: bool = typer.Option(
        False,
        "-k",
        "--keep-inputs",
        help="Include input fields (input, content, source, filename, basename) in output"
    ),
    template: Optional[Path] = typer.Option(
        None,
        "-t",
        "--template",
        help="Jinja2 template file to apply to non-JSON outputs"
    ),
    model_name: Optional[str] = typer.Option(
        env_config("DEFAULT_LLM", default=None, cast=str),
        "-m", "--model",
        help="LLM model name (overrides DEFAULT_LLM env var)",
    ),
    max_concurrent: int = typer.Option(
        20,
        "-c",
        "--concurrency",
        help="Maximum number of concurrent API requests"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
):
    """
    Process multiple inputs in batch mode.

    Examples:
        sd batch -i '*.txt' -p prompt.sd -o results.json
        sd batch -i 'data.json' "welcome for {{name}} [[msg]]"
        cat file.txt | sd batch "extract [[name]]"
        sd batch -i '*.txt' -p prompt.sd -o results.json -o report.html -t template.j2

    Multiple outputs: Use -t flag to apply a Jinja2 template to all non-JSON outputs.
    JSON outputs always use standard JSON format. Without -t, output format is inferred from extension.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("struckdown").setLevel(logging.DEBUG)

    # Validate template usage
    if template:
        if not output:
            typer.echo("Error: -t/--template requires at least one -o/--output", err=True)
            raise typer.Exit(1)

        # Check if there's at least one non-JSON output
        has_non_json = any(
            not str(out_path).lower().endswith('.json')
            for out_path in output
        )
        if not has_non_json:
            typer.echo("Error: -t/--template requires at least one non-JSON output file", err=True)
            raise typer.Exit(1)

        # Validate template file exists
        if not template.exists():
            typer.echo(f"Error: Template file not found: {template}", err=True)
            raise typer.Exit(1)

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

    if input_files:
        # Process file arguments (may include globs)
        file_paths = []
        for pattern in input_files:
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

    # Call async batch processing
    anyio.run(
        batch_async,
        prompt,
        input_data,
        output,
        keep_inputs,
        template,
        model_name,
        max_concurrent,
        verbose,
        quiet,
    )


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
            "source": content,  # alias for compatibility
            "filename": str(path),
            "basename": path.stem
        }]


if __name__ == "__main__":
    app()
