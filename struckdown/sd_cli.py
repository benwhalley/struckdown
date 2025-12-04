import json
import logging
import sys
import traceback
from glob import glob
from pathlib import Path
from typing import List, Optional

import anyio
import typer
from decouple import config as env_config
from jinja2 import Environment, meta
from lark.exceptions import UnexpectedCharacters, UnexpectedToken
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeRemainingColumn)

from . import (ACTION_LOOKUP, LLM, CostSummary, LLMCredentials, StruckdownLLMError,
               StruckdownTemplateError, __version__, chatter, chatter_async,
               progress_tracking)
from .output_formatters import render_template, write_output
from .actions import discover_actions, load_actions
from .type_loader import discover_yaml_types, load_yaml_types

app = typer.Typer(help="struckdown: structured conversations with language models")

logger = logging.getLogger(__name__)


def format_parse_error(e: UnexpectedToken | UnexpectedCharacters) -> str:
    """Format Lark parse error with user-friendly message."""
    parts = [f"Error: Parse error at line {e.line}, column {e.column}"]

    if hasattr(e, 'expected') and e.expected:
        parts.append(f"  Expected: {', '.join(sorted(e.expected))}")

    if hasattr(e, 'token'):
        token_type = e.token.type
        token_value = repr(e.token.value)
        # For anonymous tokens, just show the value
        if token_type.startswith('__ANON'):
            parts.append(f"  Got: {token_value}")
        else:
            parts.append(f"  Got: {token_type} ({token_value})")
    elif hasattr(e, 'char'):
        parts.append(f"  Got: {repr(e.char)}")

    return '\n'.join(parts)

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


def _resolve_template_includes(prompt_file: Path) -> str:
    """Resolve both <include> tags and Jinja2 includes in a template file.

    Args:
        prompt_file: Path to the template file

    Returns:
        The template content with all includes resolved

    Raises:
        Exception: If template rendering fails (e.g., include file not found)
    """
    from jinja2 import Environment, FileSystemLoader
    from struckdown import KeepUndefined
    from struckdown.parsing import resolve_includes

    # Read template
    template_text = prompt_file.read_text()

    # Configure search paths (same as chatter)
    search_paths = [
        prompt_file.parent,
        prompt_file.parent / 'templates',
        Path.cwd(),
        Path.cwd() / 'includes',
        Path.cwd() / 'templates',
        Path.home() / '.struckdown' / 'includes'
    ]
    search_paths = [p for p in search_paths if p.exists() and p.is_dir()]

    if not search_paths:
        search_paths = [prompt_file.parent]  # At minimum, template directory

    # First resolve <include src="..."/> tags (compile-time struckdown includes)
    template_text = resolve_includes(template_text, prompt_file.parent, search_paths)

    # Then resolve Jinja2 {% include %} tags
    # NOTE: We don't use struckdown_finalize here because we're just expanding
    # includes, not substituting variables. The struckdown syntax must remain intact.
    env = Environment(
        undefined=KeepUndefined,
        loader=FileSystemLoader(search_paths)
    )
    template = env.from_string(template_text)
    # Render with empty context (just expand includes, don't substitute {{vars}})
    return template.render()


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



def setup_logging(verbosity: int):
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

@app.command()
def chat(
    prompt: Optional[List[str]] = typer.Argument(
        None, help="Prompt with slots, e.g. tell a joke [[joke]]"
    ),
    prompt_file: Optional[Path] = typer.Option(
        None, "-p", "--prompt-file", help="Path to file containing the prompt"
    ),
    source: Optional[str] = typer.Option(
        None, "-s", "--source", help="Source file or URL to include as {{source}} in the template"
    ),
    raw_source: bool = typer.Option(
        False, "--raw", help="When source is a URL, fetch raw HTML instead of extracted markdown"
    ),
    context_vars: Optional[List[str]] = typer.Option(
        None, "-c", "--context", help="Context variable as key=value (can be repeated)"
    ),
    model_name: Optional[str] = typer.Option(
        env_config("DEFAULT_LLM", default=None, cast=str),
        help="LLM model name (overrides DEFAULT_LLM env var)",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible outputs (if supported by model)",
    ),
    show_context: bool = typer.Option(False, help="Print the resolved prompt context"),
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,
        help="-v for info, -vv for debug",
    ),
    include_paths: Optional[List[Path]] = typer.Option(
        None,
        "-I",
        "--include-path",
        help="Additional directories to search for includes (can be repeated)",
    ),
    type_files: Optional[List[Path]] = typer.Option(
        None,
        "--type",
        help="YAML type definition file or directory (can be repeated)",
    ),
    tools_files: Optional[List[Path]] = typer.Option(
        None,
        "--tools",
        help="Python tools file or directory (can be repeated)",
    ),
):
    """
    Run a single chatter prompt (interactive mode).

    Examples:
        sd chat "tell a joke [[joke]]"
        cat prompt.sd | sd chat
        sd chat -p prompt.sd
        sd chat -p prompt.sd -s input.txt       # {{source}} available in template
        sd chat -p prompt.sd -c topic=sun       # {{topic}} = "sun"
        sd chat -p prompt.sd -c x=1 -c y=2      # multiple context vars
        sd chat -p prompt.sd -I ./includes      # search ./includes for <include> files
    """
    # start new run for cache detection
    from struckdown import new_run

    new_run()

    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # load custom types and actions
    if type_files:
        loaded_types = load_yaml_types(type_files)
        if verbose and loaded_types:
            typer.echo(f"Loaded types: {', '.join(loaded_types)}", err=True)

    if tools_files:
        loaded_actions = load_actions(tools_files)
        if verbose and loaded_actions:
            typer.echo(f"Loaded actions: {', '.join(loaded_actions)}", err=True)

    # auto-discover types/ and actions/ directories
    discovered_types = discover_yaml_types(template_path=prompt_file)
    discovered_actions = discover_actions(template_path=prompt_file)
    if verbose:
        if discovered_types:
            typer.echo(f"Discovered types: {', '.join(discovered_types)}", err=True)
        if discovered_actions:
            typer.echo(f"Discovered actions: {', '.join(discovered_actions)}", err=True)

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
        typer.echo(
            "Error: No prompt provided. Use a positional argument, -p/--prompt-file, or pipe to stdin.",
            err=True,
        )
        raise typer.Exit(1)
    credentials = LLMCredentials()
    model = LLM(model_name=model_name)

    # build extra_kwargs for API parameters
    extra_kwargs = {}
    if seed is not None:
        extra_kwargs["seed"] = seed

    # build context from source file/URL and context variables
    context = {}
    if source:
        from struckdown.actions import is_url, fetch_and_parse
        from urllib.parse import urlparse

        source_path = Path(source)
        if source_path.exists():
            # Treat as local file (prioritise local files over URLs)
            source_content = source_path.read_text(encoding="utf-8")
            context["source"] = source_content
            context["input"] = source_content
            context["content"] = source_content
            context["filename"] = str(source_path)
            context["basename"] = source_path.stem
        elif is_url(source):
            # Fetch URL content
            from struckdown.errors import StruckdownFetchError

            typer.echo(f"Fetching URL: {source}", err=True)
            try:
                source_content = fetch_and_parse(source, raw=raw_source)
            except StruckdownFetchError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1)
            context["source"] = source_content
            context["input"] = source_content
            context["content"] = source_content
            context["url"] = source
            context["filename"] = source
            # Extract last path component as basename
            path = urlparse(source).path
            context["basename"] = path.split('/')[-1] if path else "index"
        else:
            typer.echo(f"Error: Source not found: {source}", err=True)
            raise typer.Exit(1)

    # parse context variables from -c key=value options
    if context_vars:
        for var in context_vars:
            if "=" not in var:
                typer.echo(f"Error: Context variable must be key=value format: {var}", err=True)
                raise typer.Exit(1)
            key, value = var.split("=", 1)
            context[key.strip()] = value.strip()

    # Build include paths: cwd/templates as default, plus any user-provided -I paths
    all_include_paths = []
    if (Path.cwd() / 'templates').is_dir():
        all_include_paths.append(Path.cwd() / 'templates')
    if include_paths:
        all_include_paths.extend(include_paths)

    try:
        result = chatter(
            multipart_prompt=prompt_str,
            model=model,
            credentials=credentials,
            context=context,
            extra_kwargs=extra_kwargs if extra_kwargs else None,
            template_path=prompt_file,  # Enable {% include %} for file-based templates
            include_paths=all_include_paths if all_include_paths else None,
        )
    except StruckdownTemplateError as e:
        e.template_path = prompt_file
        typer.echo(str(e), err=True)
        if verbose:
            typer.echo("\n" + traceback.format_exc(), err=True)
        raise typer.Exit(1)
    except StruckdownLLMError as e:
        typer.echo(str(e), err=True)
        if verbose:
            typer.echo("\n" + traceback.format_exc(), err=True)
        raise typer.Exit(1)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        if verbose:
            typer.echo("\n" + traceback.format_exc(), err=True)
        raise typer.Exit(1)
    except (UnexpectedToken, UnexpectedCharacters) as e:
        typer.echo(format_parse_error(e), err=True)
        if verbose:
            typer.echo("\n" + traceback.format_exc(), err=True)
        raise typer.Exit(1)

    # Check if execution was terminated by break action
    break_result = None
    for key, seg_result in result.results.items():
        if seg_result.action == 'break':
            break_result = seg_result
            break

    if verbose:
        typer.echo("\n" + "="*80)
        typer.echo("VERBOSE OUTPUT - Message Threads")
        typer.echo("="*80 + "\n")

        for idx, (key, seg_result) in enumerate(result.results.items(), 1):
            # Skip break action (shown separately at the end)
            if seg_result.action == 'break':
                continue

            typer.echo(f"Segment {idx}: `{key}`")
            typer.echo("-" * 80)

            if seg_result.messages:
                for msg in seg_result.messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    # Bold role name
                    typer.echo(f"\033[1m{role}:\033[0m")
                    typer.echo(content)
                    typer.echo()

                # Show response schema if available
                if seg_result.response_schema:
                    typer.echo("\033[1m\033[33mResponse Schema (Tool):\033[0m")
                    # Get schema summary from SegmentResult method
                    schema_summary = seg_result.get_schema_summary()
                    typer.echo(f"\033[33m{schema_summary}\033[0m")
                    typer.echo()

                # Completion in red at the end
                if isinstance(seg_result.output, (dict, list)):
                    formatted = json.dumps(seg_result.output, indent=2, default=str)
                    typer.echo(f"\033[91m\033[1m{key}\033[0m\033[91m: {formatted}\033[0m\n")
                else:
                    typer.echo(f"\033[91m\033[1m{key}\033[0m\033[91m: {seg_result.output}\033[0m\n")

                
            else:
                # Fallback if messages not available
                if isinstance(seg_result.output, (dict, list)):
                    typer.echo(f"Output: {json.dumps(seg_result.output, indent=2, default=str)}")
                else:
                    typer.echo(f"Output: {seg_result.output}")

            typer.echo()  # Blank line after segment

        # Show break notice if execution was terminated
        if break_result:
            typer.echo("="*80)
            typer.echo("\033[1m⚠ EXECUTION TERMINATED BY BREAK\033[0m")
            if break_result.output:
                typer.echo(f"\033[1mReason:\033[0m {break_result.output}")
            typer.echo("="*80 + "\n")

        # Pretty print accumulated context
        typer.echo("="*80)
        typer.echo("\033[1mAccumulated Context:\033[0m")
        typer.echo("-"*80)
        typer.echo(json.dumps(result.outputs, indent=2, default=str))
        typer.echo("="*80 + "\n")

    else:
        for k, v in result.results.items():
            # Skip break action (control flow, not a completion)
            if v.action == 'break':
                continue
            # Format dict/list as JSON for valid output
            if isinstance(v.output, (dict, list)):
                formatted = json.dumps(v.output, indent=2, default=str)
                typer.echo(f"\033[1m{k}\033[0m: {formatted}")
            else:
                typer.echo(f"\033[1m{k}\033[0m: {v.output}")

        # Show break notice if execution was terminated
        if break_result:
            typer.echo(f"\n\033[1m⚠ Break:\033[0m {break_result.output or 'execution terminated'}")

    if show_context:
        typer.echo("\nFinal context:")
        typer.echo(result.outputs)

    # print cost summary to stderr (always visible)
    summary = CostSummary.from_results([result])
    typer.echo(summary.format_summary(), err=True)


def _merge_result_with_input(
    item: dict, result, keep_inputs: bool
) -> dict:
    """
    Merge input data with completion results, handling column name clashes.

    When keep_inputs is True and a completion key matches an input column:
    - Input column is renamed to {key}.data
    - Completion is stored as {key}.predicted

    Also stores _completion_slots metadata listing the actual column names
    where completions were stored (after any renaming).
    """
    # Get completion keys
    completion_keys = set(result.results.keys())

    # Get input keys (excluding internal metadata)
    input_keys = {k for k in item.keys() if not k.startswith("_")}

    # Find clashes
    clashing_keys = completion_keys & input_keys

    # Track actual completion slot names (after renaming)
    completion_slots = []

    if keep_inputs:
        output_item = {}
        for k, v in item.items():
            if k.startswith("_"):
                # Keep internal metadata as-is
                output_item[k] = v
            elif k in clashing_keys:
                output_item[f"{k}.data"] = v
            else:
                output_item[k] = v
    else:
        output_item = {}
        # Keep filename for traceability
        if "filename" in item:
            output_item["filename"] = item["filename"]

    # Add completion results
    for key, segment_result in result.results.items():
        if key in clashing_keys:
            actual_key = f"{key}.predicted"
            output_item[actual_key] = segment_result.output
            completion_slots.append(actual_key)
        else:
            output_item[key] = segment_result.output
            completion_slots.append(key)

    # Store completion slots as metadata
    output_item["_completion_slots"] = completion_slots

    return output_item


async def batch_async(
    prompt: str,
    input_data: List[dict],
    output: Optional[List[Path]],
    keep_inputs: bool,
    template: Optional[Path],
    model_name: Optional[str],
    extra_kwargs: Optional[dict],
    max_concurrent: int,
    verbose: bool,
    quiet: bool,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    compare: Optional[List[str]] = None,
    statsonly: bool = False,
    classification_errors: Optional[int] = None,
    min_n_compare: int = 1,
):
    """
    Async implementation of batch processing with concurrent execution.
    """
    # Process each input
    credentials = LLMCredentials()
    model = LLM(model_name=model_name)
    results = [None] * len(input_data)
    errors = []

    # cost tracking - collect all ChatterResults for CostSummary
    chatter_results = []
    cost_lock = anyio.Lock()

    # API call counter for progress display
    api_call_count = [0]  # Use list to allow mutation in nested function

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
            task = progress.add_task("Processing (0 API calls)", total=len(input_data))

            async with anyio.create_task_group() as tg:
                for idx, input_item in enumerate(input_data):

                    async def run_and_store(
                        index=idx,
                        item=input_item,
                        progress_bar=progress,
                        progress_task=task,
                    ):
                        async with sem:
                            try:
                                # Progress callback for per-API-call updates
                                def on_api_call():
                                    api_call_count[0] += 1
                                    if progress_bar is not None:
                                        progress_bar.update(
                                            progress_task,
                                            description=f"Processing ({api_call_count[0]} API calls)"
                                        )

                                # Execute chatter_async with progress tracking
                                # Use strict_undefined to catch column name mismatches
                                with progress_tracking(on_api_call=on_api_call):
                                    result = await chatter_async(
                                        multipart_prompt=prompt,
                                        model=model,
                                        credentials=credentials,
                                        context=item,
                                        extra_kwargs=extra_kwargs,
                                        template_path=template_path,
                                        include_paths=include_paths,
                                        strict_undefined=True,
                                    )

                                # collect result for cost tracking
                                async with cost_lock:
                                    chatter_results.append(result)

                                # Merge input data with extracted results
                                output_item = _merge_result_with_input(item, result, keep_inputs)
                                results[index] = output_item

                                if verbose:
                                    console.print(
                                        f"Processed item {index+1}/{len(input_data)}: {output_item.get('filename', f'item_{index}')}"
                                    )

                            except Exception as e:
                                error_msg = f"Error processing item {index+1}: {e}"
                                logger.error(error_msg)
                                errors.append(error_msg)
                                if verbose:
                                    import traceback

                                    console.print(traceback.format_exc())

                            finally:
                                # Update progress bar on completion with API call count
                                if progress_bar is not None:
                                    progress_bar.update(
                                        progress_task,
                                        advance=1,
                                        description=f"Processing ({api_call_count[0]} API calls)"
                                    )

                    tg.start_soon(run_and_store)
    else:
        # No progress bar, just process concurrently
        async with anyio.create_task_group() as tg:
            for idx, input_item in enumerate(input_data):

                async def run_and_store(index=idx, item=input_item):
                    async with sem:
                        try:
                            # Progress callback for per-API-call updates (no progress bar)
                            def on_api_call():
                                api_call_count[0] += 1

                            # Execute chatter_async with progress tracking
                            # Use strict_undefined to catch column name mismatches
                            with progress_tracking(on_api_call=on_api_call):
                                result = await chatter_async(
                                    multipart_prompt=prompt,
                                    model=model,
                                    credentials=credentials,
                                    context=item,
                                    extra_kwargs=extra_kwargs,
                                    template_path=template_path,
                                    include_paths=include_paths,
                                    strict_undefined=True,
                                )

                            # collect result for cost tracking
                            async with cost_lock:
                                chatter_results.append(result)

                            # Merge input data with extracted results
                            output_item = _merge_result_with_input(item, result, keep_inputs)
                            results[index] = output_item

                            if verbose:
                                console.print(
                                    f"Processed item {index+1}/{len(input_data)} ({api_call_count[0]} API calls): {output_item.get('filename', f'item_{index}')}"
                                )

                        except Exception as e:
                            error_msg = f"Error processing item {index+1}: {e}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                            if verbose:
                                import traceback

                                console.print(traceback.format_exc())

                tg.start_soon(run_and_store)

    # print cost summary to stderr (always visible)
    summary = CostSummary.from_results(chatter_results)
    typer.echo(summary.format_summary(), err=True)

    # Report errors if any
    if errors:
        typer.echo(f"\nCompleted with {len(errors)} error(s):", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)

    # Write output(s)
    if not results or all(r is None for r in results):
        typer.echo("Error: No results produced", err=True)
        raise typer.Exit(1)

    # Filter out None results from errors, keeping input_data aligned
    paired = [(inp, res) for inp, res in zip(input_data, results) if res is not None]
    if paired:
        input_data_filtered, results = zip(*paired)
        input_data_filtered = list(input_data_filtered)
        results = list(results)
    else:
        input_data_filtered = []
        results = []

    # Calculate comparison statistics if requested
    stats = None
    error_examples = None
    if compare:
        from .stats import (
            calculate_batch_stats,
            collect_error_examples,
            format_error_examples,
            format_stats_table,
            parse_compare_spec,
            stats_to_json,
        )

        # Get available column names from data
        input_cols = set()
        result_cols = set()
        completion_slots = set()
        if input_data_filtered:
            input_cols = {k for k in input_data_filtered[0].keys() if not k.startswith("_")}
        if results:
            result_cols = {k for k in results[0].keys() if not k.startswith("_")}
            # Get completion slots from metadata
            completion_slots = set(results[0].get("_completion_slots", []))

        # Parse and resolve compare specs (handle renamed columns)
        compare_specs = []

        # Handle wildcard: --compare=all finds input columns matching completion slots
        if "all" in compare:
            # For each completion slot, find the matching input column
            for slot in sorted(completion_slots):
                # Get base name (strip .predicted suffix if present)
                if slot.endswith(".predicted"):
                    base_name = slot[:-10]
                else:
                    base_name = slot

                # Find matching input column (either base_name.data or base_name)
                if f"{base_name}.data" in input_cols:
                    compare_specs.append((f"{base_name}.data", slot))
                elif base_name in input_cols:
                    compare_specs.append((base_name, slot))

        # Process explicit specs (non-wildcard)
        for spec in compare:
            if spec == "all":
                continue  # Already handled above
            col, comp = parse_compare_spec(spec)

            # Auto-resolve renamed columns due to clashes
            actual_col = f"{col}.data" if f"{col}.data" in input_cols else col
            actual_comp = f"{comp}.predicted" if f"{comp}.predicted" in result_cols else comp

            compare_specs.append((actual_col, actual_comp))

        stats = calculate_batch_stats(input_data_filtered, results, compare_specs, min_n_compare)

        # Collect error examples if requested
        if classification_errors is not None:
            max_examples = None if classification_errors == -1 else classification_errors
            error_examples = collect_error_examples(
                input_data_filtered, results, compare_specs, max_per_type=max_examples
            )

        if statsonly:
            # Output only stats as JSON to stdout
            output_data = stats_to_json(stats)
            if error_examples:
                output_data["_error_examples"] = error_examples
            print(json.dumps(output_data, indent=2))
            return

    if output:
        # Write to multiple outputs
        for output_path in output:
            # Check if this is a JSON file
            is_json = str(output_path).lower().endswith(".json")

            if is_json or not template:
                # Use format auto-detection for JSON or when no template specified
                write_output(results, output_path)
            else:
                # Use template rendering for non-JSON outputs when template is specified
                render_template(results, output_path, template)
    elif not compare:
        # No outputs specified and no compare - write results to stdout
        write_output(results, None)

    # Print stats summary if calculated
    if stats:
        # Print to stdout if no output files, stderr otherwise
        err_target = bool(output)

        # Print error examples first (if collected)
        if error_examples:
            typer.echo(format_error_examples(error_examples), err=err_target)

        # Then print confusion matrices / stats table
        typer.echo(format_stats_table(stats), err=err_target)


@app.command()
def batch(
    prompt: Optional[str] = typer.Argument(
        None,
        help="Prompt with slots, e.g. 'extract name [[name]]'. Omit if using --prompt/-p flag.",
    ),
    input_files: Optional[List[str]] = typer.Option(
        None,
        "-i",
        "--input",
        help="Input files or glob patterns (e.g., -i '*.txt' -i 'data.xlsx'). Supports .txt, .json, .csv, .xlsx (each row processed separately). Can be specified multiple times.",
    ),
    prompt_file: Optional[Path] = typer.Option(
        None, "-p", "--prompt", help="Path to file containing the prompt"
    ),
    output: Optional[List[Path]] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file (format inferred from extension: .json, .csv, .xlsx, .md, .txt). Can be specified multiple times.",
    ),
    keep_inputs: bool = typer.Option(
        False,
        "-k",
        "--keep-inputs",
        help="Include input fields (input, content, source, filename, basename) in output",
    ),
    template: Optional[Path] = typer.Option(
        None,
        "--template",
        help="Jinja2 template file to apply to non-JSON outputs",
    ),
    model_name: Optional[str] = typer.Option(
        env_config("DEFAULT_LLM", default=None, cast=str),
        "-m",
        "--model",
        help="LLM model name (overrides DEFAULT_LLM env var)",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible outputs (if supported by model)",
    ),
    max_concurrent: int = typer.Option(
        20, "-j", "--concurrency", help="Maximum number of concurrent API requests"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
    include_paths: Optional[List[Path]] = typer.Option(
        None,
        "-I",
        "--include-path",
        help="Additional directories to search for includes (can be repeated)",
    ),
    compare: Optional[List[str]] = typer.Option(
        None,
        "-c",
        "--compare",
        help="Compare column to completion for stats. Use 'col' for same-name or 'col=completion' for mapping. Can be repeated.",
    ),
    statsonly: bool = typer.Option(
        False,
        "--statsonly",
        help="Output only comparison statistics as JSON (requires --compare)",
    ),
    head: Optional[int] = typer.Option(
        None,
        "-h",
        "--head",
        help="Limit to first N inputs",
    ),
    classification_errors: Optional[int] = typer.Option(
        None,
        "-e",
        "--classification-errors",
        help="Show examples of misclassifications. Use -e for all, -e N for N examples per error type.",
        is_flag=False,
        flag_value=-1,  # -1 means show all
    ),
    min_n_compare: int = typer.Option(
        1,
        "--min-n-compare",
        help="Minimum ground truth count for a category to be included in aggregate metrics (macro/weighted F1).",
    ),
    type_files: Optional[List[Path]] = typer.Option(
        None,
        "-t",
        "--type",
        help="YAML type definition file or directory (can be repeated)",
    ),
    tools_files: Optional[List[Path]] = typer.Option(
        None,
        "--tools",
        help="Python tools file or directory (can be repeated)",
    ),
):
    """
    Process multiple inputs in batch mode.

    Examples:
        sd batch -i '*.txt' -p prompt.sd -o results.json
        sd batch -i 'data.json' "welcome for {{name}} [[msg]]"
        cat file.txt | sd batch "extract [[name]]"
        sd batch -i '*.txt' -p prompt.sd -o results.json -o report.html -t template.j2
        sd batch -i '*.txt' -p prompt.sd -I ./includes -o results.json

    Multiple outputs: Use -t flag to apply a Jinja2 template to all non-JSON outputs.
    JSON outputs always use standard JSON format. Without -t, output format is inferred from extension.
    """
    # start new run for cache detection
    from struckdown import new_run

    new_run()

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("struckdown").setLevel(logging.DEBUG)

    # load custom types and tools
    if type_files:
        loaded_types = load_yaml_types(type_files)
        if verbose and loaded_types:
            typer.echo(f"Loaded types: {', '.join(loaded_types)}", err=True)

    if tools_files:
        loaded_actions = load_actions(tools_files)
        if verbose and loaded_actions:
            typer.echo(f"Loaded actions: {', '.join(loaded_actions)}", err=True)

    # auto-discover types/ and actions/ directories
    discovered_types = discover_yaml_types(template_path=prompt_file)
    discovered_actions = discover_actions(template_path=prompt_file)
    if verbose:
        if discovered_types:
            typer.echo(f"Discovered types: {', '.join(discovered_types)}", err=True)
        if discovered_actions:
            typer.echo(f"Discovered actions: {', '.join(discovered_actions)}", err=True)

    # Validate --statsonly requires --compare
    if statsonly and not compare:
        typer.echo("Error: --statsonly requires at least one --compare option", err=True)
        raise typer.Exit(1)

    # Validate template usage
    if template:
        if not output:
            typer.echo(
                "Error: -t/--template requires at least one -o/--output", err=True
            )
            raise typer.Exit(1)

        # Check if there's at least one non-JSON output
        has_non_json = any(
            not str(out_path).lower().endswith(".json") for out_path in output
        )
        if not has_non_json:
            typer.echo(
                "Error: -t/--template requires at least one non-JSON output file",
                err=True,
            )
            raise typer.Exit(1)

        # Validate template file exists
        if not template.exists():
            typer.echo(f"Error: Template file not found: {template}", err=True)
            raise typer.Exit(1)

    # Validate prompt arguments
    if prompt_file and prompt:
        typer.echo(
            "Error: Cannot specify both inline prompt and --prompt file", err=True
        )
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
                typer.echo(
                    f"Error: stdin JSON must be dict or list, got {type(data)}",
                    err=True,
                )
                raise typer.Exit(1)
        except json.JSONDecodeError:
            # Treat as plain text
            input_data = [
                {
                    "input": stdin_content,
                    "content": stdin_content,
                    "filename": "<stdin>",
                }
            ]

    else:
        typer.echo(
            "Error: No input provided (specify files or pipe to stdin)", err=True
        )
        raise typer.Exit(1)

    # Limit to first N inputs if --head specified
    if head is not None and head > 0:
        input_data = input_data[:head]

    # build extra_kwargs for API parameters
    extra_kwargs = {}
    if seed is not None:
        extra_kwargs["seed"] = seed

    # Build include paths: cwd/templates as default, plus any user-provided -I paths
    all_include_paths = []
    if (Path.cwd() / 'templates').is_dir():
        all_include_paths.append(Path.cwd() / 'templates')
    if include_paths:
        all_include_paths.extend(include_paths)

    # Call async batch processing
    anyio.run(
        batch_async,
        prompt,
        input_data,
        output,
        keep_inputs,
        template,
        model_name,
        extra_kwargs,
        max_concurrent,
        verbose,
        quiet,
        prompt_file,  # Pass prompt_file as template_path for includes
        all_include_paths if all_include_paths else None,
        compare,
        statsonly,
        classification_errors,
        min_n_compare,
    )


def _is_spreadsheet(path: Path) -> bool:
    """Check if file is spreadsheet (CSV or XLSX) by extension."""
    return path.suffix.lower() in [".csv", ".xlsx"]


def _extract_spreadsheet_rows(path: Path) -> tuple[List[dict], List[str]]:
    """Extract rows from CSV or XLSX file as list of dictionaries.

    Each row becomes a dictionary with column names as keys.
    NaN values are converted to None.

    Returns:
        (rows, original_columns): rows as dicts and list of original column names
    """
    import pandas as pd

    suffix = path.suffix.lower()

    # Disable default NA parsing to preserve strings like "n/a", "NA" as literal text
    if suffix == ".csv":
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    elif suffix == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl", keep_default_na=False, na_values=[""])
    else:
        raise ValueError(f"Unsupported spreadsheet format: {suffix}")

    # capture original column names
    original_columns = list(df.columns)

    # convert NaN to None, convert to list of dicts
    rows = df.where(pd.notna(df), None).to_dict("records")

    logger.info(
        f"Loaded {len(rows)} rows from {path.name} with columns: {original_columns}"
    )
    return rows, original_columns


def _read_input_file(path: Path) -> List[dict]:
    """
    Read an input file and return a list of input items.

    For spreadsheets (CSV/XLSX): returns list of dicts, one per row with columns as keys
    For text files: returns [{"input": "...", "content": "...", "filename": "..."}]
    For JSON files:
        - If dict: returns [dict]
        - If list: returns list
    """
    extension = path.suffix.lower()

    # handle spreadsheets (CSV/XLSX)
    if _is_spreadsheet(path):
        rows, original_columns = _extract_spreadsheet_rows(path)
        result = []
        for idx, row_data in enumerate(rows):
            result.append({
                "_original_columns": original_columns,  # track for output ordering
                **row_data  # unpack all column data (original columns only)
            })
        return result

    elif extension == ".json":
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

        return [
            {
                "input": content,
                "content": content,
                "source": content,  # alias for compatibility
                "filename": str(path),
                "basename": path.stem,
            }
        ]


@app.command()
def graph(
    prompt_file: Path = typer.Argument(..., help="Path to .sd prompt file"),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file for Mermaid diagram text. If not specified, prints to stdout.",
    ),
):
    """Generate section dependency graph as Mermaid diagram text.

    Creates Mermaid diagram code showing:
    - Section nodes with slot names
    - Dependencies between sections
    - External inputs

    For visual rendering, use 'sd explain -o output.html' instead.

    Examples:
        sd graph prompt.sd                  # Print to stdout
        sd graph prompt.sd -o diagram.mmd   # Write to file
    """
    from .parsing import parse_syntax
    from .visualize import analyze_sections, generate_section_dag

    console = Console()

    # Check input file exists
    if not prompt_file.exists():
        console.print(f"[red]Error: {prompt_file} not found[/red]")
        raise typer.Exit(1)

    # Parse prompt (resolve includes first)
    try:
        template = _resolve_template_includes(prompt_file)
        sections = parse_syntax(template)
    except Exception as e:
        console.print(f"[red]Error parsing prompt: {e}[/red]")
        raise typer.Exit(1)

    # Analyze structure
    structure = analyze_sections(sections)
    typer.echo(
        f"Analyzed: {len(structure['sections'])} sections, "
        f"{len(structure['all_completions'])} completions",
        err=True
    )

    # Generate simplified section DAG
    mermaid = generate_section_dag(structure)

    if output:
        # Write to file
        output.write_text(mermaid)
        typer.echo(f"✓ Wrote Mermaid diagram to {output}", err=True)
    else:
        # Print to stdout
        print(mermaid)


@app.command()
def explain(
    prompt_file: Path = typer.Argument(..., help="Path to .sd prompt file"),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file (.html for HTML format, otherwise plain text)",
    ),
):
    """Explain prompt structure and display execution plan.

    Parses the struckdown prompt and displays:
    - External inputs required
    - Sections with completion slots
    - Dependencies for each completion
    - Line numbers in source file
    - Any parsing errors

    Examples:
        sd explain prompt.sd
        sd explain prompt.sd -o plan.html
    """
    from .parsing import parse_syntax
    from .visualize import analyze_sections, build_execution_plan_data, render_execution_plan

    console = Console()

    # Check input file exists
    if not prompt_file.exists():
        console.print(f"[red]Error: {prompt_file} not found[/red]")
        raise typer.Exit(1)

    # Parse prompt (resolve includes first)
    try:
        template = _resolve_template_includes(prompt_file)
        sections = parse_syntax(template)
        console.print(f"[green]✓[/green] Syntax OK")
    except Exception as e:
        console.print(f"[red]✗ Parsing error:[/red] {e}")
        raise typer.Exit(1)

    # Analyze and build execution plan data
    structure = analyze_sections(sections)
    console.print("[cyan]Generating summaries of prompts...[/cyan]")
    plan_data = build_execution_plan_data(
        structure,
        prompt_name=prompt_file.stem,
        sections_data=sections,
        summarize=True
    )

    if output:
        # Write to file
        ext = output.suffix.lower()
        if ext in [".html", ".htm"]:
            # Render as HTML
            html_content = render_execution_plan(plan_data, format='html')
            output.write_text(html_content)
            console.print(f"[green]✓[/green] Wrote HTML to [cyan]{output}[/cyan]")
        else:
            # Render as plain text
            text_content = render_execution_plan(plan_data, format='text')
            output.write_text(text_content)
            console.print(f"[green]✓[/green] Wrote plan to [cyan]{output}[/cyan]")
    else:
        # Print to stdout
        text_content = render_execution_plan(plan_data, format='text')
        print(text_content)


@app.command(name="check", hidden=True)
def check_alias(
    prompt_file: Path = typer.Argument(..., help="Path to .sd prompt file"),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file (.html for HTML format, otherwise plain text)",
    ),
):
    """Alias for 'explain' command (deprecated, use 'explain' instead)."""
    # Just call explain with the same arguments
    explain(prompt_file, output)


@app.command()
def preview(
    prompt_file: Optional[Path] = typer.Argument(
        None, help="Path to .sd file (reads from stdin if --fragment)"
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output HTML file (default: open in browser)"
    ),
    raw: bool = typer.Option(
        False, "--raw", "-r", help="Show raw file without resolving includes"
    ),
    fragment: bool = typer.Option(
        False, "--fragment", "-f", help="Output HTML fragment to stdout (for embedding)"
    ),
):
    """Preview .sd file with syntax highlighting.

    By default resolves all includes and opens the preview in your default browser.
    Use -o to save to a file instead.
    Use --fragment to output just the highlighted HTML (no page wrapper) to stdout.

    Examples:
        sd preview prompt.sd              # Opens in browser (includes resolved)
        sd preview prompt.sd -o out.html  # Saves to file
        sd preview prompt.sd --raw        # Don't resolve includes
        sd preview prompt.sd --fragment   # Output HTML fragment to stdout
        sd preview --fragment < prompt.sd # Fragment from stdin
    """
    import tempfile
    import webbrowser
    from .highlight import render_preview_html, highlight_struckdown_with_system_blocks

    console = Console()

    # handle stdin for fragment mode
    if prompt_file is None:
        if not fragment:
            console.print("[red]Error: prompt_file required (or use --fragment with stdin)[/red]")
            raise typer.Exit(1)
        content = sys.stdin.read()
    else:
        if not prompt_file.exists():
            console.print(f"[red]Error: {prompt_file} not found[/red]")
            raise typer.Exit(1)

        # read content (resolve <include> tags by default, but don't execute Jinja)
        if raw:
            content = prompt_file.read_text(encoding="utf-8")
        else:
            from struckdown.parsing import resolve_includes
            try:
                content = prompt_file.read_text(encoding="utf-8")
                # only resolve <include src="..."/> tags, not Jinja {% include %}
                search_paths = [
                    prompt_file.parent,
                    prompt_file.parent / 'templates',
                    Path.cwd(),
                    Path.cwd() / 'includes',
                    Path.cwd() / 'templates',
                ]
                search_paths = [p for p in search_paths if p.exists() and p.is_dir()]
                content = resolve_includes(content, prompt_file.parent, search_paths)
            except Exception as e:
                console.print(f"[red]Error resolving includes:[/red] {e}")
                raise typer.Exit(1)

    # fragment mode: output just highlighted HTML to stdout
    if fragment:
        highlighted = highlight_struckdown_with_system_blocks(content)
        print(highlighted, end="")
        return

    # full preview mode
    html = render_preview_html(content, filename=prompt_file.name if prompt_file else "stdin")

    if output:
        output.write_text(html, encoding="utf-8")
        console.print(f"[green]Wrote preview to[/green] [cyan]{output}[/cyan]")
    else:
        # write to /tmp (more accessible to sandboxed apps) or system temp
        import platform
        if platform.system() == 'Darwin':
            temp_dir = Path('/tmp')
        else:
            temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f'{prompt_file.stem if prompt_file else "preview"}.html'
        temp_path.write_text(html, encoding='utf-8')

        file_url = f'file://{temp_path}'
        webbrowser.open(file_url)

        # copy file to clipboard on macOS
        import subprocess
        system = platform.system()
        if system == 'Darwin':
            subprocess.run([
                'osascript', '-e',
                f'set the clipboard to POSIX file "{temp_path}"'
            ], check=False, capture_output=True)

        console.print(f"[green]Opened preview in browser[/green]")
        console.print(f"[dim]{temp_path}[/dim]")


@app.command()
def flat(
    prompt_file: Path = typer.Argument(..., help="Path to .sd prompt file"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file (defaults to stdout)"
    ),
):
    """Flatten template by resolving all {% include %} directives.

    Shows the fully resolved template with all includes expanded.
    Useful for debugging, inspection, and creating self-contained templates.

    Examples:
        sd flat prompt.sd                    # Output to stdout
        sd flat prompt.sd -o flattened.sd    # Save to file
    """
    console = Console()

    # Check input file exists
    if not prompt_file.exists():
        console.print(f"[red]Error: {prompt_file} not found[/red]")
        raise typer.Exit(1)

    # Resolve includes
    try:
        flattened = _resolve_template_includes(prompt_file)

        if output:
            output.write_text(flattened)
            console.print(f"[green]✓[/green] Wrote flattened template to [cyan]{output}[/cyan]")
        else:
            # Use print() instead of console.print() to avoid Rich markup interpretation
            # which would strip out [[...]] struckdown completion slots
            print(flattened)

    except Exception as e:
        console.print(f"[red]Error rendering template:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def edit(
    path: Optional[Path] = typer.Argument(
        None,
        help="File or directory to edit (default: current directory as workspace)"
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
    reload: bool = typer.Option(
        False, "--reload", "-r",
        help="Auto-reload server on file changes (development mode)"
    ),
    models: Optional[str] = typer.Option(
        None, "--models", "-m",
        help="Comma-separated list of allowed models (e.g. 'gpt-4o,gpt-4o-mini')"
    ),
):
    """Open interactive playground for editing struckdown prompts.

    Starts a local web server with a browser-based editor for creating
    and testing struckdown prompts interactively.

    Examples:
        sd edit                     # Open workspace browser for current dir
        sd edit myfile.sd           # Edit specific file
        sd edit ./prompts/          # Open workspace browser for prompts/
        sd edit -p 8080             # Use specific port
        sd edit -I ./custom         # Include custom actions/types
        sd edit --reload            # Auto-reload on file changes
        sd edit --models=gpt-4o,gpt-4o-mini  # Restrict model selector
    """
    import threading
    import webbrowser

    from struckdown.playground import create_app, find_available_port

    console = Console()

    # Determine workspace_dir and prompt_file based on path
    if path is None:
        # No path: use current directory as workspace, no initial file
        workspace_dir = Path.cwd()
        prompt_file = None
    elif path.is_dir():
        # Directory: use as workspace, no initial file
        workspace_dir = path.resolve()
        prompt_file = None
    else:
        # File: use parent as workspace, file as initial
        workspace_dir = path.parent.resolve()
        prompt_file = path.resolve()

        # Create file if it doesn't exist
        if not prompt_file.exists():
            prompt_file.write_text("# Your struckdown prompt\n\n[[response]]\n")
            console.print(f"[green]Created[/green] {prompt_file}")

    # Resolve include paths
    include_paths = [p.resolve() for p in include if p.exists()]
    include_paths.append(Path.cwd())

    # Find available port
    if port is None:
        try:
            port = find_available_port()
        except RuntimeError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Parse allowed models
    allowed_models = None
    if models:
        allowed_models = [m.strip() for m in models.split(",") if m.strip()]

    # Create Flask app
    flask_app = create_app(
        prompt_file=prompt_file,
        workspace_dir=workspace_dir,
        include_paths=include_paths,
        remote_mode=False,
        allowed_models=allowed_models,
    )

    url = f"http://localhost:{port}"
    console.print(f"[green]Playground:[/green] {url}")
    console.print(f"[dim]Workspace: {workspace_dir}[/dim]")
    if prompt_file:
        console.print(f"[dim]Editing: {prompt_file.name}[/dim]")
    if reload:
        console.print("[dim]Auto-reload enabled - watching for file changes[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    if not no_browser:
        # Delay browser open slightly to let server start
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    # Run Flask app (blocks until Ctrl+C)
    try:
        flask_app.run(
            host='localhost',
            port=port,
            debug=reload,
            use_reloader=reload,
            threaded=True
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped[/dim]")


@app.command()
def serve(
    port: int = typer.Option(
        8000, "-p", "--port",
        help="Port to run server on"
    ),
    host: str = typer.Option(
        "0.0.0.0", "-h", "--host",
        help="Host to bind to"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="Server-side API key (if not set, users must provide their own)"
    ),
    models: Optional[str] = typer.Option(
        None, "--models", "-m",
        help="Comma-separated list of allowed models (e.g. 'gpt-4o,gpt-4o-mini'). Falls back to STRUCKDOWN_ALLOWED_MODELS env var."
    ),
):
    """Run playground in remote/server mode for deployment.

    This starts the playground without local file access, suitable for
    hosting as a public web service.

    By default, users must provide their own API keys via the settings panel.
    Use --api-key to provide a server-side key (e.g. for internal deployments).

    Examples:
        sd serve                              # Users provide their own keys
        sd serve --api-key=$MY_API_KEY        # Use server-side key
        sd serve -p 9000                      # Use specific port
        sd serve -h 127.0.0.1                 # Bind to localhost only
        sd serve --models=gpt-4o,gpt-4o-mini  # Restrict to specific models

    Environment variables:
        STRUCKDOWN_ALLOWED_MODELS: Comma-separated list of allowed models
                                   (fallback if --models not provided)

    For production, use with gunicorn:
        gunicorn -w 4 -b 0.0.0.0:8000 \\
            "struckdown.playground:create_app(remote_mode=True)"
    """
    import os
    from struckdown.playground import create_app

    console = Console()

    # Parse allowed models (CLI option takes precedence over env var)
    allowed_models = None
    models_str = models or os.environ.get("STRUCKDOWN_ALLOWED_MODELS")
    if models_str:
        allowed_models = [m.strip() for m in models_str.split(",") if m.strip()]

    # Create Flask app in remote mode
    flask_app = create_app(
        prompt_file=None,
        include_paths=[],
        remote_mode=True,
        server_api_key=api_key,
        allowed_models=allowed_models,
    )

    url = f"http://{host}:{port}"
    console.print(f"[green]Playground (remote mode):[/green] {url}")
    if api_key:
        console.print("[dim]Using server-side API key[/dim]")
    else:
        console.print("[dim]Users must provide their own API keys[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    # Run Flask app (blocks until Ctrl+C)
    try:
        flask_app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped[/dim]")


if __name__ == "__main__":
    app()
