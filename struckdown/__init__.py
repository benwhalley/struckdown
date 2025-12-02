"""Struckdown -- structured LLM prompting with template syntax."""

import logging
import warnings
from pathlib import Path
from typing import List, Optional

import anyio
from jinja2 import Environment

# Version - reads from package metadata (set in pyproject.toml)
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("struckdown")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

# Suppress Pydantic serialization warnings from OpenAI/Anthropic SDK completion objects
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

logger = logging.getLogger(__name__)

# Re-export from errors module
from .errors import (
    StruckdownSafe,
    StruckdownTemplateError,
    StruckdownLLMError,
)

# Re-export from results module
from .results import (
    SegmentResult,
    ChatterResult,
    CostSummary,
    StruckdownEarlyTermination,
    get_run_id,
    new_run,
    progress_tracking,
)

# Re-export from jinja_utils module
from .jinja_utils import (
    KeepUndefined,
    make_strict_undefined,
    mark_struckdown_safe,
    struckdown_finalize,
    escape_struckdown_syntax,
    escape_context_dict,
    extract_jinja_variables,
)

# Re-export from llm module
from .llm import (
    LC,
    LLM,
    LLMCredentials,
    structured_chat,
    get_embedding,
    _call_llm_cached,
)

# Re-export from execution module
from .execution import (
    process_single_segment_,
    SegmentDependencyGraph,
    merge_contexts,
)

# Re-export from other modules
from .actions import Actions
from .cache import clear_cache
from .response_types import ResponseTypes
from .return_type_models import ACTION_LOOKUP, LLMConfig
from .validation import ParsedOptions, parse_options, validate_number_constraints

# Import internal modules for chatter implementation
from .jinja_analysis import analyze_template, TemplateAnalysis
from .parsing import (
    _add_default_completion_if_needed,
    parser,
    parser_with_state,
    resolve_includes,
    split_by_checkpoint,
)
from .segment_processor import process_segment_with_delta


async def chatter_async(
    multipart_prompt: str,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    context={},
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
):
    """
    Process a struckdown template and return results.

    Example:
        chatter("tell a joke [[joke]]")

    Processing happens in two phases:
    1. Compile time: <include> tags resolved, template split by <checkpoint>
    2. Execution time: Each segment is Jinja2-rendered with accumulated context,
       then parsed and executed. Jinja conditionals can react to slot values
       filled within the same segment via delta-based re-rendering.

    Args:
        multipart_prompt: Struckdown template string
        model: LLM configuration (default: from environment)
        credentials: API credentials (default: from environment)
        context: Initial context variables
        extra_kwargs: Additional LLM parameters
        template_path: Path to template file (for includes)
        include_paths: Additional directories to search for <include> files
        strict_undefined: If True, raise error when template variables not found
    """
    if model is None:
        model = LLM()

    logger.debug(f"\n\n{LC.ORANGE}Chatter Prompt: {multipart_prompt}{LC.RESET}\n\n")

    # Configure search paths for includes
    search_paths = [template_path.parent] if template_path else []
    if include_paths:
        search_paths.extend(include_paths)
    search_paths = [p for p in search_paths if p.exists() and p.is_dir()]

    # COMPILE TIME: Resolve <include> tags
    resolved_template = resolve_includes(
        multipart_prompt, template_path.parent if template_path else None, search_paths
    )

    # COMPILE TIME: Split by <checkpoint> tags
    raw_segments = split_by_checkpoint(resolved_template)
    logger.debug(f"Split into {len(raw_segments)} raw segments")

    # EXECUTION TIME: Process each segment with accumulated context
    final = ChatterResult()
    accumulated_context = context.copy()

    # Track global system messages (persist across checkpoints)
    accumulated_globals: List[str] = []

    for seg_idx, (raw_segment_text, segment_name) in enumerate(raw_segments):
        from .segment_processor import extract_system_message

        # Extract system message from this segment
        system_template, body_template = extract_system_message(raw_segment_text)

        # System messages accumulate across segments (globals persist)
        if system_template:
            env = Environment(undefined=KeepUndefined, finalize=struckdown_finalize)
            rendered_system = env.from_string(system_template).render(**accumulated_context)
            accumulated_globals.append(rendered_system)

        # Analyze template structure for smart re-rendering
        analysis = analyze_template(body_template)
        logger.debug(
            f"Segment {seg_idx} analysis: "
            f"{len(analysis.slots)} slots, triggers={analysis.triggers}"
        )

        try:
            result = await process_segment_with_delta(
                body_template,
                accumulated_context,
                model,
                credentials,
                analysis=analysis,
                global_system_messages=accumulated_globals,
                **(extra_kwargs or {}),
            )
            final.update(result.results)

            # Update accumulated context with results for next segment
            for key, seg_result in result.results.items():
                escaped_value, _ = escape_struckdown_syntax(seg_result.output, var_name=key)
                accumulated_context[key] = escaped_value

        except Exception as e:
            logger.error(f"Segment {seg_idx} error: {e}")
            raise

    logger.debug(f"\n\n{LC.GREEN}Chatter Response: {final.response}{LC.RESET}\n\n")
    return final


def chatter(
    multipart_prompt: str,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    context={},
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
):
    """Synchronous wrapper for chatter_async."""
    if model is None:
        model = LLM()
    return anyio.run(
        chatter_async,
        multipart_prompt,
        model,
        credentials,
        context,
        extra_kwargs,
        template_path,
        include_paths,
        strict_undefined,
    )


# Public API exports
__all__ = [
    # Version
    "__version__",
    # Main entry points
    "chatter",
    "chatter_async",
    "structured_chat",
    "get_embedding",
    # Results
    "SegmentResult",
    "ChatterResult",
    "CostSummary",
    # LLM
    "LLM",
    "LLMCredentials",
    "LLMConfig",
    # Errors
    "StruckdownSafe",
    "StruckdownTemplateError",
    "StruckdownLLMError",
    "StruckdownEarlyTermination",
    # Actions
    "Actions",
    "ResponseTypes",
    "ACTION_LOOKUP",
    # Utilities
    "mark_struckdown_safe",
    "escape_struckdown_syntax",
    "escape_context_dict",
    "extract_jinja_variables",
    "get_run_id",
    "new_run",
    "progress_tracking",
    "clear_cache",
    # Validation
    "ParsedOptions",
    "parse_options",
    "validate_number_constraints",
    # Internal (for advanced use)
    "process_single_segment_",
    "SegmentDependencyGraph",
    "merge_contexts",
    "KeepUndefined",
    "make_strict_undefined",
    "struckdown_finalize",
    "LC",
]
