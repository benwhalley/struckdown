"""Struckdown -- structured LLM prompting with template syntax."""

# Fix SSL certificate issues on macOS with uv-managed Python
# Must be set before any httpx clients are created
import os
try:
    import certifi
    if not os.environ.get("SSL_CERT_FILE"):
        os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass

import logging
import warnings
from functools import partial
from pathlib import Path
from typing import AsyncGenerator, Generator, List, Optional, Union

import anyio
from jinja2.sandbox import ImmutableSandboxedEnvironment

# Version - reads from package metadata (set in pyproject.toml)
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("struckdown")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

# Suppress Pydantic serialization warnings from OpenAI/Anthropic SDK completion objects
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

logger = logging.getLogger(__name__)

# Re-export from other modules
from .actions import Actions
from .cache import clear_cache
# Re-export from errors module
from .errors import (
    Safe,
    TemplateError,
    LLMError,
    ContentFilterError,
    RateLimitError,
    ContextWindowError,
    AuthError,
    BadRequestError,
    ConnectionError,
    FetchError,
)
# Re-export from execution module
from .execution import SegmentDependencyGraph, merge_contexts
# Re-export from incremental module
from .incremental import (CheckpointReached, IncrementalEvent,
                          ProcessingComplete, ProcessingError, SlotCompleted)
# Import internal modules for chatter implementation
from .jinja_analysis import TemplateAnalysis, analyze_template
# Re-export from jinja_utils module
from .jinja_utils import (SilentUndefined, escape_context_dict,
                          escape_struckdown_syntax, extract_jinja_variables,
                          make_strict_undefined, mark_struckdown_safe,
                          struckdown_finalize)
# Re-export from llm module
from .llm import (LC, LLM, MAX_LLM_CONCURRENCY, LLMCredentials,
                  ProgressCallback, EmbeddingCostCallback, _call_llm_cached,
                  disable_api_debug, enable_api_debug, get_embedding,
                  get_embedding_async, get_cross_encoder_scores,
                  get_llm_semaphore, structured_chat,
                  EmbeddingResult, EmbeddingResultList)
from .parsing import (_add_default_completion_if_needed,
                      extract_slot_variable_refs, parser, parser_with_state,
                      resolve_includes, split_by_checkpoint)
from .response_types import ResponseTypes
# Re-export from results module
from .results import (ChatterResult, CostSummary, SegmentResult,
                      StruckdownEarlyTermination, progress_tracking)
from .return_type_models import ACTION_LOOKUP, LLMConfig
from .segment_processor import (process_segment_with_delta,
                                process_segment_with_delta_incremental)
from .validation import (ParsedOptions, parse_options,
                         validate_number_constraints)


async def _chatter_single_async(
    multipart_prompt: str,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    context={},
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
) -> ChatterResult:
    """Internal: process a single context through a struckdown template."""
    import asyncio

    from .parsing import parse_syntax
    from .segment_processor import (extract_header_message,
                                    extract_system_message)

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

    # EXECUTION TIME: Process segments in batches (parallel within each batch)
    final = ChatterResult()
    accumulated_context = context.copy()

    # Track global system messages (persist across checkpoints)
    accumulated_globals: List[str] = []
    # Track global header messages (persist across checkpoints, sent as user role)
    accumulated_header_globals: List[str] = []

    # Pre-extract system/header messages and analyze each segment
    segment_data = []
    for seg_idx, (raw_segment_text, segment_name) in enumerate(raw_segments):
        system_template, body_after_system = extract_system_message(raw_segment_text)
        header_template, body_template = extract_header_message(body_after_system)
        analysis = analyze_template(body_template)
        segment_data.append(
            {
                "idx": seg_idx,
                "name": segment_name,
                "system_template": system_template,
                "header_template": header_template,
                "body_template": body_template,
                "analysis": analysis,
                "has_slots": len(analysis.slots) > 0,
            }
        )

    # Build dependency graph from raw segments (only those with slots)
    # Map: parsed_idx -> raw_idx for segments that have slots
    slots_by_raw_idx = {}
    for data in segment_data:
        if data["has_slots"]:
            # analysis.slots is a list of SlotDependency namedtuples
            slots_by_raw_idx[data["idx"]] = {s.key for s in data["analysis"].slots}

    # Build dependency graph: which raw segments depend on which others
    def get_referenced_vars(body: str) -> set:
        """Get all variable references from template body (Jinja + action params)."""
        return extract_jinja_variables(body) | extract_slot_variable_refs(body)

    def find_dependencies(raw_idx: int, referenced_vars: set) -> set:
        """Find earlier segments that define variables this segment references."""
        return {
            earlier_idx
            for earlier_idx in range(raw_idx)
            if earlier_idx in slots_by_raw_idx
            and referenced_vars & slots_by_raw_idx[earlier_idx]
        }

    raw_dependencies = {
        raw_idx: find_dependencies(
            raw_idx, get_referenced_vars(segment_data[raw_idx]["body_template"])
        )
        for raw_idx in slots_by_raw_idx
    }

    # Build execution plan from raw segment dependencies
    def get_raw_execution_plan():
        remaining = set(raw_dependencies.keys())
        plan = []
        while remaining:
            ready = {
                idx
                for idx in remaining
                if all(dep not in remaining for dep in raw_dependencies[idx])
            }
            if not ready and remaining:
                logger.warning(f"Circular dependency in segments: {remaining}")
                plan.extend([[idx] for idx in remaining])
                break
            plan.append(sorted(ready))
            remaining -= ready
        return plan

    execution_plan = get_raw_execution_plan()
    logger.debug(f"Execution plan (raw indices): {execution_plan}")

    async def process_single_segment(
        seg_idx: int,
        ctx_snapshot: dict,
        globals_snapshot: list,
        header_globals_snapshot: list,
    ):
        """Process a single segment with the given context snapshot."""
        data = segment_data[seg_idx]
        body_template = data["body_template"]

        # Extract and render system/header messages with current context
        local_globals = globals_snapshot.copy()
        local_header_globals = header_globals_snapshot.copy()

        if data["system_template"]:
            env = ImmutableSandboxedEnvironment(
                undefined=SilentUndefined, finalize=struckdown_finalize
            )
            rendered_system = env.from_string(data["system_template"]).render(
                **ctx_snapshot
            )
            local_globals.append(rendered_system)

        if data["header_template"]:
            env = ImmutableSandboxedEnvironment(
                undefined=SilentUndefined, finalize=struckdown_finalize
            )
            rendered_header = env.from_string(data["header_template"]).render(
                **ctx_snapshot
            )
            local_header_globals.append(rendered_header)

        logger.debug(
            f"Segment {seg_idx} analysis: "
            f"{len(data['analysis'].slots)} slots, triggers={data['analysis'].triggers}"
        )

        result = await process_segment_with_delta(
            body_template,
            ctx_snapshot.copy(),
            model,
            credentials,
            analysis=data["analysis"],
            global_system_messages=local_globals,
            global_header_messages=local_header_globals,
            strict_undefined=strict_undefined,
            **(extra_kwargs or {}),
        )
        return seg_idx, result, data["system_template"], data["header_template"]

    # Process segments without slots first (they just accumulate system/header messages)
    for seg_idx, data in enumerate(segment_data):
        if not data["has_slots"]:
            if data["system_template"]:
                env = ImmutableSandboxedEnvironment(
                    undefined=SilentUndefined, finalize=struckdown_finalize
                )
                rendered_system = env.from_string(data["system_template"]).render(
                    **accumulated_context
                )
                accumulated_globals.append(rendered_system)
            if data["header_template"]:
                env = ImmutableSandboxedEnvironment(
                    undefined=SilentUndefined, finalize=struckdown_finalize
                )
                rendered_header = env.from_string(data["header_template"]).render(
                    **accumulated_context
                )
                accumulated_header_globals.append(rendered_header)

    # Process batches according to execution plan
    for batch in execution_plan:
        if len(batch) > 1:
            logger.debug(f"Processing {len(batch)} segments in parallel: {batch}")

        # For parallel batches, pre-collect system/header from all segments in the batch
        # so all segments see globals from earlier-indexed segments in the same batch
        batch_globals = accumulated_globals.copy()
        batch_header_globals = accumulated_header_globals.copy()

        for seg_idx in sorted(batch):
            data = segment_data[seg_idx]
            if data["system_template"]:
                env = ImmutableSandboxedEnvironment(
                    undefined=SilentUndefined, finalize=struckdown_finalize
                )
                rendered = env.from_string(data["system_template"]).render(
                    **accumulated_context
                )
                batch_globals.append(rendered)
            if data["header_template"]:
                env = ImmutableSandboxedEnvironment(
                    undefined=SilentUndefined, finalize=struckdown_finalize
                )
                rendered = env.from_string(data["header_template"]).render(
                    **accumulated_context
                )
                batch_header_globals.append(rendered)

        tasks = [
            process_single_segment(
                seg_idx,
                accumulated_context,
                batch_globals,
                batch_header_globals,
            )
            for seg_idx in batch
        ]

        try:
            results = await asyncio.gather(*tasks)

            # Process results in segment order for deterministic output
            for seg_idx, result, sys_tpl, hdr_tpl in sorted(
                results, key=lambda x: x[0]
            ):
                final.update(result.results)

                # Accumulate system/header messages
                if sys_tpl:
                    env = ImmutableSandboxedEnvironment(
                        undefined=SilentUndefined, finalize=struckdown_finalize
                    )
                    rendered_system = env.from_string(sys_tpl).render(
                        **accumulated_context
                    )
                    accumulated_globals.append(rendered_system)
                if hdr_tpl:
                    env = ImmutableSandboxedEnvironment(
                        undefined=SilentUndefined, finalize=struckdown_finalize
                    )
                    rendered_header = env.from_string(hdr_tpl).render(
                        **accumulated_context
                    )
                    accumulated_header_globals.append(rendered_header)

                # Update context
                for key, seg_result in result.results.items():
                    escaped_value, _ = escape_struckdown_syntax(
                        seg_result.output, var_name=key
                    )
                    accumulated_context[key] = escaped_value

        except Exception as e:
            logger.error(f"Batch {batch} error: {e}")
            raise

    logger.debug(f"\n\n{LC.GREEN}Chatter Response: {final.response}{LC.RESET}\n\n")
    return final


async def chatter_async(
    multipart_prompt: str,
    context: Union[dict, List[dict]] = {},
    *,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
    max_concurrent: Optional[int] = None,
    on_complete: Optional[callable] = None,
) -> Union[ChatterResult, List[ChatterResult]]:
    """
    Process a struckdown template with one or more contexts.

    Accepts either a single context dict or a list of context dicts.
    When given a list, processes all contexts in parallel with concurrency control.

    Examples:
        # Single context - returns ChatterResult
        result = await chatter_async("tell a joke [[joke]]", {"topic": "cats"})

        # Multiple contexts - returns List[ChatterResult]
        results = await chatter_async(
            "tell a joke about {{topic}} [[joke]]",
            [{"topic": "cats"}, {"topic": "dogs"}]
        )

    Args:
        multipart_prompt: Struckdown template string
        context: Single context dict OR list of context dicts
        model: LLM configuration (default: from environment)
        credentials: API credentials (default: from environment)
        extra_kwargs: Additional LLM parameters
        template_path: Path to template file (for includes)
        include_paths: Additional directories to search for <include> files
        strict_undefined: If True, raise error when template variables not found
        max_concurrent: Maximum concurrent requests when processing list (default: SD_MAX_CONCURRENCY)
        on_complete: Optional callback(index, result) called after each completion (list mode only)

    Returns:
        ChatterResult for single context, List[ChatterResult] for list of contexts.
    """
    if isinstance(context, list):
        # Multiple contexts - process in parallel
        sem = anyio.Semaphore(max_concurrent) if max_concurrent else get_llm_semaphore()
        results: List[ChatterResult | Exception] = [None] * len(context)

        async with anyio.create_task_group() as tg:
            for idx, ctx in enumerate(context):

                async def run_and_store(index=idx, context_item=ctx):
                    async with sem:
                        try:
                            result = await _chatter_single_async(
                                multipart_prompt=multipart_prompt,
                                context=context_item,
                                model=model,
                                credentials=credentials,
                                extra_kwargs=extra_kwargs,
                                template_path=template_path,
                                include_paths=include_paths,
                                strict_undefined=strict_undefined,
                            )
                            results[index] = result
                            if on_complete:
                                on_complete(index, result)
                        except Exception as e:
                            logger.error(f"Error processing context {index}: {e}")
                            results[index] = e
                            if on_complete:
                                on_complete(index, e)

                tg.start_soon(run_and_store)

        return results
    else:
        # Single context
        return await _chatter_single_async(
            multipart_prompt=multipart_prompt,
            context=context,
            model=model,
            credentials=credentials,
            extra_kwargs=extra_kwargs,
            template_path=template_path,
            include_paths=include_paths,
            strict_undefined=strict_undefined,
        )


def chatter(
    multipart_prompt: str,
    context: Union[dict, List[dict]] = {},
    *,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
    max_concurrent: Optional[int] = None,
    on_complete: Optional[callable] = None,
) -> Union[ChatterResult, List[ChatterResult]]:
    """Synchronous wrapper for chatter_async. Accepts single dict or list of dicts."""
    return anyio.run(
        partial(
            chatter_async,
            multipart_prompt,
            context,
            model=model,
            credentials=credentials,
            extra_kwargs=extra_kwargs,
            template_path=template_path,
            include_paths=include_paths,
            strict_undefined=strict_undefined,
            max_concurrent=max_concurrent,
            on_complete=on_complete,
        )
    )


async def chatter_incremental_async(
    multipart_prompt: str,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    context={},
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
) -> AsyncGenerator[IncrementalEvent, None]:
    """
    Process a struckdown template, yielding events as slots complete.

    This is the incremental (generator) version of chatter_async. It yields
    events after each slot completion, allowing consumers to display progress
    in real-time. Independent segments are processed in parallel.

    Example:
        async for event in chatter_incremental_async("tell a joke [[joke]]"):
            if event.type == "slot_completed":
                print(f"{event.slot_key}: {event.result.output}")

    Args:
        multipart_prompt: Struckdown template string
        model: LLM configuration (default: from environment)
        credentials: API credentials (default: from environment)
        context: Initial context variables
        extra_kwargs: Additional LLM parameters
        template_path: Path to template file (for includes)
        include_paths: Additional directories to search for <include> files
        strict_undefined: If True, raise error when template variables not found

    Yields:
        IncrementalEvent objects:
        - SlotCompleted: after each slot is filled
        - CheckpointReached: after a <checkpoint> boundary
        - ProcessingComplete: final event with aggregated ChatterResult
        - ProcessingError: if an error occurs (includes partial results)
    """
    import asyncio

    from .segment_processor import (extract_header_message,
                                    extract_system_message)

    if model is None:
        model = LLM()

    logger.debug(
        f"\n\n{LC.ORANGE}Chatter Incremental Prompt: {multipart_prompt}{LC.RESET}\n\n"
    )

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

    # EXECUTION TIME: Process segments in batches (parallel within each batch)
    all_results = ChatterResult()
    accumulated_context = context.copy()

    # Track global system messages (persist across checkpoints)
    accumulated_globals: List[str] = []
    # Track global header messages (persist across checkpoints, sent as user role)
    accumulated_header_globals: List[str] = []

    # Pre-extract system/header messages and analyze each segment
    segment_data = []
    for seg_idx, (raw_segment_text, segment_name) in enumerate(raw_segments):
        system_template, body_after_system = extract_system_message(raw_segment_text)
        header_template, body_template = extract_header_message(body_after_system)
        analysis = analyze_template(body_template)
        segment_data.append(
            {
                "idx": seg_idx,
                "name": segment_name,
                "system_template": system_template,
                "header_template": header_template,
                "body_template": body_template,
                "analysis": analysis,
                "has_slots": len(analysis.slots) > 0,
            }
        )

    # Build dependency graph from raw segments (only those with slots)
    slots_by_raw_idx = {}
    for data in segment_data:
        if data["has_slots"]:
            slots_by_raw_idx[data["idx"]] = {s.key for s in data["analysis"].slots}

    # Build dependency graph
    def get_referenced_vars(body: str) -> set:
        """Get all variable references from template body (Jinja + action params)."""
        return extract_jinja_variables(body) | extract_slot_variable_refs(body)

    def find_dependencies(raw_idx: int, referenced_vars: set) -> set:
        """Find earlier segments that define variables this segment references."""
        return {
            earlier_idx
            for earlier_idx in range(raw_idx)
            if earlier_idx in slots_by_raw_idx
            and referenced_vars & slots_by_raw_idx[earlier_idx]
        }

    raw_dependencies = {
        raw_idx: find_dependencies(
            raw_idx, get_referenced_vars(segment_data[raw_idx]["body_template"])
        )
        for raw_idx in slots_by_raw_idx
    }

    # Build execution plan
    def get_raw_execution_plan():
        remaining = set(raw_dependencies.keys())
        plan = []
        while remaining:
            ready = {
                idx
                for idx in remaining
                if all(dep not in remaining for dep in raw_dependencies[idx])
            }
            if not ready and remaining:
                logger.warning(f"Circular dependency in segments: {remaining}")
                plan.extend([[idx] for idx in remaining])
                break
            plan.append(sorted(ready))
            remaining -= ready
        return plan

    execution_plan = get_raw_execution_plan()
    logger.debug(f"Incremental execution plan (raw indices): {execution_plan}")

    # Process segments without slots first (they just accumulate system/header messages)
    for seg_idx, data in enumerate(segment_data):
        if not data["has_slots"]:
            if data["system_template"]:
                env = ImmutableSandboxedEnvironment(
                    undefined=SilentUndefined, finalize=struckdown_finalize
                )
                rendered_system = env.from_string(data["system_template"]).render(
                    **accumulated_context
                )
                accumulated_globals.append(rendered_system)
            if data["header_template"]:
                env = ImmutableSandboxedEnvironment(
                    undefined=SilentUndefined, finalize=struckdown_finalize
                )
                rendered_header = env.from_string(data["header_template"]).render(
                    **accumulated_context
                )
                accumulated_header_globals.append(rendered_header)

    async def process_segment_collect_events(
        seg_idx: int,
        ctx_snapshot: dict,
        globals_snapshot: list,
        header_globals_snapshot: list,
    ):
        """Process a segment and collect all events."""
        data = segment_data[seg_idx]
        body_template = data["body_template"]

        local_globals = globals_snapshot.copy()
        local_header_globals = header_globals_snapshot.copy()

        if data["system_template"]:
            env = ImmutableSandboxedEnvironment(
                undefined=SilentUndefined, finalize=struckdown_finalize
            )
            rendered_system = env.from_string(data["system_template"]).render(
                **ctx_snapshot
            )
            local_globals.append(rendered_system)

        if data["header_template"]:
            env = ImmutableSandboxedEnvironment(
                undefined=SilentUndefined, finalize=struckdown_finalize
            )
            rendered_header = env.from_string(data["header_template"]).render(
                **ctx_snapshot
            )
            local_header_globals.append(rendered_header)

        events = []
        async for event in process_segment_with_delta_incremental(
            body_template,
            ctx_snapshot.copy(),
            model,
            credentials,
            analysis=data["analysis"],
            global_system_messages=local_globals,
            global_header_messages=local_header_globals,
            segment_index=seg_idx,
            strict_undefined=strict_undefined,
            **(extra_kwargs or {}),
        ):
            events.append(event)

        return (
            seg_idx,
            events,
            data["system_template"],
            data["header_template"],
            data["name"],
        )

    try:
        # Process batches according to execution plan
        for batch in execution_plan:
            if len(batch) > 1:
                logger.debug(f"Processing {len(batch)} segments in parallel: {batch}")

            # For parallel batches, pre-collect system/header from all segments in the batch
            # so all segments see globals from earlier-indexed segments in the same batch
            batch_globals = accumulated_globals.copy()
            batch_header_globals = accumulated_header_globals.copy()

            for seg_idx in sorted(batch):
                data = segment_data[seg_idx]
                if data["system_template"]:
                    env = ImmutableSandboxedEnvironment(
                        undefined=SilentUndefined, finalize=struckdown_finalize
                    )
                    rendered = env.from_string(data["system_template"]).render(
                        **accumulated_context
                    )
                    batch_globals.append(rendered)
                if data["header_template"]:
                    env = ImmutableSandboxedEnvironment(
                        undefined=SilentUndefined, finalize=struckdown_finalize
                    )
                    rendered = env.from_string(data["header_template"]).render(
                        **accumulated_context
                    )
                    batch_header_globals.append(rendered)

            # Launch all segment tasks in parallel with batch-accumulated globals
            tasks = [
                process_segment_collect_events(
                    seg_idx,
                    accumulated_context,
                    batch_globals,
                    batch_header_globals,
                )
                for seg_idx in batch
            ]

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # Yield events in segment order for determinism
            for seg_idx, events, sys_tpl, hdr_tpl, seg_name in sorted(
                results, key=lambda x: x[0]
            ):
                for event in events:
                    all_results[event.slot_key] = event.result
                    yield event

                # Yield checkpoint event
                yield CheckpointReached(
                    segment_index=seg_idx,
                    segment_name=seg_name,
                    accumulated_results=dict(all_results.results),
                )

                # Accumulate system/header messages
                if sys_tpl:
                    env = ImmutableSandboxedEnvironment(
                        undefined=SilentUndefined, finalize=struckdown_finalize
                    )
                    rendered_system = env.from_string(sys_tpl).render(
                        **accumulated_context
                    )
                    accumulated_globals.append(rendered_system)
                if hdr_tpl:
                    env = ImmutableSandboxedEnvironment(
                        undefined=SilentUndefined, finalize=struckdown_finalize
                    )
                    rendered_header = env.from_string(hdr_tpl).render(
                        **accumulated_context
                    )
                    accumulated_header_globals.append(rendered_header)

                # Update context from this segment's results
                for event in events:
                    escaped_value, _ = escape_struckdown_syntax(
                        event.result.output, var_name=event.slot_key
                    )
                    accumulated_context[event.slot_key] = escaped_value

    except Exception as e:
        logger.error(f"Incremental processing error: {e}")
        yield ProcessingError(
            segment_index=0,
            slot_key=None,
            error_message=str(e),
            partial_results=all_results,
        )
        return

    logger.debug(
        f"\n\n{LC.GREEN}Chatter Incremental Complete: {all_results.response}{LC.RESET}\n\n"
    )
    yield ProcessingComplete(result=all_results, early_termination=False)


def chatter_incremental(
    multipart_prompt: str,
    model: LLM = None,
    credentials: Optional[LLMCredentials] = None,
    context={},
    extra_kwargs=None,
    template_path: Optional[Path] = None,
    include_paths: Optional[List[Path]] = None,
    strict_undefined: bool = False,
) -> Generator[IncrementalEvent, None, None]:
    """
    Synchronous wrapper for chatter_incremental_async.

    Note: This collects all events then yields them. For true incremental
    processing, use chatter_incremental_async() in an async context.
    """
    if model is None:
        model = LLM()

    async def collect():
        return [
            e
            async for e in chatter_incremental_async(
                multipart_prompt,
                model,
                credentials,
                context,
                extra_kwargs,
                template_path,
                include_paths,
                strict_undefined,
            )
        ]

    events = anyio.run(collect)
    yield from events


# Public API exports
__all__ = [
    # Version
    "__version__",
    # Main entry points
    "chatter",
    "chatter_async",
    "chatter_many",
    "chatter_many_async",
    "chatter_incremental",
    "chatter_incremental_async",
    "structured_chat",
    "get_embedding",
    "get_embedding_async",
    "get_cross_encoder_scores",
    "EmbeddingResult",
    "EmbeddingResultList",
    # Incremental events
    "IncrementalEvent",
    "SlotCompleted",
    "CheckpointReached",
    "ProcessingComplete",
    "ProcessingError",
    # Results
    "SegmentResult",
    "ChatterResult",
    "CostSummary",
    # LLM
    "LLM",
    "LLMCredentials",
    "LLMConfig",
    # Errors (new names)
    "Safe",
    "TemplateError",
    "LLMError",
    "ContentFilterError",
    "RateLimitError",
    "ContextWindowError",
    "AuthError",
    "BadRequestError",
    "ConnectionError",
    "FetchError",
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
    "progress_tracking",
    "clear_cache",
    # Validation
    "ParsedOptions",
    "parse_options",
    "validate_number_constraints",
    # Internal (for advanced use)
    "SegmentDependencyGraph",
    "merge_contexts",
    "SilentUndefined",
    "make_strict_undefined",
    "struckdown_finalize",
    "LC",
]
