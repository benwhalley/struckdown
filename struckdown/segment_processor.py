"""Delta-based segment processing with conditional re-rendering.

Processes template segments by re-rendering after each slot completion,
enabling Jinja conditionals to react to filled values.
"""

import asyncio
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from jinja2.sandbox import ImmutableSandboxedEnvironment

from .jinja_analysis import (
    TemplateAnalysis,
    analyze_template,
    find_slots_with_positions,
)
from .jinja_utils import SilentUndefined
from .parsing import parser_with_state, PromptPart

logger = logging.getLogger(__name__)


def _strip_html_comments(text: str) -> str:
    """Remove HTML comments from text."""
    return re.sub(r'<!--(.|\n)*?-->', '', text)


def split_content_by_role(content: str) -> List[Dict[str, str]]:
    """Split content by <user> and <assistant> tags into separate messages.

    Args:
        content: Content string potentially containing role tags

    Returns:
        List of {"role": "user"|"assistant", "content": "..."} dicts.
        Unmarked content defaults to "user" role.

    Example:
        >>> split_content_by_role("Hello <assistant>Hi there</assistant> Bye")
        [{"role": "user", "content": "Hello"},
         {"role": "assistant", "content": "Hi there"},
         {"role": "user", "content": "Bye"}]
    """
    # Pattern matches <user>...</user> or <assistant>...</assistant>
    pattern = re.compile(
        r'<(user|assistant)>(.*?)</\1>',
        re.DOTALL | re.IGNORECASE
    )

    messages = []
    last_end = 0

    for match in pattern.finditer(content):
        # Add any content before this tag as user
        before = content[last_end:match.start()].strip()
        if before:
            messages.append({"role": "user", "content": before})

        # Add tagged content with its role
        role = match.group(1).lower()
        tagged_content = match.group(2).strip()
        if tagged_content:
            messages.append({"role": role, "content": tagged_content})

        last_end = match.end()

    # Add any remaining content as user
    remaining = content[last_end:].strip()
    if remaining:
        messages.append({"role": "user", "content": remaining})

    return messages


def extract_system_message(template_str: str) -> Tuple[str, str]:
    """Extract <system>...</system> content from template.

    Args:
        template_str: Raw template string

    Returns:
        Tuple of (system_message, template_without_system)
    """
    # Strip HTML comments first to avoid matching tags inside comments
    clean_template = _strip_html_comments(template_str)

    # Pattern to match <system>...</system> blocks
    pattern = re.compile(r'<system[^>]*>(.*?)</system>', re.DOTALL | re.IGNORECASE)

    system_parts = []
    remaining = clean_template

    for match in pattern.finditer(clean_template):
        system_parts.append(match.group(1).strip())
        remaining = remaining.replace(match.group(0), '', 1)

    system_message = '\n\n'.join(system_parts) if system_parts else ''
    return system_message, remaining.strip()


def extract_header_message(template_str: str) -> Tuple[str, str]:
    """Extract <header>...</header> content from template.

    Args:
        template_str: Raw template string

    Returns:
        Tuple of (header_message, template_without_header)
    """
    # Strip HTML comments first to avoid matching tags inside comments
    clean_template = _strip_html_comments(template_str)

    # Pattern to match <header>...</header> blocks
    pattern = re.compile(r'<header[^>]*>(.*?)</header>', re.DOTALL | re.IGNORECASE)

    header_parts = []
    remaining = clean_template

    for match in pattern.finditer(clean_template):
        header_parts.append(match.group(1).strip())
        remaining = remaining.replace(match.group(0), '', 1)

    header_message = '\n\n'.join(header_parts) if header_parts else ''
    return header_message, remaining.strip()


def build_slot_info_map(body_template: str) -> Dict[str, PromptPart]:
    """Parse template and build map of slot_key -> PromptPart.

    Uses the main Lark parser to get authoritative slot information.

    Args:
        body_template: Template body (without system tags)

    Returns:
        Dict mapping slot key to PromptPart with return_type, options, etc.
    """
    try:
        lark, _ = parser_with_state()
        parsed_segments = lark.parse(body_template)

        # NamedSegment is an OrderedDict of (key, PromptPart) pairs
        slot_map = {}
        for segment in parsed_segments:
            for key, part in segment.items():
                if key:
                    slot_map[key] = part

        return slot_map
    except ValueError:
        # Re-raise ValueError (e.g. unknown action errors) - these are user errors
        raise
    except Exception as e:
        logger.warning(f"Failed to parse template for slot info: {e}")
        return {}


def render_template(
    template_str: str,
    context: Dict[str, Any],
    strict_undefined: bool = False,
) -> str:
    """Render Jinja template with context.

    Args:
        template_str: Template string
        context: Variable context
        strict_undefined: If True, raise error for undefined variables

    Returns:
        Rendered string
    """
    from jinja2 import StrictUndefined
    undefined_class = StrictUndefined if strict_undefined else SilentUndefined
    env = ImmutableSandboxedEnvironment(undefined=undefined_class)
    template = env.from_string(template_str)
    return template.render(**context)


async def _process_together_group(
    together_group: str,
    slot_info_map: Dict[str, PromptPart],
    unfilled_slots: List[Tuple[str, int, int, str]],
    rendered: str,
    messages: List[Dict[str, str]],
    accumulated_context: Dict[str, Any],
    filled_slots: Dict[str, Any],
    llm,
    credentials,
    segment_index: int,
    extra_kwargs: dict,
) -> AsyncGenerator:
    """Process all slots in a together group simultaneously.

    Slots in a together group are evaluated in parallel with the same
    initial context -- they don't see each other's results.

    Args:
        together_group: The group ID to process
        slot_info_map: Map of slot_key -> PromptPart
        unfilled_slots: List of (key, start, end, inner) for unfilled slots
        rendered: Current rendered template string
        messages: Snapshot of messages at start of together block
        accumulated_context: Current context (will be updated with results)
        filled_slots: Dict of filled slots (will be updated)
        llm: LLM configuration
        credentials: API credentials
        segment_index: Index of this segment
        extra_kwargs: Additional LLM parameters

    Yields:
        SlotCompleted events for each slot in the group
    """
    from .results import SegmentResult, get_progress_callback, get_run_id
    from .incremental import SlotCompleted
    from .llm import structured_chat
    from .jinja_utils import escape_struckdown_syntax
    from struckdown.actions import MessageList
    import anyio

    # Find all unfilled slots in this together group
    group_slots = [
        (key, start, end, inner)
        for key, start, end, inner in unfilled_slots
        if slot_info_map.get(key) and slot_info_map[key].together_group == together_group
    ]

    if not group_slots:
        return

    # Sort by position in the rendered text
    group_slots_sorted = sorted(group_slots, key=lambda x: x[1])

    logger.debug(f"Processing together group {together_group} with {len(group_slots_sorted)} slots")

    # Find the start position for content calculation
    # (position after the last filled slot, or 0)
    all_slots_in_render = find_slots_with_positions(rendered)
    filled_in_render = [
        (key, start, end)
        for key, start, end, _ in all_slots_in_render
        if key in filled_slots
    ]
    base_position = max((end for _, _, end in filled_in_render), default=0)

    # Find the <together> tag position to split shared preamble from slot-specific content
    # The preamble is everything from base_position up to (but not including) <together>
    together_tag_pos = rendered.find('<together>', base_position)
    if together_tag_pos == -1:
        # Fallback if tag not found (shouldn't happen, but be safe)
        together_tag_pos = base_position
        shared_preamble = ""
        together_content_start = base_position
    else:
        shared_preamble = rendered[base_position:together_tag_pos]
        # Find end of <together> tag (position after the tag and any following whitespace)
        together_tag_end = rendered.find('>', together_tag_pos) + 1
        # Skip any whitespace/newline after the tag
        while together_tag_end < len(rendered) and rendered[together_tag_end] in ' \t\n':
            together_tag_end += 1
        together_content_start = together_tag_end

    async def process_single_slot(
        slot_key: str,
        slot_start: int,
        content_before: str,
    ) -> Tuple[str, Any, Any, float, PromptPart, str, List[Dict[str, str]]]:
        """Process a single slot within the together group."""
        slot_info = slot_info_map[slot_key]
        return_type = slot_info.return_type
        is_action = slot_info.is_function or hasattr(return_type, '_executor')

        # Build messages for this slot: base messages + content before this slot
        slot_messages = messages.copy()
        if content_before.strip():
            role_segments = split_content_by_role(content_before)
            slot_messages.extend(role_segments)

        start_time = time.monotonic()

        if is_action and hasattr(return_type, '_executor'):
            # Execute action function
            logger.debug(f"Together group action: {slot_info.action_type}:{slot_key}")
            res, completion_obj = return_type._executor(
                accumulated_context, content_before, **extra_kwargs
            )
        else:
            # Call LLM
            logger.debug(f"Together: STARTING LLM call for {slot_key}")
            res, completion_obj = await anyio.to_thread.run_sync(
                lambda rt=return_type, msgs=slot_messages: structured_chat(
                    messages=msgs,
                    return_type=rt,
                    llm=llm,
                    credentials=credentials,
                    extra_kwargs=extra_kwargs or {},
                ),
                abandon_on_cancel=True,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(f"Together: COMPLETED LLM call for {slot_key} ({elapsed_ms:.0f}ms)")
        return slot_key, res, completion_obj, elapsed_ms, slot_info, content_before, slot_messages

    # Build tasks for parallel execution
    # Each slot gets: shared_preamble + its own question (not other slots' questions)
    tasks = []
    prev_end = together_content_start

    for key, start, end, inner in group_slots_sorted:
        # Slot-specific content is from prev_end to this slot's start
        slot_specific = rendered[prev_end:start]
        # Full content is shared preamble + slot-specific question
        content_before = shared_preamble + slot_specific
        tasks.append(process_single_slot(key, start, content_before))
        prev_end = end

    # Execute all slots in parallel, respecting shared LLM concurrency limit
    from .llm import get_llm_semaphore
    sem = get_llm_semaphore()

    async def run_with_semaphore(task_coro):
        async with sem:
            return await task_coro

    results = await asyncio.gather(*[run_with_semaphore(t) for t in tasks])

    # Process results and yield events
    for slot_key, res, completion_obj, elapsed_ms, slot_info, content_before, slot_messages in results:
        extracted_value = res.response if hasattr(res, 'response') else res

        # Build segment result
        segment_result = SegmentResult(
            name=slot_key,
            output=extracted_value,
            completion=completion_obj,
            prompt=content_before,
            action=slot_info.action_type,
            options=slot_info.options if slot_info.options else None,
            messages=slot_messages,
        )

        # Determine if result was cached
        current_run_id = get_run_id()
        was_cached = False
        if completion_obj and hasattr(completion_obj, 'get'):
            was_cached = completion_obj.get("_run_id") != current_run_id

        # Yield the event
        yield SlotCompleted(
            segment_index=segment_index,
            slot_key=slot_key,
            result=segment_result,
            elapsed_ms=elapsed_ms,
            was_cached=was_cached,
        )

        # Fire progress callback
        callback = get_progress_callback()
        if callback:
            callback()

        # Update context and filled_slots
        escaped_value, _ = escape_struckdown_syntax(extracted_value, var_name=slot_key)
        accumulated_context[slot_key] = escaped_value
        filled_slots[slot_key] = escaped_value


async def process_segment_with_delta_incremental(
    template_str: str,
    initial_context: Dict[str, Any],
    llm,
    credentials=None,
    analysis: Optional[TemplateAnalysis] = None,
    global_system_messages: Optional[List[str]] = None,
    global_header_messages: Optional[List[str]] = None,
    segment_index: int = 0,
    strict_undefined: bool = False,
    **extra_kwargs,
) -> AsyncGenerator:
    """Process a template segment, yielding SlotCompleted events as each slot is filled.

    This is the incremental (generator) version of process_segment_with_delta.
    It yields SlotCompleted events after each slot completion, allowing consumers
    to display progress in real-time.

    Args:
        template_str: Raw template string (may contain <system> tags)
        initial_context: Initial variable context
        llm: LLM configuration
        credentials: API credentials
        analysis: Pre-computed template analysis (optional)
        global_system_messages: System messages from previous segments (globals persist)
        global_header_messages: Header messages from previous segments (sent as user role)
        segment_index: Index of this segment (for event metadata)
        **extra_kwargs: Additional LLM parameters

    Yields:
        SlotCompleted events as each slot is filled
    """
    # Import here to avoid circular imports
    from .results import SegmentResult, get_progress_callback, get_run_id
    from .incremental import SlotCompleted
    from .llm import structured_chat
    from .jinja_utils import escape_struckdown_syntax
    import anyio

    # template_str is the body only (system already extracted by caller)
    body_template = template_str

    # Parse template once with main parser to get slot info (return types, options, etc.)
    slot_info_map = build_slot_info_map(body_template)

    # Analyze template for conditional dependencies if not provided
    if analysis is None:
        analysis = analyze_template(body_template)

    # State
    filled_slots: Dict[str, Any] = {}
    accumulated_context = initial_context.copy()
    messages: List[Dict[str, str]] = []
    last_slot_end = 0  # Position after the last filled slot placeholder
    processed_together_groups: Set[str] = set()  # Track processed parallel groups

    # Add system message from globals (caller passes accumulated globals)
    if global_system_messages:
        combined_system = "\n\n".join(global_system_messages)
        messages.append({"role": "system", "content": combined_system})

    # Add header message from globals (sent as user role, appears after system)
    if global_header_messages:
        combined_header = "\n\n".join(global_header_messages)
        messages.append({"role": "user", "content": combined_header})

    # Initial render of body
    rendered = render_template(body_template, accumulated_context, strict_undefined)

    while True:
        # Find all slots in current render
        all_slots = find_slots_with_positions(rendered)

        # Find first unfilled slot
        unfilled_slots = [
            (key, start, end, inner)
            for key, start, end, inner in all_slots
            if key not in filled_slots
        ]

        if not unfilled_slots:
            # No more slots - we're done
            break

        slot_key, slot_start, slot_end, slot_inner = unfilled_slots[0]

        # Look up slot info from pre-parsed map
        slot_info = slot_info_map.get(slot_key)
        if slot_info is None:
            logger.warning(f"Slot {slot_key} not found in parsed info, skipping")
            filled_slots[slot_key] = None
            continue

        # Check if this slot is part of a together group
        together_group = slot_info.together_group

        if together_group and together_group not in processed_together_groups:
            # Process entire together group at once (parallel execution)
            processed_together_groups.add(together_group)

            async for event in _process_together_group(
                together_group=together_group,
                slot_info_map=slot_info_map,
                unfilled_slots=unfilled_slots,
                rendered=rendered,
                messages=messages.copy(),  # Snapshot messages for parallel slots
                accumulated_context=accumulated_context,
                filled_slots=filled_slots,
                llm=llm,
                credentials=credentials,
                segment_index=segment_index,
                extra_kwargs=extra_kwargs,
            ):
                yield event

            # Re-render after together block
            needs_rerender = any(
                analysis.triggers_rerender(k)
                for k in filled_slots
                if slot_info_map.get(k) and slot_info_map[k].together_group == together_group
            )
            if needs_rerender:
                rendered = render_template(body_template, accumulated_context, strict_undefined)
                last_slot_end = 0

            continue

        # Skip slots from already-processed together groups
        if together_group and together_group in processed_together_groups:
            continue

        # --- Sequential processing for non-together slots ---

        # Find delta start position
        # If we have filled slots, find where the last one ends in this render
        if filled_slots:
            filled_in_render = [
                (key, start, end)
                for key, start, end, _ in all_slots
                if key in filled_slots
            ]
            if filled_in_render:
                # Use the end position of the last filled slot (by position)
                last_slot_end = max(end for _, _, end in filled_in_render)

        # Content between last filled slot and next unfilled slot
        content_before = rendered[last_slot_end:slot_start]

        # Add messages for content before this slot
        # Split by <user> and <assistant> tags to handle role markers
        if content_before.strip():
            role_segments = split_content_by_role(content_before)
            messages.extend(role_segments)

        # Get return type from parsed slot info
        return_type = slot_info.return_type

        # Check if this is an action (function call) vs LLM completion
        is_action = slot_info.is_function or hasattr(return_type, '_executor')

        # Track timing
        start_time = time.monotonic()

        if is_action and hasattr(return_type, '_executor'):
            # Execute action function
            logger.debug(f"Executing action: {slot_info.action_type}:{slot_key}")
            res, completion_obj = return_type._executor(
                accumulated_context, content_before, **extra_kwargs
            )
        else:
            # Call LLM
            logger.debug(f"LLM completion: {slot_info.action_type}:{slot_key}")
            res, completion_obj = await anyio.to_thread.run_sync(
                lambda: structured_chat(
                    messages=messages.copy(),
                    return_type=return_type,
                    llm=llm,
                    credentials=credentials,
                    extra_kwargs=extra_kwargs or {},
                ),
                abandon_on_cancel=True,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Extract response value
        extracted_value = res.response if hasattr(res, 'response') else res

        # Import MessageList to check for multi-message returns
        from struckdown.actions import MessageList

        # Handle MessageList (multi-message) vs single value returns
        if isinstance(extracted_value, MessageList):
            # Action returned multiple messages - add each with its role
            for msg in extracted_value:
                messages.append({"role": msg["role"], "content": msg["content"]})
            # For context storage, use string representation
            completion_str = str(extracted_value)
        else:
            # Single message - use registered role for actions, "assistant" for LLM
            completion_str = str(extracted_value)
            action_role = getattr(return_type, '_role', 'assistant')
            messages.append({"role": action_role, "content": completion_str})

        # Build result
        segment_result = SegmentResult(
            name=slot_key,
            output=extracted_value,
            completion=completion_obj,
            prompt=content_before,
            action=slot_info.action_type,
            options=slot_info.options if slot_info.options else None,
            messages=messages.copy(),
        )

        # Determine if result was cached
        current_run_id = get_run_id()
        was_cached = False
        if completion_obj and hasattr(completion_obj, 'get'):
            was_cached = completion_obj.get("_run_id") != current_run_id

        # Yield the event
        yield SlotCompleted(
            segment_index=segment_index,
            slot_key=slot_key,
            result=segment_result,
            elapsed_ms=elapsed_ms,
            was_cached=was_cached,
        )

        # Fire progress callback for backward compatibility
        callback = get_progress_callback()
        if callback:
            callback()

        # Update context with escaped value
        escaped_value, _ = escape_struckdown_syntax(extracted_value, var_name=slot_key)
        accumulated_context[slot_key] = escaped_value
        filled_slots[slot_key] = escaped_value

        # Update last_slot_end for next iteration (if no re-render)
        last_slot_end = slot_end

        # Re-render if this slot triggers conditional changes
        if analysis.triggers_rerender(slot_key):
            logger.debug(f"Slot {slot_key} triggers re-render")
            rendered = render_template(body_template, accumulated_context, strict_undefined)
            # Reset last_slot_end since positions changed
            # Will be recalculated from filled_slots on next iteration
            last_slot_end = 0


async def process_segment_with_delta(
    template_str: str,
    initial_context: Dict[str, Any],
    llm,
    credentials=None,
    analysis: Optional[TemplateAnalysis] = None,
    global_system_messages: Optional[List[str]] = None,
    global_header_messages: Optional[List[str]] = None,
    strict_undefined: bool = False,
    **extra_kwargs,
):
    """Process a template segment using delta-based re-rendering.

    This is the non-incremental version that returns a ChatterResult after
    all slots are filled. It internally consumes the incremental generator.

    Args:
        template_str: Raw template string (may contain <system> tags)
        initial_context: Initial variable context
        llm: LLM configuration
        credentials: API credentials
        analysis: Pre-computed template analysis (optional)
        global_system_messages: System messages from previous segments (globals persist)
        global_header_messages: Header messages from previous segments (sent as user role)
        **extra_kwargs: Additional LLM parameters

    Returns:
        ChatterResult with all slot completions
    """
    from .results import ChatterResult

    results = ChatterResult()

    async for event in process_segment_with_delta_incremental(
        template_str=template_str,
        initial_context=initial_context,
        llm=llm,
        credentials=credentials,
        analysis=analysis,
        global_system_messages=global_system_messages,
        global_header_messages=global_header_messages,
        segment_index=0,
        strict_undefined=strict_undefined,
        **extra_kwargs,
    ):
        results[event.slot_key] = event.result

    return results
