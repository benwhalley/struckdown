"""Built-in @history action for loading conversation history from file.

This action loads conversation turns from a text file and returns them as
a MessageList. It can be overridden in applications (like mindframe) to
provide history from a database instead.
"""

import logging
from pathlib import Path
from typing import Optional

from . import Actions, MessageList

logger = logging.getLogger(__name__)


@Actions.register("history", on_error="propagate", default_save=False)
def history_action(
    context: dict,
    filename: Optional[str] = None,
    first: str = "assistant_first",
    n: Optional[int] = None,
    role: Optional[str] = None,
    **kwargs,
) -> MessageList:
    """Load conversation history from a text file.

    Each non-empty line becomes a message, alternating between roles.
    Useful for testing prompts with canned conversation data.

    The filename can be:
    1. Passed directly: [[@history|conversation.txt]]
    2. Set via context['_history_file'] (from CLI --history option or chatter() call)

    Usage:
        [[@history]]                                    # uses context['_history_file']
        [[@history|conversation.txt]]                   # explicit file
        [[@history|conversation.txt,user_first]]        # user speaks first
        [[@history|n=5]]                                # last 5 turns only
        [[@history|role=assistant]]                     # assistant turns only
        [[@history|conversation.txt,n=3,role=user]]     # combined filters

    Args:
        context: Struckdown context dict (may contain '_history_file')
        filename: Path to text file (relative to cwd). If None, uses context['_history_file']
        first: Who speaks first - "assistant_first" (default) or "user_first"
        n: Only return last n turns (after role filtering)
        role: Filter to only messages with this role ("user", "assistant", or "system")
        **kwargs: Unknown filters are logged as warnings (allows mindframe-specific filters)

    Returns:
        MessageList with alternating user/assistant messages

    Example file (conversation.txt):
        Hello, how can I help you today?
        I'm feeling anxious about work
        That sounds difficult. Can you tell me more?

    With assistant_first, this becomes:
        assistant: Hello, how can I help you today?
        user: I'm feeling anxious about work
        assistant: That sounds difficult. Can you tell me more?
    """
    # Warn about unknown filters (e.g., step=xxyy from mindframe)
    for key, value in kwargs.items():
        logger.warning(f"@history: ignoring unknown filter '{key}={value}'")

    # Validate first parameter
    if first not in ("assistant_first", "user_first"):
        raise ValueError(f"first must be 'assistant_first' or 'user_first', got '{first}'")

    # Validate role filter if provided
    if role is not None and role not in ("user", "assistant", "system"):
        raise ValueError(f"role must be 'user', 'assistant', or 'system', got '{role}'")

    # Check for in-memory history first (from interactive mode)
    # This takes precedence over file-based history
    if "_history_messages" in context:
        messages = list(context["_history_messages"])  # Copy to avoid mutation
    else:
        # Determine filename: explicit param > context > error
        if filename is None:
            filename = context.get("_history_file")

        if filename is None:
            raise ValueError(
                "No history file specified. Either pass a filename to @history or "
                "set context['_history_file'] (via CLI --history or chatter() context)"
            )

        # Read file relative to cwd
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"History file not found: {filename}")

        content = filepath.read_text()
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        # Build messages with alternating roles
        messages = []
        roles = ["assistant", "user"] if first == "assistant_first" else ["user", "assistant"]

        for i, line in enumerate(lines):
            msg_role = roles[i % 2]
            messages.append({"role": msg_role, "content": line})

    # Apply role filter if specified
    if role is not None:
        messages = [m for m in messages if m["role"] == role]

    # Apply n filter (last n turns)
    if n is not None:
        messages = messages[-n:]

    return MessageList(messages)
