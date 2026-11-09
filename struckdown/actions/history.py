"""Built-in @history action: the conversation so far, from the context.

The turns come from ``context["_history_messages"]`` (a list of
``{"role", "content"}`` dicts), which the caller puts there: ``sd chat`` reads
``--history FILE`` with :func:`messages_from_text`, the playground sends the
chat it holds, an application passes its own. The action itself never reads
a file, so a template cannot point it at one. Applications (like mindframe)
can override it to fetch history from a database instead.
"""

import logging
from typing import Optional

from . import Actions, MessageList

logger = logging.getLogger(__name__)


def messages_from_text(text: str, first: str = "assistant_first") -> list[dict]:
    """One message per non-empty line, roles alternating from ``first``
    ("assistant_first" or "user_first")."""
    if first not in ("assistant_first", "user_first"):
        raise ValueError(f"first must be 'assistant_first' or 'user_first', got '{first}'")
    roles = ["assistant", "user"] if first == "assistant_first" else ["user", "assistant"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [{"role": roles[i % 2], "content": line} for i, line in enumerate(lines)]


@Actions.register("history", on_error="propagate", default_save=False)
def history_action(
    context: dict,
    n: Optional[int] = None,
    role: Optional[str] = None,
    **kwargs,
) -> MessageList:
    """The conversation in ``context["_history_messages"]``, optionally filtered.

    Usage:
        [[@history]]                       # every turn
        [[@history|n=5]]                   # last 5 turns only
        [[@history|role=assistant]]        # assistant turns only
        [[@history|n=3,role=user]]         # combined filters

    Args:
        context: Struckdown context dict; must contain '_history_messages'
        n: Only return last n turns (after role filtering)
        role: Filter to only messages with this role ("user", "assistant", or "system")
        **kwargs: Unknown filters are logged as warnings (allows mindframe-specific filters)
    """
    for key, value in kwargs.items():
        logger.warning(f"@history: ignoring unknown filter '{key}={value}'")

    if role is not None and role not in ("user", "assistant", "system"):
        raise ValueError(f"role must be 'user', 'assistant', or 'system', got '{role}'")

    if "_history_messages" not in context:
        raise ValueError(
            "No history: pass the conversation as context['_history_messages'] "
            "(sd chat --history FILE does this)"
        )
    messages = list(context["_history_messages"])  # copy to avoid mutation

    if role is not None:
        messages = [m for m in messages if m["role"] == role]
    if n is not None:
        messages = messages[-n:]
    return MessageList(messages)
