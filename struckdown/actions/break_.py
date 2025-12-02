"""Built-in @break action for early termination."""

from . import Actions


@Actions.register("break", on_error="propagate", default_save=True)
def break_action(context, message="", **kwargs):
    """Built-in action for early termination.

    Usage: [[@break|reason for breaking]]

    This stops execution of the current template and returns partial results.
    The message (after the pipe) is stored in context['_break_message'] and
    returned as the action output.
    """
    context["_break_requested"] = True
    context["_break_message"] = message
    return message
