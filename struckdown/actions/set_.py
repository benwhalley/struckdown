"""Built-in @set action for setting variables without LLM calls."""

from . import Actions


@Actions.register("set", on_error="propagate", default_save=True)
def set_action(context, value="", **kwargs):
    """Built-in action to set a variable without an LLM call.

    Usage: [[@set:varname|newvalue]]
    or: [[@set:varname|value=some_value]]
    or: [[@set:varname|value={{other_var}}]]

    This is useful for creating dependencies between segments without extra API calls.
    """
    return ""
