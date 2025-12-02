"""Backwards compatibility module -- redirects to struckdown.actions.loader.

This module is deprecated. Please use struckdown.actions instead:
    from struckdown.actions import load_actions, discover_actions
"""

import warnings

from struckdown.actions import (
    load_action_file as load_tools_file,
    load_action_directory as load_tools_directory,
    load_actions as load_tools,
    discover_actions as discover_tools,
)

# re-export with old names for backwards compatibility
__all__ = ["load_tools_file", "load_tools_directory", "load_tools", "discover_tools"]

# emit deprecation warning on import
warnings.warn(
    "struckdown.tools_loader is deprecated. Use struckdown.actions instead: "
    "from struckdown.actions import load_actions, discover_actions",
    DeprecationWarning,
    stacklevel=2,
)
